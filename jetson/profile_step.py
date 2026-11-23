"""Why does PyTorch decode slow down in one step between 6K and 8K cached tokens? (set O)

For each cache length: prefill the first N tokens of data/corpus.txt (fp16 weights, sdpa attention, no
eviction), then run --steps single-token decode steps, the same forward pass model.generate() makes. Records
  * wall time per step (CUDA-synchronised) and the GPU kernel time per step (torch.profiler)
  * the GPU kernels and CPU-side ops that take the time, with call counts
  * CUDA memory: allocated, reserved, and the cache size
so the kernels and allocator calls at 6K can be compared with those at 8K.
--mode picks the decode path: "loop" (the forward pass alone, no attention mask), "mask" (the same loop with the
2-D attention mask generate() passes), "generate" (model.generate() itself, timed from the first token, as in
bench_evict.py), "genpad0" (generate() the way bench_evict.py called it up to v4: no attention mask
and pad_token_id=0, so every "!" token in the prompt is masked as padding). The prefill keeps only the last position's logits (logits_to_keep=1), as generate() does.

  python3 jetson/profile_step.py --model Qwen/Qwen2.5-0.5B-Instruct --contexts 4096,5120,6144,7168,8192
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parent.parent


class FirstToken(StoppingCriteria):
    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores, **kw):
        if self.t is None:
            torch.cuda.synchronize()
            self.t = time.monotonic()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def run_generate(model, ids, steps, pad0=False):
    first = FirstToken()
    kw = {'pad_token_id': 0} if pad0 else {'attention_mask': torch.ones_like(ids), 'pad_token_id': 0}
    out = model.generate(ids, max_new_tokens=steps + 1, min_new_tokens=steps + 1, do_sample=False,
                         stopping_criteria=StoppingCriteriaList([first]), return_dict_in_generate=True, **kw)
    torch.cuda.synchronize()
    wall = (time.monotonic() - first.t) / steps
    cache = out.past_key_values
    return {'wall_ms_per_step': wall * 1000, 'gpu_ms_per_step': None, 'top_gpu': [], 'top_cpu': [],
            'cuda_alloc_mib': torch.cuda.memory_allocated() / 2**20,
            'cuda_reserved_mib': torch.cuda.memory_reserved() / 2**20,
            'cache_mib': sum(l.keys.numel() * 2 + l.values.numel() * 2 for l in cache.layers) / 2**20}


@torch.no_grad()
def run(model, ids, steps, prof_steps, mask=False):
    cache = DynamicCache()
    am = torch.ones_like(ids) if mask else None
    out = model(ids, past_key_values=cache, use_cache=True, attention_mask=am, logits_to_keep=1)
    tok = out.logits[:, -1:].argmax(-1)
    pos = ids.shape[1]

    def step():
        nonlocal tok, pos, am
        kw = {}
        if mask:
            am = torch.cat([am, am.new_ones((1, 1))], dim=1)
            kw['attention_mask'] = am
        o = model(tok, past_key_values=cache, use_cache=True,
                  cache_position=torch.tensor([pos], device=ids.device), **kw)
        tok = o.logits[:, -1:].argmax(-1)
        pos += 1

    for _ in range(4):  # warm-up at this length
        step()
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    wall = (time.monotonic() - t0) / steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(prof_steps):
            step()
        torch.cuda.synchronize()
    ev = p.key_averages()
    gpu = sorted(((e.key, e.self_device_time_total / prof_steps / 1000, e.count / prof_steps) for e in ev
                  if e.self_device_time_total > 0), key=lambda x: -x[1])
    cpu = sorted(((e.key, e.self_cpu_time_total / prof_steps / 1000, e.count / prof_steps) for e in ev
                  if e.self_cpu_time_total > 0), key=lambda x: -x[1])
    cache_mib = sum(l.keys.numel() * 2 + l.values.numel() * 2 for l in cache.layers) / 2**20
    return {'wall_ms_per_step': wall * 1000, 'gpu_ms_per_step': sum(g[1] for g in gpu),
            'top_gpu': [(k, round(ms, 3), c) for k, ms, c in gpu[:12]],
            'top_cpu': [(k, round(ms, 3), c) for k, ms, c in cpu[:15]],
            'cuda_alloc_mib': torch.cuda.memory_allocated() / 2**20,
            'cuda_reserved_mib': torch.cuda.memory_reserved() / 2**20, 'cache_mib': cache_mib}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--contexts', default='4096,5120,6144,7168,8192')
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--prof-steps', type=int, default=8)
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--mode', choices=['loop', 'mask', 'generate', 'genpad0'], default='loop')
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map='cuda:0',
                                                 attn_implementation=args.attn).eval()
    ids_all = tok((ROOT / 'data/corpus.txt').read_text(errors='ignore'), return_tensors='pt')['input_ids'][0]
    out = ROOT / f'results/profstep_{args.model.split("/")[-1]}_{args.attn}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    run(model, ids_all[:512][None].cuda(), 4, 1)  # warm-up
    for c in [int(x) for x in args.contexts.split(',')]:
        torch.cuda.empty_cache()
        ids = ids_all[:c][None].cuda()
        try:
            res = (run_generate(model, ids, args.steps, args.mode == 'genpad0') if args.mode.startswith('gen')
                   else run(model, ids, args.steps, args.prof_steps, mask=args.mode == 'mask'))
        except torch.OutOfMemoryError as e:
            print(f'ctx={c:>6} out of memory: {str(e)[:120]}', flush=True)
            continue
        r = {'model': args.model, 'attn': args.attn, 'mode': args.mode, 'context_len': c, **res}
        with open(out, 'a') as f:
            f.write(json.dumps(r) + '\n')
        print(f"ctx={c:>6} {args.mode} wall {r['wall_ms_per_step']:.1f} ms/step  "
              f"cache {r['cache_mib']:.0f} MiB  reserved {r['cuda_reserved_mib']:.0f} MiB", flush=True)
        print('   top cpu:', r['top_cpu'][:5], flush=True)
    print('->', out)


if __name__ == '__main__':
    main()
