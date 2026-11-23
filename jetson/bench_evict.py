"""Energy benchmark of KV-cache eviction (NVIDIA kvpress on Hugging Face transformers) on Jetson Orin Nano.

The eviction counterpart of bench_kv.py: same power meter, same prompts (data/corpus.txt), exactly --gen
generated tokens, shuffled rounds, one JSONL row per run with the same field names, so analyze_v2.py can
pair every method with its baseline. Rows use kv_type "f16" for the uncompressed baseline (f16 weights and
cache in PyTorch) and "<method>@<ratio>" for an eviction press, e.g. "snapkv@0.5" = SnapKV removing 50%
of the prompt's KV entries. "quanto4", "quanto2", "hqq8", "hqq4" = no eviction but transformers' QuantizedCache
(KIVI-style quantization in PyTorch, newest 128 tokens kept fp16), so both methods can be compared in one engine.

kvpress compresses the cache once, at the end of prefill; decode then runs on the smaller cache.
Prefill/decode split: a stopping criterion records the time the first generated token appears (end of
prefill); decode energy per token is the rest divided by gen - 1.

Setup (separate venv, see jetson/setup_evict.sh), then:
  python3 jetson/bench_evict.py --tag 15W-evict
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from power import PowerMeter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ['Qwen/Qwen2.5-0.5B-Instruct', 'unsloth/Llama-3.2-1B-Instruct']  # unsloth = ungated copy of Meta's


def make_press(method, ratio):
    from kvpress import SnapKVPress, StreamingLLMPress
    if method == 'streamingllm':
        return StreamingLLMPress(compression_ratio=ratio, n_sink=4)
    if method == 'snapkv':
        return SnapKVPress(compression_ratio=ratio, window_size=64)
    raise ValueError(method)


class FirstToken(StoppingCriteria):
    """Never stops; records when the first generated token exists (= end of prefill)."""
    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores, **kw):
        if self.t is None:
            torch.cuda.synchronize()
            self.t = time.monotonic()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def make_quant_cache(cfg, model):
    """'quanto4' -> QuantizedCache(backend='quanto', nbits=4); None for every other config."""
    import re
    from transformers import QuantizedCache
    m = re.fullmatch(r'(quanto|hqq)(\d)', cfg)
    return QuantizedCache(m[1], model.config, nbits=int(m[2]), residual_length=128) if m else None


def nbytes(o, depth=0):
    """Bytes held by plain tensors, by quantized tensor subclasses (their packed parts) and by containers."""
    if depth > 4 or o is None:
        return 0
    if isinstance(o, torch.Tensor):
        if type(o) in (torch.Tensor, torch.nn.Parameter):
            return o.numel() * o.element_size()
        names = o.__tensor_flatten__()[0] if hasattr(o, '__tensor_flatten__') else []
        return sum(nbytes(getattr(o, n), depth + 1) for n in names)
    if isinstance(o, (list, tuple)):
        return sum(nbytes(x, depth + 1) for x in o)
    if isinstance(o, dict):
        return sum(nbytes(x, depth + 1) for x in o.values())
    return 0


def cache_mib(cache):
    if cache is None:
        return None
    if hasattr(cache, 'layers'):
        names = ('keys', 'values', '_quantized_keys', '_quantized_values')
        return sum(nbytes(getattr(l, n, None)) for l in cache.layers for n in names) / 2**20
    ts = list(getattr(cache, 'key_cache', [])) + list(getattr(cache, 'value_cache', []))
    return sum(t.numel() * t.element_size() for t in ts if t is not None) / 2**20


@torch.no_grad()
def measure(meter, model, ids, gen, cfg, idle_w, rail, pad_id=0):
    press = make_press(cfg.split('@')[0], float(cfg.split('@')[1])) if '@' in cfg else None
    first = FirstToken()
    kw = dict(max_new_tokens=gen, min_new_tokens=gen, do_sample=False, return_dict_in_generate=True,
              stopping_criteria=StoppingCriteriaList([first]), pad_token_id=pad_id)
    qcache = make_quant_cache(cfg, model)
    if qcache is not None:
        kw['past_key_values'] = qcache
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    temps0 = meter.temps()
    t0 = time.monotonic()
    if press is None:
        out = model.generate(ids, **kw)
    else:
        with press(model):
            out = model.generate(ids, **kw)
    torch.cuda.synchronize()
    t1 = time.monotonic()
    t_split = first.t or t1
    g = out.sequences.shape[1] - ids.shape[1]
    dec_s = max(t1 - t_split, 1e-9)
    r = {
        'prompt_tokens': ids.shape[1], 'gen_tokens': g, 'latency_s': t1 - t0,
        'prefill_s': t_split - t0, 'decode_s': dec_s, 'decode_window_s': dec_s,
        'prefill_tps': ids.shape[1] / max(t_split - t0, 1e-9), 'decode_tps': (g - 1) / dec_s,
        'kv_used_mib': cache_mib(out.past_key_values),
        'peak_cuda_mib': torch.cuda.max_memory_allocated() / 2**20,
        'gpu_temp_start': temps0.get('gpu-thermal'), 'gpu_temp_end': meter.temps().get('gpu-thermal'),
        'gpu_freq_mhz_end': meter.gpu_freq_mhz(), 'min_mem_available_mb': meter.min_mem_available(t0, t1),
        'gpu_load_prefill': meter.gpu_load(t0, t_split), 'gpu_load_decode': meter.gpu_load(t_split, t1),
        'cpu_busy_prefill': meter.cpu_busy(t0, t_split), 'cpu_busy_decode': meter.cpu_busy(t_split, t1),
    }
    for rl in meter.rails():
        e, n = meter.energy(t0, t1, rl)
        ep, _ = meter.energy(t0, t_split, rl)
        ed, _ = meter.energy(t_split, t1, rl)
        idle = idle_w.get(rl, 0.0)
        k = rl.lower()
        r[f'{k}_energy_j'] = e
        r[f'{k}_energy_net_j'] = e - idle * (t1 - t0)
        r[f'{k}_decode_energy_net_j'] = ed - idle * dec_s
        r[f'{k}_avg_w'] = meter.mean_power(t0, t1, rl)
        r[f'{k}_decode_avg_w'] = meter.mean_power(t_split, t1, rl)
        if rl == rail:
            r.update({
                'n_samples': n, 'energy_j': e, 'energy_net_j': e - idle * (t1 - t0),
                'prefill_energy_j': ep, 'prefill_energy_net_j': ep - idle * (t_split - t0),
                'decode_energy_j': ed, 'decode_energy_net_j': ed - idle * dec_s,
                'decode_avg_w': meter.mean_power(t_split, t1, rl), 'prefill_avg_w': meter.mean_power(t0, t_split, rl),
                'energy_per_request_j': e, 'energy_per_request_net_j': e - idle * (t1 - t0),
                'decode_j_per_token': (ed - idle * dec_s) / max(g - 1, 1),
                'decode_j_per_token_gross': ed / max(g - 1, 1),
                'total_j_per_token': (e - idle * (t1 - t0)) / g if g else None,
                'total_j_per_token_gross': e / g if g else None,
            })
    return r


def measure_idle(meter, secs):
    t0 = time.monotonic()
    time.sleep(secs)
    t1 = time.monotonic()
    idle = {r: meter.mean_power(t0, t1, r) for r in meter.rails()}
    print('idle W:', {k: round(v, 2) for k, v in idle.items()})
    return idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--methods', default='f16,streamingllm@0.5,snapkv@0.5,streamingllm@0.75,snapkv@0.75')
    ap.add_argument('--contexts', default='1024,4096,8192,16384')
    ap.add_argument('--gen', type=int, default=128)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--corpus', default=str(ROOT / 'data/corpus.txt'))
    ap.add_argument('--rail', default='VDD_IN')
    ap.add_argument('--idle-secs', type=float, default=10)
    ap.add_argument('--cooldown', type=float, default=2)
    ap.add_argument('--tag', default='')
    ap.add_argument('--quick', action='store_true', help='first model, 2 contexts, 1 round')
    ap.add_argument('--pad', default='eos',
                    help='pad_token_id for generate(): "eos" (default since 17 Nov) or a token id. Up to v4 it was 0, '
                         'which is "!" in both tokenizers: generate() then masks every "!" in the prompt as padding '
                         '(the first is at token ~7.9K-8.1K of the corpus), and attention takes a slower masked path')
    args = ap.parse_args()
    cfgs = args.methods.split(',')
    ctxs = [int(c) for c in args.contexts.split(',')]
    if args.quick:
        args.models, ctxs, args.rounds = args.models[:1], [1024, 4096], 1

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / f'results/evict_{stamp}.jsonl'
    meta = {'tag': args.tag, 'gen': args.gen, 'methods': cfgs, 'contexts': ctxs, 'rounds': args.rounds,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'started': stamp, 'pad': args.pad}
    try:
        import kvpress
        import transformers
        meta.update({'kvpress': getattr(kvpress, '__version__', '?'), 'transformers': transformers.__version__})
    except ImportError:
        pass
    print(json.dumps(meta, indent=1))

    meter = PowerMeter().start()
    idle_w = measure_idle(meter, args.idle_secs)
    meta['idle_w'] = idle_w
    corpus = Path(args.corpus).read_text(errors='ignore')
    total = len(args.models) * len(cfgs) * len(ctxs) * args.rounds
    done = 0
    with open(out, 'a') as fh:
        for name in args.models:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map='cuda:0',
                                                         attn_implementation='sdpa').eval()
            ids_all = tok(corpus, return_tensors='pt')['input_ids'][0]
            pad_id = tok.eos_token_id if args.pad == 'eos' else int(args.pad)
            model.generate(ids_all[:256][None].cuda(), max_new_tokens=8, min_new_tokens=8)  # warm-up
            short = name.split('/')[-1]
            for rnd in range(args.rounds):
                jobs = [(c, k) for c in ctxs for k in cfgs]
                random.shuffle(jobs)
                for c, cfg in jobs:
                    ids = ids_all[:c][None].cuda()
                    row = {'model': short, 'kv_type': cfg, 'context_len': c, 'n_parallel': 1, 'round': rnd,
                           'tag': args.tag, 'engine': 'transformers+kvpress', 'idle_w': idle_w.get(args.rail),
                           'rail': args.rail, 'power_backend': meter.backend}
                    try:
                        row['pad_token_id'] = pad_id
                        row.update(measure(meter, model, ids, args.gen, cfg, idle_w, args.rail, pad_id))
                        row['status'] = 'ok'
                    except torch.cuda.OutOfMemoryError as e:
                        row.update({'status': 'oom', 'error': str(e)[:300]})
                    except Exception as e:  # noqa: BLE001  record and go on
                        row.update({'status': 'error', 'error': f'{type(e).__name__}: {e}'[:300]})
                    torch.cuda.empty_cache()
                    fh.write(json.dumps(row) + '\n')
                    fh.flush()
                    done += 1
                    if row['status'] == 'ok':
                        print(f"[{done}/{total}] {short} {cfg:<18} ctx={c:>6} decode {row['decode_tps']:.1f} tok/s "
                              f"{row['decode_j_per_token']:.3f} J/tok  req {row['energy_per_request_net_j']:.1f} J  "
                              f"kv {row['kv_used_mib']:.0f} MiB")
                    else:
                        print(f"!! [{done}/{total}] {short} {cfg} ctx={c}: {row['status']} {row.get('error', '')}")
                    time.sleep(args.cooldown)
            del model
            torch.cuda.empty_cache()
    meta['idle_w_end'] = measure_idle(meter, args.idle_secs)
    meter.stop()
    out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=1))
    print(f'done -> {out}')


if __name__ == '__main__':
    main()
