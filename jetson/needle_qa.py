"""Accuracy of KV-cache compression on a long-context retrieval task (needle in a haystack), on the Jetson.

Perplexity cannot score eviction: kvpress compresses only once, after the prompt, so a perplexity run never
uses the compressed cache. This test can: one fact (the "needle") is hidden in a long text, a question about
it comes at the end, and the model must answer from the cache. It scores quantization and eviction on the
same prompts, the same tokens and the same rule:
  haystack: Pride and Prejudice (data/corpus.txt, the energy prompts), a different start for every trial
  needle:   "The secret code for <city> is <7 digits>." placed at depth 10/30/50/70/90% of the text, plus
            3 distractors (other cities, other codes) at random depths
  question: "What is the secret code for <city>? Answer with the number only."
  correct:  the 7 digits appear in the first 16 generated tokens (greedy)

Engines (--engine):
  hf     PyTorch fp16 weights through kvpress's own pipeline (context prefilled with the press, then greedy
         decoding). Configs: "none"; "<press>@<ratio>:qa" = question inside the compressed prompt (as in the
         energy runs, so SnapKV sees the question in its window); "<press>@<ratio>:ag" = question-agnostic,
         the question comes after compression (the cache is compressed before the question is known, as when
         one document is asked several questions); "quanto4", "quanto2", "hqq8", "hqq4" = transformers'
         QuantizedCache (KIVI-style, the newest 128 tokens stay fp16).
  llama  llama.cpp llama-server with Q8_0 GGUF weights; configs are cache formats, "q8_0" or "q8_0/q4_0" (K/V).
Both engines send the same token ids (HF tokenizer, chat template applied), so only the cache differs.

  python3 jetson/needle_qa.py --engine hf --configs none,snapkv@0.5:qa,...
  python3 jetson/needle_qa.py --engine llama --llama-bin llama.cpp/build-allq/bin --configs f16,q8_0,...
"""
import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HF = {'Qwen2.5-0.5B-Instruct': 'Qwen/Qwen2.5-0.5B-Instruct',
      'Llama-3.2-1B-Instruct': 'unsloth/Llama-3.2-1B-Instruct'}
GGUF = {'Qwen2.5-0.5B-Instruct': 'models/qwen2.5-0.5b-instruct-q8_0.gguf',
        'Llama-3.2-1B-Instruct': 'models/Llama-3.2-1B-Instruct-Q8_0.gguf'}
CITIES = ['Lisbon', 'Nairobi', 'Osaka', 'Quito', 'Tallinn', 'Hobart', 'Tbilisi', 'Windhoek', 'Bergen', 'Cusco',
          'Dakar', 'Hanoi', 'Krakow', 'Lyon', 'Medan', 'Perth', 'Riga', 'Sapporo', 'Tunis', 'Valletta']
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
QUESTION = 'What is the secret code for {city}? Answer with the number only.'


def make_trials(tok, corpus, ctx, n_per_depth, seed, n_distract=3):
    """Prompts of about ctx tokens (chat template included). Same seed -> same trials for every config.
    Besides the needle, n_distract other cities with other codes sit at random depths (RULER's multi-key
    variant), so a model that kept only part of the cache cannot answer with the one code it saw."""
    rng = random.Random(seed * 100003 + ctx)
    trials = []
    for d in DEPTHS:
        for i in range(n_per_depth):
            cities = rng.sample(CITIES, 1 + n_distract)
            codes = [str(rng.randint(1_000_000, 9_999_999)) for _ in cities]
            needles = [f' The secret code for {c} is {k}.' for c, k in zip(cities, codes)]
            depths = [d] + [rng.random() for _ in range(n_distract)]
            q = QUESTION.format(city=cities[0])
            start = rng.randrange(0, len(corpus) // 2)
            budget = ctx - len(tok.encode(''.join(needles) + '\n\n' + q, add_special_tokens=False)) - 48
            hay = tok.decode(tok.encode(corpus[start:start + 8 * ctx], add_special_tokens=False)[:budget])
            cuts = []
            for dep in depths:
                cut = hay.find('. ', int(len(hay) * dep))  # after a full sentence
                cuts.append(len(hay) if cut < 0 else cut + 1)
            for cut, nd in sorted(zip(cuts, needles), key=lambda x: -x[0]):  # insert from the end
                hay = hay[:cut] + nd + hay[cut:]
            trials.append({'depth': d, 'trial': i, 'city': cities[0], 'code': codes[0], 'distractors': codes[1:],
                           'context': hay, 'question': q})
    return trials


def split_prompt(tok, context, question):
    """kvpress pipeline layout: chat-templated context, then question + answer prefix (never compressed)."""
    sep = '#' * 40
    text = tok.apply_chat_template([{'role': 'user', 'content': context + sep}], add_generation_prompt=True,
                                   tokenize=False)
    ctx_text, suffix = text.split(sep)
    return ctx_text, question + suffix


def correct(code, text):
    return code in re.sub(r'[^0-9]', '', text) or code in text


# ---------------------------------------------------------------- engines

class HFEngine:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import kvpress  # noqa: F401  registers the "kv-press-text-generation" pipeline
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(HF[name])
        self.model = AutoModelForCausalLM.from_pretrained(HF[name], dtype=torch.float16, device_map='cuda:0',
                                                          attn_implementation='sdpa').eval()
        self.pipe = pipeline('kv-press-text-generation', model=self.model, tokenizer=self.tok)

    def press(self, spec):
        from kvpress import SnapKVPress, StreamingLLMPress
        method, ratio = spec.split('@')
        if method == 'streamingllm':
            return StreamingLLMPress(compression_ratio=float(ratio), n_sink=4)
        return SnapKVPress(compression_ratio=float(ratio), window_size=64)

    def cache(self, cfg):
        from transformers import QuantizedCache
        backend, bits = re.fullmatch(r'(quanto|hqq)(\d)', cfg).groups()
        return QuantizedCache(backend, self.model.config, nbits=int(bits), residual_length=128)

    def answer(self, cfg, context, question, max_new):
        kw = {'max_new_tokens': max_new}
        spec, _, mode = cfg.partition(':')
        if spec == 'none' or mode == 'qa':  # question inside the prompt that gets compressed
            context, question = context + '\n\n' + question, ''
        if '@' in spec:
            kw['press'] = self.press(spec)
        elif spec != 'none':
            # quantized cache: plain generate() on the whole prompt (kvpress's pipeline fails with HQQ layers,
            # and for quantization it makes no difference where the question sits)
            c, q = split_prompt(self.tok, context + '\n\n' + question, '')
            ids = self.tok.encode(c, add_special_tokens=False) + self.tok.encode(q, add_special_tokens=False)
            ids = self.torch.tensor([ids], device=self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), past_key_values=self.cache(spec),
                                          max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
            self.torch.cuda.empty_cache()
            return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        with self.torch.no_grad():
            out = self.pipe(context, question=question or None, **kw)
        self.torch.cuda.empty_cache()
        return out['answer']

    def n_tokens(self, context, question):
        c, q = split_prompt(self.tok, context + '\n\n' + question, '')
        return len(self.tok.encode(c, add_special_tokens=False)) + len(self.tok.encode(q, add_special_tokens=False))


class LlamaEngine:
    """llama-server per cache format; prompt sent as the HF tokenizer's token ids."""
    def __init__(self, name, llama_bin, max_ctx):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(HF[name])
        self.gguf, self.bin, self.max_ctx = ROOT / GGUF[name], Path(llama_bin), max_ctx
        self.srv, self.cfg = None, None

    def _server(self, cfg):
        from bench_kv import Server
        if self.cfg != cfg:
            if self.srv:
                self.srv.stop()
            log = ROOT / f'results/logs/needle_{self.gguf.stem}_{cfg.replace("/", "-")}.log'
            self.srv = Server(self.bin, self.gguf, cfg, self.max_ctx + 128, log, 1, ['--cache-ram', '0'])
            if not self.srv.start():
                raise RuntimeError(f'llama-server failed for {cfg}')
            self.cfg = cfg
        return self.srv

    def answer(self, cfg, context, question, max_new):
        from bench_kv import http
        c, q = split_prompt(self.tok, context + '\n\n' + question, '')
        ids = self.tok.encode(c, add_special_tokens=False) + self.tok.encode(q, add_special_tokens=False)
        r = http(self._server(cfg).port, '/completion', {'prompt': ids, 'n_predict': max_new, 'temperature': 0.0,
                                                         'cache_prompt': False}, timeout=600)
        return r.get('content', '')

    def n_tokens(self, context, question):
        c, q = split_prompt(self.tok, context + '\n\n' + question, '')
        return len(self.tok.encode(c, add_special_tokens=False)) + len(self.tok.encode(q, add_special_tokens=False))

    def close(self):
        if self.srv:
            self.srv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['hf', 'llama'], required=True)
    ap.add_argument('--models', default='Qwen2.5-0.5B-Instruct,Llama-3.2-1B-Instruct')
    ap.add_argument('--configs', required=True)
    ap.add_argument('--contexts', default='1024,4096,8192')
    ap.add_argument('--per-depth', type=int, default=6, help='trials per needle depth')
    ap.add_argument('--distractors', type=int, default=3)
    ap.add_argument('--max-new', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--llama-bin', default=str(ROOT / 'llama.cpp/build-allq/bin'))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    cfgs, ctxs = args.configs.split(','), [int(c) for c in args.contexts.split(',')]
    corpus = (ROOT / 'data/corpus.txt').read_text(errors='ignore')
    out = ROOT / f'results/needle_{args.engine}_{datetime.now():%Y%m%d_%H%M%S}.jsonl'
    print('->', out, flush=True)
    for name in args.models.split(','):
        eng = HFEngine(name) if args.engine == 'hf' else LlamaEngine(name, args.llama_bin, max(ctxs))
        for ctx in ctxs:
            trials = make_trials(eng.tok, corpus, ctx, args.per_depth, args.seed, args.distractors)
            for cfg in cfgs:
                ok = 0
                for t in trials:
                    t0 = time.monotonic()
                    try:
                        ans, err = eng.answer(cfg, t['context'], t['question'], args.max_new), None
                    except Exception as e:  # noqa: BLE001  record and go on (e.g. out of memory)
                        ans, err = '', f'{type(e).__name__}: {e}'[:300]
                    row = {'engine': args.engine, 'model': name, 'config': cfg, 'context_len': ctx,
                           'prompt_tokens': eng.n_tokens(t['context'], t['question']), 'depth': t['depth'],
                           'trial': t['trial'], 'city': t['city'], 'code': t['code'], 'answer': ans,
                           'correct': correct(t['code'], ans) if err is None else None, 'error': err,
                           'distractor_answer': any(correct(k, ans) for k in t['distractors']),
                           'seconds': time.monotonic() - t0, 'tag': args.tag}
                    ok += bool(row['correct'])
                    with open(out, 'a') as f:
                        f.write(json.dumps(row) + '\n')
                print(f'{name} ctx={ctx} {cfg:<22} {ok}/{len(trials)} correct', flush=True)
        if args.engine == 'llama':
            eng.close()
        del eng


if __name__ == '__main__':
    main()
