"""Statistics and plots for the v4 sets (L mixed K/V, M accuracy, N PyTorch QuantizedCache, O decode-step profile,
P T4-matched eviction). Energy cells and paired changes come from analyze_v2.py (run it first: it reads every
kv_*/evict_* file, the v4 ones included). This script adds what analyze_v2 does not cover and writes
plots_v2/stats_v4.json and figures 13-17.

  python3 analysis/analyze_v2.py && python3 analysis/analyze_v4.py
"""
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from analyze_v2 import KV_COLOR, SHORT, ci, load_runs, paired_change  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / 'results', ROOT / 'plots_v2'
THESIS = ROOT.parent  # the T4 thesis data (results.json = Qwen, results_llama.json = Llama)
MIX = [('q8_0', '8-bit K and V', KV_COLOR['q8_0'], '-'), ('q4_0', '4-bit K and V', KV_COLOR['q4_0'], '-'),
       ('q8_0/q4_0', '8-bit K, 4-bit V', '#8E4FB5', '--'), ('f16/q4_0', 'f16 K, 4-bit V', '#1E8C76', '--')]


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion, as (p, low, high)."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- M: accuracy

def needle_stats():
    rows = [json.loads(l) for p in sorted(glob.glob(str(RES / 'needle_*.jsonl'))) for l in open(p) if l.strip()]
    rows = [r for r in rows if r.get('tag') != 'smoke']
    cell = defaultdict(list)
    for r in rows:
        cell[(r['engine'], r['model'], r['config'], r['context_len'])].append(r)
    out = {}
    for k, rs in cell.items():
        ok = [r for r in rs if r['error'] is None]
        n_ok = sum(bool(r['correct']) for r in ok)
        p, lo, hi = wilson(n_ok, len(ok))
        by_depth = {str(d): sum(bool(r['correct']) for r in ok if r['depth'] == d) / max(1, sum(r['depth'] == d for r in ok))
                    for d in sorted({r['depth'] for r in ok})}
        out['|'.join(map(str, k))] = {'n': len(ok), 'correct': n_ok, 'acc': p, 'lo': lo, 'hi': hi,
                                      'errors': len(rs) - len(ok), 'by_depth': by_depth,
                                      'distractor': sum(bool(r.get('distractor_answer')) and not r['correct'] for r in ok),
                                      'seconds': st.mean(r['seconds'] for r in rs)}
    return out


NEEDLE_ORDER = [('llama', 'f16', 'llama.cpp f16 cache'), ('llama', 'q8_0', 'llama.cpp 8-bit'),
                ('llama', 'q8_0/q4_0', 'llama.cpp 8-bit K, 4-bit V'), ('llama', 'f16/q4_0', 'llama.cpp f16 K, 4-bit V'),
                ('llama', 'q4_0', 'llama.cpp 4-bit'),
                ('hf', 'none', 'PyTorch, no compression'),
                ('hf', 'hqq8', 'PyTorch HQQ 8-bit'), ('hf', 'hqq4', 'PyTorch HQQ 4-bit'),
                ('hf', 'quanto4', 'PyTorch quanto 4-bit'), ('hf', 'quanto2', 'PyTorch quanto 2-bit'),
                ('hf', 'snapkv@0.5:qa', 'SnapKV 50%, question inside'), ('hf', 'snapkv@0.75:qa', 'SnapKV 75%, question inside'),
                ('hf', 'snapkv@0.5:ag', 'SnapKV 50%, question after'), ('hf', 'snapkv@0.75:ag', 'SnapKV 75%, question after'),
                ('hf', 'streamingllm@0.5:qa', 'StreamingLLM 50%, question inside'),
                ('hf', 'streamingllm@0.75:qa', 'StreamingLLM 75%, question inside'),
                ('hf', 'streamingllm@0.5:ag', 'StreamingLLM 50%, question after'),
                ('hf', 'streamingllm@0.75:ag', 'StreamingLLM 75%, question after')]
NEEDLE_MODELS = ['Qwen2.5-0.5B-Instruct', 'Llama-3.2-1B-Instruct']


def needle_plot(ns, path):
    if not ns:
        return
    ctxs = sorted({int(k.split('|')[3]) for k in ns})
    colors = {1024: '#9DC3E6', 4096: '#2F6FA8', 8192: '#14365C', 16384: '#000000'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), sharey=True)
    labels = [lab for e, c, lab in NEEDLE_ORDER]
    for ax, m in zip(axes, NEEDLE_MODELS):
        h = 0.8 / len(ctxs)
        for i, c in enumerate(ctxs):
            ys, xs, errs = [], [], [[], []]
            for j, (e, cfg, _) in enumerate(NEEDLE_ORDER):
                v = ns.get(f'{e}|{m}|{cfg}|{c}')
                if not v:
                    continue
                ys.append(j + (i - (len(ctxs) - 1) / 2) * h)
                xs.append(100 * v['acc'])
                errs[0].append(100 * (v['acc'] - v['lo']))
                errs[1].append(100 * (v['hi'] - v['acc']))
            ax.barh(ys, xs, h * 0.9, xerr=errs, color=colors[c], label=f'{c // 1024}K prompt',
                    error_kw={'lw': 0.6, 'ecolor': '#777', 'capsize': 1.5})
        for y in (4.5, 9.5):
            ax.axhline(y, color='#bbb', lw=0.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel('correct answers (%), 95% Wilson CI')
        ax.set_title(SHORT.get(m, m).replace(' (PyTorch, f16)', ''), fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('Retrieval accuracy (needle + 3 distractors, 30 prompts per bar): cache formats in llama.cpp, '
                 'quantized caches and eviction in PyTorch', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


# ---------------------------------------------------------------- L, N: energy plots from paired changes

def change_lines(rows, tag, series, models, metric, path, title, ylabel, base='f16'):
    ch = paired_change(rows, metric, base)
    models = [m for m in models if any(k[:2] == (tag, m) for k in ch)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for kv, label, color, ls in series:
            pts = sorted((k[3], v) for k, v in ch.items() if k[:3] == (tag, m, kv) and k[4] == 1)
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], color=color,
                            ls=ls, lw=2, marker='o', ms=5, capsize=2, label=label)
        ax.axhline(0, color='#888', lw=0.8)
        ax.set_xscale('log', base=2)
        cs = sorted({k[3] for k in ch if k[:2] == (tag, m)})
        ax.set_xticks(cs, [f'{c // 1024}K' for c in cs])
        ax.set_xlabel('prompt length')
        ax.set_title(SHORT.get(m, m), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


# ---------------------------------------------------------------- O: decode-step profile

def prof_stats():
    out = defaultdict(list)
    for p in sorted(glob.glob(str(RES / 'profstep_*.jsonl'))):
        for l in open(p):
            r = json.loads(l)
            out[f"{r['model'].split('/')[-1]}|{r['attn']}"].append(r)
    return dict(out)


def prof_plot(ps, path):
    if not ps:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    style = {'Qwen2.5-0.5B-Instruct|sdpa': ('#2F6FA8', 'Qwen-0.5B'), 'Llama-3.2-1B-Instruct|sdpa': ('#E07A2E', 'Llama-1B'),
             'Qwen2.5-0.5B-Instruct|eager': ('#9AA5B1', 'Qwen-0.5B, eager attention')}
    for k, rs in ps.items():
        color, lab = style.get(k, ('#333', k))
        cs = [r['context_len'] for r in rs]
        ax.plot(cs, [r['wall_ms_per_step'] for r in rs], color=color, lw=2, marker='o', ms=5, label=f'{lab}: wall time')
        ax.plot(cs, [r['gpu_ms_per_step'] for r in rs], color=color, lw=1.5, ls='--', marker='s', ms=4,
                label=f'{lab}: GPU kernel time')
    ax.set_xlabel('cached tokens')
    ax.set_ylabel('ms per generated token')
    ax.set_xticks([2048, 4096, 6144, 8192, 12288], ['2K', '4K', '6K', '8K', '12K'])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title('PyTorch decode step (fp16, no eviction): wall time vs GPU kernel time per token', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


# ---------------------------------------------------------------- P: T4 vs Jetson

def t4_stats():
    """T4 thesis cells at batch 1, 50 generated tokens, 50% eviction, tasks where all 50 tokens were generated."""
    out = {}
    for fn in ('results.json', 'results_llama.json'):
        p = THESIS / fn
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r['batch_size'] != 1 or r['output_len'] != 50:
                continue
            g = [x['total_gen_tokens'] for x in r['runs']]
            if min(g) != 50:
                continue
            m = r['model'].split('/')[-1]
            k = f"{m}|{r['method']}|{r['context_len']}"
            out.setdefault(k, []).extend(x['energy_j'] for x in r['runs'])
            out.setdefault(k + '|lat', []).extend(x['latency_s'] for x in r['runs'])
    return {k: ci(v) for k, v in out.items()}


def t4_plot(rows, t4, path):
    tag = '15W-t4match'
    jet = paired_change(rows, 'energy_per_request_net_j')
    if not any(k[0] == tag for k in jet):
        return
    cells = [('Qwen2.5-0.5B-Instruct', 2048), ('Qwen2.5-0.5B-Instruct', 8192), ('Llama-3.2-1B-Instruct', 2048),
             ('Llama-3.2-1B-Instruct', 8192)]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    series = [('T4', 'streamingllm', '#9DC3E6'), ('T4', 'snapkv', '#C9B8F0'),
              ('Jetson', 'streamingllm@0.5', '#1baf7a'), ('Jetson', 'snapkv@0.5', '#6250d6')]
    w = 0.8 / len(series)
    for i, (dev, meth, color) in enumerate(series):
        xs, ys, es = [], [], []
        for j, (m, c) in enumerate(cells):
            if dev == 'T4':
                b, e = t4.get(f'{m}|baseline|{c}'), t4.get(f'{m}|{meth}|{c}')
                if not (b and e):
                    continue
                ys.append(100 * (e[0] / b[0] - 1))
                es.append(0)
            else:
                v = jet.get((tag, m, meth, c, 1))
                if not v:
                    continue
                ys.append(v[0])
                es.append(v[1])
            xs.append(j + (i - 1.5) * w)
        name = 'StreamingLLM' if 'stream' in meth else 'SnapKV'
        ax.bar(xs, ys, w * 0.92, yerr=es, color=color, label=f'{dev}, {name} 50%', capsize=2,
               error_kw={'lw': 0.8, 'ecolor': '#555'})
    ax.set_xticks(range(len(cells)), [f"{SHORT[m].split(' (')[0]}, {c // 1024}K" for m, c in cells], fontsize=8)
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_ylabel('change in energy per request vs no eviction (%)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Same eviction settings on the T4 (thesis) and the Jetson: 50 generated tokens, batch 1', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def main():
    files = sorted(p for p in glob.glob(str(RES / 'kv_*.jsonl')) if '190420' not in p)
    files += sorted(glob.glob(str(RES / 'evict_*.jsonl')))
    rows = load_runs(files)
    ns, ps, t4 = needle_stats(), prof_stats(), t4_stats()
    (OUT / 'stats_v4.json').write_text(json.dumps({'needle': ns, 'prof': ps, 't4': t4}, indent=1, default=str))
    print('wrote', OUT / 'stats_v4.json')
    change_lines(rows, '15W-mixkv', MIX, ['Llama-3.2-1B-Instruct-Q8_0', 'qwen2.5-0.5b-instruct-q8_0',
                                          'qwen2.5-1.5b-instruct-q4_k_m'], 'energy_per_request_net_j',
                 OUT / '13_mixkv.png', 'Mixed K/V cache formats (one llama.cpp build, 15W): energy per request vs f16 '
                 'cache, paired 95% CI', 'change in energy per request vs f16 (%)')
    needle_plot(ns, OUT / '14_accuracy.png')
    change_lines(rows, '15W-pytq', [('quanto4', 'quanto 4-bit', '#2F6FA8', '-'), ('quanto2', 'quanto 2-bit', '#14365C', '--'),
                                    ('hqq8', 'HQQ 8-bit', '#E07A2E', '-'), ('hqq4', 'HQQ 4-bit', '#B65A17', '--'),
                                    ('snapkv@0.5', 'SnapKV 50% (eviction)', '#6250d6', ':')],
                 ['Qwen2.5-0.5B-Instruct', 'Llama-3.2-1B-Instruct'], 'energy_per_request_net_j', OUT / '15_pytq.png',
                 'Quantized cache vs eviction in one engine (PyTorch, 15W): energy per request vs f16 cache, paired 95% CI',
                 'change in energy per request vs f16 (%)')
    prof_plot(ps, OUT / '16_decode_step.png')
    t4_plot(rows, t4, OUT / '17_t4_vs_jetson.png')


if __name__ == '__main__':
    main()
