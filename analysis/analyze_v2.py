"""Statistics, tables and plots for the v2 Jetson experiments (and the v1 data, for comparison).

Every cell is reported as mean with a 95% confidence interval (Student t, n = rounds).
Changes versus f16 are *paired*: each format is compared with f16 from the same round
(same server-restart cycle, minutes apart, same thermal state), and the CI is taken over
the per-round ratios. A change whose CI includes zero is marked "n.s." (not separable from zero).

Usage:
  python3 analysis/analyze_v2.py --out plots_v2
Reads results/*.jsonl, results/pptg_*.jsonl, results/ppl*.csv and results/prof/*kern_sum*.csv
(whatever is present) and writes <out>/stats.json, <out>/tables.md and PNG charts.
"""
import argparse
import csv
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
KV_COLOR = {'f16': '#4A5568', 'f16@off': '#9AA5B1', 'q8_0': '#2F6FA8', 'q4_0': '#E07A2E'}
KV_LABEL = {'f16': 'f16 (flash attn on)', 'f16@off': 'f16 (flash attn off)', 'q8_0': 'q8_0 (8-bit KV)',
            'q4_0': 'q4_0 (4-bit KV)'}
SHORT = {'Llama-3.2-1B-Instruct-Q8_0': 'Llama-1B', 'qwen2.5-0.5b-instruct-q8_0': 'Qwen-0.5B',
         'Llama-3.2-1B-Instruct-Q4_K_M': 'Llama-1B (4-bit weights)',
         'qwen2.5-0.5b-instruct-q4_k_m': 'Qwen-0.5B (4-bit weights)',
         'Llama-3.2-3B-Instruct-Q4_K_M': 'Llama-3B (4-bit weights)',
         'qwen2.5-1.5b-instruct-q4_k_m': 'Qwen-1.5B (4-bit weights)',
         'qwen2.5-3b-instruct-q4_k_m': 'Qwen-3B (4-bit weights)',
         'gemma-3-1b-it-Q4_K_M': 'Gemma-1B (4-bit weights)',
         'Qwen2.5-0.5B-Instruct': 'Qwen-0.5B (PyTorch, f16)', 'Llama-3.2-1B-Instruct': 'Llama-1B (PyTorch, f16)'}
# size/family set (v3), in plot order: Qwen by size, Llama by size, then Gemma
SIZE_MODELS = ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-1.5b-instruct-q4_k_m', 'qwen2.5-3b-instruct-q4_k_m',
               'Llama-3.2-1B-Instruct-Q4_K_M', 'Llama-3.2-3B-Instruct-Q4_K_M', 'gemma-3-1b-it-Q4_K_M']
# rows replaced by a later rerun: the 31 Oct v3 run crashed (out of memory) during Llama-3B; Llama-3B was
# rerun from round 0 on 1 Nov with --cache-ram 0, so its partial rows from the crashed file are dropped
SUPERSEDED = {('kv_20261031_201343.jsonl', 'Llama-3.2-3B-Instruct-Q4_K_M')}


def ci(xs):
    """(mean, half-width of 95% CI, sd, n)."""
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    m = st.mean(xs)
    if n < 2:
        return m, float('nan'), float('nan'), n
    sd = st.stdev(xs)
    return m, T95.get(n - 1, 1.96) * sd / math.sqrt(n), sd, n


def rebase_idle(r, meta):
    """Recompute net energy against the lower of the start/end idle readings.

    bench_kv.py subtracts the idle power measured at the start of a set. The 7W main grid started right after the
    power-mode reboot, while the board was still settling (3.68 W vs 3.03 W at the end), which would understate
    every 7W net energy. A transient can only add power, so the lower reading is the better estimate of idle.
    """
    start, end = meta.get('idle_w') or {}, meta.get('idle_w_end') or {}
    for rail, old in start.items():
        if rail not in end or end[rail] >= old:
            continue
        d = old - end[rail]
        k = rail.lower()
        dec_s = r.get('decode_window_s') or 0
        for f, secs in ((f'{k}_energy_net_j', r['latency_s']), (f'{k}_decode_energy_net_j', dec_s)):
            if f in r:
                r[f] += d * secs
        if rail != r.get('rail') or not r.get('idle_w'):
            continue
        pre_s = (r['prefill_energy_j'] - r['prefill_energy_net_j']) / r['idle_w']
        dec_tokens = r['decode_energy_j'] / r['decode_j_per_token_gross'] if r.get('decode_j_per_token_gross') else 0
        r['energy_net_j'] += d * r['latency_s']
        r['prefill_energy_net_j'] += d * pre_s
        r['decode_energy_net_j'] += d * dec_s
        r['energy_per_request_net_j'] = r['energy_net_j'] / r['n_parallel']
        if dec_tokens:
            r['decode_j_per_token'] = r['decode_energy_net_j'] / dec_tokens
        if r.get('gen_tokens'):
            r['total_j_per_token'] = r['energy_net_j'] / r['gen_tokens']
        r['idle_w'] = end[rail]
    r['_idle'] = {rail: min(v, end.get(rail, v)) for rail, v in start.items()}


def swa_kv_size(r):
    """Gemma-3-1B keeps the full context in only 4 of its 26 layers; the other 22 are sliding-window layers
    (window 512) that llama.cpp gives n_swa + n_ubatch = 1,024 cells (server log: 65.0 + 22.0 MiB at 16K f16).
    bench_kv.py sized the cache as if all 26 layers were full-context, so correct it here."""
    n = r['context_len'] + (r.get('gen_tokens') or 128)
    full = r['kv_used_mib']
    r['kv_used_mib'] = full * 4 / 26 + full / n * 22 / 26 * min(n, 1024)


def load_runs(paths):
    rows = []
    for p in paths:
        meta_p = Path(p).with_suffix('.meta.json')
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if (Path(p).name, r.get('model')) in SUPERSEDED:
                continue
            if r.get('model', '').startswith('gemma-3-1b') and r.get('kv_used_mib'):
                swa_kv_size(r)
            r.setdefault('status', 'ok')
            r.setdefault('n_parallel', 1)
            r['_file'] = Path(p).name
            r['_idle'] = meta.get('idle_w', {})
            if r['status'] == 'ok':
                # v1 rows lack these fields: derive them the same way the v2 code does
                n = r['n_parallel']
                r.setdefault('energy_per_request_j', r['energy_j'] / n)
                r.setdefault('energy_per_request_net_j', r['energy_net_j'] / n)
                dec_s = r.get('decode_window_s') or (r['latency_s'] - min(r['prefill_s'], r['latency_s']))
                r.setdefault('decode_energy_j', r['decode_energy_net_j'] + r['idle_w'] * dec_s)
                r.setdefault('decode_j_per_token_gross', r['decode_energy_j'] / r['gen_tokens'])
                r.setdefault('decode_avg_w', r['decode_energy_j'] / dec_s)
                for rail in ('VDD_CPU_GPU_CV', 'VDD_SOC', 'VDD_IN'):
                    k = rail.lower()
                    if f'{k}_energy_net_j' not in r and f'{k}_energy_j' in r and rail in r['_idle']:
                        r[f'{k}_energy_net_j'] = r[f'{k}_energy_j'] - r['_idle'][rail] * r['latency_s']
                rebase_idle(r, meta)
            rows.append(r)
    return rows


def key(r):
    return (r['tag'], r['model'], r['kv_type'], r['context_len'], r['n_parallel'])


def cell_stats(rows, metric):
    d = defaultdict(list)
    for r in rows:
        if r['status'] == 'ok' and r.get(metric) is not None:
            d[key(r)].append(r[metric])
    return {k: ci(v) for k, v in d.items()}


def paired_change(rows, metric, base='f16'):
    """{key: (mean %, CI half-width %, n, significant)} of metric vs `base` in the same round."""
    by = {}
    for r in rows:
        if r['status'] == 'ok' and r.get(metric) is not None:
            by[(*key(r), r['round'])] = r[metric]
    out = {}
    groups = defaultdict(list)
    for (tag, m, kv, c, n, rnd), v in by.items():
        b = by.get((tag, m, base, c, n, rnd))
        if kv != base and b:
            groups[(tag, m, kv, c, n)].append(100 * (v / b - 1))
    for k, v in groups.items():
        mean, half, _, n = ci(v)
        out[k] = (mean, half, n, (not math.isnan(half)) and abs(mean) > half)
    return out


def fmt_change(t):
    mean, half, n, sig = t
    s = f'{mean:+.1f}%' if math.isnan(half) else f'{mean:+.1f}% ± {half:.1f}'
    return s if sig else s + ' (n.s.)'


# ---------------------------------------------------------------- plots

def grid_plot(rows, metric, ylabel, title, path, change=False, tags=None, kvs=('f16', 'f16@off', 'q8_0', 'q4_0')):
    ok = [r for r in rows if r['status'] == 'ok' and r['n_parallel'] == 1 and (tags is None or r['tag'] in tags)]
    if not ok:
        return
    tag_list = sorted({r['tag'] for r in ok}, key=lambda t: (t.startswith('7'), t))
    models = sorted({r['model'] for r in ok})
    fig, axes = plt.subplots(len(tag_list), len(models), figsize=(6.2 * len(models), 4.3 * len(tag_list)),
                             squeeze=False, sharey='row' if change else False)
    pc = paired_change(ok, metric) if change else None
    cs = cell_stats(ok, metric)
    for i, tag in enumerate(tag_list):
        for j, m in enumerate(models):
            ax = axes[i][j]
            for kv in kvs:
                if change and kv == 'f16':
                    continue
                src = pc if change else cs
                pts = sorted((k[3], v) for k, v in src.items() if k[:3] == (tag, m, kv) and k[4] == 1)
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1][0] for p in pts]
                es = [0 if math.isnan(p[1][1]) else p[1][1] for p in pts]
                ax.errorbar(xs, ys, yerr=es, color=KV_COLOR.get(kv, '#888'), marker='o', capsize=3, lw=2,
                            label=KV_LABEL.get(kv, kv))
                if change:  # hollow markers for changes not separable from zero
                    for x, (mean, half, n, sig) in pts:
                        if not sig:
                            ax.plot([x], [mean], 'o', mfc='white', mec=KV_COLOR.get(kv, '#888'), ms=9, zorder=5)
            if change:
                ax.axhline(0, color='#B03A2E', ls='--', lw=1)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('context length (tokens)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{SHORT.get(m, m)} · {tag}', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def concurrency_plot(rows, path):
    ok = [r for r in rows if r['tag'].endswith('-np')]
    if not ok:
        return
    models = sorted({r['model'] for r in ok})
    ctxs = sorted({r['context_len'] for r in ok})
    fig, axes = plt.subplots(len(ctxs), len(models), figsize=(6.2 * len(models), 4.3 * len(ctxs)), squeeze=False)
    cs = cell_stats(ok, 'total_j_per_token')
    fails = [r for r in ok if r['status'] != 'ok']
    for i, c in enumerate(ctxs):
        for j, m in enumerate(models):
            ax = axes[i][j]
            for kv in ('f16', 'q8_0', 'q4_0'):
                pts = sorted((k[4], v) for k, v in cs.items() if k[1:4] == (m, kv, c))
                if pts:
                    ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts],
                                yerr=[0 if math.isnan(p[1][1]) else p[1][1] for p in pts],
                                color=KV_COLOR[kv], marker='o', capsize=3, lw=2, label=KV_LABEL[kv])
                for n in sorted({r['n_parallel'] for r in fails
                                 if (r['model'], r['kv_type'], r['context_len']) == (m, kv, c)}):
                    y = ax.get_ylim()[0]
                    ax.plot([n], [y], 'x', color=KV_COLOR[kv], ms=12, mew=3, clip_on=False)
                    ax.annotate(f'{kv}: out of memory', (n, y), textcoords='offset points', xytext=(-8, 10),
                                ha='right', fontsize=8, color=KV_COLOR[kv])
            ax.set_xscale('log', base=2)
            ax.set_xlabel('parallel sessions')
            ax.set_ylabel('request energy per generated token (J, net)')
            ax.set_title(f'{SHORT.get(m, m)} · {c} tokens per session', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle('Concurrency: energy per generated token (x = did not fit)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def fit_plot(rows, path):
    fit = [r for r in rows if '-fit' in r['tag']]
    if not fit:
        return
    models = sorted({r['model'] for r in fit})
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.5), squeeze=False)
    cs = cell_stats(fit, 'energy_per_request_net_j')
    for j, m in enumerate(models):
        ax = axes[0][j]
        for kv in ('f16', 'q8_0', 'q4_0'):
            pts = sorted((k[3], v) for k, v in cs.items() if k[1:3] == (m, kv))
            if pts:
                ax.plot([p[0] for p in pts], [p[1][0] for p in pts], color=KV_COLOR[kv], marker='o', lw=2,
                        label=KV_LABEL[kv])
            failed = sorted({r['context_len'] for r in fit if r['model'] == m and r['kv_type'] == kv
                             and r['status'] != 'ok'})
            for c in failed:
                ax.axvline(c, color=KV_COLOR[kv], ls=':', lw=1.5)
                ax.text(c, 0.02, f' {kv} failed', color=KV_COLOR[kv], rotation=90, va='bottom',
                        transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('context length (tokens)')
        ax.set_ylabel('energy per request (J, net)')
        ax.set_title(SHORT.get(m, m), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Long contexts: energy per request, and where each format fails to load')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


# ---------------------------------------------------------------- llama-bench, perplexity, profiles

def load_pptg(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    d = defaultdict(list)
    for r in rows:
        if r.get('status') == 'ok':
            d[(r['tag'], r['model'], r['kv_type'], r['test'], r['n'], r['depth'])].append(r)
    out = {}
    for k, rs in d.items():
        out[k] = {m: ci([r[m] for r in rs]) for m in ('tps', 'j_per_token', 'j_per_token_gross', 'avg_w',
                                                      'gpu_load', 'cpu_busy') if all(r.get(m) is not None for r in rs)}
    return out


def load_ppl(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            r.setdefault('type_k', r.get('kv_type'))
            r.setdefault('type_v', r.get('kv_type'))
            r.setdefault('build', 'build')
            r['_file'] = Path(p).name
            rows.append(r)
    return rows


DECODE_CALLS = 65  # profile = fill 16,384 tokens, then 1 warm-up + 64 generated tokens


def kernel_summary(prof_dir):
    """Per profile: where attention time goes. Decode kernels are the ones launched once per layer per
    generated token (instances divisible by 65); prefill ones once per layer per 512-token batch."""
    out = {}
    for p in sorted(glob.glob(str(prof_dir / '*cuda_gpu_kern_sum*.csv'))):
        tag = Path(p).name.split('_cuda_gpu_kern_sum')[0]
        d = {'total_ms': 0.0, 'decode_attn_ms': 0.0, 'decode_attn_calls': 0, 'decode_attn_kernel': '',
             'prefill_attn_ms': 0.0, 'kv_quant_ms': 0.0, 'softmax_gemm_ms': 0.0}
        for r in csv.DictReader(open(p)):
            name, t, n = r['Name'], float(r['Total Time (ns)']) / 1e6, int(float(r['Instances']))
            d['total_ms'] += t
            if name.lstrip('void ').startswith('flash_attn_ext'):
                if n % DECODE_CALLS == 0:
                    d['decode_attn_ms'] += t
                    d['decode_attn_calls'] += n
                    kern = name.split('(')[0].replace('void ', '')
                    d['decode_attn_kernel'] = ('vec (no tensor cores)' if '_vec' in kern else
                                               'f16 mma (tensor cores)')
                else:
                    d['prefill_attn_ms'] += t
            elif any(x in name for x in ('dequantize_block', 'fwht_cuda', 'k_set_rows_quant')):
                d['kv_quant_ms'] += t
            elif any(x in name for x in ('soft_max', 'gemm', 'cutlass', 'convert_unary')):
                d['softmax_gemm_ms'] += t
        if d['decode_attn_calls']:
            d['decode_attn_us_per_call'] = 1000 * d['decode_attn_ms'] / d['decode_attn_calls']
            d['decode_attn_ms_per_token'] = d['decode_attn_ms'] / DECODE_CALLS
        out[tag] = d
    return out


def kernel_plot(ks, path):
    tags = [t for t in ks if ks[t].get('decode_attn_calls')]
    if not tags:
        return
    models = [m for m in ('Llama-3.2-1B-Instruct-Q8_0', 'qwen2.5-0.5b-instruct-q8_0') if any(t.startswith(m) for t in tags)]
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        kvs = [kv for kv in ('f16', 'q8_0', 'q4_0') if f'{m}_{kv}_faon' in ks]
        vals = [ks[f'{m}_{kv}_faon']['decode_attn_us_per_call'] for kv in kvs]
        bars = ax.bar(range(len(kvs)), vals, color=[KV_COLOR[k] for k in kvs])
        ax.bar_label(bars, [f'{v:.0f} µs' for v in vals], fontsize=8)
        ax.set_xticks(range(len(kvs)), [KV_LABEL[k].replace(' (', '\n(') for k in kvs], fontsize=8)
        ax.set_ylabel('decode attention kernel time per call (µs)')
        ax.set_title(f'{SHORT.get(m, m)}: one layer, one token, 16K cache', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Decode attention at 16K: tensor-core kernel (f16) vs vec kernel (8-bit, 4-bit)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


SIZE_LABEL = {'qwen2.5-0.5b-instruct-q4_k_m': 'Qwen\n0.5B', 'qwen2.5-1.5b-instruct-q4_k_m': 'Qwen\n1.5B',
              'qwen2.5-3b-instruct-q4_k_m': 'Qwen\n3B', 'Llama-3.2-1B-Instruct-Q4_K_M': 'Llama\n1B',
              'Llama-3.2-3B-Instruct-Q4_K_M': 'Llama\n3B', 'gemma-3-1b-it-Q4_K_M': 'Gemma\n1B'}


def _change_bars(ax, ch, tag, metric, series, c=16384):
    """Grouped bars of the paired change vs f16 for SIZE_MODELS; series = [(kv, label, color)]."""
    models = [m for m in SIZE_MODELS if (tag, m, series[0][0], c, 1) in ch]
    w = 0.8 / len(series)
    for i, (kv, label, color) in enumerate(series):
        cells = [ch.get((tag, m, kv, c, 1)) for m in models]
        xs = [j + (i - (len(series) - 1) / 2) * w for j in range(len(models))]
        vals = [x[0] if x else 0 for x in cells]
        bars = ax.bar(xs, vals, w * 0.92, color=color, label=label,
                      yerr=[x[1] if x else 0 for x in cells], capsize=2, error_kw={'lw': 0.8, 'ecolor': '#555'})
        ax.bar_label(bars, [f'+{v:.0f}%' for v in vals], fontsize=7, padding=2)
    for x in (2.5, 4.5):  # family separators
        ax.axvline(x, color='#ccc', lw=0.8)
    ax.set_xticks(range(len(models)), [SIZE_LABEL[m] for m in models], fontsize=8)
    ax.axhline(0, color='#888', lw=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)


def size_family_plot(rows, path):
    tag = '15W-size'
    if not any(r['tag'] == tag for r in rows):
        return
    series = [('q8_0', KV_LABEL['q8_0'], KV_COLOR['q8_0']), ('q4_0', KV_LABEL['q4_0'], KV_COLOR['q4_0'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, metric, title in ((axes[0], 'energy_per_request_net_j', 'Energy per request (net)'),
                              (axes[1], 'decode_j_per_token', 'Decode energy per token (net)')):
        _change_bars(ax, paired_change(rows, metric), tag, metric, series)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('change vs f16 cache at 16K (%)')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Model size and family (4-bit weights, 15W): extra energy of a quantized KV cache at 16K, '
                 'paired, 95% CI', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def gen_length_plot(rows, path):
    if not any(r['tag'] == '15W-gen512' for r in rows):
        return
    ch = paired_change(rows, 'energy_per_request_net_j')
    # relabel the 512-token set into the size set's key space so both draw as series of one chart
    merged = dict(ch)
    for (tag, m, kv, c, n), v in ch.items():
        if tag == '15W-gen512':
            merged[('15W-size', m, kv + '@512', c, n)] = v
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, kv in zip(axes, ('q8_0', 'q4_0')):
        _change_bars(ax, merged, '15W-size', 'energy_per_request_net_j',
                     [(kv, '128 generated tokens', '#1E8C76'), (kv + '@512', '512 generated tokens', '#8E4FB5')])
        ax.set_title(f'{KV_LABEL[kv]}, 16K prompt', fontsize=10)
    axes[0].set_ylabel('change in energy per request vs f16 (%)')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Longer replies: the penalty per request grows 1.7–2.2× with 512 generated tokens (Gemma: unchanged) '
                 '(paired, 95% CI)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def evict_plot(rows, path):
    tag = '15W-evict'
    ev = [r for r in rows if r['tag'] == tag]
    if not ev:
        return
    models = [m for m in ('Qwen2.5-0.5B-Instruct', 'Llama-3.2-1B-Instruct') if any(r['model'] == m for r in ev)]
    ch = paired_change(rows, 'energy_per_request_net_j')
    series = [('streamingllm@0.5', 'StreamingLLM, 50% removed', '#1baf7a', '-'),
              ('streamingllm@0.75', 'StreamingLLM, 75% removed', '#1baf7a', '--'),
              ('snapkv@0.5', 'SnapKV, 50% removed', '#6250d6', '-'),
              ('snapkv@0.75', 'SnapKV, 75% removed', '#6250d6', '--')]
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for kv, label, color, ls in series:
            pts = sorted((k[3], v) for k, v in ch.items() if k[:3] == (tag, m, kv) and k[4] == 1)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ax.errorbar(xs, [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], color=color, ls=ls, lw=2,
                        marker='o', ms=6, capsize=2, label=label)
        ax.axhline(0, color='#888', lw=0.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1024, 4096, 8192, 16384], ['1K', '4K', '8K', '16K'])
        ax.set_xlabel('prompt length')
        ax.set_title(SHORT.get(m, m), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('change in energy per request vs no eviction (%)')
    axes[0][0].legend(fontsize=8)
    fig.suptitle('Eviction (kvpress, PyTorch, 15W) vs no eviction: energy per request, paired 95% CI', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(ROOT / 'results'))
    ap.add_argument('--out', default=str(ROOT / 'plots_v2'))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kv_files = sorted(p for p in glob.glob(str(res / 'kv_*.jsonl')) if '190420' not in p)  # skip v1 quick test
    rows = load_runs(kv_files)
    evict_files = sorted(glob.glob(str(res / 'evict_*.jsonl')))  # eviction (kvpress), kv_type f16 = no eviction
    rows += load_runs(evict_files)
    kv_files += evict_files
    v2 = [r for r in rows if r['tag'].endswith('-v2')]
    v1 = [r for r in rows if r['tag'] in ('15W', '7W')]
    print(f'{len(rows)} rows ({len(v1)} v1, {len(v2)} v2 main grid)')

    stats = {}
    metrics = ['energy_per_request_net_j', 'energy_per_request_j', 'decode_j_per_token',
               'decode_j_per_token_gross', 'decode_tps', 'prefill_tps', 'decode_avg_w', 'prefill_avg_w',
               'cpu_busy_decode', 'gpu_load_decode', 'total_j_per_token', 'total_j_per_token_gross',
               'vdd_cpu_gpu_cv_decode_energy_net_j', 'vdd_soc_decode_energy_net_j', 'vdd_in_decode_energy_net_j',
               'vdd_cpu_gpu_cv_energy_net_j', 'vdd_soc_energy_net_j', 'kv_used_mib', 'kv_alloc_mib',
               'min_mem_available_mb', 'gpu_temp_end', 'gpu_freq_mhz_end']
    for m in metrics:
        stats[m] = {'|'.join(map(str, k)): v for k, v in cell_stats(rows, m).items()}
    for m in ('energy_per_request_net_j', 'energy_per_request_j', 'decode_j_per_token', 'decode_j_per_token_gross',
              'decode_tps', 'decode_avg_w', 'total_j_per_token'):
        stats[f'change_{m}'] = {'|'.join(map(str, k)): v for k, v in paired_change(rows, m).items()}
    stats['failures'] = [{k: r.get(k) for k in ('tag', 'model', 'kv_type', 'context_len', 'n_parallel', 'round',
                                                 'status', 'kv_alloc_mib', 'log_tail')}
                         for r in rows if r['status'] != 'ok']
    stats['pptg'] = {'|'.join(map(str, k)): v for k, v in load_pptg(sorted(glob.glob(str(res / 'pptg_*.jsonl')))).items()}
    stats['ppl'] = load_ppl(sorted(glob.glob(str(res / 'ppl*.csv'))))
    stats['kernels'] = kernel_summary(res / 'prof')
    stats['meta'] = {Path(p).name: json.loads(Path(p).with_suffix('.meta.json').read_text())
                     for p in kv_files + sorted(glob.glob(str(res / 'pptg_*.jsonl')))
                     if Path(p).with_suffix('.meta.json').exists()}
    (out / 'stats.json').write_text(json.dumps(stats, indent=1, default=str))
    print('wrote', out / 'stats.json')

    main_tags = {r['tag'] for r in v2} or {'15W', '7W'}
    grid_plot(rows, 'energy_per_request_net_j', 'change in energy per request vs f16 (%)',
              'Per-request energy (16K prompt + 128 tokens) vs f16, paired by round, 95% CI',
              out / '01_request_energy_change.png', change=True, tags=main_tags)
    grid_plot(rows, 'decode_j_per_token', 'change in decode J/token vs f16 (%)',
              'Decode energy per token vs f16, paired by round, 95% CI (hollow = not separable from 0)',
              out / '02_decode_energy_change.png', change=True, tags=main_tags)
    grid_plot(rows, 'decode_tps', 'decode speed (tokens/s)', 'Decode speed, mean and 95% CI',
              out / '03_decode_tps.png', tags=main_tags)
    grid_plot(rows, 'decode_avg_w', 'decode-only average power, VDD_IN (W, gross)',
              'Average module power during decoding only', out / '04_decode_power.png', tags=main_tags)
    grid_plot(rows, 'energy_per_request_net_j', 'energy per request (J, net)', 'Energy per request',
              out / '05_request_energy.png', tags=main_tags)
    w4 = {r['tag'] for r in rows if r['tag'].endswith('-w4')}
    if w4:
        grid_plot(rows, 'energy_per_request_net_j', 'change in energy per request vs f16 (%)',
                  'Q4_K_M weights: per-request energy vs f16 KV, paired, 95% CI',
                  out / '06_w4_request_change.png', change=True, tags=w4, kvs=('q8_0', 'q4_0'))
    concurrency_plot(rows, out / '07_concurrency.png')
    fit_plot(rows, out / '08_fit.png')
    kernel_plot(stats['kernels'], out / '09_kernels.png')
    size_family_plot(rows, out / '10_size_family.png')
    gen_length_plot(rows, out / '11_gen512.png')
    evict_plot(rows, out / '12_eviction.png')


if __name__ == '__main__':
    main()
