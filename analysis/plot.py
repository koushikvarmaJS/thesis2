"""Turn Jetson benchmark JSONL (+ optional perplexity CSV) into plots and a summary table.

Usage:  python3 analysis/plot.py results/kv_*.jsonl [--ppl results/ppl_*.csv] [--out plots]
Needs only matplotlib (pip3 install matplotlib).
"""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

KV_ORDER = ['f16', 'q8_0', 'q4_0']
KV_STYLE = {'f16': ('#4A5568', 'o', 'f16 (no compression)'),
            'q8_0': ('#2F6FA8', 's', 'q8_0 (8-bit KV)'),
            'q4_0': ('#E07A2E', '^', 'q4_0 (4-bit KV)')}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def agg(rows, metric):
    """{(group, model, kv, ctx): (mean, std, n)}; group = tag or nvpmodel."""
    d = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None:
            d[(r.get('tag') or r.get('nvpmodel', ''), r['model'], r['kv_type'], r['context_len'])].append(v)
    return {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) for k, v in d.items()}


def line_plot(rows, metric, ylabel, title, path, pct_vs_f16=False):
    a = agg(rows, metric)
    groups = sorted({k[0] for k in a}, key=lambda g: (-int(g.rstrip('W')) if g.rstrip('W').isdigit() else 0, g))
    models = sorted({k[1] for k in a})
    fig, axes = plt.subplots(len(groups), len(models), figsize=(6 * len(models), 4.2 * len(groups)),
                             squeeze=False, sharey='row' if pct_vs_f16 else False)
    for ax, (grp, model) in zip(axes.flat, [(g, m) for g in groups for m in models]):
        for kv in KV_ORDER:
            pts = sorted((k[3], v) for k, v in a.items() if k[:3] == (grp, model, kv))
            if not pts or (pct_vs_f16 and kv == 'f16'):
                continue
            xs = [p[0] for p in pts]
            if pct_vs_f16:
                base = {k[3]: v[0] for k, v in a.items() if k[:3] == (grp, model, 'f16')}
                ys = [100 * (base[x] - v[0]) / base[x] for x, v in pts if x in base]
                xs = [x for x, _ in pts if x in base]
                es = None
            else:
                ys = [v[0] for _, v in pts]
                es = [v[1] for _, v in pts]
            c, m, lab = KV_STYLE.get(kv, ('#888', 'x', kv))
            ax.errorbar(xs, ys, yerr=es, color=c, marker=m, label=lab, capsize=3, lw=2)
        if pct_vs_f16:
            ax.axhline(0, color='#B03A2E', ls='--', lw=1)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('context length (tokens)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{model}\n{grp}', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def kv_mem_plot(rows, path):
    d, ctx_of = {}, {}
    for r in rows:  # KV memory actually filled at each model's largest context
        if r.get('kv_used_mib') is None:
            continue
        if r['context_len'] >= ctx_of.get(r['model'], 0):
            if r['context_len'] > ctx_of.get(r['model'], 0):
                d = {k: v for k, v in d.items() if k[0] != r['model']}
            ctx_of[r['model']] = r['context_len']
            d[(r['model'], r['kv_type'])] = r['kv_used_mib']
    if not d:
        return
    models = sorted({k[0] for k in d})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.25
    for i, kv in enumerate(KV_ORDER):
        ys = [d.get((m, kv), 0) for m in models]
        bars = ax.bar([j + (i - 1) * w for j in range(len(models))], ys, w,
                      color=KV_STYLE[kv][0], label=KV_STYLE[kv][2])
        ax.bar_label(bars, fmt='%.0f', fontsize=8)
    ax.set_xticks(range(len(models)), [f'{m}\n@{ctx_of[m]} tokens' for m in models], fontsize=8)
    ax.set_ylabel('KV cache size (MiB)')
    ax.set_title('KV cache memory at the largest context tested')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def summary(rows, ppl_rows, path):
    keys = ['decode_j_per_token', 'decode_tps', 'prefill_tps', 'energy_net_j', 'vdd_in_avg_w']
    A = {k: agg(rows, k) for k in keys}
    ppl = {(r['model'], r['kv_type']): r['ppl'] for r in ppl_rows}
    lines = ['| setting | model | KV | ctx | decode J/tok | saving vs f16 | decode tok/s | prefill tok/s | '
             'request J (net) | avg W | WikiText-2 PPL | n |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in sorted(A['decode_j_per_token'], key=lambda k: (k[0], k[1], k[3], KV_ORDER.index(k[2]) if k[2] in KV_ORDER else 9)):
        grp, model, kv, ctx = k
        j = A['decode_j_per_token'][k]
        base = A['decode_j_per_token'].get((grp, model, 'f16', ctx))
        save = f'{100 * (base[0] - j[0]) / base[0]:+.1f}%' if base and kv != 'f16' else '—'
        g = lambda m: A[m].get(k, (float('nan'),))[0]  # noqa: E731
        lines.append(f'| {grp} | {model} | {kv} | {ctx} | {j[0]:.4f} ± {j[1]:.4f} | {save} | '
                     f'{g("decode_tps"):.1f} | {g("prefill_tps"):.0f} | {g("energy_net_j"):.1f} | '
                     f'{g("vdd_in_avg_w"):.2f} | {ppl.get((model, kv), "")} | {j[2]} |')
    Path(path).write_text('\n'.join(lines) + '\n')
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', nargs='+')
    ap.add_argument('--ppl', nargs='*', default=[])
    ap.add_argument('--out', default='plots')
    args = ap.parse_args()
    rows = load(args.jsonl)
    ppl_rows = []
    for p in args.ppl:
        with open(p) as f:
            ppl_rows += list(csv.DictReader(f))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(rows)} runs loaded')

    line_plot(rows, 'decode_j_per_token', 'decode energy per token (J, idle-subtracted)',
              'Decode energy per token vs context length', out / '01_decode_j_per_token.png')
    line_plot(rows, 'decode_j_per_token', 'energy saving vs f16 KV (%)',
              'Decode energy saving from KV quantization', out / '02_saving_vs_f16.png', pct_vs_f16=True)
    line_plot(rows, 'decode_tps', 'decode throughput (tokens/s)',
              'Decode speed vs context length', out / '03_decode_tps.png')
    line_plot(rows, 'prefill_tps', 'prefill throughput (tokens/s)',
              'Prefill speed vs context length', out / '04_prefill_tps.png')
    line_plot(rows, 'vdd_in_avg_w', 'average module power VDD_IN (W)',
              'Average power during a request', out / '05_avg_power.png')
    kv_mem_plot(rows, out / '06_kv_cache_mib.png')
    summary(rows, ppl_rows, out / 'summary.md')


if __name__ == '__main__':
    main()
