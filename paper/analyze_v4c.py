"""Numbers from the 17 Nov follow-up runs (run_v4c.sh, run_v4d.sh) for the paper -> paper/v4c.json

  prof:  ms per decode step by mode (loop / mask / generate / genpad0) and cache length, per model
  ppl:   set Q perplexities (Qwen2.5-1.5B, 8K)
  fix:   sets K, N, P rerun at >= 8K with pad_token_id = eos: baseline J and tok/s, paired changes
"""
import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'analysis'))
from analyze_v2 import key, load_runs, paired_change  # noqa: E402


def prof():
    out = {}
    for p in sorted(glob.glob(str(ROOT / 'results/profstep_*_2026111[7-9]_*.jsonl'))):
        for line in open(p):
            r = json.loads(line)
            m = r['model'].split('/')[-1].split('-')[0]  # Qwen2.5 / Llama
            out.setdefault(m, {}).setdefault(r['mode'], {})[str(r['context_len'])] = round(r['wall_ms_per_step'], 1)
    return out


def ppl():
    files = sorted(glob.glob(str(ROOT / 'results/ppl_v4_*.csv')))
    rows = {}
    for p in files:
        for r in csv.DictReader(open(p)):
            if r['ppl']:
                rows[f"{r['type_k']}/{r['type_v']}"] = [float(r['ppl']), float(r['ppl_err'])]
    return rows


def fix():
    tags = ('15W-evict-fix', '15W-pytq-fix', '15W-t4match-fix')
    files = [p for p in sorted(glob.glob(str(ROOT / 'results/evict_2026111[7-9]_*.jsonl')))]
    rows = [r for r in load_runs(files) if r.get('tag') in tags]
    out = {'change': {}, 'base': {}}
    for metric in ('energy_net_j', 'decode_j_per_token', 'prefill_energy_net_j'):
        for k, v in paired_change(rows, metric).items():
            tag, m, cfg, c, _ = k
            out['change'].setdefault(metric, {})[f'{tag}|{m}|{cfg}|{c}'] = [round(v[0], 1), round(v[1], 1), v[3]]
    acc = {}
    for r in rows:
        if r['status'] == 'ok':
            acc.setdefault('|'.join(map(str, key(r)[:4])), []).append(r)
    for k, rs in acc.items():
        n = len(rs)
        out['base'][k] = {'J': round(sum(r['energy_net_j'] for r in rs) / n, 1),
                          'tps': round(sum(r['decode_tps'] for r in rs) / n, 1),
                          'kv_mib': round(sum(r['kv_used_mib'] or 0 for r in rs) / n), 'n': n}
    return out


if __name__ == '__main__':
    res = {'prof': prof(), 'ppl': ppl(), 'fix': fix()}
    (HERE / 'v4c_numbers.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['prof'], indent=0)[:1500])
    print(res['ppl'])
    print(len(res['fix']['base']), 'fix cells')
