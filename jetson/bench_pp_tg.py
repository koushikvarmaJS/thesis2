"""Prefill and decode energy measured separately with llama-bench (no timestamp split).

llama-bench runs prompt processing (pp) and token generation (tg) as separate tests.
With --progress it prints a line just before each timed repetition, e.g.
  "llama-bench: benchmark 1/1: generation run 3/5"
and reports each repetition's duration (samples_ns). The energy of a repetition is the
INA3221 energy from that line's timestamp over samples_ns. A tg test at depth d first fills
the KV cache with d tokens (untimed, "depth run"), so "tg128 @ d16384" is pure decode energy
at a 16K context.

Usage: python3 jetson/bench_pp_tg.py --tag 15W [--depths 1024,16384] [--reps 5]
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from bench_kv import ROOT, parse_cfg, system_meta
from bench_kv import measure_idle
from power import PowerMeter

RUN_RX = re.compile(r'benchmark \d+/\d+: (prompt|generation) run (\d+)/(\d+)')


def run_bench(bench, model, cfg, test, reps, meter, idle_w, rail, log_path):
    """test = ('pp', n_prompt, 0) or ('tg', n_gen, depth). Returns one row per repetition."""
    k, v, fa = parse_cfg(cfg)
    kind, n, depth = test
    cmd = [str(bench), '-m', str(model), '-ngl', '99', '-fa', fa, '-ctk', k, '-ctv', v,
           '-r', str(reps), '--progress', '-o', 'jsonl',
           '-p', str(n if kind == 'pp' else 0), '-n', str(n if kind == 'tg' else 0), '-d', str(depth)]
    starts = []
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        for line in proc.stderr:
            t = time.monotonic()
            log.write(line)
            if RUN_RX.search(line):
                starts.append(t)
        out = proc.stdout.read()
        proc.wait()
    res = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    if proc.returncode != 0 or not res:
        return [{'status': 'failed', 'returncode': proc.returncode}]
    samples = res[-1]['samples_ns']
    rows = []
    for i, (t0, ns) in enumerate(zip(starts, samples)):
        t1 = t0 + ns / 1e9
        e, n_s = meter.energy(t0, t1, rail)
        net = e - idle_w[rail] * (t1 - t0)
        rows.append({'status': 'ok', 'rep': i, 'seconds': ns / 1e9, 'tokens': n,
                     'tps': n / (ns / 1e9), 'energy_j': e, 'energy_net_j': net,
                     'j_per_token': net / n, 'j_per_token_gross': e / n, 'n_samples': n_s,
                     'avg_w': meter.mean_power(t0, t1, rail),
                     'gpu_load': meter.gpu_load(t0, t1), 'cpu_busy': meter.cpu_busy(t0, t1),
                     **{f'{r.lower()}_energy_net_j': meter.energy(t0, t1, r)[0] - idle_w[r] * (t1 - t0)
                        for r in meter.rails()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default=str(ROOT / 'llama.cpp/build/bin/llama-bench'))
    ap.add_argument('--models', nargs='+', default=sorted(str(p) for p in (ROOT / 'models').glob('*.gguf')))
    ap.add_argument('--kv', default='f16,f16@off,q8_0,q4_0')
    ap.add_argument('--prompts', default='1024,16384', help='pp test sizes')
    ap.add_argument('--depths', default='1024,16384', help='context depth for the tg test')
    ap.add_argument('--gen', type=int, default=128)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--rail', default='VDD_IN')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = ROOT / f'results/pptg_{stamp}.jsonl'
    log_dir = ROOT / 'results/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = {**system_meta(), 'tag': args.tag, 'kv_types': args.kv.split(','), 'reps': args.reps,
            'prompts': args.prompts, 'depths': args.depths, 'gen': args.gen, 'started': stamp}
    meter = PowerMeter().start()
    idle_w, _ = measure_idle(meter, 10)

    tests = [('pp', int(p), 0) for p in args.prompts.split(',')] + \
            [('tg', args.gen, int(d)) for d in args.depths.split(',')]
    jobs = [(m, c, t) for m in args.models for c in args.kv.split(',') for t in tests]
    random.shuffle(jobs)  # no fixed order between formats
    for i, (model, cfg, test) in enumerate(jobs, 1):
        mname = Path(model).stem
        log_path = log_dir / f'pptg_{stamp}_{mname}_{cfg.replace("/", "-")}_{test[0]}{test[1]}_d{test[2]}.log'
        time.sleep(2)
        rows = run_bench(args.bench, model, cfg, test, args.reps, meter, idle_w, args.rail, log_path)
        with open(out_path, 'a') as f:
            for r in rows:
                f.write(json.dumps({'model': mname, 'kv_type': cfg, 'test': test[0], 'n': test[1],
                                    'depth': test[2], 'tag': args.tag, 'idle_w': idle_w[args.rail], **r}) + '\n')
        ok = [r for r in rows if r['status'] == 'ok']
        if ok:
            print(f'[{i}/{len(jobs)}] {mname} {cfg:9s} {test[0]}{test[1]}@d{test[2]}: '
                  f'{sum(r["tps"] for r in ok) / len(ok):.1f} tok/s  '
                  f'{sum(r["j_per_token"] for r in ok) / len(ok):.4f} J/tok', flush=True)
        else:
            print(f'[{i}/{len(jobs)}] {mname} {cfg} {test}: FAILED (see {log_path})', flush=True)
    idle_end, _ = measure_idle(meter, 10)
    meter.stop()
    meta.update({'idle_w': idle_w, 'idle_w_end': idle_end, 'finished': datetime.now().isoformat()})
    out_path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=1))
    print('->', out_path)


if __name__ == '__main__':
    sys.exit(main())
