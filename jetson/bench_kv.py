"""Energy benchmark of KV-cache quantization (llama.cpp) on Jetson Orin Nano.

For every (model, KV cache type, context length) it sends one long-context
request to a local llama-server and measures, from the INA3221 power monitor:
  * total / prefill / decode energy (J), idle-subtracted and raw
  * decode energy per generated token (J/token)
  * prefill and decode throughput (tokens/s, from llama.cpp's own timings)
  * KV cache size reported by llama.cpp (MiB) and system memory used
  * GPU temperature and clock (to spot throttling)

Design choices that fix problems found in the thesis data:
  * ignore_eos=True  -> every run generates exactly --gen tokens (no early-EOS skew)
  * cache_prompt=False -> the prompt is really processed every run
  * rounds with shuffled method/context order -> no "baseline always first" bias

Each run is appended to a JSONL file as soon as it finishes, so a crash loses nothing.
"""
import argparse
import functools
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from power import PowerMeter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ---------------------------------------------------------------- helpers

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f'ERR {e!r}'


def http(port, path, payload=None, timeout=600):
    url = f'http://127.0.0.1:{port}{path}'
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def system_meta():
    return {
        'hostname': socket.gethostname(),
        'l4t_release': sh('head -1 /etc/nv_tegra_release'),
        'jetpack': sh("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null"),
        'nvpmodel': sh('nvpmodel -q 2>/dev/null | head -1'),
        'kernel': sh('uname -r'),
        'model_name': sh("tr -d '\\0' < /proc/device-tree/model"),
    }


# ---------------------------------------------------------------- llama-server

class Server:
    def __init__(self, llama_bin, model, kv, ctx, log_path, extra=()):
        self.port = free_port()
        self.log_path = log_path
        base = [str(llama_bin / 'llama-server'), '-m', str(model), '-c', str(ctx),
                '-ngl', '99', '-np', '1', '-ctk', kv, '-ctv', kv,
                '--host', '127.0.0.1', '--port', str(self.port), *extra]
        # Newer llama.cpp: "-fa on"; older: bare "-fa". Quantized V cache needs flash attention.
        self.variants = [base + ['-fa', 'on'], base + ['-fa']]
        self.proc = None

    def start(self, timeout=180):
        for cmd in self.variants:
            self.log = open(self.log_path, 'w')
            self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.proc.poll() is not None:
                    break  # died: try next variant
                try:
                    if http(self.port, '/health', timeout=2).get('status') == 'ok':
                        self.cmd = cmd
                        return True
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.5)
            self.stop()
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, 'log', None):
            self.log.close()



# ---------------------------------------------------------------- KV cache size

# bytes per stored element: q8_0 / q4_0 store 32 values + one fp16 scale per block
KV_BYTES = {'f32': 4, 'f16': 2, 'bf16': 2, 'q8_0': 34 / 32, 'q5_0': 22 / 32, 'q5_1': 24 / 32,
            'q4_0': 18 / 32, 'q4_1': 20 / 32, 'iq4_nl': 18 / 32}


@functools.lru_cache(maxsize=None)
def gguf_metadata(path, wanted_suffixes=('block_count', 'head_count', 'head_count_kv',
                                         'embedding_length', 'key_length', 'value_length')):
    """Read the few architecture keys we need from a GGUF header."""
    import struct
    scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    out = {}
    with open(path, 'rb') as f:
        rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))[0]  # noqa: E731
        rstr = lambda: f.read(rd('Q')).decode('utf-8', 'replace')  # noqa: E731

        def value(t):
            if t in scalar:
                return rd(scalar[t])
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd('I'), rd('Q')
                if et in scalar:  # skip numeric arrays fast
                    data = f.read(struct.calcsize(scalar[et]) * n)
                    return list(struct.unpack('<' + scalar[et] * n, data)) if n < 4096 else None
                items = [value(et) for _ in range(n)]  # e.g. tokenizer strings: must read through
                return items if n < 4096 else None
            raise ValueError(f'unknown gguf type {t}')

        if f.read(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        rd('I'); rd('Q')  # version, tensor count
        for _ in range(rd('Q')):
            key, t = rstr(), rd('I')
            v = value(t)
            if key == 'general.architecture' or key.split('.')[-1] in wanted_suffixes:
                out[key] = v
    return out


def kv_cache_mib(model_path, n_tokens, kv_type):
    """Exact KV cache size for n_tokens: tokens x layers x kv_heads x head_dim x (K+V) x bytes."""
    try:
        md = gguf_metadata(model_path)
        a = md['general.architecture']
        n_layer = md[f'{a}.block_count']
        n_head = md[f'{a}.attention.head_count']
        n_kv = md[f'{a}.attention.head_count_kv']
        n_kv = max(n_kv) if isinstance(n_kv, list) else n_kv
        n_head = max(n_head) if isinstance(n_head, list) else n_head
        k_dim = md.get(f'{a}.attention.key_length') or md[f'{a}.embedding_length'] // n_head
        v_dim = md.get(f'{a}.attention.value_length') or k_dim
        return n_tokens * n_layer * n_kv * (k_dim + v_dim) * KV_BYTES[kv_type] / 2**20
    except Exception as e:  # noqa: BLE001
        print(f'  (could not compute KV size: {e!r})')
        return None


# ---------------------------------------------------------------- one measurement

def measure(meter, srv, prompt_ids, gen, idle_w, rail):
    payload = {'prompt': prompt_ids, 'n_predict': gen, 'ignore_eos': True,
               'temperature': 0.0, 'cache_prompt': False, 'stream': False}
    temps0 = meter.temps()
    t0 = time.monotonic()
    resp = http(srv.port, '/completion', payload)
    t1 = time.monotonic()
    tm = resp.get('timings', {})
    prompt_s = tm.get('prompt_ms', 0) / 1000.0
    t_split = min(t1, t0 + prompt_s)  # approx. end of prefill

    out = {
        'prompt_tokens': tm.get('prompt_n'),
        'gen_tokens': tm.get('predicted_n'),
        'latency_s': t1 - t0,
        'prefill_s': prompt_s,
        'decode_s': tm.get('predicted_ms', 0) / 1000.0,
        'prefill_tps': tm.get('prompt_per_second'),
        'decode_tps': tm.get('predicted_per_second'),
        'gpu_temp_start': temps0.get('gpu-thermal') or temps0.get('GPU-therm'),
        'gpu_temp_end': (lambda t: t.get('gpu-thermal') or t.get('GPU-therm'))(meter.temps()),
        'gpu_freq_mhz_end': meter.gpu_freq_mhz(),
        'min_mem_available_mb': meter.min_mem_available(t0, t1),
    }
    for r in meter.rails():
        e, n = meter.energy(t0, t1, r)
        ep, _ = meter.energy(t0, t_split, r)
        ed, _ = meter.energy(t_split, t1, r)
        key = r.lower()
        out[f'{key}_energy_j'] = e
        out[f'{key}_avg_w'] = meter.mean_power(t0, t1, r)
        if r == rail:
            idle = idle_w.get(r, 0.0)
            out['n_samples'] = n
            out['energy_j'] = e
            out['energy_net_j'] = e - idle * (t1 - t0)
            out['prefill_energy_net_j'] = ep - idle * (t_split - t0)
            out['decode_energy_net_j'] = ed - idle * (t1 - t_split)
            g = out['gen_tokens'] or 0
            out['decode_j_per_token'] = out['decode_energy_net_j'] / g if g else None
            out['total_j_per_token'] = out['energy_net_j'] / g if g else None
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llama-bin', default=str(ROOT / 'llama.cpp/build/bin'))
    ap.add_argument('--models', nargs='+', default=sorted(str(p) for p in (ROOT / 'models').glob('*.gguf')))
    ap.add_argument('--kv', default='f16,q8_0,q4_0', help='KV cache types to compare')
    ap.add_argument('--contexts', default='1024,4096,8192,16384')
    ap.add_argument('--gen', type=int, default=128, help='tokens generated per request')
    ap.add_argument('--rounds', type=int, default=3, help='repeats (each in shuffled order)')
    ap.add_argument('--corpus', default=str(ROOT / 'data/corpus.txt'))
    ap.add_argument('--rail', default='VDD_IN', help='rail used for headline energy')
    ap.add_argument('--idle-secs', type=float, default=10)
    ap.add_argument('--cooldown', type=float, default=2)
    ap.add_argument('--tag', default='', help='free-text label, e.g. "25W"')
    ap.add_argument('--quick', action='store_true', help='1 model, 2 contexts, 1 round (~5 min)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.quick:
        args.models = args.models[:1]
        args.contexts = '1024,4096'
        args.rounds = 1
    kvs = args.kv.split(',')
    ctxs = [int(c) for c in args.contexts.split(',')]
    llama_bin = Path(args.llama_bin)
    if not (llama_bin / 'llama-server').exists():
        sys.exit(f'llama-server not found in {llama_bin} (run setup.sh first)')
    if not args.models:
        sys.exit('no models found in models/ (run setup.sh first)')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(args.out or ROOT / f'results/kv_{stamp}.jsonl')
    log_dir = out_path.parent / 'logs'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = system_meta()
    meta.update({'tag': args.tag, 'gen': args.gen, 'kv_types': kvs, 'contexts': ctxs,
                 'rounds': args.rounds, 'started': stamp})
    print(json.dumps(meta, indent=1))

    meter = PowerMeter().start()
    print(f'power backend: {meter.backend}; rails: {meter.rails()}')
    if args.rail not in meter.rails():
        sys.exit(f'rail {args.rail} not found; choose one of {meter.rails()} with --rail')

    # Idle baseline: nothing running.
    print(f'measuring idle power for {args.idle_secs:.0f}s ...')
    t0 = time.monotonic()
    time.sleep(args.idle_secs)
    t1 = time.monotonic()
    idle_w = {r: meter.mean_power(t0, t1, r) for r in meter.rails()}
    idle_mem = meter.min_mem_available(t0, t1)
    print('idle W:', {k: round(v, 2) for k, v in idle_w.items()}, f'mem_avail={idle_mem:.0f}MB')

    corpus = Path(args.corpus).read_text(errors='ignore')
    n_done = 0
    total = len(args.models) * len(kvs) * len(ctxs) * args.rounds

    for model in args.models:
        mname = Path(model).stem
        prompts = None
        for rnd in range(args.rounds):
            order = kvs[:]
            random.shuffle(order)
            for kv in order:
                log_path = log_dir / f'{stamp}_{mname}_{kv}_r{rnd}.log'
                srv_ctx = max(ctxs) + args.gen + 64
                srv = Server(llama_bin, model, kv, srv_ctx, log_path)
                if not srv.start():
                    print(f'!! server failed: {mname} kv={kv} (see {log_path})')
                    continue
                try:
                    if prompts is None:  # tokenize once per model
                        ids = http(srv.port, '/tokenize', {'content': corpus})['tokens']
                        if len(ids) < max(ctxs):
                            sys.exit(f'corpus has only {len(ids)} tokens; need {max(ctxs)}')
                        prompts = {c: ids[:c] for c in ctxs}
                    kv_alloc = kv_cache_mib(model, srv_ctx, kv)
                    load_mem = meter.min_mem_available(time.monotonic() - 1, time.monotonic())
                    # warm-up (short) so CUDA kernels are loaded before measuring
                    http(srv.port, '/completion', {'prompt': prompts[min(ctxs)], 'n_predict': 16,
                                                   'cache_prompt': False, 'ignore_eos': True})
                    c_order = ctxs[:]
                    random.shuffle(c_order)
                    for ctx in c_order:
                        time.sleep(args.cooldown)
                        r = measure(meter, srv, prompts[ctx], args.gen, idle_w, args.rail)
                        row = {'model': mname, 'kv_type': kv, 'context_len': ctx, 'round': rnd,
                               'tag': args.tag, 'nvpmodel': meta['nvpmodel'],
                               'kv_alloc_mib': kv_alloc,
                               'kv_used_mib': kv_cache_mib(model, ctx + args.gen, kv),
                               'server_mem_mb': idle_mem - load_mem,
                               'idle_w': idle_w.get(args.rail), 'rail': args.rail,
                               'power_backend': meter.backend, **r}
                        with open(out_path, 'a') as f:
                            f.write(json.dumps(row) + '\n')
                        n_done += 1
                        print(f'[{n_done}/{total}] {mname} kv={kv:5s} ctx={ctx:6d} '
                              f'decode {r["decode_tps"]:.1f} tok/s  '
                              f'{(r.get("decode_j_per_token") or float("nan")):.3f} J/tok  '
                              f'avg {r.get(args.rail.lower() + "_avg_w", float("nan")):.1f} W  '
                              f'gpu {r["gpu_temp_end"]}C')
                except Exception as e:  # noqa: BLE001
                    print(f'!! error {mname} kv={kv}: {e!r}')
                finally:
                    srv.stop()

    meter.stop()
    meta_path = out_path.with_suffix('.meta.json')
    meta.update({'idle_w': idle_w, 'idle_mem_available_mb': idle_mem,
                 'power_backend': meter.backend, 'finished': datetime.now().isoformat()})
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f'\ndone: {n_done} runs -> {out_path}\nmeta -> {meta_path}')


if __name__ == '__main__':
    main()
