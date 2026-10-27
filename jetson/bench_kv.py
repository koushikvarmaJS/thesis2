"""Energy benchmark of KV-cache quantization (llama.cpp) on Jetson Orin Nano.

For every (model, KV cache config, context length, parallel sessions) it sends
long-context requests to a local llama-server and measures, from the INA3221 power monitor:
  * total / prefill / decode energy (J), gross and idle-subtracted (net), per rail
  * decode energy per generated token (J/token) and decode-only average power
  * prefill and decode throughput (tokens/s, from llama.cpp's own timings)
  * GPU load and CPU utilization during prefill and decode
  * KV cache size (calculated from the GGUF header) and system memory available
  * GPU temperature and clock (to spot throttling)

KV configs (--kv, comma-separated): "q8_0" = K and V both q8_0; "f16/q4_0" = K f16, V q4_0;
a suffix "@off" turns flash attention off (e.g. "f16@off"), default is on.

Design choices that fix problems found in the thesis data:
  * ignore_eos=True  -> every run generates exactly --gen tokens (no early-EOS skew)
  * cache_prompt=False -> the prompt is really processed every run
  * rounds with shuffled method/context order -> no "baseline always first" bias
  * a server that fails to start (e.g. KV cache does not fit) is recorded as a row, not skipped

Each run is appended to a JSONL file as soon as it finishes, so a crash loses nothing.
"""
import argparse
import functools
import json
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def parse_cfg(spec):
    """'q8_0' -> (q8_0, q8_0, on); 'f16/q4_0' -> (f16, q4_0, on); 'f16@off' -> (f16, f16, off)."""
    kv, _, fa = spec.partition('@')
    k, _, v = kv.partition('/')
    return k, v or k, fa or 'on'


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
    def __init__(self, llama_bin, model, cfg, ctx, log_path, n_par=1, extra=()):
        self.port = free_port()
        self.log_path = log_path
        k, v, fa = parse_cfg(cfg)
        base = [str(llama_bin / 'llama-server'), '-m', str(model), '-c', str(ctx),
                '-ngl', '99', '-np', str(n_par), '-ctk', k, '-ctv', v,
                '--host', '127.0.0.1', '--port', str(self.port), *extra]
        # Newer llama.cpp: "-fa on"; older: bare "-fa". Quantized V cache needs flash attention.
        self.variants = [base + ['-fa', 'on'], base + ['-fa']] if fa == 'on' else [base + ['-fa', 'off']]
        self.proc = None

    def start(self, timeout=180):
        for i, cmd in enumerate(self.variants):
            self.log = open(self.log_path, 'w' if i == 0 else 'a')  # keep the first attempt's error
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

    def log_tail(self, n=6):
        try:
            return Path(self.log_path).read_text(errors='ignore').splitlines()[-n:]
        except OSError:
            return []

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
def gguf_metadata(path, wanted_suffixes=('block_count', 'head_count', 'head_count_kv', 'context_length',
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


def kv_cache_mib(model_path, n_tokens, cfg):
    """Exact KV cache size for n_tokens: tokens x layers x kv_heads x (K dim x K bytes + V dim x V bytes)."""
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
        tk, tv, _ = parse_cfg(cfg)
        return n_tokens * n_layer * n_kv * (k_dim * KV_BYTES[tk] + v_dim * KV_BYTES[tv]) / 2**20
    except Exception as e:  # noqa: BLE001
        print(f'  (could not compute KV size: {e!r})')
        return None


def ctx_train(model_path):
    md = gguf_metadata(model_path)
    return md.get(f"{md.get('general.architecture')}.context_length")


# ---------------------------------------------------------------- one measurement

def complete(port, prompt_ids, gen, timeout):
    payload = {'prompt': prompt_ids, 'n_predict': gen, 'ignore_eos': True,
               'temperature': 0.0, 'cache_prompt': False, 'stream': False}
    resp = http(port, '/completion', payload, timeout=timeout)
    return time.monotonic(), resp.get('timings', {})


def measure(meter, srv, prompt_ids, gen, idle_w, rail, n_par=1, timeout=1800):
    """One request (n_par=1) or n_par identical requests sent at once to n_par server slots."""
    temps0 = meter.temps()
    t0 = time.monotonic()
    if n_par == 1:
        done = [complete(srv.port, prompt_ids, gen, timeout)]
    else:
        with ThreadPoolExecutor(n_par) as ex:
            done = list(ex.map(lambda _: complete(srv.port, prompt_ids, gen, timeout), range(n_par)))
    t1 = max(d[0] for d in done)
    tms = [d[1] for d in done]
    if n_par == 1:
        # prefill ends at send time + llama.cpp's prompt time; the rest is decode
        t_split = min(t1, t0 + tms[0].get('prompt_ms', 0) / 1000.0)
        t_dec1 = t1
    else:
        # window in which every session is decoding: from the last decode start to the first finish
        starts = [d[0] - d[1].get('predicted_ms', 0) / 1000.0 for d in done]
        t_split, t_dec1 = max(starts), min(d[0] for d in done)
    dec_s = max(t_dec1 - t_split, 1e-9)
    agg_tps = sum(tm.get('predicted_per_second') or 0 for tm in tms)

    out = {
        'n_parallel': n_par,
        'prompt_tokens': sum(tm.get('prompt_n') or 0 for tm in tms),
        'gen_tokens': sum(tm.get('predicted_n') or 0 for tm in tms),
        'latency_s': t1 - t0,
        'prefill_s': max(tm.get('prompt_ms', 0) for tm in tms) / 1000.0,
        'decode_s': max(tm.get('predicted_ms', 0) for tm in tms) / 1000.0,
        'decode_window_s': dec_s,
        'prefill_tps': sum(tm.get('prompt_per_second') or 0 for tm in tms) if n_par == 1 else
                       sum(tm.get('prompt_n') or 0 for tm in tms) / max(t_split - t0, 1e-9),
        'decode_tps': agg_tps,  # all sessions together
        'decode_tps_per_session': agg_tps / n_par,
        'gpu_temp_start': temps0.get('gpu-thermal') or temps0.get('GPU-therm'),
        'gpu_temp_end': (lambda t: t.get('gpu-thermal') or t.get('GPU-therm'))(meter.temps()),
        'gpu_freq_mhz_end': meter.gpu_freq_mhz(),
        'min_mem_available_mb': meter.min_mem_available(t0, t1),
        'gpu_load_prefill': meter.gpu_load(t0, t_split),
        'gpu_load_decode': meter.gpu_load(t_split, t_dec1),
        'cpu_busy_prefill': meter.cpu_busy(t0, t_split),
        'cpu_busy_decode': meter.cpu_busy(t_split, t_dec1),
    }
    for r in meter.rails():
        e, n = meter.energy(t0, t1, r)
        ep, _ = meter.energy(t0, t_split, r)
        ed, _ = meter.energy(t_split, t_dec1, r)
        idle = idle_w.get(r, 0.0)
        key = r.lower()
        out[f'{key}_energy_j'] = e
        out[f'{key}_energy_net_j'] = e - idle * (t1 - t0)
        out[f'{key}_decode_energy_net_j'] = ed - idle * dec_s
        out[f'{key}_avg_w'] = meter.mean_power(t0, t1, r)
        out[f'{key}_decode_avg_w'] = meter.mean_power(t_split, t_dec1, r)
        if r == rail:
            out['n_samples'] = n
            out['energy_j'] = e
            out['energy_net_j'] = e - idle * (t1 - t0)
            out['prefill_energy_j'] = ep
            out['prefill_energy_net_j'] = ep - idle * (t_split - t0)
            out['decode_energy_j'] = ed
            out['decode_energy_net_j'] = ed - idle * dec_s
            out['decode_avg_w'] = meter.mean_power(t_split, t_dec1, r)
            out['prefill_avg_w'] = meter.mean_power(t0, t_split, r)
            out['energy_per_request_j'] = e / n_par
            out['energy_per_request_net_j'] = out['energy_net_j'] / n_par
            g = out['gen_tokens'] or 0
            # tokens generated inside the decode window (= gen_tokens when n_par == 1)
            dec_tokens = g if n_par == 1 else agg_tps * dec_s
            out['decode_j_per_token'] = out['decode_energy_net_j'] / dec_tokens if dec_tokens else None
            out['decode_j_per_token_gross'] = ed / dec_tokens if dec_tokens else None
            out['total_j_per_token'] = out['energy_net_j'] / g if g else None
            out['total_j_per_token_gross'] = e / g if g else None
    return out


# ---------------------------------------------------------------- main

def measure_idle(meter, secs):
    print(f'measuring idle power for {secs:.0f}s ...')
    t0 = time.monotonic()
    time.sleep(secs)
    t1 = time.monotonic()
    idle_w = {r: meter.mean_power(t0, t1, r) for r in meter.rails()}
    print('idle W:', {k: round(v, 2) for k, v in idle_w.items()})
    return idle_w, meter.min_mem_available(t0, t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llama-bin', default=str(ROOT / 'llama.cpp/build/bin'))
    ap.add_argument('--models', nargs='+', default=sorted(str(p) for p in (ROOT / 'models').glob('*.gguf')))
    ap.add_argument('--kv', default='f16,q8_0,q4_0',
                    help='KV configs: "q8_0", "f16/q4_0" (K/V), "f16@off" (flash attention off)')
    ap.add_argument('--contexts', default='1024,4096,8192,16384')
    ap.add_argument('--np', default='1', help='parallel sessions, comma-separated, e.g. 1,2,4,8')
    ap.add_argument('--per-context-server', action='store_true',
                    help='start a server sized for each context (fit test) instead of one for the largest')
    ap.add_argument('--gen', type=int, default=128, help='tokens generated per request')
    ap.add_argument('--rounds', type=int, default=5, help='repeats (each in shuffled order)')
    ap.add_argument('--corpus', default=str(ROOT / 'data/corpus.txt'))
    ap.add_argument('--rail', default='VDD_IN', help='rail used for headline energy')
    ap.add_argument('--idle-secs', type=float, default=10)
    ap.add_argument('--cooldown', type=float, default=2)
    ap.add_argument('--timeout', type=float, default=1800, help='seconds per request')
    ap.add_argument('--extra', default='', help='extra llama-server arguments, e.g. "-fit off"')
    ap.add_argument('--tag', default='', help='free-text label, e.g. "25W"')
    ap.add_argument('--quick', action='store_true', help='1 model, 2 contexts, 1 round (~5 min)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.quick:
        args.models = args.models[:1]
        args.contexts = '1024,4096'
        args.rounds = 1
    cfgs = args.kv.split(',')
    ctxs = [int(c) for c in args.contexts.split(',')]
    nps = [int(n) for n in args.np.split(',')]
    extra = shlex.split(args.extra)
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
    meta.update({'tag': args.tag, 'gen': args.gen, 'kv_types': cfgs, 'contexts': ctxs, 'n_parallel': nps,
                 'per_context_server': args.per_context_server, 'extra_args': extra,
                 'llama_bin': str(llama_bin), 'rounds': args.rounds, 'started': stamp,
                 'top_at_start': sh('top -b -n 1 | sed -n 7,12p')})
    print(json.dumps(meta, indent=1))

    meter = PowerMeter().start()
    print(f'power backend: {meter.backend}; rails: {meter.rails()}')
    if args.rail not in meter.rails():
        sys.exit(f'rail {args.rail} not found; choose one of {meter.rails()} with --rail')
    idle_w, idle_mem = measure_idle(meter, args.idle_secs)
    print(f'mem_avail={idle_mem:.0f}MB')

    corpus = Path(args.corpus).read_text(errors='ignore')
    n_done = 0
    total = len(args.models) * len(cfgs) * len(ctxs) * len(nps) * args.rounds

    def write(row):
        with open(out_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    for model in args.models:
        mname = Path(model).stem
        m_ctxs = [c for c in ctxs if not ctx_train(model) or c + args.gen <= ctx_train(model)]
        if m_ctxs != ctxs:
            print(f'{mname}: skipping contexts above its trained length {ctx_train(model)}')
        prompts = None
        for rnd in range(args.rounds):
            order = cfgs[:]
            random.shuffle(order)
            for cfg in order:
                np_order = nps[:]
                random.shuffle(np_order)
                for n_par in np_order:
                    # one server for all contexts, or one per context (sized to it)
                    units = ([[c] for c in m_ctxs] if args.per_context_server else [m_ctxs[:]])
                    random.shuffle(units)
                    for unit in units:
                        slot_ctx = max(unit) + args.gen + 64
                        tk, tv, fa = parse_cfg(cfg)
                        base_row = {'model': mname, 'kv_type': cfg, 'type_k': tk, 'type_v': tv,
                                    'flash_attn': fa, 'n_parallel': n_par, 'round': rnd, 'tag': args.tag,
                                    'nvpmodel': meta['nvpmodel'], 'model_file_mib': Path(model).stat().st_size / 2**20,
                                    'kv_alloc_mib': kv_cache_mib(model, slot_ctx * n_par, cfg),
                                    'idle_w': idle_w.get(args.rail), 'rail': args.rail,
                                    'power_backend': meter.backend}
                        tagp = f'_np{n_par}' if n_par > 1 else ''
                        tagc = f'_c{unit[0]}' if args.per_context_server else ''
                        log_path = log_dir / f'{stamp}_{mname}_{cfg.replace("/", "-")}{tagp}{tagc}_r{rnd}.log'
                        srv = Server(llama_bin, model, cfg, slot_ctx * n_par, log_path, n_par, extra)
                        if not srv.start():
                            print(f'!! server failed: {mname} kv={cfg} np={n_par} ctx={unit} (see {log_path})')
                            for ctx in unit:
                                write({**base_row, 'context_len': ctx, 'status': 'server_failed',
                                       'log_tail': srv.log_tail()})
                                n_done += 1
                            continue
                        try:
                            if prompts is None:  # tokenize once per model
                                ids = http(srv.port, '/tokenize', {'content': corpus})['tokens']
                                if len(ids) < max(m_ctxs):
                                    print(f'!! corpus has only {len(ids)} tokens; longer contexts skipped')
                                prompts = {c: ids[:c] for c in m_ctxs if c <= len(ids)}
                                warm = ids[:1024]
                            load_mem = meter.min_mem_available(time.monotonic() - 1, time.monotonic())
                            # warm-up (short) so CUDA kernels are loaded before measuring
                            http(srv.port, '/completion', {'prompt': warm, 'n_predict': 16,
                                                           'cache_prompt': False, 'ignore_eos': True},
                                 timeout=args.timeout)
                            c_order = [c for c in unit if c in prompts]
                            random.shuffle(c_order)
                            for ctx in c_order:
                                time.sleep(args.cooldown)
                                try:
                                    r = measure(meter, srv, prompts[ctx], args.gen, idle_w, args.rail,
                                                n_par, args.timeout)
                                except Exception as e:  # noqa: BLE001
                                    print(f'!! request failed {mname} kv={cfg} ctx={ctx}: {e!r}')
                                    write({**base_row, 'context_len': ctx, 'status': 'request_failed',
                                           'error': repr(e), 'log_tail': srv.log_tail()})
                                    n_done += 1
                                    continue
                                write({**base_row, 'context_len': ctx, 'status': 'ok',
                                       'kv_used_mib': kv_cache_mib(model, ctx + args.gen, cfg),
                                       'server_mem_mb': idle_mem - load_mem, **r})
                                n_done += 1
                                print(f'[{n_done}/{total}] {mname} kv={cfg:9s} np={n_par} ctx={ctx:6d} '
                                      f'decode {r["decode_tps"]:.1f} tok/s  '
                                      f'{(r.get("decode_j_per_token") or float("nan")):.3f} J/tok  '
                                      f'req {r["energy_per_request_net_j"]:.1f} J  '
                                      f'gpu {r["gpu_temp_end"]}C  cpu {(r["cpu_busy_decode"] or 0):.0%}', flush=True)
                        except Exception as e:  # noqa: BLE001
                            print(f'!! error {mname} kv={cfg}: {e!r}')
                        finally:
                            srv.stop()

    idle_end, _ = measure_idle(meter, args.idle_secs)
    meter.stop()
    meta_path = out_path.with_suffix('.meta.json')
    meta.update({'idle_w': idle_w, 'idle_w_end': idle_end, 'idle_mem_available_mb': idle_mem,
                 'power_backend': meter.backend, 'finished': datetime.now().isoformat()})
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f'\ndone: {n_done} runs -> {out_path}\nmeta -> {meta_path}')


if __name__ == '__main__':
    main()
