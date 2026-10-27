"""Sections of Jetson_KV_Report_v2.pdf. Layout helpers and data access live in make_report.py."""
import json
import re

from make_report import (CTX, KVN, L3, LL, LL4, NAME, PLOTS, QW, QW4, ROOT, S, P, PageBreak, Spacer, chg, f, figure,
                         get, mean, sig, st_body, st_box, st_code, st_h1, st_h2, st_small, st_sub, st_title, table)

M = (LL, QW)


def title_page():
    return [
        Spacer(1, 30),
        P('KV-Cache Quantization on NVIDIA Jetson Orin Nano', st_title),
        P('Energy, speed, memory and accuracy of f16, 8-bit and 4-bit KV caches. '
          'Revised report (v2): addresses the review of the v1 report and adds seven new experiments.', st_sub),
        P('Koushik Bhupathiraju · Supervisors: Prof. Taskin Kocak, Prof. Rongyu Lin', st_body),
        P('v1 experiments: 19 October 2026 · v2 experiments: 26–27 October 2026', st_body),
        P('Code and raw data: <u>github.com/koushikvarmaJS/thesis2</u>', st_body),
        Spacer(1, 14),
        P('<b>Contents</b>'),
        P('0. Response to the review<br/>1. Summary and main claim<br/>2. Background and thesis scope<br/>'
          '3. What happened, step by step<br/>4. Hardware<br/>5. Software, libraries and tools<br/>6. Models<br/>'
          '7. KV cache formats<br/>8. How energy was measured<br/>9. Experiments and parameters<br/>10. Results<br/>'
          '11. Data quality<br/>12. Scope and limitations<br/>13. Next steps<br/>'
          'Appendix A. Reproduction · B. Files · C. v1 per-round values · D. Glossary'),
        PageBreak(),
    ]


def setup_sections():
    s = []
    s.append(P('4. Hardware', st_h1))
    s.append(table([
        ['Component', 'Detail'],
        ['Board', 'NVIDIA Jetson Orin Nano Developer Kit (8 GB), hostname "ranjotsingh", runs as user koushik'],
        ['CPU', '6 × Arm Cortex-A78AE (4 online in 7W mode)'],
        ['GPU', 'NVIDIA Ampere, compute capability 8.7 (sm_87), 1024 CUDA cores, 32 tensor cores'],
        ['Memory', '8 GB LPDDR5 shared by CPU and GPU; 7,486 MB visible to Linux; no swap'],
        ['Storage', 'microSD card, 113 GB root partition'],
        ['Power sensor', 'Onboard TI INA3221, three channels: VDD_IN (whole module), VDD_CPU_GPU_CV (CPU, GPU and '
                         'vision accelerators on one rail), VDD_SOC'],
        ['Cooling', 'Stock dev-kit heatsink and fan'],
        ['Control', 'SSH over Wi-Fi from a MacBook (Apple M2); long runs inside tmux'],
    ], [32, 142]))
    s.append(Spacer(1, 6))
    s.append(table([
        ['Power mode', 'nvpmodel ID', 'GPU max clock', 'CPU cores', 'Idle VDD_IN, v1', 'Idle VDD_IN, v2'],
        ['15W (default)', '0', '624.75 MHz', '6', '3.57 W', idle('15W-v2')],
        ['7W', '1', '408 MHz', '4', '3.41 W', idle('7W-v2')],
    ], [28, 22, 28, 20, 38, 38]))
    s.append(P('Only these two modes exist in /etc/nvpmodel.conf on this image (no 25W / MAXN Super). Switching needs '
               'a reboot. v2 idle power is lower than v1 because the desktop\'s Software Updater, which starts after '
               'every boot and kept one CPU core busy, was closed before every experiment set (section 9).', st_small))

    s.append(P('5. Software, libraries and tools', st_h1))
    s.append(table([
        ['Item', 'Version / setting'],
        ['Operating system', 'Ubuntu 24.04.4 LTS, Linux 6.8.12-1021-tegra'],
        ['Jetson Linux', 'L4T R39.2.0 (JetPack 7 line)'],
        ['CUDA', 'CUDA Toolkit 13.2.2 (nvcc 13.2.86), cuBLAS 13.4.1'],
        ['Build tools', 'GCC 13.3.0, CMake 3.28.3'],
        ['Inference engine', 'llama.cpp commit 7adf160 (0.7.0-dev). Main build: -DGGML_CUDA=ON '
                             '-DCMAKE_CUDA_ARCHITECTURES=87, Release. Used for all energy runs, v1 and v2.'],
        ['Second build', 'Same commit and flags plus -DGGML_CUDA_FA_ALL_QUANTS=ON. Without it, GPU flash attention '
                         'only exists for matching K/V types (f16/f16, q8_0/q8_0, q4_0/q4_0); a mixed pair such as '
                         'K f16 + V q4_0 would run attention on the CPU. Used only for the K/V split perplexity tests.'],
        ['llama.cpp programs', 'llama-server (energy), llama-bench (separate prefill/decode), llama-perplexity '
                               '(accuracy)'],
        ['Profilers', 'NVIDIA Nsight Systems 2025.6.3 (nsys), Nsight Compute 2026.1.1 (ncu)'],
        ['Benchmark code', 'Python 3.12.3, standard library only: power.py, bench_kv.py, bench_pp_tg.py, '
                           'ppl_v2.sh, profile.sh, run_v2.sh'],
        ['Analysis (Mac)', 'Python 3.14 + matplotlib 3.11.2 (analyze_v2.py), reportlab 5.1.0 (this PDF)'],
    ], [32, 142]))

    s.append(P('6. Models', st_h1))
    s.append(table([
        ['Property', 'Llama-3.2-1B-Instruct', 'Qwen2.5-0.5B-Instruct', 'Llama-3.2-3B-Instruct'],
        ['Used in', 'all experiments', 'all except fit test', 'fit test only'],
        ['Weights (main)', 'Q8_0, 1.32 GB (bartowski GGUF)', 'Q8_0, 0.68 GB (Qwen GGUF)', '–'],
        ['Weights (4-bit set)', 'Q4_K_M, 0.81 GB', 'Q4_K_M, 0.49 GB', 'Q4_K_M, 2.02 GB'],
        ['Layers', '16', '24', '28'],
        ['Query heads / KV heads', '32 / 8', '14 / 2', '24 / 8'],
        ['Head size', '64', '64', '128'],
        ['Trained context', '131,072', '32,768', '131,072'],
        ['KV cache per token, f16', '32 KiB', '12 KiB', '112 KiB'],
    ], [40, 45, 45, 44]))
    s.append(P('Values read from each GGUF header. KV bytes per token = layers × KV heads × head size × 2 (K and V) '
               '× bytes per value.', st_small))

    s.append(P('7. KV cache formats', st_h1))
    s.append(table([
        ['Format', 'How values are stored', 'Bytes / value', 'Size vs f16'],
        ['f16', '16-bit floating point, no compression', '2', '100%'],
        ['q8_0', 'blocks of 32: 32 × int8 + one fp16 scale (34 bytes); symmetric, no offset', '1.0625', '53%'],
        ['q4_0', 'blocks of 32: 32 × 4-bit + one fp16 scale (18 bytes); symmetric, no offset', '0.5625', '28%'],
        ['q4_1', 'blocks of 32: 32 × 4-bit + fp16 scale + fp16 minimum (20 bytes); has an offset', '0.625', '31%'],
    ], [18, 104, 24, 28]))
    s.append(P('In q4_0 the scale is the block value with the largest magnitude divided by −8, so that value is stored '
               'exactly and the rest are rounded to 16 levels around zero. q4_1 stores a minimum as well, so a block '
               'whose values all sit far from zero (an offset) still gets its 16 levels where the values are. In the '
               'energy runs K and V always use the same format. Flash attention is on in all runs except the '
               '"f16, flash attn off" baseline (section 10.3).'))

    s.append(P('8. How energy was measured', st_h1))
    for t in [
        '<b>Sensor and sampling.</b> A background thread reads voltage and current of the three INA3221 rails from '
        'sysfs every 50 ms (about 18 samples/s); power = mV × mA / 10<super>6</super>. Each sample also stores '
        'available memory, the GPU load (0–100%, from the GPU\'s sysfs <i>load</i> file) and the CPU busy time '
        '(from /proc/stat). Energy over any window is the trapezoid integral of the samples.',
        '<b>Gross and net energy.</b> Gross energy is what the module actually drew. Net energy subtracts the idle '
        'power (10 s with nothing running, measured before and after every experiment set) times the duration. '
        'Both are reported: net isolates the work, gross is what a battery pays. The difference matters most when '
        'run times differ a lot, as between 7W and 15W.',
        '<b>Per request.</b> Energy of one request = from sending it to receiving the reply. This is the headline '
        'metric in v2.',
        '<b>Decode energy per token (server runs).</b> Prefill is taken to end at send time + llama.cpp\'s own '
        'prompt time; the rest is decode. Decode J/token = decode energy ÷ 128. This split is approximate, so v2 '
        'also measures prefill and decode as separate llama-bench runs (below).',
        '<b>Decode energy per token (llama-bench).</b> llama-bench runs prompt processing (pp) and generation (tg) '
        'as separate tests. With --progress it prints a line just before each timed repetition and reports each '
        'repetition\'s duration; the energy of a repetition is integrated from that line\'s timestamp over that '
        'duration. "tg128 at depth 16,384" first fills the cache with 16,384 tokens (untimed), then times 128 '
        'generated tokens: pure decode energy, no split needed.',
        '<b>Power columns.</b> "Decode power" is the mean gross VDD_IN power inside the decode window only. v1\'s '
        'slide 13 showed the mean over the whole request (prefill + decode), which is lower because prefill draws '
        'less power; that is why it did not match decode joules ÷ decode time.',
        '<b>Concurrent sessions.</b> With N sessions the requests overlap. Energy per generated token = request '
        'energy ÷ (N × 128). Decode J/token uses the window in which all N sessions are decoding (from the last '
        'session to start decoding until the first to finish), divided by the tokens generated in it.',
        '<b>Statistics.</b> Every cell is a mean with a 95% confidence interval (Student t). Changes against f16 are '
        'paired: each format is compared with f16 from the same round (minutes apart, same thermal state) and the '
        'CI is taken over the per-round ratios. A change whose CI includes zero is marked (n.s.), not separable '
        'from zero.',
    ]:
        s.append(P('• ' + t))
    return s


def idle(tag):
    for name, meta in S.get('meta', {}).items():
        if meta.get('tag') == tag and 'idle_w' in meta:
            a = meta['idle_w'].get('VDD_IN')
            b = meta.get('idle_w_end', {}).get('VDD_IN')
            return f'{a:.2f} W (start) / {b:.2f} W (end)' if b else f'{a:.2f} W'
    return 'see section 11'


def appendix_repro():
    s = [P('Appendix A. Reproduction', st_h1)]
    s.append(P('One-time setup on the Jetson (sudo needed once):', st_body))
    s.append(P('sudo apt install -y openssh-server cuda-toolkit-13-2 cmake tmux<br/>'
               'bash jetson/setup.sh          # llama.cpp + models + data (v1)<br/>'
               'bash jetson/setup_v2.sh       # llama-bench, all-quants build, Q4_K_M models', st_code))
    s.append(P('All v2 experiments (inside tmux):', st_body))
    s.append(P('bash jetson/run_v2.sh 15W 2>&amp;1 | tee results/run_v2_15W.log<br/>'
               'sudo bash jetson/profile.sh ncu            # needs root for GPU counters<br/>'
               'sudo nvpmodel -m 1 --force                 # reboot into 7W<br/>'
               'bash jetson/run_v2.sh 7W 2>&amp;1 | tee results/run_v2_7W.log<br/>'
               'sudo nvpmodel -m 0 --force                 # back to 15W', st_code))
    s.append(P('On the Mac:', st_body))
    s.append(P('rsync -a &lt;user&gt;@&lt;jetson&gt;:~/thesis2/results/ results/<br/>'
               'python3 analysis/analyze_v2.py --out plots_v2<br/>'
               'python3 analysis/make_report.py', st_code))
    return s


# ---------------------------------------------------------------- results

def tags_present():
    return [t for t in ('15W-v2', '7W-v2') if any(k.startswith(t + '|') for k in S['energy_per_request_net_j'])]


def change_table(metric, tag, kvs=('q8_0', 'q4_0'), models=M, ctxs=CTX, np_=1):
    head = ['Context'] + [f'{NAME[m].split(" (")[0]} {KVN[kv]}' for m in models for kv in kvs]
    rows = [head]
    for c in ctxs:
        rows.append([f'{c // 1024}K'] + [chg(metric, tag, m, kv, c, np_) for m in models for kv in kvs])
    w = 174 / len(head)
    return table(rows, [w] * len(head), hl_rows=(len(rows) - 1,))


def results_energy():
    s = [P('10. Results', st_h1), P('10.1 Headline: energy per request', st_h2)]
    s.append(P('One request = a prompt of the given length plus exactly 128 generated tokens, the whole thing from '
               'send to reply. Change against f16 (flash attention on), paired by round, mean ± 95% CI over 5 rounds. '
               'Positive = the compressed cache uses <b>more</b> energy. (n.s.) = the CI includes zero.'))
    for tag in tags_present():
        mode = tag.split('-')[0]
        s.append(P(f'<b>{mode}, net energy (idle subtracted)</b>'))
        s.append(change_table('energy_per_request_net_j', tag))
        s.append(Spacer(1, 4))
        s.append(P(f'<b>{mode}, gross energy (what the module drew)</b>'))
        s.append(change_table('energy_per_request_j', tag))
        s.append(Spacer(1, 4))
    rows = [['f16 baseline, J per request (net / gross)'] + [f'{c // 1024}K' for c in CTX]]
    for tag in tags_present():
        for m in M:
            rows.append([f'{NAME[m]}, {tag.split("-")[0]}'] +
                        [f'{f(mean("energy_per_request_net_j", tag, m, "f16", c))} / '
                         f'{f(mean("energy_per_request_j", tag, m, "f16", c))}' for c in CTX])
    s.append(table(rows, [54, 30, 30, 30, 30]))
    s += figure(PLOTS / '01_request_energy_change.png',
                'Figure 1. Change in energy per request against f16, paired by round, 95% CI. Hollow markers: not '
                'separable from zero. File: plots_v2/01_request_energy_change.png')

    s.append(P('10.2 Decode energy per generated token', st_h2))
    s.append(P('The same runs, counting only the decode part (energy after the prompt has been read, ÷ 128). This is '
               'the v1 headline metric. It isolates the cost of reading the cache for each new token, so the '
               'changes are larger than per request, where prefill dilutes them.'))
    for tag in tags_present():
        s.append(P(f'<b>{tag.split("-")[0]}, decode J/token, net</b>'))
        s.append(change_table('decode_j_per_token', tag))
        s.append(Spacer(1, 4))
    s += figure(PLOTS / '02_decode_energy_change.png',
                'Figure 2. Change in decode energy per token against f16, paired by round, 95% CI. '
                'File: plots_v2/02_decode_energy_change.png')

    s.append(P('10.3 Baseline strength: f16 with flash attention off', st_h2))
    s.append(P('v1 forced flash attention on for f16 too, because a quantized V cache needs it. If f16 were faster '
               'without flash attention, the baseline would have been held back. v2 ran f16 both ways.'))
    rows = [['Model, mode, context', 'Decode tok/s, FA on', 'Decode tok/s, FA off', 'Request energy, FA off vs on']]
    for tag in tags_present():
        for m in M:
            for c in (1024, 16384):
                rows.append([f'{NAME[m]}, {tag.split("-")[0]}, {c // 1024}K',
                             f(mean('decode_tps', tag, m, 'f16', c)), f(mean('decode_tps', tag, m, 'f16@off', c)),
                             chg('energy_per_request_net_j', tag, m, 'f16@off', c)])
    s.append(table(rows, [52, 36, 36, 50]))
    return s


def pp(tag, m, kv, test, n, d, metric):
    v = S.get('pptg', {}).get(f'{tag}|{m}|{kv}|{test}|{n}|{d}', {}).get(metric)
    return v


def results_pptg():
    s = [P('10.4 Prefill and decode measured separately (llama-bench)', st_h2)]
    s.append(P('llama-bench timed prompt processing (pp) and generation (tg) as separate tests, 5 repetitions each '
               '(section 8). "tg128 at 16K" = 128 tokens generated after the cache holds 16,384 tokens. The last '
               'column repeats the server-run decode J/token (timestamp split) for comparison.'))
    for mode in ('15W', '7W'):
        if not any(k.startswith(mode + '|') for k in S.get('pptg', {})):
            continue
        rows = [['Model, KV', 'pp16384 tok/s', 'pp16384 mJ/token', 'tg128 at 16K tok/s', 'tg128 at 16K J/token',
                 'vs f16', 'server split J/token']]
        for m in M:
            base = pp(mode, m, 'f16', 'tg', 128, 16384, 'j_per_token')
            for kv in ('f16', 'f16@off', 'q8_0', 'q4_0'):
                t = pp(mode, m, kv, 'tg', 128, 16384, 'j_per_token')
                rows.append([f'{NAME[m]}, {KVN[kv]}',
                             f(pp(mode, m, kv, 'pp', 16384, 0, 'tps')[0], 0),
                             f(1000 * pp(mode, m, kv, 'pp', 16384, 0, 'j_per_token')[0], 2),
                             f(pp(mode, m, kv, 'tg', 128, 16384, 'tps')[0]),
                             f'{t[0]:.3f} ± {t[1]:.3f}',
                             '–' if kv == 'f16' else f'{100 * (t[0] / base[0] - 1):+.1f}%',
                             f(mean('decode_j_per_token', f'{mode}-v2', m, kv, 16384), 3)])
        s.append(P(f'<b>{mode}</b>'))
        s.append(table(rows, [36, 22, 22, 24, 28, 18, 24]))
    s.append(P('Net energy (idle subtracted), mean ± 95% CI over 5 repetitions. mJ = millijoules.', st_small))
    return s


def results_mechanism():
    s = [P('10.5 Why: what the extra energy is made of', st_h2)]
    rows = [['At 16K tokens', 'f16', '8-bit', '4-bit']]
    for tag in tags_present():
        md = tag.split('-')[0]
        for m in M:
            rows.append([f'{NAME[m]} decode tok/s, {md}'] + [f(mean('decode_tps', tag, m, kv, 16384)) for kv in
                                                              ('f16', 'q8_0', 'q4_0')])
            rows.append([f'{NAME[m]} decode power, W gross, {md}'] +
                        [f(mean('decode_avg_w', tag, m, kv, 16384), 2) for kv in ('f16', 'q8_0', 'q4_0')])
            rows.append([f'{NAME[m]} CPU busy / GPU load in decode, {md}'] +
                        [f'{mean("cpu_busy_decode", tag, m, kv, 16384):.0%} / {mean("gpu_load_decode", tag, m, kv, 16384):.0%}'
                         for kv in ('f16', 'q8_0', 'q4_0')])
    s.append(table(rows, [74, 33, 33, 34]))
    s.append(P('Decode power here is the mean gross VDD_IN power inside the decode window only (section 8). '
               'CPU busy is the share of all six CPU cores\' time that was busy (/proc/stat); GPU load is the GPU\'s '
               'own busy counter.', st_small))
    return s


def kern(model, kv, fa='on'):
    return S.get('kernels', {}).get(f'{model}_{kv}_fa{fa}', {})


def pct(a, b):
    return f'{100 * (a / b - 1):+.0f}%' if a and b else '–'


def ppl_rows():
    return S.get('ppl', [])


def ppl_v2_done():
    return any(r.get('_file', '').startswith('ppl_v2') for r in ppl_rows())


def fit_tags():
    return [t for t in ('15W-fit1b', '15W-fit3b')
            if any(k.startswith(t + '|') for k in S.get('energy_per_request_net_j', {}))
            or any(r.get('tag') == t for r in S.get('failures', []))]


def has_7w():
    return '7W-v2' in tags_present()


def headline(m, kv, c=16384, tag='15W-v2'):
    return chg('energy_per_request_net_j', tag, m, kv, c, show_ci=False)


# ---------------------------------------------------------------- 0-3

def response_to_review():
    s = [P('0. Response to the review', st_h1)]
    s.append(P('Each point of Prof. Lin\'s review, what was done about it, and where the answer is. All v2 numbers come '
               'from new runs on 26–27 October; the v1 numbers are only quoted for comparison.'))
    k_ll, k_ll4 = kern(LL, 'f16'), kern(LL, 'q4_0')
    rows = [['Review point', 'What was done', 'Where'],
            ['Flash-attention-off baseline: was f16 held back by forcing FA on?',
             'f16 run with FA on and off in every v2 grid (5 rounds). FA off is much worse: '
             f'{headline(LL, "f16@off")} (Llama) and {headline(QW, "f16@off")} (Qwen) energy per request at 16K. '
             'FA on is the stronger baseline, so v1 compared against the right one.', '10.3'],
            ['Gross vs net energy; the 7W/15W comparison should show both',
             'Every table now has net (idle subtracted) and gross. Idle is measured at the start and end of every '
             'set.', '8, 10.1, 10.11'],
            ['CPU fallback? Per-rail energy',
             'CPU busy and GPU load are now sampled with the power. In decode the CPU is 2–3% busy for every format and '
             'the GPU 90–97%, so nothing falls back to the CPU. The extra energy of 8-bit/4-bit is almost all on the '
             'CPU+GPU rail; the SoC rail barely changes.', '10.5, 10.6'],
            ['Prefill and decode measured separately (llama-bench)',
             'New set B: pp and tg as separate llama-bench tests with energy per repetition. tg128 at 16K agrees with '
             'the server timestamp split within a few percent.', '10.4'],
            ['Concurrency (np = 2/4/8)',
             'New set D: 1, 2, 4, 8 parallel sessions at 4K and 16K per session. The penalty stays under batching; '
             'f16 with 8 × 16K does not fit on Llama, 8-bit and 4-bit do.', '10.7'],
            ['4-bit weights (Q4_K_M)',
             'New set E: both models with Q4_K_M weights, full context grid, 5 rounds. Same result as with Q8_0 '
             'weights.', '10.8'],
            ['At least 5 paired rounds, CIs',
             'Main grid, weights set and llama-bench: 5 rounds; concurrency 3 rounds. All changes are paired by round '
             'with 95% CIs; (n.s.) marks a CI that includes zero.', '8, 11'],
            ['Is the 7W Qwen 16K cell an outlier?',
             'No: the 3 v1 rounds are within 1% of each other (Appendix C), and the 7W rerun in v2 '
             + ('confirms it.' if has_7w() else 'is listed in section 13.'), 'App. C'],
            ['Power column did not match J ÷ s on slide 13',
             'v1 showed whole-request power; the v2 tables show decode-window power, which matches decode J ÷ s.',
             '8, 10.5'],
            ['Accuracy at long context; is 4-bit K or V the problem?',
             'Perplexity at 8K for both models (16K runs out of memory in llama-perplexity), plus a K/V split for '
             'Qwen and Llama with the all-quants build: 4-bit keys, not values, break Qwen.' + ('' if ppl_v2_done() else ' (running at the time of writing)'), '10.9'],
            ['Why is it slower? Mechanism',
             'Nsight Systems kernel profile at 16K: with an 8-bit or 4-bit cache, decode attention moves from a '
             'tensor-core kernel to a CUDA-core "vec" kernel that takes '
             f'{pct(k_ll4.get("decode_attn_us_per_call"), k_ll.get("decode_attn_us_per_call"))} longer per call on '
             'Llama (4-bit). Nsight Compute: the f16 kernel is memory-bound (88% of peak memory, 10% compute); '
             'the 4-bit kernel is compute-heavy (74% compute) and re-reads the cache for every query head.', '10.5'],
            ['Where does compression help? Memory fit test',
             'New set G: Llama-1B at 32K/64K and Llama-3B at 16K/32K/64K, auto-fit off. On the 3B model f16 fits only '
             '16K, 8-bit 32K and 4-bit 64K.',
             '10.10'],
            ['Thesis main question and a falsifiable claim', 'Proposed in section 1 (please confirm).', '1'],
            ['Public repository with raw data', 'github.com/koushikvarmaJS/thesis2: code, every raw JSONL/CSV, '
             'analysis scripts and this PDF\'s generator.', 'App. B'],
            ]
    s.append(table(rows, [52, 104, 18]))
    return s


def summary():
    s = [P('1. Summary and main claim', st_h1)]
    s.append(P('<b>Proposed thesis main question.</b> On a memory-limited edge device, does compressing the KV cache '
               'reduce the energy of LLM inference, and if not, what does it buy?', st_box))
    s.append(P('<b>Proposed falsifiable claim.</b> On a Jetson Orin Nano running llama.cpp, an 8-bit or 4-bit KV cache '
               'uses <b>more</b> energy per request than an f16 cache at every context length from 1K to 16K tokens, '
               'and the gap grows with context. The cause is compute, not memory traffic: the quantized cache moves '
               'decode attention off the tensor-core kernel. Compression only pays as capacity: it lets contexts and '
               'session counts run that do not fit with f16. The claim is refuted by any paired cell where a quantized '
               'cache is significantly cheaper than f16, or by f16 fitting everything the quantized caches fit.',
               st_box))
    rows = [['At 16K tokens, 15W', 'Llama-1B 8-bit', 'Llama-1B 4-bit', 'Qwen-0.5B 8-bit', 'Qwen-0.5B 4-bit'],
            ['Energy per request, net'] + [headline(m, kv) for m in M for kv in ('q8_0', 'q4_0')],
            ['Decode energy per token, net'] + [chg('decode_j_per_token', '15W-v2', m, kv, 16384, show_ci=False)
                                                for m in M for kv in ('q8_0', 'q4_0')],
            ['Decode speed'] + [chg('decode_tps', '15W-v2', m, kv, 16384, show_ci=False)
                                for m in M for kv in ('q8_0', 'q4_0')],
            ['KV cache memory'] + [pct(mean('kv_used_mib', '15W-v2', m, kv, 16384),
                                       mean('kv_used_mib', '15W-v2', m, 'f16', 16384))
                                   for m in M for kv in ('q8_0', 'q4_0')]]
    if has_7w():
        rows.insert(2, ['Energy per request, net, 7W'] + [headline(m, kv, tag='7W-v2') for m in M
                                                           for kv in ('q8_0', 'q4_0')])
    s.append(table(rows, [46, 32, 32, 32, 32]))
    s.append(Spacer(1, 4))
    for t in [
        'Every quantized cell in the 15W main grid uses more energy than f16, and every change is significant '
        '(paired, 5 rounds). At 1K the penalty is 2.6–4.6%; at 16K it is 13–18% per request and 29–58% per decoded '
        'token.',
        'The f16 baseline was not held back: turning flash attention off makes f16 3.3–3.7× more expensive per request at 16K.',
        'Nothing runs on the CPU. The extra energy is GPU work: decode attention on an 8-bit/4-bit cache uses a '
        '"vec" kernel without tensor cores, 2.1–2.3× slower per call than f16 on Llama and 3.5–3.9× on Qwen (whose '
        '7 query heads per KV head make the f16 tensor-core kernel especially efficient). Decode power barely '
        'changes; time grows, and energy with it.',
        'The same holds with 4-bit weights and with up to 8 concurrent sessions. What compression buys is room: '
        'Llama-1B cannot hold 8 sessions × 16K tokens in f16 (4.1 GiB of cache) but can in 8-bit or 4-bit.',
    ] + ([
        'At 7W the result is the same: all 48 quantized cells are significantly more expensive than f16 (+2.6% at 1K '
        'up to +28% per request at 16K). 7W itself is the more expensive mode: +21–31% net and +60–73% gross energy '
        'per request against 15W (section 10.11).',
    ] if has_7w() else []):
        s.append(P('• ' + t))
    return s


def background():
    s = [P('2. Background and thesis scope', st_h1)]
    for t in [
        'The thesis (spring 2026) measured the energy of KV-cache <i>eviction</i> methods (StreamingLLM and SnapKV '
        'through NVIDIA\'s kvpress library) on a Tesla T4 cloud GPU, with Qwen2.5-0.5B and Llama-3.2-1B. The paper '
        'extends it to an edge device, the Jetson Orin Nano, where memory is the limiting resource.',
        'This report covers the first edge method family: KV-cache <i>quantization</i> (storing K and V in 8 or 4 '
        'bits) in llama.cpp. Eviction on the Jetson needs PyTorch and kvpress and comes next (section 13).',
        '<b>Correction to v1.</b> v1 said quantized KV caches could not run on the T4. More precisely: the published '
        'KIVI implementations failed there. optimum-quanto\'s Marlin kernel and KIVI\'s own CUDA extension need an '
        'Ampere GPU (sm_80 or newer; the T4 is sm_75), and the hqq backend failed to install. llama.cpp\'s quantized '
        'KV cache would have built for the T4. The move to the Jetson is about the edge question, not a T4 limit.',
        '<b>Why perplexity is meaningful here.</b> In the thesis, kvpress compressed the cache only after the prompt '
        'was read, so perplexity over the prompt never saw the compression. In llama.cpp every token\'s K and V are '
        'written to the quantized cache immediately and attention reads them back from it, so perplexity includes '
        'the quantization error.',
        '<b>Related work.</b> Server-GPU work (QServe, BitDecoding, KVmix and others) reports that low-bit KV caches '
        'make decode attention compute-bound because of dequantization, and builds special kernels to hide it. No '
        'measurement of KV quantization energy on an edge device was found; that is this paper\'s contribution.',
    ]:
        s.append(P(t))
    return s


def run_log_stages():
    out = []
    for mode in ('15W', '7W'):
        p = ROOT / 'results' / f'run_v2_{mode}.log'
        if p.exists():
            for line in p.read_text(errors='ignore').splitlines():
                mm = re.match(r'=== (\S+ \S+) (\w+): (.*)', line)
                if mm:
                    out.append((mm.group(1), mm.group(2), mm.group(3)))
    return out


def timeline():
    s = [P('3. What happened, step by step', st_h1)]
    rows = [['When', 'What'],
            ['19 Oct', 'v1: llama.cpp built on the Jetson; 3 rounds × 2 models × 3 formats × 4 contexts at 15W, then 7W; '
                       'perplexity at 2K. Results deck and report (v1).'],
            ['21–25 Oct', 'Review by Prof. Lin. v2 code: K/V pairs and FA on/off, CPU/GPU load sampling, '
                          'concurrency, llama-bench wrapper, perplexity at 16K, profiling, fit test.'],
            ['26 Oct', 'Found the desktop\'s Software Updater using a full CPU core after every boot (raised idle power '
                       'in v1 by about 0.3–0.5 W). It is closed before every v2 set.']]
    manual = [('2026-10-27 01:39:57', '15W', 'C (rerun) perplexity at 8K and the K/V split at 8K: 16K ran out of '
                                             'memory (section 10.9)'),
              ('2026-10-27 02:16:00', '15W', 'Check: Llama-1B f16 with 8 sessions × 16K; out of memory, board rebooted'),
              ('2026-10-27 02:20:00', '15W', 'F (ncu) Nsight Compute counters of the decode attention kernel (root)')]
    for when, mode, what in sorted(run_log_stages() + manual):
        rows.append([f'{when[5:16]} ({mode})', what])
    s.append(table(rows, [34, 140]))
    s.append(P('Times from results/run_v2_*.log. Each set\'s raw data is a separate file (Appendix B).', st_small))
    return s


def experiments():
    s = [P('9. Experiments and parameters', st_h1)]
    rows = [['Set', 'What', 'Models', 'Formats', 'Contexts', 'Rounds'],
            ['A', 'Main grid, llama-server, 128 generated tokens', 'Llama-1B, Qwen-0.5B (Q8_0 weights)',
             'f16 (FA on), f16 (FA off), q8_0, q4_0', '1K, 4K, 8K, 16K', '5'],
            ['B', 'llama-bench: pp at 1K/16K, tg128 at depth 1K/16K', 'same', 'same', '–', '5 reps'],
            ['C', 'Perplexity, WikiText-2; K/V split', 'same', 'f16, q8_0, q4_0, q4_1, mixed K/V', '2K, 8K',
             '1 (± error)'],
            ['D', 'Concurrency: N sessions sent at once', 'same', 'f16, q8_0, q4_0', '4K, 16K per session; N = 1, 2, 4, 8',
             '3'],
            ['E', 'Main grid with 4-bit weights', 'Llama-1B, Qwen-0.5B (Q4_K_M)', 'f16, q8_0, q4_0', '1K–16K', '5'],
            ['F', 'Kernel profile (nsys), attention counters (ncu)', 'Llama-1B, Qwen-0.5B', 'f16 (FA on/off), q8_0, q4_0',
             'depth 16K + 64 tokens', '1'],
            ['G', 'Fit test: largest context per format', 'Llama-1B, Llama-3B (Q4_K_M)', 'f16, q8_0, q4_0',
             '1B: 32K/64K; 3B: 16K/32K/64K', '2'],
            ]
    s.append(table(rows, [9, 50, 36, 34, 30, 15]))
    s.append(Spacer(1, 4))
    for t in [
        '<b>Common settings.</b> All layers on the GPU (-ngl 99), flash attention on unless stated, -c = context × '
        'sessions, one server start per format and round. The prompt is WikiText-2 test text cut to context − 128 − '
        '16 tokens, sent as token IDs; ignore_eos makes every reply exactly 128 tokens; cache_prompt=false makes '
        'every request process its prompt. Temperature 0.',
        '<b>Order.</b> Inside each round, formats and contexts run in a shuffled order (fixed seed), so no format '
        'always runs first or on a cooler board. A 1K-token warm-up request runs after each server start.',
        '<b>Sets A and B run at 15W and at 7W</b>; C–G at 15W only (the 7W slowdown affects all formats alike in v1).',
        '<b>Before every set</b> the Software Updater and GNOME Software are closed and idle power is measured for 10 s.',
    ]:
        s.append(P('• ' + t))
    return s


# ---------------------------------------------------------------- results 10.5+

def results_kernels():
    s = [P('<b>Kernel profile.</b> Nsight Systems recorded every GPU kernel while llama-bench filled the cache to '
           '16,384 tokens and then generated 64 tokens (+1 warm-up). Decode attention is the flash-attention kernel '
           'launched once per layer per generated token.')]
    rows = [['Model, KV', 'Decode attention kernel', 'µs per call', 'ms per token', 'vs f16', 'Prefill attention, ms',
             'Quantize / dequantize / rotate, ms']]
    for m in M:
        base = kern(m, 'f16').get('decode_attn_us_per_call')
        for kv in ('f16', 'q8_0', 'q4_0'):
            k = kern(m, kv)
            if not k:
                continue
            rows.append([f'{NAME[m]}, {KVN[kv]}', k.get('decode_attn_kernel', ''),
                         f(k.get('decode_attn_us_per_call'), 0), f(k.get('decode_attn_ms_per_token'), 1),
                         '–' if kv == 'f16' else pct(k.get('decode_attn_us_per_call'), base),
                         f(k.get('prefill_attn_ms'), 0), f(k.get('kv_quant_ms'), 0)])
    s.append(table(rows, [34, 34, 18, 18, 16, 24, 30]))
    s.append(P('The last column sums the kernels that only exist with a quantized cache: writing new K/V into the '
               'cache in 8 or 4 bits, converting the cache back to f16 before each prefill batch, and a Walsh–Hadamard '
               'rotation (fwht) that llama.cpp applies to K and Q with a quantized cache. Prefill attention itself '
               'takes the same time for all formats because it runs on an f16 copy. With flash attention off, '
               f'Llama spends {f(kern(LL, "f16", "off").get("softmax_gemm_ms"), 0)} ms in separate softmax and matrix '
               f'kernels (against {f(kern(LL, "f16").get("total_ms"), 0)} ms for everything with FA on).',
               st_small))
    s += figure(PLOTS / '09_kernels.png', 'Figure 3. Decode attention kernel time per call at 16K. '
                'File: plots_v2/09_kernels.png', width=165)
    s.append(P('<b>Reading.</b> With f16, llama.cpp runs decode attention on a tensor-core (mma) kernel that handles '
               'all query heads sharing a KV head together. With q8_0 or q4_0 it switches to the "vec" kernel, which '
               'works one query row at a time on the CUDA cores and dequantizes as it reads. It moves fewer bytes but '
               'does more instructions, and on this GPU the instructions cost more than the bytes saved. Qwen has 7 '
               'query heads per KV head (Llama 4), so the tensor-core kernel helps it more and the switch hurts it more '
               '(3.5–3.9× against 2.1–2.3×). This matches the larger energy penalty for Qwen in 10.1–10.2.'))
    ncu = sorted((ROOT / 'results' / 'prof').glob('*_ncu.csv')) if (ROOT / 'results' / 'prof').exists() else []
    if ncu:
        s += ncu_table(ncu)
    return s


def ncu_table(paths):
    """Mean over the profiled decode launches (one token, all layers) of the Nsight Compute summary metrics."""
    import csv
    want = {'Compute (SM) Throughput': 'sm', 'Memory Throughput': 'mem', 'L2 Hit Rate': 'l2',
            'L1/TEX Hit Rate': 'l1', 'Duration': 'dur'}
    rows = [['Model, KV', 'Kernel', 'Compute (SM), % of peak', 'Memory, % of peak', 'L1 / L2 hit rate',
             'µs per call (clocks locked)']]
    for p in paths:
        vals = {}
        try:
            lines = [ln for ln in p.read_text(errors='ignore').splitlines() if ln.startswith('"')]
            for r in csv.DictReader(lines):
                k = want.get(r.get('Metric Name', ''))
                if k:
                    vals.setdefault(k, []).append(float(r['Metric Value'].replace(',', '')))
        except (OSError, ValueError, KeyError):
            continue
        if not vals:
            continue
        tag = p.name.replace('_ncu.csv', '')
        model = LL if tag.startswith('Llama') else QW
        kv = tag[len(model) + 1:].rsplit('_fa', 1)[0]
        avg = lambda k: sum(vals[k]) / len(vals[k]) if vals.get(k) else float('nan')  # noqa: E731
        rows.append([f'{NAME[model]}, {KVN.get(kv, kv)}', 'mma (tensor cores)' if kv == 'f16' else 'vec',
                     f(avg('sm'), 0), f(avg('mem'), 0), f'{f(avg("l1"), 0)}% / {f(avg("l2"), 0)}%',
                     f(avg('dur') / 1000, 0)])
    if len(rows) == 1:
        return []
    return [P('<b>Hardware counters (Nsight Compute).</b> The decode attention kernel for one generated token (all '
              'layers) at a 16K cache, as a share of the GPU\'s peak. Nsight Compute locks the clocks to base '
              'frequency, so times are longer than in the table above; the shares are what matters.'),
            table(rows, [34, 30, 28, 24, 30, 28]),
            P('The f16 kernel is memory-bound: it streams the cache at close to the peak memory throughput and '
              'leaves the SMs 90% idle. Halving or quartering the bytes should have made decode attention up to 2–4× '
              'faster. The vec kernels instead keep the SMs 64–74% busy (unpacking, scaling, one query row at a '
              'time) and still run the memory system at 63–90% of peak. Their high L1/L2 hit rates show why: each '
              'query head re-reads the K/V block it shares with the other heads of its group, where the f16 kernel '
              'reads it once for the whole group. The compressed cache is smaller, but it is read 4 (Llama) or 7 '
              '(Qwen) times and unpacked each time.', st_small)]


def results_rails():
    s = [P('10.6 Per-rail energy', st_h2)]
    s.append(P('The INA3221 has three channels. VDD_CPU_GPU_CV feeds the CPU cores, the GPU and the vision '
               'accelerators together, so it cannot separate CPU from GPU on its own; the CPU busy counter (10.5) '
               'shows the CPU is nearly idle in decode, so changes on this rail are GPU. VDD_SOC feeds the rest of the '
               'chip, including the memory controller. The remainder of VDD_IN (memory, regulators, fan, I/O) is shown '
               'as "other". Decode energy, net, at 16K, 15W.'))
    rows = [['Model, KV', 'VDD_IN, J', 'CPU+GPU rail, J', 'SoC rail, J', 'Other, J', 'CPU+GPU vs f16', 'SoC vs f16']]
    for m in M:
        b = {r: mean(f'{r}_decode_energy_net_j', '15W-v2', m, 'f16', 16384) for r in ('vdd_cpu_gpu_cv', 'vdd_soc')}
        for kv in ('f16', 'q8_0', 'q4_0'):
            vin = mean('vdd_in_decode_energy_net_j', '15W-v2', m, kv, 16384)
            cg = mean('vdd_cpu_gpu_cv_decode_energy_net_j', '15W-v2', m, kv, 16384)
            soc = mean('vdd_soc_decode_energy_net_j', '15W-v2', m, kv, 16384)
            rows.append([f'{NAME[m]}, {KVN[kv]}', f(vin), f(cg), f(soc), f(vin - cg - soc),
                         '–' if kv == 'f16' else pct(cg, b['vdd_cpu_gpu_cv']),
                         '–' if kv == 'f16' else pct(soc, b['vdd_soc'])])
    s.append(table(rows, [36, 20, 24, 20, 18, 28, 28]))
    s.append(P('The extra decode energy of a quantized cache lands on the CPU+GPU rail (+60–90%); the SoC rail, which '
               'includes the memory controller, changes by 5–32%. If the cost were memory traffic, the SoC and '
               '"other" shares would grow instead.', st_small))
    return s


def results_concurrency():
    s = [P('10.7 Concurrency: several sessions at once', st_h2)]
    s.append(P('N identical requests sent at the same moment to a server with N slots, each slot holding the full '
               'context. Energy per generated token over the whole batch (request energy ÷ (N × 128)), net, mean of '
               '3 rounds, and the paired change against f16 with the same N.'))
    for c in (4096, 16384):
        rows = [[f'{c // 1024}K per session'] + [f'N = {n}' for n in (1, 2, 4, 8)]]
        for m in M:
            for kv in ('f16', 'q8_0', 'q4_0'):
                cells = []
                for n in (1, 2, 4, 8):
                    v = get('total_j_per_token', '15W-np', m, kv, c, n)
                    if not v:
                        fail = any(r.get('tag') == '15W-np' and r['model'] == m and r['kv_type'] == kv and
                                   r['n_parallel'] == n for r in S.get('failures', []))
                        cells.append('did not fit' if fail else '–')
                    else:
                        d = chg('total_j_per_token', '15W-np', m, kv, c, n, False)
                        cells.append(f'{v[0]:.3f}' + ('' if kv == 'f16' or d == '–' else f' ({d})'))
                rows.append([f'{NAME[m]}, {KVN[kv]}'] + cells)
        s.append(table(rows, [42, 33, 33, 33, 33]))
        s.append(Spacer(1, 4))
    s += figure(PLOTS / '07_concurrency.png', 'Figure 4. Energy per generated token against the number of sessions. '
                'File: plots_v2/07_concurrency.png', width=160)
    s.append(P('Batching lowers the energy per token for every format, because the model weights are read once per '
               'step for all sessions. The quantized caches stay more expensive at every N. The one place they win is '
               'capacity: Llama-1B with 8 sessions × 16K tokens needs 4.1 GiB of f16 cache, which does not fit next to '
               'the weights in 8 GB of shared memory, while 8-bit (2.2 GiB) and 4-bit (1.2 GiB) ran. A separate check '
               'confirmed it: llama.cpp\'s memory check reports "failed to fit params to free device memory", and '
               'starting the server anyway froze the board until it rebooted (results/logs/np8_f16_oomcheck.log). (The 4K row for N = 8 is missing for the same reason: one server, sized for 16K per slot, served '
               'both contexts.)'))
    return s


def results_w4():
    s = [P('10.8 4-bit weights (Q4_K_M)', st_h2)]
    s.append(P('Same grid as set A with both models\' weights in Q4_K_M instead of Q8_0. Smaller weights leave more of '
               'each decode step to attention, so if anything the cache format should matter more.'))
    s.append(P('<b>Energy per request, net, change against f16</b>'))
    s.append(change_table('energy_per_request_net_j', '15W-w4', models=(LL4, QW4)))
    s.append(Spacer(1, 4))
    s.append(P('<b>Decode energy per token, net, change against f16</b>'))
    s.append(change_table('decode_j_per_token', '15W-w4', models=(LL4, QW4)))
    s += figure(PLOTS / '06_w4_request_change.png', 'Figure 5. Q4_K_M weights: change in energy per request against '
                'f16. File: plots_v2/06_w4_request_change.png', width=160)
    s.append(P('The penalty is the same as with 8-bit weights within a few points. The result is not specific to how '
               'the weights are stored.'))
    return s


def results_ppl():
    s = [P('10.9 Accuracy: perplexity', st_h2)]
    rows = ppl_rows()
    if not ppl_v2_done():
        s.append(P('<i>The v2 runs (16K and the K/V split) were not finished at the time of writing; v1 values at 2K '
                   'below.</i>'))
    s.append(P('WikiText-2 test set, llama-perplexity, which scores the second half of each window. At 8K the text '
               'is cut into 32 windows of 8,192 tokens; at 2K into 20 windows of 2,048. Lower is better; ± is '
               'llama.cpp\'s standard error. Mixed K/V pairs need the all-quants build (section 5). 16K was planned but '
               'does not fit: llama-perplexity keeps the logits of the scored half of a window (8,192 tokens × a '
               '128K–152K-token vocabulary × 4 bytes = 4–5 GB) in memory next to the model, and Linux killed the '
               'process.'))
    t = [['Model', 'K', 'V', 'Context', 'Run', 'Perplexity', 'vs f16/f16']]
    base = {(r['model'], int(r['ctx'])): float(r['ppl']) for r in rows
            if r['type_k'] == 'f16' and r['type_v'] == 'f16' and r.get('ppl')}
    for r in sorted(rows, key=lambda r: (r['model'], int(r['ctx']), r['type_k'], r['type_v'])):
        if not r.get('ppl'):
            continue  # the 16K attempts (out of memory, see above)
        b = base.get((r['model'], int(r['ctx'])))
        t.append([NAME.get(r['model'], r['model']), r['type_k'], r['type_v'], f'{int(r["ctx"]) // 1024}K', src(r),
                  f'{float(r["ppl"]):.3f} ± {float(r["ppl_err"]):.3f}',
                  f'{100 * (float(r["ppl"]) / b - 1):+.1f}%' if b else '–'])
    s.append(table(t, [40, 16, 16, 18, 30, 32, 22]))
    split = {(r['model'], r['type_k'], r['type_v']): float(r['ppl']) for r in rows
             if r.get('ppl') and int(r['ctx']) == 2048}
    split8 = {(r['model'], r['type_k'], r['type_v']): float(r['ppl']) for r in rows
              if r.get('ppl') and int(r['ctx']) == 8192}
    q = lambda k, v: split.get((QW, k, v))  # noqa: E731
    q8 = lambda k, v: split8.get((QW, k, v))  # noqa: E731
    if q('f16', 'q4_0') and q('q4_0', 'f16'):
        at8 = ''
        if q8('f16', 'q4_0') and q8('q4_0', 'f16'):
            at8 = (f' At 8K the split gives the same answer: f16 K + 4-bit V {q8("f16", "q4_0"):.2f} '
                   f'(f16: {q8("f16", "f16"):.2f}), 4-bit K + f16 V {q8("q4_0", "f16"):.1f}, q4_1 '
                   f'{f(q8("q4_1", "q4_1"), 1)}.')
        s.append(P('<b>Which half breaks Qwen at 4 bits: the keys.</b> At 2K, with 4-bit V and f16 K, Qwen\'s perplexity is '
                   f'{q("f16", "q4_0"):.2f} (f16: {q("f16", "f16") or 13.02:.2f}); with 4-bit K and f16 V it is '
                   f'{q("q4_0", "f16"):.1f}. 8-bit K with 4-bit V is also fine ({f(q("q8_0", "q4_0"), 2)}). q4_1, which '
                   f'stores a minimum per block, does not rescue it ({f(q("q4_1", "q4_1"), 1)}), so an offset is not the '
                   'problem. The likely cause is outlier channels: keys are known to have a few channels with very '
                   'large values (Qwen2.5 also adds a bias in its K projection), and with one scale per 32 values the '
                   'small values in a block round to zero. Llama behaves the same way on a smaller scale (4-bit K +5%, 4-bit V +0.4%).'
                   + at8 + ' For the energy question this means 8-bit K + 4-bit V is the useful configuration to test next: '
                   'nearly the memory of 4-bit with the accuracy of 8-bit.'))
    return s


def src(r):
    if not r.get('_file', '').startswith('ppl_v2'):
        return 'v1'
    return 'v2, all-quants' if 'allq' in r.get('build', '') else 'v2'


def results_fit():
    s = [P('10.10 Where compression helps: the memory fit test', st_h2)]
    tags = fit_tags()
    if not tags:
        s.append(P('<i>Running at the time of writing; see section 13.</i>'))
        return s
    s.append(P('One server per format and context, auto-fit off (-fit off) so llama.cpp cannot quietly shrink the '
               'context. "did not fit" = the server failed to start (out of memory). 2 rounds.'))
    rows = [['Model, context', 'f16', '8-bit', '4-bit']]
    for tag, m, ctxs in (('15W-fit1b', LL, (32768, 65536)), ('15W-fit3b', L3, (16384, 32768, 65536))):
        if tag not in tags:
            continue
        for c in ctxs:
            cells = []
            for kv in ('f16', 'q8_0', 'q4_0'):
                v = get('energy_per_request_net_j', tag, m, kv, c)
                alloc = next((r.get('kv_alloc_mib') for r in S.get('failures', [])
                              if r.get('tag') == tag and r['model'] == m and r['kv_type'] == kv and
                              r['context_len'] == c), None)
                if v:
                    cells.append(f'{v[0]:.0f} J, {f(mean("decode_tps", tag, m, kv, c))} tok/s')
                elif alloc:
                    cells.append(f'did not fit ({alloc / 1024:.1f} GiB cache)')
                else:
                    cells.append('–')
            rows.append([f'{NAME[m]}, {c // 1024}K'] + cells)
    s.append(table(rows, [44, 43, 43, 44]))
    s.append(P('Cells show net energy per request and decode speed, mean of 2 rounds. Llama-1B at 128K was not run: '
               'a 131,072-token prompt plus 128 generated tokens exceeds its trained context.', st_small))
    s.append(P('Llama-1B fits 64K in every format, so on the 1B model compression only costs energy. On the 3B model '
               'the formats separate: with f16 the cache for 32K (3.5 GiB) does not fit next to the 2 GB of weights, '
               '8-bit reaches 32K and 4-bit 64K. Where both run, 4-bit still uses more energy per request than f16 '
               '(at 16K), but it is the only way to serve 64K at all. This is the trade the thesis question asks about: '
               'on an 8 GB device, KV quantization buys context length, and pays for it in energy and speed.'))
    return s


def results_modes():
    s = [P('10.11 7W against 15W, gross and net', st_h2)]
    if not has_7w():
        s.append(P('<i>The 7W rerun was not finished at the time of writing; v1 found 7W uses more energy per request '
                   'than 15W for f16 (6–23% net) and doubles the quantization penalty.</i>'))
        return s
    s.append(P('Same requests at both power modes. Positive = 7W uses more energy. Gross includes the idle power '
               'that the longer 7W runs keep paying; net does not.'))
    rows = [['Model, KV, context', 'Net J, 15W', 'Net J, 7W', '7W vs 15W, net', 'Gross J, 15W', 'Gross J, 7W',
             '7W vs 15W, gross']]
    for m in M:
        for kv in ('f16', 'q8_0', 'q4_0'):
            for c in (1024, 16384):
                n15, n7 = (mean('energy_per_request_net_j', t, m, kv, c) for t in ('15W-v2', '7W-v2'))
                g15, g7 = (mean('energy_per_request_j', t, m, kv, c) for t in ('15W-v2', '7W-v2'))
                rows.append([f'{NAME[m]}, {KVN[kv]}, {c // 1024}K', f(n15), f(n7), pct(n7, n15), f(g15), f(g7),
                             pct(g7, g15)])
    s.append(table(rows, [44, 20, 20, 24, 22, 20, 24]))
    s.append(P('<b>The low-power mode costs energy, both net and gross.</b> At 7W the GPU runs at 408 MHz instead of '
               '625 MHz and decode is 2.0–2.4× slower (Llama f16 at 16K: 22.8 → 10.3 tok/s), while decode power only '
               'falls from about 10 W to 7 W. The work therefore costs 21–31% more net energy, and the idle power '
               'paid over the longer run raises the gross cost by 60–73%. This confirms v1 ("race to idle") with '
               'more rounds. The quantization penalty is somewhat larger at 7W (16K: Llama +21% / +18%, Qwen +28% / '
               '+21% for 8-bit / 4-bit, against +13–18% at 15W) because the slower GPU makes the vec kernel\'s extra '
               'compute weigh more. 7W is only worth using under a hard power or thermal limit.'))
    s.append(P('<i>Idle correction.</i> The 7W grid started right after the reboot that the power-mode switch needs, '
               'and its start-of-set idle reading (3.68 W) was taken while the board was still settling; the reading '
               'at the end of the set was 3.03 W, and the 7W llama-bench set measured 2.92–3.05 W. Net energy in this '
               'report therefore subtracts the lower of the start and end idle readings of each set. This changes '
               'only the 7W grid (the 15W sets start lower than they end; one differs by 0.01 W). Gross energy is '
               'not affected.', st_small))
    return s


# ---------------------------------------------------------------- 11-13, appendices

def data_quality():
    s = [P('11. Data quality', st_h1)]
    fails = S.get('failures', [])
    by = {}
    for r in fails:
        by.setdefault((r.get('tag'), r['model'], r['kv_type'], r.get('n_parallel', 1)), []).append(r)
    for t in [
        f'<b>Runs.</b> Every row of every result file is kept, including failures ({len(fails)} failed runs, all '
        'listed below). Analysis uses only runs with status "ok".',
        '<b>Spread.</b> Every paired 95% CI on energy per request in the 15W main grid is within ±3.6 percentage '
        'points, most within ±2 (tables in 10.1). The 7W grid is tighter: all within ±1.6 points per request and '
        '±2.5 per decoded token.',
        f'<b>Idle drift.</b> Idle VDD_IN at the start and end of the 15W main grid: {idle("15W-v2")}. The drift '
        '(about 0.25 W, fan and board warming up) is small against 9–11 W decode power and affects all formats '
        'equally because they are interleaved within each round. 7W: 3.68 W at the start (just after the reboot) '
        'and 3.03 W at the end; net energy uses the lower reading (section 10.11).',
        '<b>Background load.</b> v1 ran with the Software Updater active (one core busy). v2 closes it first; the '
        'v1 and v2 15W percentages agree within a few points, so it did not change the conclusions.',
        '<b>Other software.</b> An Ollama service (installed on the board before this project) runs in the '
        'background in all v1 and v2 runs. It is idle (about 40 MB of memory, no CPU or GPU use), so its draw is '
        'part of the measured idle power.',
        '<b>Thermals.</b> GPU temperature and clock are logged after every run. In sets A, D and E the GPU clock '
        'was at the 15W maximum (624.75 MHz) after every run and the temperature stayed between 48 and 54 °C. In '
        'the 7W grid the clock was at the 7W maximum (408 MHz) after every run, at 45–47 °C. No throttling.',
    ]:
        s.append(P('• ' + t))
    if by:
        rows = [['Set, model, format, sessions', 'Failed runs', 'Reason']]
        for (tag, m, kv, n), rs in sorted(by.items()):
            why = 'out of memory at server start' if rs[0]['status'] == 'server_failed' else rs[0]['status']
            rows.append([f'{tag}, {NAME.get(m, m)}, {kv}, N = {n}', str(len(rs)), why])
        s.append(table(rows, [90, 22, 62]))
    return s


def scope():
    s = [P('12. Scope and limitations', st_h1)]
    for t in [
        'One device (Jetson Orin Nano 8 GB), one engine (llama.cpp 7adf160), two power modes. Other engines with '
        'fused low-bit attention kernels (e.g. TensorRT-LLM, MLC) could change the sign of the result; the claim is '
        'about this software stack, which is the one most edge users run.',
        'Small models (0.5B–3B) and greedy decoding with fixed 128-token replies. Longer replies shift the balance '
        'further toward decode, where the penalty is largest.',
        'Energy is measured at the module input (VDD_IN). It includes the fan and regulators but not the carrier '
        'board or power adapter.',
        'Only quantization is covered. Eviction methods (StreamingLLM, SnapKV) are the next experiment.',
    ]:
        s.append(P('• ' + t))
    return s


def next_steps():
    s = [P('13. Next steps', st_h1)]
    items = []
    if not has_7w():
        items.append('Finish the 7W rerun of sets A and B (section 10.11).')
    if not ppl_v2_done():
        items.append('Finish the perplexity runs (10.9).')
    if not fit_tags():
        items.append('Finish the fit test (10.10).')
    items += [
        'Eviction on the Jetson: StreamingLLM and SnapKV through kvpress (PyTorch for JetPack), same energy method.',
        'Comparison with the thesis T4 data as the server-side reference point.',
        'Confirm the main question and claim (section 1) with the supervisors, then start the paper draft.',
    ]
    for t in items:
        s.append(P('• ' + t))
    return s


def appendix_files():
    s = [P('Appendix B. Files', st_h1)]
    rows = [['File', 'Contents']]
    for name, meta in sorted(S.get('meta', {}).items()):
        if name.startswith('pptg'):
            rows.append([f'results/{name}', f'set B, llama-bench, {meta.get("tag", "")} (+ .meta.json)'])
            continue
        rows.append([f'results/{name}', f'{meta.get("tag", "")}: '
                     + {'15W': 'v1 main grid, 15W', '7W': 'v1 main grid, 7W', '15W-v2': 'set A, 15W',
                        '7W-v2': 'set A, 7W', '15W-np': 'set D, concurrency', '15W-w4': 'set E, Q4_K_M weights',
                        '15W-fit1b': 'set G, Llama-1B', '15W-fit3b': 'set G, Llama-3B'}.get(meta.get('tag'), 'set B')
                     + ' (+ .meta.json: settings, idle power, versions)'])
    rows += [['results/ppl_*.csv', 'perplexity (v1 at 2K; v2 at 8K and the K/V split; the 16K attempts are the rows '
              'without a value)'],
             ['results/prof/', 'nsys kernel summaries (*_cuda_gpu_kern_sum.csv) and ncu counters (*_ncu.csv)'],
             ['results/logs/', 'llama-server log of every server start'],
             ['plots_v2/stats.json', 'every number in this report, with CIs'],
             ['analysis/analyze_v2.py, make_report.py, report_content.py', 'statistics, figures and this PDF']]
    s.append(table(rows, [70, 104]))
    return s


def appendix_v1_rounds():
    s = [P('Appendix C. v1 per-round values: Qwen-0.5B, 7W, 16K', st_h1)]
    p = ROOT / 'results' / 'kv_20261019_194427.jsonl'
    if not p.exists():
        return s
    rs = [json.loads(ln) for ln in open(p) if ln.strip()]
    rows = [['Format', 'Round', 'Decode J/token, net', 'Decode tok/s', 'Request energy, net J', 'GPU °C at end']]
    for kv in ('f16', 'q8_0', 'q4_0'):
        for r in sorted((r for r in rs if r['model'] == QW and r['kv_type'] == kv and r['context_len'] == 16384),
                        key=lambda r: r['round']):
            rows.append([kv, str(r['round'] + 1), f(r['decode_j_per_token'], 3), f(r['decode_tps']),
                         f(r['energy_net_j']), f(r.get('gpu_temp_end'))])
    s.append(table(rows, [24, 16, 36, 30, 38, 30]))
    s.append(P('The review asked whether this cell (decode energy roughly doubled by 8-bit) was an outlier. The '
               'rounds agree closely, so it is a stable result of the 7W mode, not noise.', st_small))
    return s


def appendix_glossary():
    s = [P('Appendix D. Glossary', st_h1)]
    rows = [['Term', 'Meaning'],
            ['KV cache', 'The keys and values of every token read so far, kept so each new token only computes its '
                         'own. Grows linearly with context.'],
            ['Prefill / decode', 'Reading the prompt (all tokens at once) / generating the reply one token at a time.'],
            ['Flash attention (FA)', 'Attention computed in one fused kernel without storing the full score matrix.'],
            ['Tensor cores', 'GPU units for small matrix multiplies; much faster than ordinary CUDA cores for that.'],
            ['mma / vec kernel', 'llama.cpp\'s tensor-core flash-attention kernel / its one-row CUDA-core kernel.'],
            ['Net / gross energy', 'Energy with idle power subtracted / energy the module actually drew.'],
            ['Paired change, 95% CI', 'Mean of per-round ratios against f16 from the same round, with the interval '
                                      'that contains the true mean with 95% confidence.'],
            ['n.s.', 'Not significant: the CI includes zero.'],
            ['Perplexity', 'How surprised the model is by real text; lower is better.'],
            ['Q8_0, Q4_K_M', 'Weight formats: 8-bit; 4-bit with per-block scales (K-quant, "medium").']]
    s.append(table(rows, [36, 138]))
    return s


def story():
    s = title_page()
    s += response_to_review()
    s.append(PageBreak())
    s += summary()
    s += background()
    s += timeline()
    s += setup_sections()
    s += experiments()
    s.append(PageBreak())
    s += results_energy()
    s += results_pptg()
    s += results_mechanism()
    s += results_kernels()
    s += results_rails()
    s += results_concurrency()
    s += results_w4()
    s += results_ppl()
    s += results_fit()
    s += results_modes()
    s += data_quality()
    s += scope()
    s += next_steps()
    s.append(PageBreak())
    s += appendix_repro()
    s += appendix_files()
    s += appendix_v1_rounds()
    s += appendix_glossary()
    return s
