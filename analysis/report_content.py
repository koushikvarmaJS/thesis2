"""Sections of Jetson_KV_Report_v2.pdf. Layout helpers and data access live in make_report.py."""
import json
import re

from make_report import (CTX, KVN, L3, LL, LL4, NAME, PLOTS, QW, QW4, ROOT, S, P, PageBreak, Spacer, chg, f, figure,
                         get, mean, sig, st_body, st_box, st_code, st_h1, st_h2, st_small, st_sub, st_title, table)

M = (LL, QW)


def title_page():
    return [
        Spacer(1, 30),
        P('KV-Cache Compression on NVIDIA Jetson Orin Nano: Quantization and Eviction', st_title),
        P('Energy, speed, memory and accuracy of f16, 8-bit and 4-bit KV caches in llama.cpp, and of the thesis '
          'eviction methods (StreamingLLM, SnapKV) in PyTorch. Report v4: the v2 response to the review; model '
          'size and family, longer replies and eviction (v3); and mixed K/V formats, retrieval accuracy of every '
          'method, quantization in PyTorch, the PyTorch decode step and the T4 comparison (v4). Every experiment run '
          'so far is in here.',
          st_sub),
        P('Koushik Bhupathiraju · Supervisors: Prof. Taskin Kocak, Prof. Rongyu Lin', st_body),
        P('v1 experiments: 19 October 2026 · v2: 26–27 October · v3 (sets H–K): 31 October – 2 November · '
          'v4 (sets L–Q): 11–12 November 2026',
          st_body),
        P('Code and raw data: <u>github.com/koushikvarmaJS/thesis2</u>', st_body),
        Spacer(1, 14),
        P('<b>Contents</b>'),
        P('0. Response to the review<br/>1. Summary and main claim<br/>2. Background and thesis scope<br/>'
          '3. What happened, step by step<br/>4. Hardware<br/>5. Software, libraries and tools<br/>6. Models<br/>'
          '7. Compression methods: KV cache formats and eviction<br/>8. How energy was measured<br/>'
          '9. Experiments and parameters<br/>10. Results (10.1–10.19), 10.20 Questions and answers<br/>'
          '11. Data quality<br/>12. Scope and limitations<br/>13. Next steps<br/>'
          'Appendix A. Reproduction · B. Files · C. v1 results · D. Glossary · E. Everything produced so far'),
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
                             '-DCMAKE_CUDA_ARCHITECTURES=87, Release. Used for all llama.cpp runs, v1 to v3.'],
        ['Second build', 'Same commit and flags plus -DGGML_CUDA_FA_ALL_QUANTS=ON. Without it, GPU flash attention '
                         'only exists for matching K/V types (f16/f16, q8_0/q8_0, q4_0/q4_0); a mixed pair such as '
                         'K f16 + V q4_0 would run attention on the CPU. Used for the K/V split perplexity tests (sets C, '
                         'J, Q) and for all of v4 set L and the llama.cpp half of set M (f16 rerun in it as baseline).'],
        ['llama.cpp programs', 'llama-server (energy), llama-bench (separate prefill/decode), llama-perplexity '
                               '(accuracy)'],
        ['Profilers', 'NVIDIA Nsight Systems 2025.6.3 (nsys), Nsight Compute 2026.1.1 (ncu)'],
        ['Benchmark code', 'Python 3.12.3, standard library only: power.py, bench_kv.py, bench_pp_tg.py, '
                           'ppl_v2.sh, ppl_v3.sh, ppl_v4.sh, profile.sh, run_v2.sh, run_v3.sh, run_v4.sh, run_v4b.sh'],
        ['Eviction (set K)', 'Separate Python 3.12 venv: PyTorch 2.15.0+cu132 (upstream cu132 wheel index; a plain '
                             'pip install gives a build that does not run on Orin), transformers 4.57.6, accelerate '
                             '1.15.0, NVIDIA kvpress 0.5.3 (the version the thesis used). Scripts: setup_evict.sh, '
                             'bench_evict.py (same power.py). v4 adds optimum-quanto 0.2.7 (its CUDA extension compiled '
                             'on the board, sm_87) and hqq 0.2.8 for transformers\' QuantizedCache, and needle_qa.py '
                             '(accuracy) and profile_step.py (torch.profiler).'],
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
    s.append(Spacer(1, 4))
    s.append(table([
        ['Added in v3 (sets H–J)', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Gemma-3-1B-it'],
        ['Weights', 'Q4_K_M, 1.12 GB (Qwen GGUF)', 'Q4_K_M, 2.10 GB (Qwen GGUF)', 'Q4_K_M, 0.81 GB (ggml-org GGUF)'],
        ['Layers', '28', '36', '26 (4 full-context, 22 sliding-window of 512 tokens)'],
        ['Query heads / KV heads', '12 / 2', '16 / 2', '4 / 1'],
        ['Head size', '128', '128', '256'],
        ['KV cache per token, f16', '28 KiB', '36 KiB', '4 KiB in the full-context layers + a fixed window'],
    ], [40, 45, 45, 44]))
    s.append(P('Set H also uses Qwen2.5-0.5B, Llama-3.2-1B and Llama-3.2-3B, all with Q4_K_M weights (table above). '
               'Set K (eviction) loads the original 16-bit weights from Hugging Face instead of GGUF files: '
               'Qwen/Qwen2.5-0.5B-Instruct (0.95 GB) and unsloth/Llama-3.2-1B-Instruct (2.4 GB; an ungated copy of '
               'Meta\'s weights), in fp16 with PyTorch\'s scaled-dot-product attention.', st_small))

    s.append(P('7. Compression methods: KV cache formats and eviction', st_h1))
    s.append(P('7.1 Quantization: KV cache formats (llama.cpp)', st_h2))
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
    s.append(P('7.2 Eviction: StreamingLLM and SnapKV (kvpress, set K)', st_h2))
    for t in [
        '<b>When it happens.</b> kvpress attaches a hook to every attention layer. During prefill each layer first '
        'computes attention over the whole prompt as usual; then the hook scores every cached token, separately '
        'for each KV head, keeps the best (1 − ratio) × N tokens of that head and deletes the rest. This happens '
        'once per request, at the end of prefill. During decode nothing more is removed: the 128 new tokens are '
        'added in full. "50% removed" (compression ratio 0.5) halves the prompt\'s cache: Qwen at 16K goes from '
        '193.5 MiB to 97.5 MiB, and 49.5 MiB with 75% removed.',
        '<b>StreamingLLM: by position.</b> Keeps the first 4 tokens (the "attention sinks": models put a large share '
        'of attention on the first tokens whatever they say, and removing them breaks generation) and the most '
        'recent tokens, and deletes the middle. It never looks at the content, so every layer and head keeps the '
        'same tokens. Cheap, but it can delete an important fact from the middle of a document.',
        '<b>SnapKV: by importance.</b> Takes the last 64 prompt tokens as an "observation window" (usually where the '
        'question is), recomputes how much attention those 64 queries pay to every earlier token, averages over the '
        '64 queries, smooths the scores with a 5-token average (so neighbouring tokens are kept together) and '
        'averages over the query heads that share a KV head. The highest-scoring tokens are kept, wherever they '
        'are, plus the window itself. Each layer and KV head chooses its own tokens. It keeps what the question '
        'needs, at the cost of extra prefill work.',
        '<b>Settings used.</b> StreamingLLMPress(n_sink=4), SnapKVPress(window_size=64, kernel_size=5 by default), '
        'each at compression ratio 0.5 and 0.75; both ran in every round, model and context.',
        '<b>Question inside or after (set M).</b> In the energy runs the whole prompt is compressed, so a question '
        'at its end sits in SnapKV\'s window ("question inside"). kvpress\'s own evaluation compresses the document '
        'first and adds the question afterwards ("question after"), as when one cached document gets several '
        'questions; then SnapKV must guess what matters without seeing the question. Set M tests both.',
    ]:
        s.append(P(t))

    s.append(P('7.3 Quantization in PyTorch (transformers QuantizedCache, sets M and N)', st_h2))
    s.append(P('transformers\' QuantizedCache follows KIVI: the newest 128 tokens stay in fp16 (the "residual"); '
               'older tokens are quantized in groups of 64 values with a scale and a zero point each, by one of two '
               'backends: optimum-quanto (4 or 2 bits) or HQQ (8 or 4 bits here). Every decode step dequantizes the '
               'whole stored cache back to fp16 in PyTorch and then runs ordinary attention, so it saves memory but '
               'adds work. It differs from llama.cpp\'s formats in group size (64 vs 32), in having a zero point and '
               'in keeping recent tokens exact. On the T4 (sm_75) quanto\'s kernels did not build; on the Orin (sm_87) '
               'they do.'))

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
        '<b>Eviction runs (set K, bench_evict.py).</b> Hugging Face generate() with greedy decoding and '
        'min_new_tokens = max_new_tokens = 128, so every reply is exactly 128 tokens. A stopping-criterion hook that '
        'never stops records the moment the first generated token exists (after a CUDA synchronize): the end of '
        'prefill, eviction included. Decode J/token = decode energy ÷ 127 (the first token belongs to prefill). The '
        'cache size is summed from the tensors of the returned cache. Same power meter, idle readings and paired '
        'statistics as the llama.cpp sets; the baseline row is the same model in PyTorch with no press.',
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
    s.append(P('v3 (sets H–J) and eviction (set K), 15W, inside tmux:', st_body))
    s.append(P('bash jetson/run_v3.sh 2>&amp;1 | tee results/run_v3.log    # RESUME=1 after a crash<br/>'
               'bash jetson/setup_evict.sh                  # PyTorch venv in ~/evict-venv<br/>'
               '~/evict-venv/bin/python -u jetson/bench_evict.py --tag 15W-evict<br/>'
               '~/evict-venv/bin/python -u jetson/bench_evict.py --models unsloth/Llama-3.2-1B-Instruct '
               '--contexts 1024,4096,8192 --tag 15W-evict<br/>'
               '~/evict-venv/bin/python -u jetson/bench_evict.py --methods f16 --contexts 2048,3072,5120,6144,12288 '
               '--rounds 3 --tag 15W-evict-scan', st_code))
    s.append(P('v4 (sets L–Q), 15W, inside tmux (quanto compiles its CUDA kernels on first use: do that alone, with '
               'MAX_JOBS=2, or the board runs out of memory):', st_body))
    s.append(P('cmake --build llama.cpp/build-allq --target llama-server<br/>'
               '~/evict-venv/bin/pip install --no-deps optimum-quanto==0.2.7 hqq==0.2.8.post1 ninja<br/>'
               'STEPS="L M" bash jetson/run_v4.sh 2&gt;&amp;1 | tee results/run_v4_LM.log<br/>'
               'bash jetson/run_v4b.sh 2&gt;&amp;1 | tee results/run_v4.log   # M (PyTorch) again, then N O P Q', st_code))
    s.append(P('On the Mac:', st_body))
    s.append(P('rsync -a &lt;user&gt;@&lt;jetson&gt;:~/thesis2/results/ results/<br/>'
               'python3 analysis/analyze_v2.py --out plots_v2<br/>'
               'python3 analysis/analyze_v4.py<br/>'
               'python3 analysis/make_report.py', st_code))
    return s


# ---------------------------------------------------------------- results

def tags_present():
    return [t for t in ('15W-v2', '7W-v2') if any(k.startswith(t + '|') for k in S['energy_per_request_net_j'])]


def has_tag(tag):
    return any(k.startswith(tag + '|') for k in S['energy_per_request_net_j'])


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


def pct1(a, b):
    return f'{100 * (a / b - 1):+.1f}%' if a and b else '–'


def ppl_rows(v3=False):
    """Perplexity rows. The v3 file (size/family models) is reported in 10.12; only its Llama-1B K/V split
    (Q8_0 weights, all-quants build) belongs to the 10.9 table."""
    rows = S.get('ppl', [])
    if v3:
        return [r for r in rows if r.get('_file', '').startswith('ppl_v3')]
    return [r for r in rows if not r.get('_file', '').startswith('ppl_v3') or 'allq' in r.get('build', '')]


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
            ['Follow-up: other model sizes and families; longer replies',
             'Set H: Qwen2.5 0.5B/1.5B/3B, Llama 3.2 1B/3B, Gemma-3 1B. Set I: 512-token replies. The result holds '
             'on all six; the penalty shrinks with size and nearly vanishes on Gemma.', '10.12, 10.13'],
            ['Follow-up: eviction on the Jetson',
             'Set K: StreamingLLM and SnapKV (kvpress, PyTorch), 50% and 75% removed, Qwen-0.5B and Llama-1B, '
             '5 rounds. Eviction saves energy only where it makes decode faster (8K and 16K).', '10.14'],
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
        'Six models (Qwen2.5 0.5B/1.5B/3B, Llama 3.2 1B/3B, Gemma-3 1B, set H): no quantized cell is cheaper than f16. '
        'The penalty shrinks with model size (Llama +16% → +11% per request at 16K) and nearly vanishes on '
        'Gemma-3 (+3–4%), which keeps the full context in only 4 of 26 layers. With 512-token replies it grows '
        '1.7–2.2× (e.g. Qwen-0.5B +18% → +39%). Every Qwen2.5 size breaks with a 4-bit cache (section 10.12).',
    ] if has_size() else []) + ([
        'Eviction (StreamingLLM, SnapKV through kvpress on PyTorch, set K) saves energy only at long prompts: '
        '−40 to −47% per request at 8K (and at 16K with 75% removed; −26% with 50%), +2 to −6% at 1K–4K. The saving comes from PyTorch decode '
        'slowing down in one step between 6K and 8K cached tokens, which removing the cache avoids; the PyTorch baseline itself '
        'uses about 4× the energy of llama.cpp at 16K (section 10.14).',
    ] if has_tag('15W-evict') else []) + ([
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
        'bits) in llama.cpp, and then the thesis eviction methods on the same board through PyTorch and kvpress '
        '(section 10.14).',
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
    for p in [ROOT / 'results' / f for f in ('run_v2_15W.log', 'run_v2_7W.log', 'run_v3.log', 'run_v3_resume.log',
                                             'run_v4_LM.log', 'run_v4.log')]:
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
              ('2026-10-27 02:20:00', '15W', 'F (ncu) Nsight Compute counters of the decode attention kernel (root)'),
              ('2026-10-31 21:34:32', '15W', 'H stopped: llama-server killed (out of memory) during Llama-3B with f16; '
                                             'board rebooted at 21:52 (section 11)'),
              ('2026-11-02 04:55:48', '15W', 'K eviction (kvpress on PyTorch): Qwen-0.5B, then Llama-1B'),
              ('2026-11-02 06:04:52', '15W', 'K stopped: Llama-1B at 16K ran out of memory in PyTorch; board '
                                             'rebooted. Qwen complete (100 runs)'),
              ('2026-11-02 06:12:20', '15W', 'K (rerun) Llama-1B at 1K–8K'),
              ('2026-11-11 18:36:00', '15W', 'v4 preparation: board rebooted (out of memory) while optimum-quanto '
                                             'compiled its CUDA kernels next to a loaded model; recompiled alone'),
              ('2026-11-11 21:17:00', '15W', 'M (PyTorch half) restarted: its result file was deleted by mistake during '
                                             'a side test, and the quantized-cache path had failed with HQQ; rerun '
                                             'from the start with the fix (section 11)')]
    for when, mode, what in sorted(run_log_stages() + manual):
        rows.append([f'{when[5:16]} ({mode})', what])
    s.append(table(rows, [34, 140]))
    s.append(P('Times from results/run_v2_*.log, run_v3*.log and run_v4*.log. Each set\'s raw data is a separate file (Appendix B).',
               st_small))
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
            ['H', 'Model size and family (follow-up, 31 Oct)', 'Qwen2.5 0.5B/1.5B/3B, Llama 3.2 1B/3B, Gemma-3 1B '
             '(all Q4_K_M)', 'f16, q8_0, q4_0', '1K–16K', '5'],
            ['I', 'Longer replies: 512 generated tokens', 'same 6', 'f16, q8_0, q4_0', '4K, 16K', '3'],
            ['J', 'Perplexity of the 6 models; Llama-1B K/V split', 'same 6; Llama-1B (Q8_0)',
             'f16, q8_0, q4_0; mixed K/V', '2K, 8K (Gemma 2K, 4K)', '1 (± error)'],
            ['K', 'Eviction: kvpress on PyTorch, 128 generated tokens', 'Qwen2.5-0.5B, Llama-3.2-1B (fp16)',
             'none; StreamingLLM, SnapKV with 50% / 75% removed', '1K–16K (Llama 1K–8K); scan without eviction 2K–12K', '5 (scan 3)'],
            ['L', 'Mixed K/V formats, energy (v4); nsys at 16K', 'Llama-1B, Qwen-0.5B (Q8_0), Qwen2.5-1.5B (Q4_K_M)',
             'f16, q8_0, q4_0, 8-bit K + 4-bit V, f16 K + 4-bit V (all-quants build)', '1K–16K', '5'],
            ['M', 'Accuracy: needle retrieval with 3 distractors, 30 prompts per cell', 'Qwen2.5-0.5B, Llama-3.2-1B',
             'llama.cpp: the 5 set-L formats; PyTorch: none, eviction (question inside / after), quanto 4/2, HQQ 8/4',
             '1K, 4K, 8K', '1 (30 prompts)'],
            ['N', 'Quantized cache in PyTorch, energy', 'Qwen2.5-0.5B, Llama-3.2-1B (fp16)',
             'none, quanto 4/2-bit, HQQ 8/4-bit, SnapKV 50%', '1K, 4K, 8K', '3'],
            ['O', 'Profile of the PyTorch decode step (torch.profiler)', 'same', 'none (sdpa; Qwen also eager)',
             '2K–12K', '32 timed + 8 profiled steps'],
            ['P', 'Eviction at the T4 thesis settings: 50 generated tokens', 'same', 'none, StreamingLLM 50%, SnapKV 50%',
             '2K, 8K', '5'],
            ['Q', 'Perplexity of the mixed formats', 'Qwen2.5-1.5B (Q4_K_M)', 'f16, 8-bit K + 4-bit V, f16 K + 4-bit V',
             '8K', '1 (± error)'],
            ]
    s.append(table(rows, [9, 50, 36, 34, 30, 15]))
    s.append(Spacer(1, 4))
    for t in [
        '<b>Common settings.</b> All layers on the GPU (-ngl 99), flash attention on unless stated, -c = (largest '
        'context + 128 + 64) × sessions, one server start per format and round. ignore_eos makes every reply '
        'exactly 128 tokens; cache_prompt=false makes every request process its prompt. Temperature 0.',
        '<b>Prompt text.</b> The energy prompts are the first N tokens (N = the context length, exactly) of a '
        'public-domain novel, <i>Pride and Prejudice</i> (Project Gutenberg #1342, including its header), as tokenized '
        'by each model\'s own tokenizer and sent as token IDs; at the end of a reply the cache holds N + 128 tokens. '
        'WikiText-2 is used only for perplexity (sets C, J and Q). Sets K, L, N and P use the same novel and the '
        'same N; set M builds its prompts from it too (section 10.16).',
        '<b>Order.</b> Inside each round, formats and contexts run in a random order drawn anew every round '
        '(Python\'s random.shuffle, not seeded), so no format always runs first or on a cooler board; the result '
        'files keep the rows in the order they ran. A 1K-token warm-up request runs after each server start '
        '(set K: an 8-token warm-up generation after loading each model, and 2 s between runs).',
        '<b>Sets A and B run at 15W and at 7W</b>; C–Q at 15W only (the 7W slowdown affects all formats alike in v1).',
        '<b>Sets H, I, L and M</b> start llama-server with --cache-ram 0 (section 11). All six models use Q4_K_M weights, so '
        'the 3B models fit next to a 16K f16 cache and every size is compared on the same weight format.',
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
    l8 = lambda k, v: split8.get((LL, k, v))  # noqa: E731
    if q('f16', 'q4_0') and q('q4_0', 'f16'):
        at8 = ''
        if q8('f16', 'q4_0') and q8('q4_0', 'f16'):
            at8 = (f' At 8K the split gives the same answer: f16 K + 4-bit V {q8("f16", "q4_0"):.2f} '
                   f'(f16: {q8("f16", "f16"):.2f}), 4-bit K + f16 V {q8("q4_0", "f16"):.1f}, q4_1 '
                   f'{f(q8("q4_1", "q4_1"), 1)}.')
        if l8('f16', 'q4_0') and l8('q4_0', 'f16') and l8('f16', 'f16'):
            at8 += (f' Llama at 8K (v3): 4-bit K only {pct1(l8("q4_0", "f16"), l8("f16", "f16"))}, 4-bit V only '
                    f'{pct1(l8("f16", "q4_0"), l8("f16", "f16"))}, 8-bit K + 4-bit V '
                    f'{pct1(l8("q8_0", "q4_0"), l8("f16", "f16"))} against f16.')
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
    f = r.get('_file', '')
    v = 'v3' if f.startswith('ppl_v3') else 'v2' if f.startswith('ppl_v2') else 'v1'
    return f'{v}, all-quants' if 'allq' in r.get('build', '') else v


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

SZ = ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-1.5b-instruct-q4_k_m', 'qwen2.5-3b-instruct-q4_k_m',
      'Llama-3.2-1B-Instruct-Q4_K_M', 'Llama-3.2-3B-Instruct-Q4_K_M', 'gemma-3-1b-it-Q4_K_M']
# name, layers (full-context), query heads, KV heads, head dim
SZ_SPEC = {SZ[0]: ('Qwen2.5-0.5B', '24', 14, 2, 64), SZ[1]: ('Qwen2.5-1.5B', '28', 12, 2, 128),
           SZ[2]: ('Qwen2.5-3B', '36', 16, 2, 128), SZ[3]: ('Llama-3.2-1B', '16', 32, 8, 64),
           SZ[4]: ('Llama-3.2-3B', '28', 24, 8, 128), SZ[5]: ('Gemma-3-1B', '26 (4)', 4, 1, 256)}


def has_size():
    return has_tag('15W-size')


def ppl3(m, kv, c):
    for r in ppl_rows(v3=True):
        if r['model'] == m and r['type_k'] == kv and r['type_v'] == kv and int(r['ctx']) == c and r.get('ppl'):
            return float(r['ppl'])
    return None


def results_size():
    s = [P('10.12 Model size and family (set H)', st_h2)]
    if not has_size():
        return s + [P('<i>Not run yet.</i>')]
    s.append(P('Requested after the v2 report: the same family at different sizes (Qwen2.5 0.5B → 1.5B → 3B, Llama 3.2 '
               '1B → 3B) and different families at the same size (Llama-3.2-1B, Qwen2.5-1.5B, Gemma-3-1B). All six '
               'with Q4_K_M weights, 1K–16K, 5 paired rounds at 15W. Changes are at 16K against the same model with an '
               'f16 cache.'))
    rows = [['Model', 'Layers', 'Query / KV heads', 'KV cache 16K, f16', 'f16: J, tok/s at 16K',
             'Request, 8-bit', 'Request, 4-bit', 'Decode/token, 8-bit', 'Decode/token, 4-bit']]
    for m in SZ:
        name, layers, qh, kh, hd = SZ_SPEC[m]
        rows.append([name, layers, f'{qh} / {kh}', f'{mean("kv_used_mib", "15W-size", m, "f16", 16384):.0f} MiB',
                     f'{mean("energy_per_request_net_j", "15W-size", m, "f16", 16384):.0f} J, '
                     f'{mean("decode_tps", "15W-size", m, "f16", 16384):.1f}']
                    + [chg(met, '15W-size', m, kv, 16384, show_ci=False)
                       for met in ('energy_per_request_net_j', 'decode_j_per_token') for kv in ('q8_0', 'q4_0')])
    s.append(table(rows, [24, 13, 17, 19, 25, 18, 18, 20, 20]))
    s.append(P('Net energy, paired by round. Gemma-3-1B keeps the full context in only 4 of its 26 layers; the other 22 '
               'are sliding-window layers (512 tokens), so it uses 86 MiB of cache at 16K (87 MiB allocated), not the 419 MiB a full-context '
               'cache of that shape would need. Every cell in the set (6 models × 2 formats × 4 contexts × 3 energy '
               'measures, 144 comparisons) is more expensive than f16; 143 significantly, one not separable from zero '
               '(Qwen-3B, 4-bit, 1K, decode per token, +1.9%).', st_small))
    s += figure(PLOTS / '10_size_family.png', 'Figure 6. Extra energy of an 8-bit and 4-bit KV cache at 16K, per model. '
                'Bars: mean of 5 paired rounds; whiskers: 95% CI.')
    for tx in [
        '<b>Same family, bigger model: smaller penalty, same sign.</b> Per request at 16K, Qwen goes from +18% (0.5B) '
        'to +14% (1.5B and 3B) with an 8-bit cache, and Llama from +16% (1B) to +11% (3B). Per decoded token the drop '
        'is larger: Qwen +60% → +49% → +38%, Llama +43% → +27%. A bigger model spends more of each decode step '
        'multiplying its weights, which costs the same with any cache, so the slower attention kernel is a smaller '
        'share of the step. Decode speed still falls by about a third on every Qwen size; what changes is power: '
        'with the quantized cache, decode power rises 8% on Qwen-0.5B but falls 5% on Qwen-3B, so the energy penalty '
        'shrinks even though the slowdown does not.',
        '<b>Same size, different family: architecture decides.</b> At about 1B parameters Llama-1B (+16% / +13% per '
        'request) and Qwen-1.5B (+14% / +13%) behave alike, but Gemma-3-1B pays only +4% / +3%. The penalty comes '
        'from decode attention over the long cache, and Gemma does that over the full 16K in only 4 layers. Its '
        'cache is also small to begin with (86 MiB), so compression saves Gemma almost nothing either: 86 → 46 → 24 '
        'MiB.',
        '<b>Accuracy follows the family, not the size.</b> Perplexity at 8K (Gemma: 4K):',
    ]:
        s.append(P(tx))
    prow = [['Model', 'Context', 'f16', '8-bit', '4-bit', '4-bit vs f16']]
    for m in SZ:
        c = 4096 if m.startswith('gemma') else 8192
        a, b, d = ppl3(m, 'f16', c), ppl3(m, 'q8_0', c), ppl3(m, 'q4_0', c)
        prow.append([SZ_SPEC[m][0], f'{c // 1024}K', f(a, 2), f(b, 2), f(d, 1 if d and d > 100 else 2),
                     pct1(d, a) if d and a and d / a < 3 else (f'×{d / a:.0f}' if d and a else '–')])
    s.append(table(prow, [40, 20, 22, 22, 26, 28]))
    s.append(P('Every Qwen2.5 size breaks with a 4-bit cache (the 1.5B model the most), while Llama and Gemma lose '
               '1.5–5%. All three Qwen sizes have only 2 KV heads and a bias on the K projection, which fits the '
               'key-outlier explanation in 10.9; it is a property of the family, not of small models. 8-bit costs at '
               'most 1.5% on any model.'))
    return s


def results_gen512():
    s = [P('10.13 Longer replies (set I)', st_h2)]
    if not has_tag('15W-gen512'):
        return s + [P('<i>Not run yet.</i>')]
    s.append(P('The same six models with 512 generated tokens instead of 128, at 4K and 16K, 3 paired rounds. A longer '
               'reply puts more of the request into decode, where the quantized cache is slow.'))
    rows = [['Model', 'f16 J, 128 tok', 'f16 J, 512 tok', '8-bit: 128 → 512 tok', '4-bit: 128 → 512 tok']]
    for m in SZ:
        rows.append([SZ_SPEC[m][0],
                     f'{mean("energy_per_request_net_j", "15W-size", m, "f16", 16384):.0f}',
                     f'{mean("energy_per_request_net_j", "15W-gen512", m, "f16", 16384):.0f}']
                    + [f'{chg("energy_per_request_net_j", "15W-size", m, kv, 16384, show_ci=False)} → '
                       f'{chg("energy_per_request_net_j", "15W-gen512", m, kv, 16384, show_ci=False)}'
                       for kv in ('q8_0', 'q4_0')])
    s.append(table(rows, [36, 26, 26, 42, 42]))
    s.append(P('16K prompt; net energy per request; changes against f16 with the same reply length. All 72 comparisons '
               'in this set (both contexts, three energy measures) are significantly more expensive than f16.',
               st_small))
    s += figure(PLOTS / '11_gen512.png', 'Figure 7. Penalty per request at 16K with 128 and with 512 generated tokens.')
    s.append(P('With 512-token replies the per-request penalty grows 1.7–2.2× on Qwen and Llama: Qwen-0.5B with an '
               '8-bit cache goes from +18% to +39%, Llama-1B from +16% to +29%, Llama-3B from +11% to +19%. It moves '
               'toward the per-token decode penalty, because decode becomes most of the request. Gemma stays at '
               '+3–4%. For chat-style use, where replies are long, the cost of a quantized cache is therefore '
               'closer to the decode-per-token numbers than to the 128-token request numbers in 10.1.'))
    return s


EV = ('Qwen2.5-0.5B-Instruct', 'Llama-3.2-1B-Instruct')
SHORT_EV = {EV[0]: 'Qwen2.5-0.5B', EV[1]: 'Llama-3.2-1B'}
EV_CFG = (('streamingllm@0.5', 'StreamingLLM 50%'), ('streamingllm@0.75', 'StreamingLLM 75%'),
          ('snapkv@0.5', 'SnapKV 50%'), ('snapkv@0.75', 'SnapKV 75%'))


def results_evict():
    s = [P('10.14 Eviction (set K)', st_h2)]
    if not has_tag('15W-evict'):
        return s + [P('<i>Not run yet.</i>')]
    s.append(P('The thesis methods on the Jetson: StreamingLLM (keeps the first 4 tokens and the most recent ones) and '
               'SnapKV (keeps the tokens the last 64 prompt tokens attend to most), through kvpress 0.5.3 on PyTorch '
               '2.15 for CUDA 13.2. Each removes 50% or 75% of the prompt cache '
               'after prefill. Models in fp16 (no weight quantization), same prompts, power meter and paired, '
               'shuffled rounds as the llama.cpp sets (5 rounds, 128 generated tokens, 15W). The baseline is the '
               'same model in the same engine without eviction. Llama-1B stops at 8K: in fp16 PyTorch a 16K prompt '
               'does not fit next to the model on the 8 GB board (the first attempt froze and rebooted it).'))
    rows = [['Model', 'Context', 'No eviction: J', 'tok/s'] + [c[1] for c in EV_CFG]]
    for m in EV:
        for c in (1024, 4096, 8192, 16384):
            e = mean('energy_per_request_net_j', '15W-evict', m, 'f16', c)
            if e != e:
                continue
            rows.append([SHORT_EV[m], f'{c // 1024}K', f'{e:.0f}',
                         f'{mean("decode_tps", "15W-evict", m, "f16", c):.1f}']
                        + [chg('energy_per_request_net_j', '15W-evict', m, kv, c) for kv, _ in EV_CFG])
    s.append(table(rows, [22, 14, 20, 12, 27, 27, 26, 26]))
    s.append(P('Net energy per request; changes against no eviction with 95% paired CIs.', st_small))
    s += figure(PLOTS / '12_eviction.png', 'Figure 8. Energy per request with eviction against the same model '
                'without eviction.')
    q = 'Qwen2.5-0.5B-Instruct'
    s.append(P(
        f'<b>Eviction saves energy only where it makes decode faster.</b> At 1K and 4K it changes energy per '
        f'request by a few percent at most. Decode in this engine runs at about '
        f'{mean("decode_tps", "15W-evict", q, "f16", 1024):.0f} tok/s at short contexts whatever the cache size, '
        'because each token costs hundreds of small kernel launches from Python, not memory traffic. Between 6K '
        'and 8K cached tokens decode speed drops in one step (table below; Qwen without eviction: '
        f'{mean("decode_tps", "15W-evict-scan", q, "f16", 6144):.1f} tok/s at 6K, '
        f'{mean("decode_tps", "15W-evict", q, "f16", 8192):.1f} at 8K, '
        f'{mean("decode_tps", "15W-evict", q, "f16", 16384):.1f} at 16K). A step rather than a slope points to '
        'PyTorch switching attention code path, not to memory traffic growing. Removing enough of the cache to stay '
        'below the step restores the short-context speed, and energy per request falls by about half: '
        f'{chg("energy_per_request_net_j", "15W-evict", q, "streamingllm@0.75", 16384, show_ci=False)} with '
        f'StreamingLLM and {chg("energy_per_request_net_j", "15W-evict", q, "snapkv@0.75", 16384, show_ci=False)} '
        'with SnapKV for Qwen at 16K. Removing only 50% at 16K keeps 8K tokens and saves less '
        f'({chg("energy_per_request_net_j", "15W-evict", q, "streamingllm@0.5", 16384, show_ci=False)}).'))
    if has_tag('15W-evict-scan'):
        cs = (1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384)
        rows = [['No eviction', *[f'{c // 1024}K' for c in cs]]]
        for m in EV:
            v = [mean('decode_tps', '15W-evict-scan', m, 'f16', c) for c in cs]
            v = [x if x == x else mean('decode_tps', '15W-evict', m, 'f16', c) for x, c in zip(v, cs)]
            rows.append([SHORT_EV[m]] + [f(x) for x in v])
        s.append(table(rows, [30] + [16] * len(cs)))
        s.append(P('Decode speed (tok/s) of the PyTorch baseline against prompt length; 2K–6K and 12K from a short '
                   'extra scan (3 rounds, tag 15W-evict-scan), the rest from the main set.', st_small))
    s.append(P(
        '<b>StreamingLLM and SnapKV cost the same energy.</b> SnapKV scores tokens during prefill, which adds '
        'energy at 1K (prefill +34%), but prefill is a small part of the request and the two methods end within '
        'about two points of each other at every length.'))
    s.append(P(
        '<b>Caveats.</b> (1) The size of the saving is a property of PyTorch decode, not of memory traffic alone: '
        'Qwen at 16K with 50% removed keeps 8K tokens but decodes at '
        f'{mean("decode_tps", "15W-evict", q, "streamingllm@0.5", 16384):.1f} tok/s, slower than an 8K prompt '
        'without eviction; we have not traced why. (2) The PyTorch baseline is much more expensive than llama.cpp: '
        f'Qwen-0.5B at 16K uses {mean("energy_per_request_net_j", "15W-evict", q, "f16", 16384):.0f} J per request '
        f'here and {mean("energy_per_request_net_j", "15W-size", "qwen2.5-0.5b-instruct-q4_k_m", "f16", 16384):.0f} J '
        'in llama.cpp (Q4_K_M weights, f16 cache, set H). Even with 75% removed, PyTorch uses about twice the '
        'energy of llama.cpp without eviction. (3) Accuracy with eviction is not measured here; the thesis measured '
        'it on the T4.'))
    s.append(P('<b>Details per method.</b> Decode speed and the change in decode energy per generated token against '
               'no eviction, with the size of the cache that decode works on.', st_body))
    rows = [['Model, context', 'Cache, MiB (none / 50% / 75%)', 'tok/s, none'] + [c[1] for c in EV_CFG]]
    for m in EV:
        for c in (1024, 4096, 8192, 16384):
            if mean('decode_tps', '15W-evict', m, 'f16', c) != mean('decode_tps', '15W-evict', m, 'f16', c):
                continue
            kv = ' / '.join(f'{mean("kv_used_mib", "15W-evict", m, k, c):.0f}'
                            for k in ('f16', 'streamingllm@0.5', 'streamingllm@0.75'))
            rows.append([f'{SHORT_EV[m]}, {c // 1024}K', kv, f'{mean("decode_tps", "15W-evict", m, "f16", c):.1f}']
                        + [f'{mean("decode_tps", "15W-evict", m, k, c):.1f} tok/s, '
                           f'{chg("decode_j_per_token", "15W-evict", m, k, c, show_ci=False)}' for k, _ in EV_CFG])
    s.append(table(rows, [26, 30, 14, 26, 26, 26, 26]))
    s.append(P('Each method cell: decode speed, then the paired change in net decode J/token against no eviction. The '
               'cache sizes are the same for both methods.', st_small))
    s.append(P(
        'Per generated token the saving at 16K with 75% removed is about three quarters (Qwen −76%); at 1K–4K it '
        'is −7% to +2%. Decode power also changes: Qwen without eviction draws '
        f'{mean("decode_avg_w", "15W-evict", q, "f16", 8192):.1f} W in decode at 8K and '
        f'{mean("decode_avg_w", "15W-evict", q, "f16", 16384):.1f} W at 16K, against 6.9–7.0 W with eviction, so on '
        'Qwen the slow path costs both time and power; on Llama decode power is about the same (9.9 W without, '
        '10.5 W with eviction at 8K). Prefill energy rises with SnapKV at short prompts (+34–36% at 1K, +9% at 4K: '
        'the window attention is extra work) and falls by 6–10% at 8K and 16K for both methods.'))
    return s


MIXM = ('Llama-3.2-1B-Instruct-Q8_0', 'qwen2.5-0.5b-instruct-q8_0', 'qwen2.5-1.5b-instruct-q4_k_m')
MIXN = {MIXM[0]: 'Llama-1B', MIXM[1]: 'Qwen-0.5B', MIXM[2]: 'Qwen-1.5B (Q4_K_M)'}
MIXKV = (('q8_0', '8-bit K+V'), ('q4_0', '4-bit K+V'), ('q8_0/q4_0', '8-bit K, 4-bit V'), ('f16/q4_0', 'f16 K, 4-bit V'))


def ppl4(model, k, v, c=8192):
    """Perplexity row (ppl, err) from any ppl CSV."""
    for r in S.get('ppl', []):
        if r.get('model') == model and r.get('type_k') == k and r.get('type_v') == v and int(r.get('ctx', 0)) == c \
                and r.get('ppl'):
            return float(r['ppl']), float(r.get('ppl_err') or 'nan')
    return None


def results_mixkv():
    tag = '15W-mixkv'
    s = [P('10.15 Mixed K/V formats (set L)', st_h2)]
    if not has_tag(tag):
        return s + [P('<i>Not run yet.</i>')]
    s.append(P('Set C and J showed that keys, not values, break with 4 bits: 8-bit or f16 keys with 4-bit values keep '
               'perplexity within 1–2% of f16. Set L measures what these mixed formats cost in energy. A mixed pair '
               'needs the second build (GGML_CUDA_FA_ALL_QUANTS=ON), so f16, q8_0 and q4_0 were run again in that build '
               'and every format is compared with f16 from the same binary, round and server cycle. Llama-1B and '
               'Qwen-0.5B (Q8_0 weights, as in set A) and Qwen2.5-1.5B (Q4_K_M, the model whose 4-bit keys fail '
               'worst), 1K–16K, 128 generated tokens, 5 rounds, 15W, --cache-ram 0.'))
    rows = [['Model, context', 'f16: J, tok/s'] + [lab for _, lab in MIXKV]]
    for m in MIXM:
        for c in CTX:
            rows.append([f'{MIXN[m]}, {c // 1024}K', f'{mean("energy_per_request_net_j", tag, m, "f16", c):.1f}, '
                         f'{mean("decode_tps", tag, m, "f16", c):.1f}']
                        + [chg('energy_per_request_net_j', tag, m, kv, c) for kv, _ in MIXKV])
    s.append(table(rows, [32, 22, 30, 30, 30, 30]))
    s.append(P('Net energy per request against f16 in the same build, paired by round, 95% CI.', st_small))
    rows = [['At 16K', 'KV cache, MiB'] + [lab for _, lab in MIXKV]]
    for m in MIXM:
        rows.append([MIXN[m], ' / '.join(f'{mean("kv_used_mib", tag, m, kv, 16384):.0f}'
                                         for kv in ('f16', 'q8_0', 'q4_0', 'q8_0/q4_0', 'f16/q4_0'))]
                    + [chg('decode_j_per_token', tag, m, kv, 16384, show_ci=False) for kv, _ in MIXKV])
    s.append(table(rows, [26, 48, 25, 25, 25, 25]))
    s.append(P('Decode energy per generated token at 16K against f16. Cache sizes in the order f16 / 8-bit / 4-bit / '
               '8-bit K + 4-bit V / f16 K + 4-bit V.', st_small))
    s += figure(PLOTS / '13_mixkv.png', 'Figure 9. Mixed K/V formats: energy per request against an f16 cache, one '
                'llama.cpp build.')
    ks = S.get('kernels', {})
    kk = {kv: ks.get(f'Llama-3.2-1B-Instruct-Q8_0_{kv}_allq', {}).get('decode_attn_us_per_call')
          for kv in ('f16-f16', 'q8_0-q8_0', 'q8_0-q4_0', 'f16-q4_0')}
    if all(kk.values()):
        s.append(P(
            f'<b>Kernel check (nsys, Llama-1B, 16K, this build).</b> Decode attention takes {kk["f16-f16"]:.0f} µs per '
            f'call with an f16 cache (tensor-core mma kernel) and {kk["f16-q4_0"]:.0f} µs with f16 K + 4-bit V, '
            f'{kk["q8_0-q8_0"]:.0f} µs with 8-bit K and V and {kk["q8_0-q4_0"]:.0f} µs with 8-bit K + 4-bit V, all '
            'on the vec kernel (1.9–2.3×). Any quantized half sends decode to the vec kernel; f16 keys only make it '
            'somewhat cheaper because the key half needs no unpacking.'))
    q15 = 'qwen2.5-1.5b-instruct-q4_k_m'
    acc = [(k, v, ppl4(q15, k, v)) for k, v in (('f16', 'f16'), ('q8_0', 'q8_0'), ('q4_0', 'q4_0'),
                                                 ('q8_0', 'q4_0'), ('f16', 'q4_0'))]
    acc_txt = ', '.join(f'{k}/{v} {p[0]:.2f}' for k, v, p in acc if p)
    s.append(P(
        '<b>Reading.</b> The mixed formats do not escape the penalty. 8-bit K + 4-bit V costs about what 8-bit K and V '
        f'cost ({chg("energy_per_request_net_j", tag, MIXM[0], "q8_0/q4_0", 16384, show_ci=False)} per request at 16K '
        f'on Llama, {chg("energy_per_request_net_j", tag, MIXM[1], "q8_0/q4_0", 16384, show_ci=False)} on Qwen-0.5B); '
        'f16 K + 4-bit V is the cheapest quantized format everywhere but still significantly more expensive than '
        f'f16 ({chg("energy_per_request_net_j", tag, MIXM[0], "f16/q4_0", 16384, show_ci=False)}, '
        f'{chg("energy_per_request_net_j", tag, MIXM[1], "f16/q4_0", 16384, show_ci=False)} and '
        f'{chg("energy_per_request_net_j", tag, MIXM[2], "f16/q4_0", 16384, show_ci=False)}). It saves the least '
        'memory of the mixed formats (64% of f16). So the accuracy-safe choice on this stack is a trade of memory for '
        'energy, not a free lunch.'
        + (f' WikiText-2 perplexity of Qwen-1.5B at 8K (set Q): {acc_txt}.' if acc_txt else '')))
    return s


def results_qa():
    s = [P('10.20 Questions and answers', st_h2)]
    s.append(P('Questions raised while reviewing the results, answered from the measurements above.'))
    qa = [
        ('Why is the penalty per generated token (+29% to +58% at 16K) so much larger than per request (+13–18%)?',
         'A request has two phases. Prefill (reading the prompt) is almost unaffected by the cache format; decode '
         '(writing the reply) is where the slower kernel runs. Per token, a speed drop of x% makes each token take '
         '1 / (1 − x) as long: −20% speed is +25% time, −36% is +56%. Decode power stays about the same, so energy per '
         'token rises by about the same amount, which is the +29% to +58% in 10.2. The per-request number also '
         'contains prefill, which at 16K is a large part of the request, and that dilutes it. With 512-token '
         'replies decode is most of the request, so the per-request penalty roughly doubles (10.13).'),
        ('Why can a quantized cache not use the tensor cores? What are mma and vec?',
         'Tensor cores multiply small blocks of numbers in fixed formats such as fp16. q8_0 and q4_0 are llama.cpp\'s '
         'own packed formats (32 values + one scale per block), which tensor cores cannot read directly: each block '
         'must first be unpacked to fp16. "mma" (matrix multiply-accumulate) is llama.cpp\'s tensor-core '
         'flash-attention kernel, used for an f16 cache; it handles all query heads that share a KV head together and '
         'is limited by memory reads (88% of peak bandwidth, compute 10% busy). "vec" is its vector kernel on '
         'ordinary CUDA cores, which unpacks as it reads, one query row at a time; it is limited by that work (compute '
         '64–74% busy) and repeats it for every query head of a group. For prompt reading, where many tokens are '
         'processed together, llama.cpp converts the cache to an f16 copy first and keeps the tensor cores (the '
         'conversion kernels in 10.5), so prefill barely changes. That decode cannot do the same is a software '
         'limit, not a hardware one: a kernel that unpacks blocks into on-chip memory and then feeds the tensor cores '
         'is possible (server work such as QServe and BitDecoding builds such kernels), but llama.cpp has none for '
         'single-token decode on this GPU.'),
        ('Why are 8-bit and 4-bit about equally expensive?',
         'Both use the vec kernel, and its time is set by unpacking and per-head repetition rather than by the bytes '
         'read. 4-bit reads half the bytes of 8-bit but unpacks a denser format, so the two end within a few points '
         'of each other; 8-bit is often slightly worse.'),
        ('What is flash attention, and why test f16 with it off?',
         'Attention compares the new token with every cached token, turns the scores into weights (softmax) and '
         'mixes the values. Without flash attention these are separate steps that write the full score table to '
         'memory and read it back. Flash attention does all of it in one kernel on small on-chip tiles. llama.cpp '
         'needs flash attention for a quantized V cache, so v1 also ran f16 with it on; the check in 10.3 shows f16 '
         'without it is 3.3–3.7× more expensive at 16K, so the baseline was the strong one.'),
        ('What are sessions ("users")?',
         'Separate conversations that one server handles at the same time (llama-server slots, -np N). Set D sent N '
         'requests at once, each with its own prompt and its own KV cache, so cache memory grows with N. 8 sessions × '
         '16K on Llama-1B need 4.1 GiB of f16 cache, which does not fit next to the model in 8 GB; 8-bit and 4-bit '
         'do.'),
        ('"Does not fit": the model fits, but not the cache?',
         'Yes. The 8 GB is shared by the CPU, the GPU, Linux, the model weights and the cache. Llama-3B\'s weights '
         '(2 GB) fit, but its f16 cache needs 3.5 GiB at 32K and 7.0 GiB at 64K, and the server then fails to '
         'allocate it. With 8-bit the 32K cache fits, with 4-bit even 64K (10.10).'),
        ('Why does PyTorch decode stay at the same speed up to 6K tokens and then drop in one step?',
         'Below about 6K, each generated token costs the same fixed overhead: Python launches several hundred small '
         'GPU operations per token, and that, not reading the cache, sets the speed (about 14 tok/s for Qwen, 17 for '
         'Llama). Between 6K and 8K the speed falls in one step (Qwen 13.6 → 8.5 tok/s, Llama 16.1 → 6.0) and then '
         'keeps falling. A step rather than a slope suggests PyTorch chooses a different, slower attention code path '
         'above some size; the cause has not been traced. For eviction this means the saving depends on whether the '
         'cache that is left is below the step: 16K with 75% removed leaves about 4K tokens and saves 47%; 16K with '
         '50% removed leaves 8K, still above the step, and saves 27%; at 1K–4K the cache is already below it and '
         'eviction changes almost nothing. Most of the saving is avoiding PyTorch\'s slow path, not reading less '
         'memory, which is why these numbers are not comparable with llama.cpp, where speed falls gradually.'),
        ('Why not run eviction in llama.cpp, or quantization in PyTorch?',
         'llama.cpp has no eviction methods; its "context shift" only drops the oldest tokens when the context is '
         'full. SnapKV needs attention scores and per-head token selection, which its cache layout does not support, '
         'so adding it would be new C++/CUDA work. PyTorch can quantize the cache (transformers QuantizedCache with '
         'the quanto or HQQ backend). It failed on the T4 in the thesis because quanto\'s fast kernel needs sm_80 or '
         'newer; the Orin is sm_87, so it may work here. It is a different implementation (recent tokens kept in full '
         'precision, unpacking in Python every step) that few edge users run, but it would compare quantization and '
         'eviction in one engine (section 13).'),
    ]
    for q, a in qa:
        s.append(P(f'<b>{q}</b> {a}'))
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
        'the 7W grid the clock was at the 7W maximum (408 MHz) after every run, at 45–47 °C. Sets H and I: 624.75 MHz '
        'after every run, 44–53 °C. No throttling.',
        '<b>Crash during set H (31 Oct).</b> llama-server keeps up to 8 GiB of earlier prompts\' KV caches in RAM by '
        'default (--cache-ram 8192). With Llama-3B and a 16K f16 cache (1.75 GiB per prompt) this filled the 8 GB '
        'board: the kernel killed the server at 21:34 (3 failed requests) and the board froze and rebooted at 21:52. '
        'Requests use cache_prompt=false, so the store never changed a measurement (the logs show all prompt tokens '
        'evaluated every time); it only used memory. Llama-3B was rerun from round 0 and Gemma run with '
        '--cache-ram 0 on 1 Nov; its partial rows from 31 Oct are not analysed. The four models that had finished '
        'are kept.',
        '<b>Crash during set K (2 Nov).</b> Llama-1B in fp16 PyTorch with a 16K prompt used up the 8 GB of shared '
        'memory (at 8K, 1.2 GB was still free); with no swap the board froze for about 30 minutes and rebooted at '
        '06:04. Qwen-0.5B had finished all 100 runs and is kept. Llama-1B was rerun from round 0 at 1K–8K; the 6 '
        'rows from before the crash are not analysed. Set K ran at 624.75 MHz after every run, 44–51 °C.',
        '<b>Gemma cache size.</b> bench_kv.py sizes the cache from the model header as if every layer kept the full '
        'context; Gemma-3 has 22 sliding-window layers. The analysis corrects it to what llama-server allocates '
        '(65 MiB full-context + 22 MiB window at 16K f16, from the server log).',
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
        'Small models (0.5B–3B, six models from three families) and greedy decoding with fixed replies of 128 '
        'tokens (512 in set I, where the penalty per request grows 1.7–2.2×). Larger models (7B+) do not fit '
        'next to a long f16 cache on this 8 GB board.',
        'Energy is measured at the module input (VDD_IN). It includes the fan and regulators but not the carrier '
        'board or power adapter.',
        'Eviction (set K) runs in a different engine (PyTorch), so its savings are compared against PyTorch without '
        'eviction, not against llama.cpp; its accuracy is not measured on the Jetson.',
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
        'Accuracy of eviction on the Jetson (a long-context QA task), so that eviction and quantization can be '
        'compared on energy and accuracy together.',
        'Energy of 8-bit K + 4-bit V, the accuracy-safe mix (10.9).',
        'Comparison with the thesis T4 data as the server-side reference point.',
        'Optional: a quantized cache in PyTorch (transformers QuantizedCache, quanto backend) so that quantization '
        'and eviction are compared in the same engine (10.15).',
        'Optional: profile PyTorch decode between 6K and 8K tokens to find which attention path causes the step '
        '(10.14).',
        'Push the v3 code and data (sets H–K) to the repository (Appendix E).',
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
                        '15W-fit1b': 'set G, Llama-1B', '15W-fit3b': 'set G, Llama-3B',
                        '15W-size': 'set H, size and family (Llama-3B rerun, Gemma)',
                        '15W-gen512': 'set I, 512-token replies', '15W-evict': 'set K, eviction (Llama-1B rerun)',
                        '15W-evict-scan': 'set K, decode-speed scan without eviction', '15W-mixkv': 'set L, mixed K/V',
                        '15W-pytq': 'set N, PyTorch quantized cache', '15W-t4match': 'set P, T4 settings'
                        }.get(meta.get('tag'), 'set B')
                     + ' (+ .meta.json: settings, idle power, versions)'])
    rows += [['results/evict_20261102_045556.jsonl', '15W-evict: set K, Qwen-0.5B (100 runs); ends in the 2 Nov '
              'crash, so it has no .meta.json; the 6 Llama-1B rows before the crash are kept only in the '
              '.partial-bak copy and not analysed'],
             ['results/kv_20261031_201343.jsonl', '15W-size: set H, Qwen 0.5B/1.5B/3B and Llama-1B; ends in the 31 Oct '
              'crash, so it has no .meta.json; its partial Llama-3B rows are superseded by the rerun and not analysed'],
             ['results/ppl_*.csv', 'perplexity (v1 at 2K; v2 at 8K and the K/V split, the 16K attempts are the rows '
              'without a value; v3: the six set-H models and the Llama-1B K/V split at 8K)'],
             ['results/prof/', 'nsys kernel summaries (*_cuda_gpu_kern_sum.csv; *_allq = set L) and ncu counters '
              '(*_ncu.csv)'],
             ['results/needle_llama_*.jsonl, needle_hf_*.jsonl', 'set M: one row per prompt (engine, model, config, '
              'context, needle depth, code, answer, correct). needle_hf_20261111_205314.aborted is the stopped first '
              'attempt (not analysed)'],
             ['results/profstep_*.jsonl', 'set O: per cache length, ms per step and the top GPU kernels and CPU ops'],
             ['results/ppl_v4_*.csv', 'set Q: perplexity of the mixed formats on Qwen2.5-1.5B'],
             ['plots_v2/stats_v4.json', 'set M accuracy with Wilson CIs, set O profile, the T4 cells used in 10.19'],
             ['results/logs/', 'llama-server log of every server start'],
             ['plots_v2/stats.json', 'every number in this report, with CIs'],
             ['analysis/analyze_v2.py, analyze_v4.py, make_report.py, report_content.py',
              'statistics, figures and this PDF']]
    s.append(table(rows, [70, 104]))
    return s


def appendix_v1_rounds():
    s = [P('Appendix C. v1 results', st_h1)]
    s.append(P('The first runs (19 October): both models with Q8_0 weights, f16 / q8_0 / q4_0 with flash attention '
               'on, 1K–16K, 3 paired rounds, at 15W and then 7W; the desktop Software Updater was still active (one CPU '
               'core busy). Changes against f16, paired by round, 95% CI. v2 repeated this with 5 rounds and a quiet '
               'board (10.1, 10.2) and agrees within a few points.'))
    for metric, label in (('energy_per_request_net_j', 'energy per request, net'),
                          ('decode_j_per_token', 'decode energy per token, net')):
        for tag in ('15W', '7W'):
            s.append(P(f'v1, {tag}, {label}', st_small))
            s.append(change_table(metric, tag))
    s.append(P('v1 per-round values: Qwen-0.5B, 7W, 16K', st_h2))
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
            ['Q8_0, Q4_K_M', 'Weight formats: 8-bit; 4-bit with per-block scales (K-quant, "medium").'],
            ['Token', 'The unit a model reads and writes; about three quarters of an English word.'],
            ['Context length', 'Number of prompt tokens in the cache (1K = 1,024 tokens).'],
            ['q8_0 / q4_0 / q4_1', 'llama.cpp cache formats: blocks of 32 values in 8 or 4 bits plus a scale (q4_1: '
                                   'and a minimum). See section 7.1.'],
            ['Dequantize', 'Unpack stored 8- or 4-bit values back to fp16 before computing with them.'],
            ['GQA (query heads per KV head)', 'Several query heads share one key/value head (Llama 4, Qwen 7, '
                                              'Gemma 4), which shrinks the cache.'],
            ['Sliding-window layer', 'A layer that only attends to the last W tokens (Gemma-3: 512) and keeps only '
                                     'those in its cache.'],
            ['Eviction', 'Deleting tokens from the cache instead of storing them with fewer bits (section 7.2).'],
            ['StreamingLLM', 'Eviction by position: keep the first 4 tokens (attention sinks) and the most recent.'],
            ['SnapKV', 'Eviction by importance: keep the tokens the last 64 prompt tokens attend to most.'],
            ['kvpress', 'NVIDIA\'s PyTorch library of eviction methods ("presses"), used in the thesis and set K.'],
            ['Compression ratio', 'Share of the prompt\'s cache removed by eviction (0.5 = 50%).'],
            ['Session (user)', 'One conversation served in parallel with others, each with its own cache.'],
            ['Launch-bound', 'Speed limited by the overhead of starting many small GPU operations, not by the work.'],
            ['SDPA', 'PyTorch\'s scaled-dot-product attention, which picks an attention kernel per call.'],
            ['VDD_IN / VDD_CPU_GPU_CV / VDD_SOC', 'The three power rails the board\'s INA3221 sensor measures.'],
            ['15W / 7W', 'The board\'s two power modes (nvpmodel 0 and 1).'],
            ['Needle in a haystack', 'A fact hidden in a long text, asked about at the end; tests whether the cache '
                                     'kept it. Here with 3 distractor facts (RULER\'s multi-key variant).'],
            ['Question inside / after', 'Whether eviction runs on a prompt that already contains the question, or '
                                        'on the document alone with the question added later.'],
            ['Wilson interval', 'A 95% confidence interval for a share of correct answers that stays within 0–100%.'],
            ['QuantizedCache, quanto, HQQ', 'transformers\' KIVI-style quantized KV cache and its two quantization '
                                            'libraries (section 7.3).'],
            ['Residual (KIVI)', 'The newest tokens a quantized cache keeps in fp16 (here 128).'],
            ['Mixed K/V format', 'Keys and values stored in different formats, e.g. 8-bit K + 4-bit V.']]
    s.append(table(rows, [36, 138]))
    return s


def appendix_produced():
    s = [P('Appendix E. Everything produced so far', st_h1)]
    rows = [['When', 'What', 'Where'],
            ['19 Oct', 'v1 benchmark (llama.cpp on the Jetson), 15W and 7W, perplexity at 2K; v1 report (17 pages) '
                       'and v1 results deck (21 slides)', 'Jetson_KV_Report.pdf; deck shared earlier'],
            ['21–27 Oct', 'Prof. Lin\'s review; v2 code; sets A–G at 15W and A–B at 7W; v2 report', 'this report, '
             'sections 0, 10.1–10.11'],
            ['27 Oct', 'v2 summary deck (7 slides, later extended to 9 with the model and eviction slides)',
             'Claude artifact (private until shared)'],
            ['31 Oct – 2 Nov', 'v3: sets H (six models), I (512-token replies), J (perplexity, Llama K/V split) and K '
                               '(eviction in PyTorch, with a decode-speed scan)', 'sections 10.12–10.14'],
            ['5 Nov', 'All-results deck: 10 slides, with speaker notes that explain every term',
             'Claude artifact (private until shared)'],
            ['10 Nov', 'This report extended to cover everything (v3), with the questions and answers in 10.15',
             'Jetson_KV_Report_v2.pdf'],
            ['ongoing', 'Results summary in Markdown and code', 'RESULTS.md, README.md, jetson/, analysis/']]
    s.append(table(rows, [24, 104, 46]))
    s.append(P('Repository status at the time of writing: github.com/koushikvarmaJS/thesis2 holds v1 and v2 (last '
               'commit 27 Oct). The v3 code (run_v3.sh, ppl_v3.sh, setup_evict.sh, bench_evict.py), the result files '
               'of sets H–K and this version of the report are in the local copy and have not been pushed yet.',
               st_small))
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
    s += results_size()
    s += results_gen512()
    s += results_evict()
    s += results_mixkv()
    s += results_qa()
    s += data_quality()
    s += scope()
    s += next_steps()
    s.append(PageBreak())
    s += appendix_repro()
    s += appendix_files()
    s += appendix_v1_rounds()
    s += appendix_glossary()
    s += appendix_produced()
    return s
