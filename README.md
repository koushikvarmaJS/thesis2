# thesis2: KV-cache energy on Jetson Orin Nano

First edge experiment for the paper *KV-Cache Compression on the Edge*.
It measures how much energy the Jetson Orin Nano Developer Kit uses to generate text with three KV-cache formats in llama.cpp:

| KV cache | What it is | KV memory |
|---|---|---|
| `f16`  | baseline, no compression | 100% |
| `q8_0` | 8-bit quantized KV cache | ~53% |
| `q4_0` | 4-bit quantized KV cache | ~28% |

This is the quantization experiment the Tesla T4 couldn't run (KIVI needed Ampere). The Orin Nano is Ampere (sm_87).
Eviction methods (StreamingLLM/SnapKV via kvpress) need PyTorch on the Jetson and come next.

**What's measured for each model × KV type × context length (1K, 4K, 8K, 16K):**
- Energy from the board's built-in INA3221 power monitor (`VDD_IN` = whole module), with idle power subtracted, split into prefill and decode
- Decode energy per token (J/token), the headline metric
- Prefill and decode speed (tokens/s), average power, KV cache size, GPU temperature and clock
- Accuracy: WikiText-2 perplexity for each KV type

**Fixes for the problems found in the thesis data:**
- `ignore_eos` makes every run generate exactly the same number of tokens.
- `cache_prompt=false` makes the prompt really get processed every time.
- Methods and context lengths run in shuffled order each round, so baseline doesn't always go first.
- Perplexity is real here: llama.cpp reads K/V back from the quantized cache.

```
thesis2/
├── README.md
├── jetson/            # runs ON the Jetson
│   ├── check_env.sh   # what board / JetPack / power mode / sensors do we have?
│   ├── setup.sh       # build llama.cpp (CUDA) + download models + text
│   ├── power.py       # INA3221 power sampler (also: live sanity check)
│   ├── bench_kv.py    # the energy benchmark
│   └── ppl_kv.sh      # accuracy (perplexity) per KV type
├── analysis/
│   └── plot.py        # runs on the Mac: results -> plots/*.png + plots/summary.md
└── results/           # JSONL/CSV outputs land here
```

---

## Step 0: Make sure the Jetson boots (skip if it already shows an Ubuntu desktop)

1. Flash a microSD card (64 GB+) with the **JetPack 6.x SD card image** from NVIDIA's Jetson Orin Nano "Getting Started" page, using Balena Etcher.
2. Insert it, connect a DisplayPort monitor, keyboard, mouse and Ethernet or Wi-Fi, then power on. Finish the Ubuntu first-boot setup (username and password).
3. **Gotcha:** a brand-new board may have old firmware that won't boot JetPack 6 (black screen or UEFI shell). If that happens, boot the JetPack 5.1.3 image once, let it update the firmware, reboot, then use the JetPack 6 card. NVIDIA's Getting Started page covers this.

## Step 1: Connect from the Mac

Use either option.

**A. USB-C cable (simplest, no network needed).** Plug the Jetson's USB-C port into the Mac. After the Jetson boots, it appears at `192.168.55.1`:
```bash
ssh <your-jetson-user>@192.168.55.1
```

**B. Same Wi-Fi as the Mac.** On the Jetson:
```bash
nmcli device wifi connect "<SSID>" password "<password>"   # if not already connected
hostname -I                                               # note the IP, e.g. 10.0.0.42
```
Then on the Mac: `ssh <user>@<that IP>`.

**Optional: passwordless login (lets Claude run everything for you):**
```bash
ssh-keygen -t ed25519            # on the Mac, if you don't have a key yet (press Enter at prompts)
ssh-copy-id <user>@<jetson-ip>
```

## Step 2: Copy this folder to the Jetson

On the Mac:
```bash
cd ~/project/school/Thesis
rsync -av --exclude results thesis2 <user>@<jetson-ip>:~/
```

## Step 3: Check the board (1 min)

On the Jetson:
```bash
cd ~/thesis2
bash jetson/check_env.sh
```
Look for:
- **L4T R36.x** (JetPack 6)
- **nvcc** found
- **power sensors** listed (e.g. `VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`) with sensible watts (a few W at idle)

If `nvcc` is missing: `sudo apt install nvidia-jetpack`.

## Step 4: Setup (30–60 min, mostly compiling, one time only)

Use `tmux` so a dropped SSH connection doesn't kill it (`sudo apt install tmux`; reattach with `tmux attach`):
```bash
tmux new -s setup
bash jetson/setup.sh
```
This builds llama.cpp with CUDA and downloads Qwen2.5-0.5B and Llama-3.2-1B (Q8_0 GGUF, ~1.9 GB together), the long-context prompt text, and WikiText-2.
For a bigger model add `WITH_3B=1` (Llama-3.2-3B Q4_K_M), then pass it with `--models`.

## Step 5: Fix the power mode and clocks

Benchmarks must run at a known, fixed setting:
```bash
sudo nvpmodel -q                 # current mode
grep "POWER_MODEL ID" /etc/nvpmodel.conf   # available modes (IDs differ per JetPack/Super)
sudo nvpmodel -m <ID>            # e.g. the highest mode (MAXN / 25W). May ask to reboot.
sudo jetson_clocks               # lock clocks at max for this mode (stable numbers)
```
Also set the fan to max if you can (`sudo jetson_clocks --fan`) to avoid thermal throttling.

## Step 6: Sanity-check the power meter (10 s)

```bash
python3 jetson/power.py --watch 10
```
You should see `VDD_IN` at a few watts idle and a sample rate around 15–20 Hz.

## Step 7: Run the benchmark

```bash
python3 jetson/bench_kv.py --quick --tag test        # ~5 min, checks everything works
python3 jetson/bench_kv.py --tag 25W                 # full: 2 models x 3 KV x 4 ctx x 3 rounds (~20-30 min)
bash jetson/ppl_kv.sh                                # accuracy, ~5-10 min
```
Each run prints decode speed, J/token, power and GPU temperature as it goes. Results go to `results/kv_<timestamp>.jsonl` (+ `.meta.json`) and `results/ppl_<timestamp>.csv`.

**Optional power-mode comparison (RQ3):** switch to a lower mode (Step 5), then run again with a new tag, e.g. `--tag 15W`. The plots show each tag as its own panel.

Useful flags: `--contexts 1024,4096,8192,16384,32768`, `--gen 256`, `--rounds 5`, `--models models/x.gguf`.

## Step 8: Copy results back and plot (on the Mac)

```bash
cd ~/project/school/Thesis/thesis2
rsync -av <user>@<jetson-ip>:~/thesis2/results/ results/
pip3 install matplotlib
python3 analysis/plot.py results/kv_*.jsonl --ppl results/ppl_*.csv --out plots
open plots
```
**Output:**
- `01_decode_j_per_token.png`: energy per generated token vs context, per KV type (the headline)
- `02_saving_vs_f16.png`: % energy saved by q8_0 / q4_0 vs f16
- `03_decode_tps.png`, `04_prefill_tps.png`: speed
- `05_avg_power.png`: average module power
- `06_kv_cache_mib.png`: KV-cache memory per KV type
- `summary.md`: one table with every number, including perplexity

---

## What to say when presenting

1. **Setup:** Jetson Orin Nano 8 GB, JetPack 6, llama.cpp with CUDA, power from the onboard INA3221 (`VDD_IN`, whole module), idle subtracted.
2. **Method:** equal output length for every run, shuffled order, 3 repeats, mean ± std.
3. **Result:** read off plot 02. How much does a 4-bit or 8-bit KV cache save per token, and does the saving grow with context? Expected: yes, because decode re-reads the whole KV cache for every token and the Orin Nano's memory bandwidth is limited. Check what the data actually shows before claiming it.
4. **Trade-off:** perplexity column in `summary.md`. What accuracy does 4-bit cost?
5. **Next:** add eviction methods (kvpress), a power-mode sweep, larger models, and a server-GPU comparison.

**Caveats to state honestly:**
- `VDD_IN` is module power, not GPU-only like NVML, so the numbers aren't directly comparable with the T4 thesis data.
- The prefill/decode split is approximate (it uses llama.cpp's timing).
- These are 0.5–1B models, where the KV cache is small relative to the weights. Savings should grow with longer contexts and bigger models.

## Troubleshooting

| Problem | Fix |
|---|---|
| `nvcc: command not found` | `sudo apt install nvidia-jetpack`, then `export PATH=/usr/local/cuda/bin:$PATH` |
| Compile killed / board freezes | Out of memory: change `-j4` to `-j2` in `setup.sh`, or add swap: `sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| `rail VDD_IN not found` | Use a rail `power.py` lists, e.g. `--rail VDD_CPU_GPU_CV` |
| `server failed` | Read the log in `results/logs/`. Usually an unknown flag in an older or newer llama.cpp. Paste it to Claude. |
| Very slow tokens/s | Check `nvpmodel -q` and run `sudo jetson_clocks`. Check the log says layers were offloaded to CUDA. |
| Can't reach `192.168.55.1` | Use a USB-C data cable (not charge-only) and wait about a minute after boot. Otherwise use Wi-Fi (option B). |
