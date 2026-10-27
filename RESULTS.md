# v2 results (2026-10-26/27): response to Prof. Lin's review

Full write-up: `Jetson_KV_Report_v2.pdf` · plots: `plots_v2/` · raw data: `results/kv_2026102*.jsonl`, `results/pptg_*.jsonl`, `results/ppl_v2_*.csv`, `results/prof/`

**Proposed main question:** On a memory-limited edge device, does compressing the KV cache reduce the energy of LLM inference, and if not, what does it buy?

**Proposed falsifiable claim:** On a Jetson Orin Nano running llama.cpp, an 8-bit or 4-bit KV cache uses **more** energy per request than an f16 cache at every context length from 1K to 16K tokens, and the gap grows with context. The cause is compute, not memory traffic. The claim is refuted by any paired cell where a quantized cache is significantly cheaper than f16, or by f16 fitting everything the quantized caches fit.

What changed since v1:
- 5 paired rounds instead of 3
- net *and* gross energy
- per-rail power
- FA-off baseline
- llama-bench pp/tg split
- concurrency at N = 1/2/4/8
- Q4_K_M weights
- nsys + ncu kernel profiling
- K/V split perplexity
- a fit test with Llama-3B

| Finding (15W, 16K context) | Llama-1B q8_0 / q4_0 | Qwen-0.5B q8_0 / q4_0 |
|---|---|---|
| Net energy per request vs f16 | +15.9% / +12.6% | +18.0% / +16.0% |
| Decode J/token vs f16 | +35.9% / +28.6% | +58.0% / +50.4% |
| FA off (f16) vs FA on | +228% | +270% |

- **Mechanism (nsys/ncu):** f16 decode attention runs on a tensor-core kernel that is limited by memory bandwidth (ncu: ~88% of memory bandwidth, ~10% SM). A quantized cache switches to the `flash_attn_ext_vec` kernel instead. That kernel has no tensor cores, keeps the SM 64–74% busy, and re-reads the cache once per query head (GQA). Per call at 16K: Llama 570 → 1201/1309 µs, Qwen 151 → 532/589 µs.
- **Per-rail:** the extra energy is on the CPU+GPU rail (+61–90% during decode). The SoC/memory rail changes by only +5–32%, so the data does not show memory traffic going down.
- **Concurrency (N = 2/4/8):** the penalty is present at every N. Llama f16 at N=8 (8 × 16K = 4.1 GiB of cache) does not fit, so only quantized caches can run it.
- **Q4_K_M weights:** same pattern, +13–18% at 16K. The result does not depend on the weight format.
- **Where quantization pays off (fit test):**
  - Llama-3B: f16 fits only 16K, q8_0 fits up to 32K, q4_0 up to 64K (2468 J per request, 3.2 tok/s).
  - Llama-1B: fits 64K in every format, and f16 stays cheapest.
- **Perplexity, 8K:**
  - Llama: f16 11.07 / q8_0 11.06 / q4_0 11.62.
  - Qwen: 12.49 / 12.65 / **155.8**.
  - K/V split: f16 K + q4 V = 12.53 (fine), q4 K + f16 V = 156.8. 4-bit *keys* break Qwen.

Deviations from the plan:
- Perplexity was run at 8K instead of 16K: llama-perplexity at 16K ran out of memory.
- 128K was not run, because the prompt plus generated tokens exceeds Llama's trained context.
- The Jetson froze and rebooted once during the deliberate N=8 out-of-memory check.

**7W mode (v2, 5 rounds, sets A and B):**
- **Quantization penalty:** all 48 quantized cells are significantly more expensive than f16, from +2.6% at 1K up to +28% per request at 16K.
  - Llama: +21% (8-bit) / +18% (4-bit)
  - Qwen: +28% / +21%
- **7W against 15W:** 7W costs **+21–31% net** and **+60–73% gross** energy per request. Decode is 2.0–2.4× slower (408 vs 625 MHz) and power only falls from about 10 W to 7 W. This confirms v1's "race to idle".
- **Idle correction:** the 7W grid started right after the power-mode reboot, and its start-of-set idle (3.68 W) was a transient: 3.03 W at the end, 2.92–3.05 W in set B. `analysis/analyze_v2.py` (`rebase_idle`) therefore subtracts the lower of the start/end idle readings. Gross energy is not affected.

---

# v1: first Jetson results: KV-cache quantization energy (2026-10-19)

**Setup:**
- Jetson Orin Nano Developer Kit, L4T R39.2 / Ubuntu 24.04, CUDA 13.2
- Two power modes: **15W** (GPU up to 625 MHz, 6 CPU cores) and **7W** (GPU up to 408 MHz, 4 CPU cores)
- llama.cpp with CUDA and flash attention, run through `llama-server`
- Models: Llama-3.2-1B-Instruct and Qwen2.5-0.5B-Instruct, both with Q8_0 weights
- Energy is `VDD_IN` (whole module) from the onboard INA3221 sensor, sampled at ~18 Hz, minus idle power (3.57 W at 15W, 3.41 W at 7W)
- Each request: a 1K / 4K / 8K / 16K-token prompt, then exactly 128 generated tokens
- 3 rounds per setting, in shuffled order, 72 runs per power mode (144 total). Run-to-run spread (std ÷ mean of the 3 rounds) is at most 6.3% at 15W and 8.5% at 7W, with a median of 1.5% and 1.0%.

Data: `results/kv_20261019_190621.jsonl` (15W), `results/kv_20261019_194427.jsonl` (7W), `results/ppl_20261019_192054.csv` · Plots: `plots/` · Full table: `plots/summary.md`

## Finding 1: quantizing the KV cache *costs* energy on the Jetson, and more at longer context

Extra decode energy per token compared with f16, **15W mode** (positive = uses **more** energy than f16):

| Context | Llama-1B q8_0 | Llama-1B q4_0 | Qwen-0.5B q8_0 | Qwen-0.5B q4_0 |
|---|---|---|---|---|
| 1K  | +0.5%  | −0.6%  | +5.3%  | +3.2%  |
| 4K  | +3.5%  | +2.3%  | +14.8% | +12.7% |
| 8K  | +17.3% | +12.0% | +32.9% | +25.0% |
| 16K | +35.4% | +28.5% | +58.9% | +50.5% |

**Why:** power stays almost the same (~9.6–10.4 W), but decoding gets slower. At 16K, Llama drops from 22.8 to about 18 tokens/s and Qwen from 50.8 to about 33. Unpacking the compressed K/V costs more than the memory traffic it saves. At 15W the GPU runs at only 625 MHz, so it's short on compute, not on memory bandwidth. This is the thesis question "do compression overheads cancel the savings?", answered with **yes** for this device and software.

In **7W mode** the penalty roughly doubles at 16K: Llama +61% (q8_0) / +52% (q4_0), and Qwen **+109%** (q8_0) / +74% (q4_0). Qwen with an 8-bit cache uses more than twice the energy per token.
This supports the explanation: the slower the GPU, the more the unpacking work dominates (plot `02_saving_vs_f16.png`, top row 15W, bottom row 7W; that plot shows *saving*, so the same numbers appear below zero there).

## Finding 2: but the memory saving is real

KV-cache size at 16K tokens: Llama **516 → 274 → 145 MiB**, Qwen **194 → 103 → 54 MiB** (f16 → q8_0 → q4_0).
On an 8 GB board shared with the OS, that's what lets longer contexts or bigger models fit at all. The trade-off is **memory vs energy**, not a free win.

## Finding 3: accuracy depends on the model, and 4-bit can break small models

WikiText-2 perplexity (lower is better):
- **Llama-1B:** f16 12.04 · q8_0 12.05 · q4_0 12.73. That's 8-bit at no cost and 4-bit at +5.7%.
- **Qwen-0.5B:** f16 13.02 · q8_0 13.19 · **q4_0 100.3**. A 4-bit KV cache breaks it: Qwen has only 2 KV heads, so every quantization error is shared widely.

## Finding 4: the low-power mode is not the energy-efficient mode

With an f16 cache, 7W draws less power (about 6 W vs 8–10 W) but runs 2–2.4× slower, so each token costs **more** energy:

| f16 decode J/token | 1K | 16K |
|---|---|---|
| Llama-1B, 15W → 7W | 0.249 → 0.278 (+12%) | 0.314 → 0.343 (+10%) |
| Qwen-0.5B, 15W → 7W | 0.089 → 0.109 (+23%) | 0.126 → 0.135 (+7%) |

Finishing fast and going idle beats running slowly ("race to idle"). 7W only makes sense when there's a hard power or thermal limit.

## What this means and next steps

- **Practical rule for this setup:** on the Jetson, at either power mode, keep the KV cache in f16 unless memory is the limit. If memory is the limit, prefer q8_0 (accuracy-safe) and accept higher energy per token.
- **Not yet claimed:** this may depend on llama.cpp's quantized flash-attention kernels. Next:
  - Super / 25W mode if this JetPack supports it
  - a larger 3B model, where the KV cache is a bigger share of memory traffic
  - eviction methods (StreamingLLM, SnapKV) for comparison
  - longer contexts (32K)

**Caveats:**
- `VDD_IN` is whole-module power, so it can't be compared directly with the GPU-only NVML numbers from the T4 thesis.
- The prefill/decode energy split uses llama.cpp's own timings, so it's approximate.
- The GPU clocks were not locked with `jetson_clocks`. The GPU did stay at 625 MHz in every run.
