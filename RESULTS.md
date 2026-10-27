# First Jetson results: KV-cache quantization energy (2026-10-19)

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
