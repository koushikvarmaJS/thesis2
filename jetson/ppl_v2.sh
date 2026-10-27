#!/usr/bin/env bash
# Accuracy v2 (WikiText-2 perplexity):
#  1. at a long context: -c 8192 (llama-perplexity scores the second half of each window, so every
#     scored token sees 4K-8K tokens of context). 16K does not fit: the logits of the scored half
#     (8,192 x vocabulary x 4 B = 4-5 GB) plus the model exceed the 8 GB, and the process is killed.
#     Override with LONG=16384 on a larger device.
#  2. Qwen K/V split, to find out which half of the cache breaks Qwen at 4 bits:
#     K f16 + V q4_0, K q4_0 + V f16, K q8_0 + V q4_0, and q4_1 (asymmetric 4-bit with an offset)
# Mixed K/V types need the GGML_CUDA_FA_ALL_QUANTS build (setup_v2.sh), or attention runs on the CPU.
# Usage: bash jetson/ppl_v2.sh            (ONLY_CTX=8192 bash jetson/ppl_v2.sh: rerun one context only)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="results/ppl_v2_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results/logs
echo "model,type_k,type_v,ctx,chunks,build,ppl,ppl_err,seconds" > "$OUT"

run() {  # build model k v ctx chunks
  local bin="llama.cpp/$1/bin/llama-perplexity" m="$2" k="$3" v="$4" c="$5" n="$6"
  local name; name="$(basename "$m" .gguf)"
  local log="results/logs/ppl_v2_${name}_${k}-${v}_c${c}_$1.log"
  local t0; t0=$(date +%s)
  [ -n "${ONLY_CTX:-}" ] && [ "$c" != "$ONLY_CTX" ] && return
  # One batch per window (-b = context): otherwise llama-perplexity reserves host memory for the logits
  # of every token (16K x vocabulary x 4 B = 8-10 GB) and fails with bad_alloc.
  "$bin" -m "$m" -f data/wiki.test.raw -c "$c" -b "$c" -ub 512 --chunks "$n" -ngl 99 -fa on -ctk "$k" -ctv "$v" \
    > "$log" 2>&1
  local line; line="$(grep -E "Final estimate: PPL" "$log" | tail -1)"
  local ppl err
  ppl="$(echo "$line" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')"
  err="$(echo "$line" | sed -E 's/.*\+\/- ([0-9.]+).*/\1/')"
  echo "$name,$k,$v,$c,$n,$1,$ppl,$err,$(( $(date +%s) - t0 ))" | tee -a "$OUT"
}

QWEN=models/qwen2.5-0.5b-instruct-q8_0.gguf
LLAMA=models/Llama-3.2-1B-Instruct-Q8_0.gguf

# 1. Long context, same build as the energy runs. WikiText-2 test has ~290K tokens.
LONG="${LONG:-8192}"
LONG_CHUNKS=$(( 262144 / LONG ))
for m in "$LLAMA" "$QWEN"; do
  for kv in f16 q8_0 q4_0; do run build "$m" "$kv" "$kv" "$LONG" "$LONG_CHUNKS"; done
done

# 2. Qwen K/V split at 2K (as in v1) and at the long context, with the all-quants build
for c in 2048 "$LONG"; do
  n=$([ "$c" = 2048 ] && echo 20 || echo "$LONG_CHUNKS")
  for kv in "q4_0 q4_0" "f16 q4_0" "q4_0 f16" "q8_0 q4_0" "q4_0 q8_0" "q4_1 q4_1"; do
    set -- $kv
    run build-allq "$QWEN" "$1" "$2" "$c" "$n"
  done
done
# same check on Llama at 2K, for comparison
for kv in "f16 q4_0" "q4_0 f16" "q4_1 q4_1"; do
  set -- $kv
  run build-allq "$LLAMA" "$1" "$2" 2048 20
done
echo "-> $OUT"
