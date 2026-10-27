#!/usr/bin/env bash
# Accuracy check: WikiText-2 perplexity for each model x KV cache type.
# In llama.cpp attention reads K/V back from the (quantized) cache, so unlike the
# thesis' kvpress setup, perplexity here really does reflect the compression.
# Usage: bash jetson/ppl_kv.sh [chunks]   (default 20 chunks of 2048 tokens, ~2-4 min per config)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CHUNKS="${1:-20}"
OUT="results/ppl_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results/logs
echo "model,kv_type,ctx,chunks,ppl,ppl_err" > "$OUT"
for m in models/*.gguf; do
  name="$(basename "$m" .gguf)"
  for kv in f16 q8_0 q4_0; do
    log="results/logs/ppl_${name}_${kv}.log"
    llama.cpp/build/bin/llama-perplexity -m "$m" -f data/wiki.test.raw -c 2048 --chunks "$CHUNKS" \
      -ngl 99 -ctk "$kv" -ctv "$kv" -fa on > "$log" 2>&1 \
    || llama.cpp/build/bin/llama-perplexity -m "$m" -f data/wiki.test.raw -c 2048 --chunks "$CHUNKS" \
      -ngl 99 -ctk "$kv" -ctv "$kv" -fa > "$log" 2>&1
    line="$(grep -E "Final estimate: PPL" "$log" | tail -1)"
    ppl="$(echo "$line" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')"
    err="$(echo "$line" | sed -E 's/.*\+\/- ([0-9.]+).*/\1/')"
    echo "$name,$kv,2048,$CHUNKS,$ppl,$err" | tee -a "$OUT"
  done
done
echo "-> $OUT"
