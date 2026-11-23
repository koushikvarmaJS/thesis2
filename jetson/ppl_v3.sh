#!/usr/bin/env bash
# Accuracy v3 (WikiText-2 perplexity) for the size and family models, f16 / q8_0 / q4_0, at 2K and 8K.
# Gemma-3 has a 262K-token vocabulary: at 8K the logits of the scored half (4,096 x 262,144 x 4 B = 4.3 GB)
# do not fit next to the model, so Gemma runs at 2K and 4K instead.
# Also the Llama-1B (Q8_0 weights) K/V split at 8K, to match the Qwen split in ppl_v2.sh.
# Usage: bash jetson/ppl_v3.sh model.gguf [model.gguf ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="results/ppl_v3_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results/logs
echo "model,type_k,type_v,ctx,chunks,build,ppl,ppl_err,seconds" > "$OUT"

run() {  # build model k v ctx chunks
  local bin="llama.cpp/$1/bin/llama-perplexity" m="$2" k="$3" v="$4" c="$5" n="$6"
  local name; name="$(basename "$m" .gguf)"
  local log="results/logs/ppl_v3_${name}_${k}-${v}_c${c}_$1.log"
  local t0; t0=$(date +%s)
  # one batch per window (-b = context), as in ppl_v2.sh
  "$bin" -m "$m" -f data/wiki.test.raw -c "$c" -b "$c" -ub 512 --chunks "$n" -ngl 99 -fa on -ctk "$k" -ctv "$v" \
    > "$log" 2>&1
  local line; line="$(grep -E "Final estimate: PPL" "$log" | tail -1)"
  local ppl err
  ppl="$(echo "$line" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')"
  err="$(echo "$line" | sed -E 's/.*\+\/- ([0-9.]+).*/\1/')"
  echo "$name,$k,$v,$c,$n,$1,$ppl,$err,$(( $(date +%s) - t0 ))" | tee -a "$OUT"
}

for m in "$@"; do
  ctxs="2048 8192"
  [[ "$(basename "$m")" == gemma* ]] && ctxs="2048 4096"
  for c in $ctxs; do
    for kv in f16 q8_0 q4_0; do run build "$m" "$kv" "$kv" "$c" $(( 262144 / c > 64 ? 64 : 262144 / c )); done
  done
done

LLAMA=models/Llama-3.2-1B-Instruct-Q8_0.gguf
for kv in "f16 q4_0" "q4_0 f16" "q8_0 q4_0"; do
  set -- $kv
  run build-allq "$LLAMA" "$1" "$2" 8192 32
done
echo "-> $OUT"
