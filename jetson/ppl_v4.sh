#!/usr/bin/env bash
# Accuracy v4 (WikiText-2 perplexity, 8K) of the mixed K/V formats on Qwen2.5-1.5B (Q4_K_M weights), the model
# set L adds and whose 4-bit keys break (ppl_v3: 8.67 -> 6,425). Same settings as ppl_v3.sh, build-allq.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="results/ppl_v4_$(date +%Y%m%d_%H%M%S).csv"
echo "model,type_k,type_v,ctx,chunks,build,ppl,ppl_err,seconds" > "$OUT"
M=models_extra/qwen2.5-1.5b-instruct-q4_k_m.gguf
name="$(basename "$M" .gguf)"
for kv in "f16 q4_0" "q8_0 q4_0" "f16 f16"; do
  set -- $kv
  log="results/logs/ppl_v4_${name}_${1}-${2}_c8192.log"
  t0=$(date +%s)
  llama.cpp/build-allq/bin/llama-perplexity -m "$M" -f data/wiki.test.raw -c 8192 -b 8192 -ub 512 --chunks 32 \
    -ngl 99 -fa on -ctk "$1" -ctv "$2" > "$log" 2>&1
  line="$(grep -E "Final estimate: PPL" "$log" | tail -1)"
  echo "$name,$1,$2,8192,32,build-allq,$(echo "$line" | sed -E 's/.*PPL = ([0-9.]+).*/\1/'),$(echo "$line" | sed -E 's/.*\+\/- ([0-9.]+).*/\1/'),$(( $(date +%s) - t0 ))" | tee -a "$OUT"
done
echo "-> $OUT"
