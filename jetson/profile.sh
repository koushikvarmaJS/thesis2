#!/usr/bin/env bash
# Mechanism check: where does decode time go at a 16K context, per KV format?
#   bash jetson/profile.sh nsys         -> per-kernel GPU time (Nsight Systems), runs as the normal user
#   sudo bash jetson/profile.sh ncu     -> compute vs memory throughput of the decode attention kernel
#                                          (Nsight Compute; GPU counters need root on Jetson)
# Each profile is one llama-bench tg run: fill the cache to 16,384 tokens, then generate 64 tokens.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:-nsys}"
BENCH=llama.cpp/build/bin/llama-bench
OUT=results/prof
mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
DEPTH=16384

for m in models/Llama-3.2-1B-Instruct-Q8_0.gguf models/qwen2.5-0.5b-instruct-q8_0.gguf; do
  name="$(basename "$m" .gguf)"
  layers=16; [[ "$name" == qwen* ]] && layers=24
  for cfg in f16 q8_0 q4_0 f16@off; do
    kv="${cfg%@*}"; fa=on; [ "$cfg" = "f16@off" ] && fa=off
    tag="${name}_${kv}_fa${fa}"
    args=(-m "$m" -ngl 99 -fa "$fa" -ctk "$kv" -ctv "$kv" -p 0 -n 64 -d "$DEPTH" -r 1)
    if [ "$MODE" = nsys ]; then
      nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o "$OUT/$tag" \
        "$BENCH" "${args[@]}" > "$OUT/$tag.nsys.log" 2>&1
      nsys stats -r cuda_gpu_kern_sum --format csv --force-export true -o "$OUT/$tag" \
        "$OUT/$tag.nsys-rep" >> "$OUT/$tag.nsys.log" 2>&1
      echo "nsys $tag: $(ls "$OUT/$tag"*kern_sum*.csv 2>/dev/null)"
    elif [ "$fa" = on ]; then
      # the flash-attention launches are: 32 prefill batches x layers (filling the 16K cache), then one
      # per layer per generated token. Skip the prefill and 2 tokens, profile one token (all layers).
      # Decode uses flash_attn_ext_f16 (tensor cores) for f16 and flash_attn_ext_vec for q8_0/q4_0.
      GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name regex:flash_attn_ext --launch-skip $((34 * layers)) \
        --launch-count "$layers" --section SpeedOfLight --section MemoryWorkloadAnalysis --csv \
        "$BENCH" "${args[@]}" > "$OUT/${tag}_ncu.csv" 2> "$OUT/${tag}_ncu.log"
      echo "ncu $tag: $(wc -l < "$OUT/${tag}_ncu.csv") lines"
    fi
  done
done
[ "$(id -u)" = 0 ] && chown -R "${SUDO_USER:-koushik}" "$OUT"
echo "-> $OUT"
