#!/usr/bin/env bash
# v3: model size and model family, then longer replies. 15W only. Start inside tmux:
#   tmux new -s v3 "cd ~/thesis2 && bash jetson/run_v3.sh 2>&1 | tee results/run_v3.log"
# All models use Q4_K_M weights so that sizes up to 3B fit next to a 16K f16 cache.
#   same family, different sizes:   Qwen2.5 0.5B / 1.5B / 3B,  Llama 3.2 1B / 3B
#   same size, different families:  Llama-3.2-1B, Qwen2.5-1.5B, Gemma-3-1B
# --cache-ram 0: llama-server otherwise keeps up to 8 GiB of earlier prompts' KV caches in RAM (prompt cache,
# default since PR 16391). With Llama-3B and a 16K f16 cache that ran the 8 GB board out of memory on 31 Oct.
# Requests use cache_prompt=false, so the cache never changed a measurement; it only used memory.
# RESUME=1 skips the downloads and the four models that finished before that crash.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
X=models_extra
MODELS=("$X/qwen2.5-0.5b-instruct-q4_k_m.gguf" "$X/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        "$X/qwen2.5-3b-instruct-q4_k_m.gguf" "$X/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        "$X/Llama-3.2-3B-Instruct-Q4_K_M.gguf" "$X/gemma-3-1b-it-Q4_K_M.gguf")

quiet() {  # close desktop updaters that wake up after boot and use a full CPU core
  pkill -f "[u]pdate-manager"; pkill -f "[g]nome-software"; sleep 5
  echo "--- top:"; top -b -n 1 | sed -n 7,12p
}
step() { echo; echo "=== $(date '+%F %T') 15W: $*"; quiet; }

CRAM="--cache-ram 0"
if [ -z "${RESUME:-}" ]; then
step "download the new models"
dl() { [ -s "$X/$2" ] || curl -L --fail -o "$X/$2" "$1"; }
dl https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf
dl https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf qwen2.5-3b-instruct-q4_k_m.gguf
dl https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf gemma-3-1b-it-Q4_K_M.gguf
ls -la "${MODELS[@]}" || exit 1

step "H size and family grid: 6 models, f16 / q8_0 / q4_0, 1K-16K, 128 tokens, 5 rounds"
python3 jetson/bench_kv.py --tag 15W-size --kv f16,q8_0,q4_0 --rounds 5 --extra "$CRAM" --models "${MODELS[@]}"
else
step "H (resumed): Llama-3B again from round 0, then Gemma-3-1B"
python3 jetson/bench_kv.py --tag 15W-size --kv f16,q8_0,q4_0 --rounds 5 --extra "$CRAM" --models "${MODELS[@]:4:2}"
fi

step "I long replies: 512 generated tokens at 4K and 16K, 3 rounds"
python3 jetson/bench_kv.py --tag 15W-gen512 --kv f16,q8_0,q4_0 --contexts 4096,16384 --gen 512 --rounds 3 --extra "$CRAM" \
  --models "${MODELS[@]}"

step "J perplexity of every model, and the Llama K/V split at 8K"
bash jetson/ppl_v3.sh "${MODELS[@]}"

echo "=== $(date '+%F %T') v3 done"
