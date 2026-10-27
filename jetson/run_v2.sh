#!/usr/bin/env bash
# All v2 experiments for one power mode, in order. Start inside tmux:
#   tmux new -s v2 "bash jetson/run_v2.sh 15W 2>&1 | tee results/run_v2_15W.log"
# 15W runs everything; 7W repeats the main grid and the llama-bench pp/tg test.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:-15W}"
L1=models/Llama-3.2-1B-Instruct-Q8_0.gguf
Q05=models/qwen2.5-0.5b-instruct-q8_0.gguf

quiet() {  # close desktop updaters that wake up after boot and use a full CPU core
  pkill -f "[u]pdate-manager"; pkill -f "[g]nome-software"; sleep 5
  echo "--- top:"; top -b -n 1 | sed -n 7,12p
}
step() { echo; echo "=== $(date '+%F %T') $MODE: $*"; quiet; }

step "A main grid: f16 (flash attention on and off), q8_0, q4_0; 5 rounds"
python3 jetson/bench_kv.py --tag "$MODE-v2" --kv f16,f16@off,q8_0,q4_0 --rounds 5

step "B llama-bench: prompt and generation measured separately"
python3 jetson/bench_pp_tg.py --tag "$MODE"

[ "$MODE" = 15W ] || { echo "=== $(date '+%F %T') $MODE done"; exit 0; }

step "D concurrency: 1, 2, 4, 8 sessions"
python3 jetson/bench_kv.py --tag 15W-np --kv f16,q8_0,q4_0 --contexts 4096,16384 --np 1,2,4,8 --rounds 3

step "E 4-bit weights (Q4_K_M)"
python3 jetson/bench_kv.py --tag 15W-w4 --kv f16,q8_0,q4_0 --rounds 5 \
  --models models_extra/Llama-3.2-1B-Instruct-Q4_K_M.gguf models_extra/qwen2.5-0.5b-instruct-q4_k_m.gguf

step "F nsys kernel profile at 16K"
bash jetson/profile.sh nsys

step "C perplexity at 16K and Qwen K/V split (finishes the all-quants build first)"
if [ ! -x llama.cpp/build-allq/bin/llama-perplexity ]; then
  PATH=/usr/local/cuda/bin:$PATH cmake --build llama.cpp/build-allq --config Release -j4 \
    --target llama-perplexity llama-bench > results/build_allq.log 2>&1
fi
bash jetson/ppl_v2.sh

step "G fit test: long contexts and a 3B model (f16 may not fit)"
python3 jetson/bench_kv.py --tag 15W-fit1b --models "$L1" --kv f16,q8_0,q4_0 \
  --contexts 32768,65536,131072 --per-context-server --rounds 2 --extra "-fit off"
python3 jetson/bench_kv.py --tag 15W-fit3b --models models_extra/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --kv f16,q8_0,q4_0 --contexts 16384,32768,65536 --per-context-server --rounds 2 --extra "-fit off"

echo "=== $(date '+%F %T') $MODE done"
