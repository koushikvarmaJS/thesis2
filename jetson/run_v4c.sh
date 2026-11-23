#!/usr/bin/env bash
# v4c (17 Nov): what run_v4b.sh left open.
#   O2 the PyTorch decode step by path: forward loop without mask / with generate()'s 2-D mask / generate() itself
#      (set O's loop stayed flat to 8K while generate() slowed down; this finds which part does it)
#   Q  perplexity of the mixed formats on Qwen2.5-1.5B (the Jetson's run_v4.sh predated step Q, so it never ran)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/evict-venv/bin:/usr/local/cuda/bin:$PATH"
PY="$HOME/evict-venv/bin/python -u"
step() { echo; echo "=== $(date '+%F %T') 15W: $*"; pkill -f "[u]pdate-manager"; pkill -f "[g]nome-software"; sleep 5; }
for mode in loop mask generate; do
  step "O2 decode step, mode $mode"
  $PY jetson/profile_step.py --model Qwen/Qwen2.5-0.5B-Instruct --mode $mode --steps 64 --prof-steps 2 \
    --contexts 2048,4096,5120,6144,6656,7168,7680,8192
  $PY jetson/profile_step.py --model unsloth/Llama-3.2-1B-Instruct --mode $mode --steps 64 --prof-steps 2 \
    --contexts 2048,4096,5120,6144,7168,8192
done
step "Q perplexity of the mixed formats, Qwen2.5-1.5B, 8K"
bash jetson/ppl_v4.sh
echo "=== $(date '+%F %T') v4c done"
