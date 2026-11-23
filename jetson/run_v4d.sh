#!/usr/bin/env bash
# v4d (17 Nov): the PyTorch "step" between 6K and 8K tokens is the benchmark's pad_token_id=0: token 0 is "!",
# the corpus has its first "!" at token ~7.9K-8.1K, and generate() without an attention mask then treats every
# "!" as padding and builds a real mask. This run (1) reproduces it with profile_step.py, (2) reruns the PyTorch
# energy cells that contain a "!" (prompts >= 8K) with pad_token_id = eos (bench_evict.py --pad eos, the default).
# Starts after run_v4c.sh (waits for its tmux session).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/evict-venv/bin:/usr/local/cuda/bin:$PATH"
PY="$HOME/evict-venv/bin/python -u"
while tmux has-session -t v4c 2>/dev/null; do sleep 30; done
step() { echo; echo "=== $(date '+%F %T') 15W: $*"; pkill -f "[u]pdate-manager"; pkill -f "[g]nome-software"; sleep 5; }
step "O3 decode step: generate() with pad_token_id=0 and no mask, around the first '!'"
$PY jetson/profile_step.py --model Qwen/Qwen2.5-0.5B-Instruct --mode genpad0 --steps 64 --prof-steps 2 \
  --contexts 6144,7680,8061,8063,8192
$PY jetson/profile_step.py --model unsloth/Llama-3.2-1B-Instruct --mode genpad0 --steps 64 --prof-steps 2 \
  --contexts 6144,7680,7884,7886,8192
step "K-fix eviction energy, pad = eos: Qwen 8K/16K, Llama 8K, 5 rounds"
$PY jetson/bench_evict.py --tag 15W-evict-fix --contexts 8192,16384 --rounds 5 \
  --models Qwen/Qwen2.5-0.5B-Instruct
$PY jetson/bench_evict.py --tag 15W-evict-fix --contexts 8192 --rounds 5 --models unsloth/Llama-3.2-1B-Instruct
step "N-fix quantized cache energy, pad = eos, 8K, 3 rounds"
$PY jetson/bench_evict.py --tag 15W-pytq-fix --methods f16,quanto4,quanto2,hqq8,hqq4,snapkv@0.5 --contexts 8192 \
  --rounds 3
step "P-fix T4 settings, pad = eos, 8K, 5 rounds"
$PY jetson/bench_evict.py --tag 15W-t4match-fix --methods f16,streamingllm@0.5,snapkv@0.5 --contexts 8192 \
  --gen 50 --rounds 5
echo "=== $(date '+%F %T') v4d done"
