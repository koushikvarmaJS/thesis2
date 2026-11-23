#!/usr/bin/env bash
# v4: the tests left open in report v3. 15W only. Start inside tmux:
#   tmux new -s v4 "cd ~/thesis2 && bash jetson/run_v4.sh 2>&1 | tee results/run_v4.log"
#   L  energy of mixed K/V cache formats (8-bit K + 4-bit V, f16 K + 4-bit V) next to f16/q8_0/q4_0, all in the
#      build-allq llama.cpp (GGML_CUDA_FA_ALL_QUANTS=ON) so the comparison uses one binary; plus nsys at 16K
#   M  accuracy on a long-context retrieval task (needle + 3 distractors): llama.cpp cache formats, kvpress
#      eviction (question inside / after the compressed prompt) and PyTorch QuantizedCache
#   N  energy of PyTorch QuantizedCache (quanto 4/2-bit, HQQ 8/4-bit) next to eviction, same engine
#   O  profile of the PyTorch decode step between 6K and 8K cached tokens
#   P  eviction energy at the T4 thesis settings (2K and 8K prompt, 50 generated tokens, 50% eviction)
#   Q  perplexity of the mixed formats on Qwen2.5-1.5B (added after L; ran right after P)
# PyTorch runs stay at <= 8K for Llama-1B (16K ran the board out of memory on 2 Nov).
# STEPS selects parts, e.g. STEPS="M N" bash jetson/run_v4.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/evict-venv/bin:/usr/local/cuda/bin:$PATH"
PY="$HOME/evict-venv/bin/python -u"
ALLQ=llama.cpp/build-allq/bin
STEPS="${STEPS:-L M N O P Q}"
has() { [[ " $STEPS " == *" $1 "* ]]; }

quiet() {
  pkill -f "[u]pdate-manager"; pkill -f "[g]nome-software"; sleep 5
  echo "--- top:"; top -b -n 1 | sed -n 7,12p
}
step() { echo; echo "=== $(date '+%F %T') 15W: $*"; quiet; }

if has L; then
step "L mixed K/V energy: 3 models, 5 formats, 1K-16K, 5 rounds (build-allq)"
python3 jetson/bench_kv.py --tag 15W-mixkv --llama-bin "$ALLQ" --kv f16,q8_0,q4_0,q8_0/q4_0,f16/q4_0 --rounds 5 \
  --extra "--cache-ram 0" --models models/Llama-3.2-1B-Instruct-Q8_0.gguf models/qwen2.5-0.5b-instruct-q8_0.gguf \
  models_extra/qwen2.5-1.5b-instruct-q4_k_m.gguf

step "L nsys: decode kernels of the mixed formats at 16K"
mkdir -p results/prof
for kv in "q8_0 q4_0" "f16 q4_0" "f16 f16" "q8_0 q8_0"; do
  set -- $kv
  tag="Llama-3.2-1B-Instruct-Q8_0_${1}-${2}_allq"
  nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o "results/prof/$tag" \
    "$ALLQ/llama-bench" -m models/Llama-3.2-1B-Instruct-Q8_0.gguf -ngl 99 -fa on -ctk "$1" -ctv "$2" \
    -p 0 -n 64 -d 16384 -r 1 > "results/prof/$tag.nsys.log" 2>&1
  nsys stats -r cuda_gpu_kern_sum --format csv --force-export true -o "results/prof/$tag" \
    "results/prof/$tag.nsys-rep" >> "results/prof/$tag.nsys.log" 2>&1
  echo "nsys $tag: $(ls results/prof/$tag*kern_sum*.csv 2>/dev/null)"
done
fi

if has M; then
step "M accuracy, llama.cpp cache formats (Q8_0 weights)"
$PY jetson/needle_qa.py --engine llama --llama-bin "$ALLQ" --configs f16,q8_0,q4_0,q8_0/q4_0,f16/q4_0 --tag mixkv

step "M accuracy, PyTorch: eviction (question inside / after the compressed prompt) and QuantizedCache"
$PY jetson/needle_qa.py --engine hf --tag evict --configs none,streamingllm@0.5:qa,streamingllm@0.75:qa,\
snapkv@0.5:qa,snapkv@0.75:qa,streamingllm@0.5:ag,streamingllm@0.75:ag,snapkv@0.5:ag,snapkv@0.75:ag,quanto4,quanto2,hqq8,hqq4
fi

if has N; then
step "N PyTorch QuantizedCache energy vs f16 and eviction: 1K-8K, 3 rounds"
$PY jetson/bench_evict.py --tag 15W-pytq --methods f16,quanto4,quanto2,hqq8,hqq4,snapkv@0.5 \
  --contexts 1024,4096,8192 --rounds 3
fi

if has O; then
step "O profile of the PyTorch decode step, 4K-8K"
$PY jetson/profile_step.py --model Qwen/Qwen2.5-0.5B-Instruct --contexts 2048,4096,5120,6144,6656,7168,7680,8192,12288
$PY jetson/profile_step.py --model unsloth/Llama-3.2-1B-Instruct --contexts 2048,4096,5120,6144,6656,7168,7680,8192
$PY jetson/profile_step.py --model Qwen/Qwen2.5-0.5B-Instruct --attn eager --contexts 4096,6144,8192
fi

if has P; then
step "P eviction at the T4 thesis settings: 2K and 8K prompts, 50 tokens, 50% eviction, 5 rounds"
$PY jetson/bench_evict.py --tag 15W-t4match --methods f16,streamingllm@0.5,snapkv@0.5 --contexts 2048,8192 \
  --gen 50 --rounds 5
fi
if has Q; then
step "Q perplexity of the mixed formats, Qwen2.5-1.5B, 8K"
bash jetson/ppl_v4.sh
fi
echo "=== $(date '+%F %T') v4 done"
