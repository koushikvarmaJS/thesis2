#!/usr/bin/env bash
# Extra setup for the v2 experiments (run after setup.sh). Safe to re-run.
#  * llama-bench + llama-batched-bench in the existing build (same binary as v1 runs)
#  * a second build with GGML_CUDA_FA_ALL_QUANTS=ON: without it, flash attention on the GPU
#    only supports matching K/V types (f16/f16, q8_0/q8_0, q4_0/q4_0); mixed pairs such as
#    K=f16,V=q4_0 would silently run attention on the CPU
#  * 4-bit-weight models (Q4_K_M) and Llama-3.2-3B for the weight-format and fit tests
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH=/usr/local/cuda/bin:$PATH

echo "== llama-bench in the main build"
cmake --build llama.cpp/build --config Release -j4 --target llama-bench llama-batched-bench

echo "== models (Q4_K_M weights)"
mkdir -p models_extra
dl() { [ -s "models_extra/$2" ] || curl -L --fail -s -o "models_extra/$2" "$1"; }
dl https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf Llama-3.2-1B-Instruct-Q4_K_M.gguf &
dl https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf qwen2.5-0.5b-instruct-q4_k_m.gguf &
dl https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf Llama-3.2-3B-Instruct-Q4_K_M.gguf &

echo "== second build with all K/V flash-attention combinations (slow: compiles many kernels)"
if [ ! -x llama.cpp/build-allq/bin/llama-perplexity ]; then
  cmake -S llama.cpp -B llama.cpp/build-allq -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON
  cmake --build llama.cpp/build-allq --config Release -j4 --target llama-perplexity llama-bench
fi
wait
ls -lh models_extra
echo "setup_v2 done"
