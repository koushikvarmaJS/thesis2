#!/usr/bin/env bash
# One-time setup on the Jetson: build llama.cpp with CUDA, download models + text.
# Safe to re-run: every step skips work that is already done.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH=/usr/local/cuda/bin:$PATH

echo "== 1/4 system packages"
need=""
for t in git cmake g++ curl unzip; do command -v "$t" >/dev/null || need="$need $t"; done
if [ -n "$need" ]; then
  echo "missing:$need -> installing (needs sudo password)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq git cmake build-essential curl unzip
fi

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found. Install the CUDA toolkit, e.g.: sudo apt install cuda-toolkit-13-2   (then re-run)"
  exit 1
fi

echo "== 2/4 build llama.cpp (CUDA, Orin = sm_87). Takes ~20-40 min the first time."
if [ ! -x llama.cpp/build/bin/llama-server ]; then
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  # -j4: more parallel jobs can run the 8 GB board out of memory while compiling CUDA
  cmake --build llama.cpp/build --config Release -j4 --target llama-server llama-perplexity llama-cli
fi
(cd llama.cpp && git rev-parse --short HEAD > ../results_llama_commit.txt) || true
llama.cpp/build/bin/llama-server --version 2>&1 | head -3 || true

echo "== 3/4 models (GGUF)"
mkdir -p models
dl() { [ -s "models/$2" ] || curl -L --fail -o "models/$2" "$1"; }
dl https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf qwen2.5-0.5b-instruct-q8_0.gguf
dl https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf Llama-3.2-1B-Instruct-Q8_0.gguf
if [ "${WITH_3B:-0}" = "1" ]; then
  mkdir -p models_extra
  [ -s models_extra/Llama-3.2-3B-Instruct-Q4_K_M.gguf ] || curl -L --fail -o models_extra/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
fi
ls -lh models

echo "== 4/4 text data"
mkdir -p data
# Long-context prompt: Pride and Prejudice (public domain, ~160K tokens)
[ -s data/corpus.txt ] || curl -L --fail -o data/corpus.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
# WikiText-2 test set for perplexity (accuracy check of quantized KV)
if [ ! -s data/wiki.test.raw ]; then
  curl -L --fail -o data/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  unzip -o -q data/wikitext-2-raw-v1.zip -d data
  cp data/wikitext-2-raw/wiki.test.raw data/wiki.test.raw
fi
wc -c data/corpus.txt data/wiki.test.raw

echo
echo "Setup done. Next: python3 jetson/power.py --watch 10"
