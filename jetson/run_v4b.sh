cd ~/thesis2
export PATH=$HOME/evict-venv/bin:/usr/local/cuda/bin:$PATH
echo "=== $(date "+%F %T") 15W: M (restart) accuracy, PyTorch: eviction and QuantizedCache"
$HOME/evict-venv/bin/python -u jetson/needle_qa.py --engine hf --tag evict --configs none,streamingllm@0.5:qa,streamingllm@0.75:qa,snapkv@0.5:qa,snapkv@0.75:qa,streamingllm@0.5:ag,streamingllm@0.75:ag,snapkv@0.5:ag,snapkv@0.75:ag,quanto4,quanto2,hqq8,hqq4
STEPS="N O P Q" bash jetson/run_v4.sh
