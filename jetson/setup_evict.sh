#!/usr/bin/env bash
# Python environment for the eviction benchmark (bench_evict.py): CUDA PyTorch for JetPack 7 (CUDA 13.2,
# Orin = sm_87) from the upstream cu132 index, plus transformers and kvpress at the versions the thesis used.
# A plain `pip install torch` gives a build for data-centre GPUs that does not run on Orin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/evict-venv}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --pre torch --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple
"$VENV/bin/pip" install -q "transformers>=4.56,<5.0" "kvpress==0.5.3" accelerate
"$VENV/bin/python" - <<'EOF'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
print('matmul ok', float((x @ x).float().abs().mean()))
EOF
echo "run: $VENV/bin/python $ROOT/jetson/bench_evict.py --quick"
