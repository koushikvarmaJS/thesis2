#!/usr/bin/env bash
# Print everything we need to know about this Jetson before benchmarking.
echo "== Board";      tr -d '\0' < /proc/device-tree/model; echo
echo "== L4T";        head -1 /etc/nv_tegra_release 2>/dev/null || echo "not found"
echo "== JetPack";    dpkg-query -W -f='${Version}\n' nvidia-jetpack 2>/dev/null || echo "nvidia-jetpack meta-package not installed (fine)"
echo "== OS";         lsb_release -ds 2>/dev/null; uname -r
echo "== Power mode"; sudo nvpmodel -q 2>/dev/null || nvpmodel -q
echo "== Modes available:"; grep -E "POWER_MODEL ID" /etc/nvpmodel.conf 2>/dev/null
echo "== CUDA";       (/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2) || echo "nvcc not found: sudo apt install nvidia-jetpack (or cuda-toolkit)"
echo "== Memory";     free -h
echo "== Disk";       df -h ~ | tail -1
echo "== Tools";      for t in cmake git gcc g++ python3 curl unzip tegrastats; do printf "%-11s %s\n" "$t" "$(command -v $t || echo MISSING)"; done
echo "== Power sensors"
python3 "$(dirname "$0")/power.py" --watch 3
