#!/bin/bash
set -eu
cd /home/kason/workspace/CalibAll
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export USE_PERFLIB=0
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec .venv/bin/python scripts/caliball_web.py --host 0.0.0.0 --port 8766 --device cuda --no-browser
