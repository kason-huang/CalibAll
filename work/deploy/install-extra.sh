#!/bin/bash
set -eu
cd /home/kason/workspace/CalibAll
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=4
uv pip install --python .venv/bin/python --no-deps timm ftfy==6.1.1 iopath fvcore portalocker ninja setuptools==75.0.0 wheel yacs websockets qpsolvers daqp plyfile huggingface-hub==0.36.2
uv pip install --python .venv/bin/python --no-deps -e . -e third_party/co-tracker -e third_party/sam3 -e third_party/moge
uv pip install --python .venv/bin/python --no-deps --no-build-isolation ./third_party/nvdiffrast
printf '%s\n' /home/kason/workspace/CalibAll/third_party/pytorch3d > .venv/lib/python3.11/site-packages/pytorch3d-source.pth
.venv/bin/python scripts/caliball_web.py --help
