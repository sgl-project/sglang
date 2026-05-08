#!/bin/bash
# DSv4-Flash on SM120 — rebased on main, with CUDA graph
set -e

export VENV=/home/scratch.alichen_sw_1/workspace/sglang_venv
export PY=$VENV/bin/python3
export HF_HOME=/home/scratch.alichen_sw_1/.hf_cache
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_HACK_FLASHMLA_BACKEND=kernel
export PYTHONUNBUFFERED=1
export CUDA_HOME=/home/scratch.alichen_sw_1/workspace/cuda-12.8
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA 13 libnvrtc needed by sgl_kernel
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

# Install rebased sglang from scratch workspace
cd /home/scratch.alichen_sw_1/sglang_rebase
pip install -e "python[all]" --no-build-isolation 2>&1 | tail -5

exec $PY -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --tp 8 \
    --trust-remote-code \
    --mem-fraction-static 0.70 \
    --port 30000 \
    --host 0.0.0.0 \
    --cuda-graph-max-bs 32 \
    --watchdog-timeout 600
