#!/bin/bash
# Two-batch overlap (TBO) + CUDA graph on a DeepSeek-V3 NVFP4 MoE model.
# TBO requires a MoE A2A backend (deepep) and DP attention.
# Stresses the tbo_backend wrapper around the underlying attention init.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="tbo_cudagraph"
MODEL_PATH="${MODEL_PATH:-/mnt/vast/models/dsv3-nvfp4}"

LAUNCH_ARGS=(
    --tp-size 4
    --moe-a2a-backend deepep
    --moe-runner-backend deep_gemm
    --enable-two-batch-overlap
    --enable-dp-attention
    --dp-size 4
    --moe-dense-tp-size 1
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
)

READY_TIMEOUT=2400
run_server_smoke
