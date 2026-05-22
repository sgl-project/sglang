#!/bin/bash
# FlashAttention-3 backend + full CUDA graph.
# Note: fa3 is hopper-default; on B200/GB300 it must be explicitly forced.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="fa3_cudagraph"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend fa3
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

run_server_smoke
