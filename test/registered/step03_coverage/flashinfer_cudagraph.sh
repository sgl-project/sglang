#!/bin/bash
# flashinfer (MHA mode) + full CUDA graph.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="flashinfer_cudagraph"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend flashinfer
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

run_server_smoke
