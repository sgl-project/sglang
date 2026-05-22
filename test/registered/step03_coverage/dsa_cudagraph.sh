#!/bin/bash
# dsa attention backend + CUDA graph (DeepSeek V3.2 sparse attention).
# Uses DeepSeek-V3.2 model — large (~600GB) but cached on cluster.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="dsa_cudagraph"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3.2}"

LAUNCH_ARGS=(
    --attention-backend dsa
    --tp-size 4
    --mem-fraction-static 0.8
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --chunked-prefill-size 4096
    --disable-piecewise-cuda-graph
)

READY_TIMEOUT=2400
run_server_smoke
