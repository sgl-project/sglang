#!/bin/bash
# trtllm_mla backend + full CUDA graph (Blackwell MLA via TRT-LLM kernels).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="trtllm_mla_cudagraph"
MODEL_PATH="${MODEL_PATH:-/mnt/vast/models/dsv3-nvfp4}"

LAUNCH_ARGS=(
    --attention-backend trtllm_mla
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 4
)

run_server_smoke
