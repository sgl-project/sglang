#!/bin/bash
# flashinfer (MLA mode, used by DSV3 family) + full CUDA graph.
# Uses DeepSeek-V3-NVFP4 on the cluster — MLA model with FP4 weights.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="flashinfer_mla_cudagraph"
# DSV3 NVFP4 model is the on-cluster MLA smoke target.
MODEL_PATH="${MODEL_PATH:-/mnt/vast/models/dsv3-nvfp4}"

LAUNCH_ARGS=(
    --attention-backend flashinfer
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 4
)

run_server_smoke
