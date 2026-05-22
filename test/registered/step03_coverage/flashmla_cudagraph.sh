#!/bin/bash
# flashmla backend + full CUDA graph (DSV3 NVFP4 MLA model).
# flashmla requires Hopper (SM 9.0a) — skip on Blackwell GPUs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="flashmla_cudagraph"

SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
if [[ "$SM_VERSION" == "10.0" || "$SM_VERSION" == "10.3" || "$SM_VERSION" == "12.0" ]]; then
    echo "SKIP: ${TEST_NAME}: flashmla requires SM 9.0a (Hopper); this GPU is SM ${SM_VERSION}"
    exit 0
fi

MODEL_PATH="${MODEL_PATH:-/mnt/vast/models/dsv3-nvfp4}"

LAUNCH_ARGS=(
    --attention-backend flashmla
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 4
)

run_server_smoke
