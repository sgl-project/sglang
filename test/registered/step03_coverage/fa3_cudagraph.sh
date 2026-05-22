#!/bin/bash
# FlashAttention-3 backend + full CUDA graph.
#
# Note: fa3 requires SM 80-90 (Hopper). Skipped on Blackwell (SM100+)
# clusters where the kernel asserts in the scheduler.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="fa3_cudagraph"

# Detect Blackwell and skip with a clear message.
SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
if [[ "$SM_VERSION" == "10.0" || "$SM_VERSION" == "10.3" || "$SM_VERSION" == "12.0" ]]; then
    echo "SKIP: ${TEST_NAME}: fa3 requires SM 8.0-9.0; this GPU is SM ${SM_VERSION}"
    exit 0
fi

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
