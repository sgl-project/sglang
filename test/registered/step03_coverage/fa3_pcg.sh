#!/bin/bash
# fa3 + piecewise CUDA graph.
# Requires SM 8.0-9.0 (Hopper); skip on Blackwell.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="fa3_pcg"

SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
if [[ "$SM_VERSION" == "10.0" || "$SM_VERSION" == "10.3" || "$SM_VERSION" == "12.0" ]]; then
    echo "SKIP: ${TEST_NAME}: fa3 requires SM 8.0-9.0; this GPU is SM ${SM_VERSION}"
    exit 0
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend fa3
    --enforce-piecewise-cuda-graph
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --tp-size 1
)

run_server_smoke
