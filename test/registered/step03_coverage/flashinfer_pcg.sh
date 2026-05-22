#!/bin/bash
# flashinfer + piecewise CUDA graph (exercises the EXTEND-mode path,
# which is the prefill kernel under PCG capture). Key for catching the
# step03 prefill regression that previously hit trtllm_mha.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="flashinfer_pcg"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend flashinfer
    --enforce-piecewise-cuda-graph
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --tp-size 1
)

run_server_smoke
