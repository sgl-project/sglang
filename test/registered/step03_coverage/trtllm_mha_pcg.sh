#!/bin/bash
# trtllm_mha + piecewise CUDA graph.
# This combo previously caught a prefill regression in the step03
# refactor (init_forward_data plumbing) — keep it as a guard test.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="trtllm_mha_pcg"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend trtllm_mha
    --enforce-piecewise-cuda-graph
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --tp-size 1
)

run_server_smoke
