#!/bin/bash
# DP-attention with a forced idle-rank scenario.
# dp_size=4 tp_size=4 + small Qwen3 MoE: only some DP groups receive
# tokens for a given request, exercising the IDLE forward-mode path
# in DP-aware attention init.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="dp_idle"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B}"

LAUNCH_ARGS=(
    --tp-size 4
    --dp-size 4
    --enable-dp-attention
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --attention-backend triton
)

READY_TIMEOUT=2400
run_server_smoke
