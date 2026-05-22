#!/bin/bash
# Breakable CUDA graph runner (BCG) — replaces torch.compile-based PCG
# with a manual graph-break mechanism. Smoke that BCG cooperates with
# the unified attention init.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="breakable_cuda_graph"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend triton
    --enable-breakable-cuda-graph
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --tp-size 1
)

run_server_smoke
