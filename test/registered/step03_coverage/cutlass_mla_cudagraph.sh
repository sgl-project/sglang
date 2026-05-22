#!/bin/bash
# cutlass_mla backend + full CUDA graph (Blackwell-only MLA kernels).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="cutlass_mla_cudagraph"
MODEL_PATH="${MODEL_PATH:-lmsys/sglang-ci-dsv3-test}"

LAUNCH_ARGS=(
    --attention-backend cutlass_mla
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

run_server_smoke
