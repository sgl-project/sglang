#!/bin/bash
# flashinfer (MLA mode, used by DSV3 family) + full CUDA graph.
# Uses lmsys ci dsv3 test model — small MLA model wired for CI.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="flashinfer_mla_cudagraph"
# DSV3 mini CI model is the canonical MLA smoke target.
MODEL_PATH="${MODEL_PATH:-lmsys/sglang-ci-dsv3-test}"

LAUNCH_ARGS=(
    --attention-backend flashinfer
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

run_server_smoke
