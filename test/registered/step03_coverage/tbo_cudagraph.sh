#!/bin/bash
# Two-batch overlap (TBO) + CUDA graph on DeepSeek-V4-Flash.
# DSV4 is the model the production TBO workload targets, and dsv4-mode
# already wires up the right MoE A2A + cuda graph configuration.
# Stresses the tbo_backend wrapper around dsv4 attention init.
#
# NOTE: as of 2026-05-22, TBO is broken on origin/main HEAD with:
#   AttributeError: 'TboAttnBackend' object has no attribute
#   '_maybe_upgrade_forward_metadata'
# Once that lands, this test should pass. Until then the test is
# expected to FAIL on a clean main checkout — keep it in the suite so
# the regression is visible.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="tbo_cudagraph"
MODEL_PATH="${MODEL_PATH:-/flash_model}"

LAUNCH_ARGS=(
    --tp-size 4
    --ep-size 4
    --moe-a2a-backend deepep
    --moe-runner-backend flashinfer_mxfp4
    --enable-two-batch-overlap
    --enable-dp-attention
    --dp-size 4
    --moe-dense-tp-size 1
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 32
    --max-running-requests 32
    --chunked-prefill-size 4096
    --disable-flashinfer-autotune
    --disable-piecewise-cuda-graph
)

READY_TIMEOUT=2400
run_server_smoke
