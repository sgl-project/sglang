#!/bin/bash
# cutlass_mla backend + full CUDA graph (B200-class MLA kernels).
# cutlass_mla_decode is only compiled for SM 10.0 (B200); SM 10.3 (GB300)
# is not supported, so skip there.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="cutlass_mla_cudagraph"

SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
if [[ "$SM_VERSION" != "10.0" ]]; then
    echo "SKIP: ${TEST_NAME}: cutlass_mla_decode requires SM 10.0 exactly; this GPU is SM ${SM_VERSION}"
    exit 0
fi

MODEL_PATH="${MODEL_PATH:-/mnt/vast/models/dsv3-nvfp4}"

LAUNCH_ARGS=(
    --attention-backend cutlass_mla
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 4
)

run_server_smoke
