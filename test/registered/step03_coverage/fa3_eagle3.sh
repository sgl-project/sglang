#!/bin/bash
# triton + EAGLE3 (multi-layer draft) + CUDA graph.
# Uses Llama-3.1-8B target + sglang EAGLE3 draft. fa3 would be ideal
# here but the cluster GPUs are Blackwell (SM>=10) — triton is the
# next-best portable choice for the smoke.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="fa3_eagle3"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

# On Blackwell pick triton; on Hopper use fa3.
SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
if [[ "$SM_VERSION" == "10.0" || "$SM_VERSION" == "10.3" || "$SM_VERSION" == "12.0" ]]; then
    BACKEND="triton"
else
    BACKEND="fa3"
fi

# Canonical EAGLE3 sglang config uses fp16 + triton, because
# bf16 + flashinfer cutlass RMSNorm hits a dtype mismatch on the
# draft model's input_layernorm on Blackwell (SM120).
# See test/registered/core/test_basic_sanity_eagle3.py for context.
LAUNCH_ARGS=(
    --dtype float16
    --attention-backend triton
    --speculative-algorithm EAGLE3
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

EXTRA_ENV=(
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
)

READY_TIMEOUT=2400
run_server_smoke
