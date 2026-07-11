#!/usr/bin/env bash
# nvidia/GLM-5.2-NVFP4 | day-0 snapshot (22dce5720) | 8xB300 | attention TP8, MoE TP8
#
# The day-0 curve runs the sglang tree from GLM-5.2 launch day, which pre-dates
# this branch. Create it once (from the repo root; the fetch makes the commit
# available even on single-branch clones — it lives on the public glm-opt branch):
#   git fetch origin glm-opt
#   git worktree add ../sglang-day0 22dce572045c277ce46f1a287c4be1112b214368
#   export DAY0_SGLANG=$(cd ../sglang-day0 && pwd)
# then launch with:
#   ./server_day0_tp8.sh
# Flags differ from v0.5.15 on purpose: no --bf16-gemm-backend and no fused
# top-k / deferred-finalize env vars (none of these existed at day-0), and the
# CUDA-graph flag is the day-0 spelling (--cuda-graph-max-bs).
set -euo pipefail
: "${HOST:=localhost}"
: "${PORT:=8002}"
: "${DAY0_SGLANG:?set DAY0_SGLANG to a checkout of 22dce572045c277ce46f1a287c4be1112b214368}"
DAY0_SGLANG=$(cd "$DAY0_SGLANG" && pwd)
if [ "$(git -C "$DAY0_SGLANG" rev-parse HEAD)" != "22dce572045c277ce46f1a287c4be1112b214368" ]; then
    echo "ERROR: DAY0_SGLANG ($DAY0_SGLANG) is not at the day-0 commit 22dce5720" >&2
    exit 1
fi
export PYTHONPATH="$DAY0_SGLANG/python"

exec python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 8 \
    --quantization modelopt_fp4 \
    --context-length 90000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs 16 \
    --mem-fraction-static 0.87 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --host "${HOST}" \
    --port "${PORT}"
