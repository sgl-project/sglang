#!/usr/bin/env bash
# nvidia/GLM-5.2-NVFP4 | release/v0.5.15 | 4xGB300 | attention TP4, MoE TP4
# Identical to ../gb300/server_glm52_v0515_tp4.sh except --context-length is
# parametrized (CONTEXT_LEN) so the ISL ladder can raise it per rung.
# Launched per rung by run_isl_client.sh — not usually run by hand.
set -euo pipefail
: "${HOST:=localhost}"
: "${PORT:=8002}"
: "${CONTEXT_LEN:=90000}"
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1

exec python3 -m sglang.launch_server \
    --model-path nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 4 \
    --quantization modelopt_fp4 \
    --context-length "${CONTEXT_LEN}" \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs-decode 16 \
    --mem-fraction-static 0.87 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --bf16-gemm-backend cutedsl \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --enable-cache-report \
    --host "${HOST}" \
    --port "${PORT}"
