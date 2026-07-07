#!/bin/bash
# gfx1250 bring-up launch script for DeepSeek-V4-Flash (single-card, TP=1, port 8100)
MODEL="${MODEL:-/data/models/DeepSeek-V4-Flash}"
PORT="${PORT:-8100}"

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
# Force the FlyDSL grouped MoE onto the a8w4 (fp8 activation)
export AITER_FORCE_A8W4=1
export SGLANG_USE_AITER_MOE_GU_ITLV=0
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0
export ENABLE_CK=0
set -x
exec sglang serve \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp 1 \
    --attention-backend dsv4 \
    --page-size 256 \
    --mem-fraction-static 0.60 \
    --swa-full-tokens-ratio 0.15 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs 512 \
    --max-running-requests 512 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --kv-cache-dtype fp8_e4m3 \
    --port "${PORT}"
