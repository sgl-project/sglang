export SGLANG_OPT_USE_OLD_COMPRESSOR=true
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false

export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false

export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false

export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=true

# changed
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_OPT_USE_TILELANG_INDEXER=true
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_REASONING_EFFORT=max
export SGLANG_FORCE_TRITON_MOE_FP8=0
export SGLANG_OPT_USE_AITER_MHC_PRE=true
export SGLANG_OPT_USE_AITER_MHC_POST=true
export SGLANG_OPT_USE_TRITON_SWA_PREPARE=true
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true

export AITER_BF16_FP8_MOE_BOUND=1

MODEL=/dockerx/data/deepseek-ai/DeepSeek-V4-Pro
#MODEL=/dockerx/data/sgl-project/DeepSeek-V4-Flash-FP8/

python3 -m sglang.launch_server \
    --model-path ${MODEL} \
    --trust-remote-code \
    --tp 8 \
    --disable-radix-cache \
    --attention-backend compressed \
    --max-running-request 256 \
    --page-size 256 \
    --chunked-prefill-size 8192 \
    --port 8000 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4
