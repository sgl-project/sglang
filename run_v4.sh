export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true

export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false

export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=true	# change

# changed
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_REASONING_EFFORT=max
export SGLANG_OPT_USE_AITER_MHC_PRE=true
export SGLANG_OPT_USE_AITER_MHC_POST=true
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_FUSE_WQA_WKV=true

export SGLANG_OPT_USE_FUSED_QK_NORM_ROPE=true
export SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL=true
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true

export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_TOPK_V2=false

SGLANG_OPT_USE_JIT_INDEXER_METADATA=false

export AITER_BF16_FP8_MOE_BOUND=1

MODEL=/dockerx/data/models/DeepSeek-V4-Pro

python3 -m sglang.launch_server \
    --model-path ${MODEL} \
    --trust-remote-code \
    --tp 8 \
    --dp 8 \
    --enable-dp-attention \
    --disable-radix-cache \
    --attention-backend compressed \
    --max-running-request 256 \
    --page-size 256 \
    --chunked-prefill-size 8192 \
    --port 8000 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4
