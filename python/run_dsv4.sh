#export CUDA_VISIBLE_DEVICES=0,1,2,3
#/dockerx/data/models/DeepSeek-V4-Flash

#### FP8 model path ####
export SGLANG_REASONING_EFFORT=max

export SGLANG_OPT_USE_FUSED_COMPRESS=false #use PyTorch implemented compressor
export SGLANG_OPT_USE_OLD_COMPRESSOR=true #use old compressor
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false #use old prepare
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false #use old topk
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false #AMD: hash_topk JIT needs CUDA toolchain

export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_HACK_FLASHMLA_BACKEND=tilelang
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false #use old prenorm

export SGLANG_OPT_USE_TILELANG_MHC_PRE=false #use torch hc_pre
export SGLANG_OPT_USE_TILELANG_MHC_POST=false #use torch hc_post

export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

export SGLANG_DSV4_FP4_EXPERTS=false

export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false #non-radix backend has no store_cache method
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false #fused_store_cache JIT needs CUDA toolchain

export SGLANG_FORCE_TRITON_MOE_FP8=1  # this is required to apply swiglu_limit clamp in fused_moe_triton

python3 -m sglang.launch_server \
    --model-path /dockerx/data2/models/DeepSeek-V4-Flash-FP8 \
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

#### FP4 model path ####
#export SGLANG_REASONING_EFFORT=max
#
#export SGLANG_OPT_USE_FUSED_COMPRESS=false #use PyTorch implemented compressor
#export SGLANG_OPT_USE_OLD_COMPRESSOR=true #use old compressor
#export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false #use old prepare
#export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false #use old topk
#export SGLANG_OPT_USE_FUSED_HASH_TOPK=false #AMD: hash_topk JIT needs CUDA toolchain
#
#export SGLANG_HACK_FLASHMLA_BACKEND=tilelang
#export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false #use old prenorm
#
#export SGLANG_OPT_USE_TILELANG_MHC_PRE=false #use torch hc_pre
#export SGLANG_OPT_USE_TILELANG_MHC_POST=false #use torch hc_post
#
#export SGLANG_ENABLE_THINKING=1
#export SGLANG_USE_AITER=1
#export SGLANG_USE_ROCM700A=1
#export SGLANG_TOPK_TRANSFORM_512_TORCH=1
#export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
#
#export SGLANG_DSV4_FP4_EXPERTS=False
#
#export SGLANG_OPT_DPSK_V4_RADIX=0
#export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false #non-radix backend has no store_cache method
#export SGLANG_OPT_USE_FUSED_STORE_CACHE=false #fused_store_cache JIT needs CUDA toolchain
#
#export SGLANG_FORCE_TRITON_MOE_FP8=1  # this is required to apply swiglu_limit clamp in fused_moe_triton
#
#python3 -m sglang.launch_server \
#    --model-path /dockerx/data2/models/DeepSeek-V4-Flash-FP8/ \
#    --trust-remote-code \
#    --tp 8 \
#    --ep 8 \
#    --disable-radix-cache \
#    --attention-backend compressed \
#    --max-running-request 256 \
#    --page-size 256 \
#    --chunked-prefill-size 8192 \
#    --port 8000 \
#    --disable-shared-experts-fusion \
#    --disable-cuda-graph \
#    --tool-call-parser deepseekv4 \
#    --reasoning-parser deepseek-v4
