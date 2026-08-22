#!/bin/bash
# Single-box repro: prefill-only + CP + EAGLE, row-clobber hunt.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export USE_NPU_MOE_GATING_TOP_K=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=8
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:14699"
export SGLANG_ZBAL_LOCAL_MEM_SIZE=62084
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_DEBUG_MEMORY_POOL=1

MODEL_PATH=/mnt/paas/weights/DeepSeek-V4-Flash-0731-w8a8
LOG=/home/w00889861/dsv4/sglang_pcp/sglang/logs_single_repro.log

nohup python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 --tp-size 8 --trust-remote-code --device npu \
    --attention-backend dsv4 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.62 \
    --prefill-max-requests 128 --max-prefill-tokens 70000 --max-running-requests 64 \
    --max-running-requests 128 \
    --chunked-prefill-size 16384 \
    \
    --moe-a2a-backend deepep --deepep-mode normal \
    --quantization modelslim \
    --kv-cache-dtype bfloat16 \
    --disable-cuda-graph \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-prefill-cp --cp-strategy interleave \
    > ${LOG} 2>&1 &
echo "launched, pid=$!, log=${LOG}"
