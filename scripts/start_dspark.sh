#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV

# skip gpu branch
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

# mtp
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# path
#export PYTHONPATH=/home/z50065439/sglang-main/sglang/python:$PYTHONPATH
export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH

MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1

export DEEPEP_HCCL_BUFFSIZE=2500
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=60

#
export SGLANG_RAGGED_VERIFY_MODE=static # static, CAP_ACCEPT, compact
export SGLANG_DSPARK_FAST_KERNEL=0
#export SGLANG_NPU_USE_MULTI_STREAM=1

export DEEPEP_HYBRID_DEPLOYMENT=1

# 不确定是否有影响
#export SGLANG_DEFAULT_THINKING=1
#export SGLANG_DSV4_REASONING_EFFORT=max

# 启动参数


# 设置默认 IP，可通过环境变量 SGLANG_HOST_IP 覆盖
IP=("${SGLANG_HOST_IP:-0.0.0.0}")


python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host "${IP[0]}" --port 30100 \
    --mem-fraction-static 0.68 \
    --prefill-max-requests 160 \
    --max-prefill-tokens 80000 \
    --chunked-prefill-size 131072 \
    --max-running-requests 160 \
    --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim --enable-dp-lm-head \
    --kv-cache-dtype bfloat16 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5 \
    --cuda-graph-bs 1 2 4 8 10 \



#    --speculative-algorithm EAGLE \
#    --speculative-num-steps 2 \
#    --speculative-eagle-topk 1 \
#    --speculative-num-draft-tokens 3