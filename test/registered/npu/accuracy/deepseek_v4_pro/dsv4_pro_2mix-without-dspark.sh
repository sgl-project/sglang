#!/bin/bash

# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

pkill -9 python  2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 VLLM   2>/dev/null || true

# ===== Environment =====
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


export DEEPEP_HCCL_BUFFSIZE=1536
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

# skip gpu branch
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False


# [DEEPEP]
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=30

# [Prefill Delay]
#export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
#export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

# perfermance
# export SGLANG_NPU_USE_MULTI_STREAM=1


# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# dspark
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0
export SGLANG_DSPARK_QUANT_AUDIT=1
export SGLANG_DSPARK_QUANT_AUDIT_STRICT=0

# path
#export PYTHONPATH=/mnt/share/r00/sglang/python:$PYTHONPATH
MODEL_PATH=/home/weights/DeepSeek-V4-Pro-0813-w4a8
SERVED_MODEL_NAME=dsv4
SERVER_PORT=6677

# ===== Cluster Config ===========================================
# 每台机器: IP + HCCL 网卡名 (一一对应)
NODE_IPS=(
  "192.168.25.216"
  "192.168.25.217"
)
HCCL_IFS=(
  "enp196s0f0"
  "enp196s0f0"
)

NUM_NPUS_PER_NODE=16          # 每机 NPU 数

export GLOO_SOCKET_IFNAME=enp196s0f0
export HCCL_HOST_SOCKET_PORT_RANGE=auto
# ================================================================

MASTER_ADDR="${NODE_IPS[0]}"
MASTER_PORT="5567"
DIST_INIT_ADDR="${MASTER_ADDR}:${MASTER_PORT}"

NNODES=${#NODE_IPS[@]}
TP_SIZE=$(( NNODES * NUM_NPUS_PER_NODE ))
DP_SIZE=$(( NNODES * NUM_NPUS_PER_NODE ))                    # DP 并行度


# ===== Auto-detect node rank by matching local IPs =============
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

NODE_RANK=""
for i in "${!NODE_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${NODE_IPS[$i]}" || "$LOCAL_HOST2" == "${NODE_IPS[$i]}" ]]; then
    NODE_RANK="$i"
    SERVER_HOST="${NODE_IPS[$i]}"
    export HCCL_SOCKET_IFNAME="${HCCL_IFS[$i]}"
    break
  fi
done

if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found in NODE_IPS=[${NODE_IPS[*]}]"
  exit 1
fi

echo "========================================"
echo "Launching GLM5.2 ${NNODES} Nodes"
echo "node-rank       : ${NODE_RANK}"
echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
echo "dist-init-addr  : ${DIST_INIT_ADDR}"
echo "nnodes          : ${NNODES}"
echo "tp-size         : ${TP_SIZE}"
echo "dp-size         : ${DP_SIZE}"
echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
echo "========================================"


# ===== Launch =====
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --tp-size "${TP_SIZE}" \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --watchdog-timeout 9000 \
  --max-running-requests 64 \
  --mem-fraction-static 0.8 \
  --quantization modelslim \
  --chunked-prefill-size 65536 \
  --kv-cache-dtype auto \
  --moe-dense-tp-size 1 \
  --cuda-graph-bs-decode 1 4 \
  --load-balance-method round_robin \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-metrics \
  --dp 16 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --disable-radix-cache
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path ${MODEL_PATH} \
  --speculative-draft-model-quantization modelslim \
  --speculative-draft-attention-backend ascend \
  --speculative-num-draft-tokens 6

