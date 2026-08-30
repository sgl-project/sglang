#!/bin/bash

# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_ENABLE_SPEC_V2 SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
unset SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS

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
source /usr/local/memfabric_hybrid/set_env.sh

# ===== Environment =====
export PYTHONPATH=/mnt/share/r00648901/sglang/python:$PYTHONPATH

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


# export HCCL_CONNECT_TIMEOUT=300
# export HCCL_EXEC_TIMEOUT=68
# export HCCL_OP_EXPANSION_MODE=AIV
# export ACL_DEVICE_SYNC_TIMEOUT=60

export MF_HYBM_USE_VMM_SEGMENT=1
export ASCEND_MF_TRANSFER_PROTOCOL="device_urma"
# [DEEPEP]
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export TRANSFORMERS_VERBOSITY=error

# [多机]
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.3001

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# ===== Model Config =====
MODEL_PATH=/mnt/share/y00882530/1600B
SERVED_MODEL_NAME=dsv4

# ===== Cluster Config ===========================================
# Prefill/Decode 每组 4 节点, 每节点 8 NPU => 32 NPU / 组
# Attention: Prefill DP16 x TP2; Decode DP32 x TP1
# ================================================================

# Prefill 节点 (4 nodes)
P_IPS=(
  "141.61.133.101"
  "141.61.133.102"
  "141.61.133.103"
  "141.61.133.104"
)
P_IFS=(
  "enp34s0f1"
  "eth2"
  "enp34s0f1"
  "eth2"
)

# Decode 节点 (4 nodes)
D_IPS=(
  "141.61.133.105"
  "141.61.133.106"
  "141.61.133.108"
  "141.61.133.115"
)
D_IFS=(
  "eth2"
  "eth2"
  "eth2"
  "eth2"
)

NUM_NPUS_PER_NODE=8

P_NNODES=${#P_IPS[@]}
D_NNODES=${#D_IPS[@]}

# 并行度 (enable-dp-attention 模式下总NPU = TP_SIZE):
#   P: TP = P_NNODES * 8 = 32, DP16 => attention TP2
#   D: TP = D_NNODES * 8 = 32, DP32 => attention TP1
P_TP_SIZE=$(( P_NNODES * NUM_NPUS_PER_NODE ))
D_TP_SIZE=$(( D_NNODES * NUM_NPUS_PER_NODE ))
P_DP_SIZE=16
D_DP_SIZE=32
# ================================================================

P_MASTER="${P_IPS[0]}"
D_MASTER="${D_IPS[0]}"
P_DIST_INIT="${P_MASTER}:5567"
D_DIST_INIT="${D_MASTER}:5569"


export ASCEND_MF_STORE_URL="tcp://141.61.133.101:31001"
# export ASCEND_MF_LOG_LEVEL=2
# ===== Auto-detect node by matching local IPs ==================
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

# ===== Launch Prefill nodes ====================================
for i in "${!P_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${P_IPS[$i]}" || "$LOCAL_HOST2" == "${P_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${P_IFS[$i]}"
    export DEEPEP_HCCL_BUFFSIZE=2048
    # [Prefill Delay]
    export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
    export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

    echo "========================================"
    echo "Launching GLM5.2 Prefill node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${P_DIST_INIT}"
    echo "nnodes          : ${P_NNODES}"
    echo "tp-size         : ${P_TP_SIZE}"
    echo "dp-size         : ${P_DP_SIZE}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "boostrap        : $((8998+$i))"
    echo "========================================"

    python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes ${P_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${P_DIST_INIT} \
      --tp-size ${P_TP_SIZE} \
      --dp ${P_DP_SIZE} \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --load-balance-method round_robin \
      --disaggregation-mode prefill \
      --disaggregation-transfer-backend ascend \
      --disaggregation-bootstrap-port 8998 \
      --trust-remote-code \
      --attention-backend ascend \
      --enable-dynamic-batch-tokenizer \
      --tokenizer-worker-num 16 \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 256 \
      --mem-fraction-static 0.83 \
      --quantization fp8 \
      --max-prefill-tokens 2048000 \
      --chunked-prefill-size 65536 \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --disable-cuda-graph \
      --enable-metrics

    exit 1
  fi
done

# ===== Launch Decode nodes =====================================
for i in "${!D_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${D_IPS[$i]}" || "$LOCAL_HOST2" == "${D_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${D_IFS[$i]}"
    export DEEPEP_HCCL_BUFFSIZE=900
    # [MTP]
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1

    echo "========================================"
    echo "Launching GLM5.2 Decode node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${D_DIST_INIT}"
    echo "nnodes          : ${D_NNODES}"
    echo "tp-size         : ${D_TP_SIZE}"
    echo "dp-size         : ${D_DP_SIZE}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"

    python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes ${D_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${D_DIST_INIT} \
      --tp-size ${D_TP_SIZE} \
      --dp ${D_DP_SIZE} \
      --enable-dp-attention \
      --enable-dp-attention-local-control-broadcast \
      --enable-dp-lm-head \
      --load-balance-method round_robin \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend ascend \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 256 \
      --mem-fraction-static 0.86 \
      --quantization fp8 \
      --max-prefill-tokens 2048000 \
      --chunked-prefill-size 16384 \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --cuda-graph-bs 1 2 4 8 \
      --tokenizer-worker-num 8 \
      --speculative-algorithm EAGLE \
      --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
      --enable-metrics

    exit 1
  fi
done

echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found in P_IPS=[${P_IPS[*]}] or D_IPS=[${D_IPS[*]}]"
exit 1

# # ===== Router (在独立节点或任一 P/D 节点上手动执行) ============
# python -m sglang_router.launch_router \
#     --pd-disaggregation --policy cache_aware \
#     --prefill http://141.61.133.130:30000 \
#     --decode http://141.61.133.132:30000 \
#     --host 141.61.133.130 --port 6611
