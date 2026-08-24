#!/bin/bash
# ============================================================================
# PD 分离 + Selective HiSparse 启动脚本（A5 / ASCEND_950，memfabric URMA）
#
# 拓扑: Prefill 2 机 16 卡 + Decode 2 机 16 卡
#   P: 141.61.133.101, 141.61.133.103
#   D: 141.61.133.102, 141.61.133.105
#
# 并行策略 (对照 run_8node_pd.sh):
#   P: TP8, PP2, CP8 → attn_tp=1, EP8   (与 8node 单组完全一致)
#   D: TP16, DP16 → attn_tp=1, EP16     (8node 的一半: TP32/DP32/EP32)
# ============================================================================

# cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

pkill -9 python  2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 VLLM   2>/dev/null || true

# ===== Environment =====
source /mnt/share/t00882532/vendors/custom_transformer/bin/set_env.bash
source /usr/local/memfabric_hybrid/set_env.sh
export PYTHONPATH=`pwd`/python:$PYTHONPATH

export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# [FIA]
export ASCEND_USE_FIA=1

# [MLAPO]
export SGLANG_NPU_USE_MLAPO=1
# export SGLANG_NPU_GLM_NEXTN_BF16_KV_CACHE=1

# [DEEPEP]  (EP16: 每 rank token 翻倍 36→72)
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=72

# [Prefill Delay]
#export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
#export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

# [MTP]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export TRANSFORMERS_VERBOSITY=error

# [多机]
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.3001

# A5 PD分离
export MF_HYBM_USE_VMM_SEGMENT=1
export ASCEND_MF_TRANSFER_PROTOCOL="device_urma"

export ASCEND_MF_STORE_URL="tcp://141.61.133.101:31001"
#export ASCEND_MF_LOG_LEVEL=1

unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true

# ===== Model Config =====
#MODEL_PATH_P=/home/cx/glm5_2_weight
MODEL_PATH=/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8
SERVED_MODEL_NAME=glm52

# Selective HiSparse: 选中的卸载层 ID
# 全部 19 层: 5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77
# 首版测试建议先用少量层, 验证通过后逐步扩展:
#   单层:  SELECTIVE_LAYER_IDS="5"
#   4 层:  SELECTIVE_LAYER_IDS="5 21 49 77"
#   全部:  SELECTIVE_LAYER_IDS="5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77"
SELECTIVE_LAYER_IDS=${SELECTIVE_LAYER_IDS:-"5"}
#SELECTIVE_LAYER_IDS=${SELECTIVE_LAYER_IDS:-"5 41 65 77"}
#SELECTIVE_LAYER_IDS=${SELECTIVE_LAYER_IDS:-"5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77"}

# ===== Graph Debug =====
# 当前配置: 单层(layer 5) + 单batch(cuda-graph-bs 1) 用于定位 graph 问题
# Debug 环境变量 (在 Decode 节点设置):
#   export SGLANG_SELECTIVE_SKIP_H2D=1  → graph 模式跳过 H2D DMA
#   export SGLANG_SELECTIVE_SKIP_D2H=1  → graph 模式跳过 D2H DMA
# 二分法: 先试 SKIP_H2D, 再试 SKIP_D2H, 看哪个能消除 507011
# ===== Cluster Config ===========================================
# 4 机: P 2 节点 × 8 卡 + D 2 节点 × 8 卡
#   P: TP8, PP2, CP8 → attn_tp = 1, EP8   (与 8node 生产单组一致)
#   D: TP16, DP16 → attn_tp = 1, EP16
# ================================================================
NUM_NPUS_PER_NODE=8

# Prefill 节点 (2 nodes, 1 group)
P_IPS=(
  "141.61.133.101"
  "141.61.133.103"
)
P_IFS=(
  "enp34s0f1"
  "enp34s0f1"
)

# Decode 节点 (2 nodes)
D_IPS=(
  "141.61.133.102"
  "141.61.133.105"
)
D_IFS=(
  "eth2"
  "eth2"
)

P_NNODES=${#P_IPS[@]}
D_NNODES=${#D_IPS[@]}

# 并行度:
#   P: TP = 8 (per PP stage), PP = 2
#   D: TP = D_NNODES * 8 = 16, DP16
D_TP_SIZE=$(( D_NNODES * NUM_NPUS_PER_NODE ))   # 16
D_DP_SIZE=16
# ================================================================

P_MASTER="${P_IPS[0]}"
D_MASTER="${D_IPS[0]}"
P_DIST_INIT="${P_MASTER}:5567"
D_DIST_INIT="${D_MASTER}:5568"

# ===== Auto-detect node by matching local IPs ==================
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')

# ===== Launch Prefill nodes ====================================
for i in "${!P_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${P_IPS[$i]}" || "$LOCAL_HOST2" == "${P_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${P_IFS[$i]}"

    echo "========================================"
    echo "Launching GLM5.2 Prefill node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${P_DIST_INIT}"
    echo "nnodes          : ${P_NNODES}"
    echo "tp-size         : 8"
    echo "pp-size         : 2"
    echo "attn-cp-size    : 8"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"
#    export DEEPEP_HCCL_BUFFSIZE=2048
    export DEEP_USE_MODE="allgather"
    export SGLANG_PP_LAYER_PARTITION="38,40"
    python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port 30123 \
      --nnodes ${P_NNODES} \
      --node-rank ${i} \
      --dist-init-addr ${P_DIST_INIT} \
      --tp-size 8 \
      --pp-size 2 \
      --enable-nsa-prefill-context-parallel \
      --dsa-prefill-cp-mode in-seq-split \
      --attn-cp-size 8 \
      --moe-dense-tp-size 1 \
      --load-balance-method round_robin \
      --disaggregation-mode prefill \
      --disaggregation-transfer-backend ascend \
      --disaggregation-bootstrap-port 8998 \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 256 \
      --mem-fraction-static 0.86 \
      --tokenizer-worker-num 12 \
      --enable-dynamic-batch-tokenizer \
      --quantization modelslim \
      --max-prefill-tokens 20480000 \
      --chunked-prefill-size 16384 \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode normal \
      --disable-cuda-graph \
      --disable-radix-cache \
      --enable-metrics

    exit 1
  fi
done

# ===== Launch Decode nodes =====================================
for i in "${!D_IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${D_IPS[$i]}" || "$LOCAL_HOST2" == "${D_IPS[$i]}" ]]; then
    export HCCL_SOCKET_IFNAME="${D_IFS[$i]}"

    echo "========================================"
    echo "Launching GLM5.2 Decode node ${i}"
    echo "node-rank       : ${i}"
    echo "local IPs       : ${LOCAL_HOST1} ${LOCAL_HOST2}"
    echo "dist-init-addr  : ${D_DIST_INIT}"
    echo "nnodes          : ${D_NNODES}"
    echo "tp-size         : ${D_TP_SIZE}"
    echo "dp-size         : ${D_DP_SIZE}"
    echo "selective layers: ${SELECTIVE_LAYER_IDS}"
    echo "HCCL interface  : ${HCCL_SOCKET_IFNAME}"
    echo "GLOO interface  : ${GLOO_SOCKET_IFNAME}"
    echo "========================================"
    export DEEPEP_HCCL_BUFFSIZE=512
    export HCCL_BUFFSIZE=140
    # LM-head TP
    export SGLANG_LM_HEAD_TP=8
#    export SGLANG_ATTN_O_TP_SIZE=4
    export SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=1
#    export SGLANG_NPU_PROFILING=1

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
      --moe-dense-tp-size 1 \
      --load-balance-method round_robin \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend ascend \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --disable-radix-cache \
      --watchdog-timeout 9000 \
      --max-running-requests 192 \
      --mem-fraction-static 0.945 \
      --quantization modelslim \
      --kv-cache-dtype "fp8_e4m3" \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --cuda-graph-bs 1 2 3 4 5 6 \
      --context-len 133120 \
      --tokenizer-worker-num 4 \
      --enable-dynamic-batch-tokenizer \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
      --speculative-draft-kv-cache-dtype bf16 \
      --npu-selective-hisparse-layer-ids ${SELECTIVE_LAYER_IDS} \
      --enable-metrics

    exit 1
  fi
done

echo "ERROR: local IPs [${LOCAL_HOST1} ${LOCAL_HOST2}] not found in P_IPS=[${P_IPS[*]}] or D_IPS=[${D_IPS[*]}]"
exit 1

# ===== Router (在独立节点或任一 P/D 节点上手动执行) ============
# python -m sglang_router.launch_router \
#     --pd-disaggregation --policy cache_aware \
#     --prefill http://141.61.133.101:30123 \
#     --decode http://141.61.133.102:30000 \
#     --host 0.0.0.0 --port 6677
