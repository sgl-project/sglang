#!/bin/bash
# Cross-node PD: decode role. Runs on the decode node inside its container.
# Listens on $HOST_IP:30100; reaches prefill at $PREFILL_IP:8998 (bootstrap).
set -ex

: "${HOST_IP:?HOST_IP required - this nodes RoCE-routable IP}"
: "${MODEL_PATH:=/data/models/DeepSeek-V4-Flash-MXFP4}"
: "${TP_SIZE:=8}"

# RDMA NICs: auto-discover the rdma* devices on this node (Pensando RoCE
# enumerates them as rdma0..rdma7). Override by exporting IB_DEVS.
IB_DEVS="${IB_DEVS:-$(ibv_devices 2>/dev/null | awk 'NR>2 && $1 ~ /^rdma/ {print $1}' | sort -V | paste -sd,)}"
: "${IB_DEVS:?could not auto-discover rdma* devices; set IB_DEVS explicitly}"

# Cross-node TCP control plane (gloo/NCCL bootstrap). eno0 is the routable mgmt
# iface on these nodes; override with SOCKET_IFNAME if different.
: "${SOCKET_IFNAME:=eno0}"

# KV transfer backend: mori (default) or mooncake.
# mooncake requires an image with the #27730 mooncake bump (d8f35569, which
# carries #2346 — the QP-teardown wild-pointer segfault fix). Older images ship
# mooncake v0.3.7.post2 which segfaults at ibv_destroy_qp on disconnect.
: "${TRANSFER_BACKEND:=mori}"

if [ "$TRANSFER_BACKEND" = "mori" ]; then
  # MORI QP caps. Pensando reports max_qp_wr=65535 / max_sge=8 so these defaults
  # are well within limits; all overridable for tuning.
  export MORI_IO_XGMI_SCATTER_GATHER_THRESHOLD="${MORI_IO_XGMI_SCATTER_GATHER_THRESHOLD:-4}"
  export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-4096}"
  export MORI_IO_QP_MAX_CQE="${MORI_IO_QP_MAX_CQE:-32768}"
  export MORI_IO_QP_MAX_SGE="${MORI_IO_QP_MAX_SGE:-4}"
  export SGLANG_MORI_QP_PER_TRANSFER="${SGLANG_MORI_QP_PER_TRANSFER:-4}"
  export SGLANG_MORI_NUM_WORKERS="${SGLANG_MORI_NUM_WORKERS:-4}"
fi

export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_AITER_AG=0
export AITER_ENABLE_EXPERIMENTAL=1
export AITER_DISABLE_FLYDSL=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_AITER_MLA_PERSIST=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_USE_ROCM700A=1

export NCCL_IB_RETRY_CNT=15
export NCCL_IB_TIMEOUT=22
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export NCCL_IB_HCA="$IB_DEVS"
export NCCL_CROSS_NIC=1
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name deepseek \
  --host "$HOST_IP" \
  --port 30100 \
  --tensor-parallel-size "$TP_SIZE" \
  --trust-remote-code \
  --attention-backend aiter \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
  --disaggregation-ib-device "$IB_DEVS" \
  --disaggregation-bootstrap-port 9001 \
  --chunked-prefill-size 131072 \
  --max-prefill-tokens 131072 \
  --mem-fraction-static 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 32 \
  --cuda-graph-max-bs 32 \
  --context-length 200000 \
  --stream-interval 30 \
  --schedule-policy lpm \
  --watchdog-timeout 10000 \
  --show-time-cost \
  --enable-metrics \
  --enable-cache-report
