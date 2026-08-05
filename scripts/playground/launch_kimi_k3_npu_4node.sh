#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
mkdir -p "${LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/weights/RadixArk-Kimi-K3-DSpark}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-192.168.25.209:29600}"
SERVER_PORT="${SERVER_PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
read -ra cuda_graph_bs_decode <<<"${CUDA_GRAPH_BS_DECODE:-1 4 16}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-deepep}"
FUSEEP_MODE="${FUSEEP_MODE:-3}"
LINEAR_ATTN_VERIFY_BACKEND="${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
HCCL_IFNAME="${HCCL_IFNAME:-enp196s0f0}"
CUSTOM_OPP_ROOT="${CUSTOM_OPP_ROOT:-/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer}"
NODE_IPS=(
  "192.168.25.209"
  "192.168.25.212"
  "192.168.25.216"
  "192.168.25.217"
)

node_rank=-1
read -ra local_ips <<<"$(hostname -I)"
for rank in "${!NODE_IPS[@]}"; do
  for local_ip in "${local_ips[@]}"; do
    if [[ "${local_ip}" == "${NODE_IPS[rank]}" ]]; then
      node_rank="${rank}"
      break 2
    fi
  done
done
if ((node_rank < 0)); then
  echo "This host does not match the configured Kimi-K3 node list: ${local_ips[*]}" >&2
  exit 1
fi

# Vendor environment scripts reference optional shell variables directly.
set +o nounset
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -o nounset

export PYTHONPATH="${REPO_DIR}/python:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_K3_SHARED_EXPERTS_ATTN_TP="${SGLANG_K3_SHARED_EXPERTS_ATTN_TP:-1}"
export SGLANG_K3_DENSE_MLP_ATTN_TP="${SGLANG_K3_DENSE_MLP_ATTN_TP:-1}"
export SGLANG_RAGGED_VERIFY_MODE=static
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_SOCKET_IFNAME="${HCCL_IFNAME}"
export GLOO_SOCKET_IFNAME="${HCCL_IFNAME}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2000}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
# DeepEP keeps DP-local verify rows scattered; graph bs16 * 8 rows = 128.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
speculative_args=()
if [[ "${ENABLE_DSPARK:-1}" == "1" ]]; then
  speculative_args=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size 7
    --speculative-draft-attention-backend ascend
    --speculative-eagle-topk 1
    --speculative-draft-model-quantization unquant
  )
fi

cuda_graph_args=(--cuda-graph-bs-decode "${cuda_graph_bs_decode[@]}")
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  cuda_graph_args=(--disable-cuda-graph)
fi

cache_args=()
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  cache_args=(--disable-radix-cache)
fi

moe_args=(--moe-a2a-backend "${MOE_A2A_BACKEND}")
if [[ "${MOE_A2A_BACKEND}" == "ascend_fuseep" ]]; then
  moe_args+=(--fuseep-mode "${FUSEEP_MODE}")
else
  # This legacy variable is translated to --fuseep-mode by server_args.  It
  # must not leak into the regular DeepEP path.
  unset SGLANG_NPU_FUSED_MOE_MODE
fi

# The 20260723 base image may contain the pre-MegaMoe DeepEP Python ABI.
# Only mode-3 Ascend FuseEP needs this ABI; regular DeepEP does not.
if [[ "${MOE_A2A_BACKEND}" == "ascend_fuseep" && "${FUSEEP_MODE}" == "3" ]]; then
  export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}"
  export LD_LIBRARY_PATH="${CUSTOM_OPP_ROOT}/op_api/lib/:${LD_LIBRARY_PATH:-}"
  python - <<'PY'
import inspect

from cann_ops_transformer.ops import mega_moe
from deep_ep import Buffer

if not callable(mega_moe):
    raise RuntimeError("cann_ops_transformer.ops.mega_moe is unavailable")

parameters = inspect.signature(Buffer.fused_deep_moe).parameters
required = {"backend", "activation", "beta", "linear_beta", "l1_bias", "l2_bias"}
missing = sorted(required - parameters.keys())
if missing:
    raise RuntimeError(
        "Kimi-K3 FuseEP mode 3 requires the MegaMoe DeepEP ABI; "
        f"missing fused_deep_moe parameters: {', '.join(missing)}"
    )
PY
fi

log_file="${LOG_DIR}/kimi_k3_tp64_rank${node_rank}_$(date +%Y-%m-%d_%H-%M-%S).log"
sglang serve \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --nnodes 4 \
  --node-rank "${node_rank}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${MODEL_PATH}" \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --quantization modelslim \
  --dtype bfloat16 \
  --tp-size 64 \
  --enable-dp-attention \
  --dp-size 4 \
  --enable-dp-lm-head \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --chunked-prefill-size 8192 \
  "${cuda_graph_args[@]}" \
  "${cache_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --reasoning-parser kimi_k3 \
  --linear-attn-verify-backend "${LINEAR_ATTN_VERIFY_BACKEND}" \
  "${moe_args[@]}" \
  --deepep-mode auto \
  "${speculative_args[@]}" \
  --watchdog-timeout 9000 2>&1 | tee "${log_file}"
