#!/usr/bin/env bash
set -euo pipefail

# HY4 + FlexKV profile for one MI355X node. This is the FlexKV-enabled
# counterpart of launch_hy4_mi355x_cptp.sh from sglang-internal MR !4.
#
#   PARALLEL_MODE=tp       Attention TP + MoE TP (default)
#   PARALLEL_MODE=attn_dp  Attention DP + MoE TP
#   PARALLEL_MODE=attn_cp  Attention CP + MoE TP (experimental for HY4)
#
# Example:
#   FLEXKV_CONFIG_FILE=/path/to/flexkv.yaml \
#   FLEXKV_PYTHONPATH=/path/to/FlexKV \
#   PARALLEL_MODE=tp TP_SIZE=4 \
#   ./scripts/launch_hy4_mi355x_cptp_flexkv.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-tencent/Hy4-preview-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-hy4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-31000}"
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-1}"
PARALLEL_MODE="${PARALLEL_MODE:-tp}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
LOAD_THREADS="${LOAD_THREADS:-16}"
CUDA_GRAPH="${CUDA_GRAPH:-on}"
MTP="${MTP:-off}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FLEXKV_CONFIG_FILE="${FLEXKV_CONFIG_FILE:-${FLEXKV_CONFIG_PATH:-${REPO_ROOT}/python/sglang/srt/mem_cache/storage/flexkv/example_config_mp.yaml}}"
FLEXKV_PYTHONPATH="${FLEXKV_PYTHONPATH:-}"

if [[ ! -r "${FLEXKV_CONFIG_FILE}" ]]; then
  echo "FlexKV config is not readable: ${FLEXKV_CONFIG_FILE}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${FLEXKV_PYTHONPATH}" ]]; then
  export PYTHONPATH="${FLEXKV_PYTHONPATH}:${PYTHONPATH}"
fi
export FLEXKV_CONFIG_PATH="${FLEXKV_CONFIG_FILE}"
export FLEXKV_DEDUP_INDEXER_GROUP="${FLEXKV_DEDUP_INDEXER_GROUP:-1}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-0}"

PARALLEL_ARGS=()
if (( EP_SIZE > 1 )); then
  if (( TP_SIZE % EP_SIZE != 0 )); then
    echo "EP_SIZE=${EP_SIZE} must divide TP_SIZE=${TP_SIZE}" >&2
    exit 2
  fi
  PARALLEL_ARGS+=(--ep-size "${EP_SIZE}")
fi

case "${PARALLEL_MODE}" in
  tp)
    ;;
  attn_dp)
    DP_SIZE="${DP_SIZE:-${TP_SIZE}}"
    if (( TP_SIZE % DP_SIZE != 0 )); then
      echo "DP_SIZE=${DP_SIZE} must divide TP_SIZE=${TP_SIZE}" >&2
      exit 2
    fi
    PARALLEL_ARGS+=(--dp-size "${DP_SIZE}" --enable-dp-attention)
    ;;
  attn_cp)
    CP_SIZE="${CP_SIZE:-${TP_SIZE}}"
    CP_STRATEGY="${CP_STRATEGY:-interleave}"
    if (( TP_SIZE % CP_SIZE != 0 )); then
      echo "CP_SIZE=${CP_SIZE} must divide TP_SIZE=${TP_SIZE}" >&2
      exit 2
    fi
    PARALLEL_ARGS+=(
      --attn-cp-size "${CP_SIZE}"
      --enable-prefill-cp
      --cp-strategy "${CP_STRATEGY}"
    )
    export SGLANG_ENABLE_CP_V2="${SGLANG_ENABLE_CP_V2:-1}"
    ;;
  *)
    echo "Unsupported PARALLEL_MODE=${PARALLEL_MODE}; use tp, attn_dp, or attn_cp" >&2
    exit 2
    ;;
esac

CHAT_ARGS=()
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${MODEL_PATH}/chat_template.jinja}"
if [[ -f "${CHAT_TEMPLATE}" ]]; then
  CHAT_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi

GRAPH_ARGS=()
if [[ "${CUDA_GRAPH}" == "off" ]]; then
  GRAPH_ARGS+=(--disable-cuda-graph)
else
  if [[ "${PARALLEL_MODE}" == "attn_cp" ]]; then
    GRAPH_MAX_BS="${GRAPH_MAX_BS:-${CP_SIZE}}"
    if [[ "${MTP}" == "on" && "${GRAPH_MAX_BS}" -lt "${CP_SIZE}" ]]; then
      echo "GRAPH_MAX_BS must be >= CP_SIZE for attention-CP + MTP" >&2
      exit 2
    fi
  else
    GRAPH_MAX_BS="${GRAPH_MAX_BS:-8}"
  fi
  GRAPH_ARGS+=(--cuda-graph-max-bs-decode "${GRAPH_MAX_BS}")
fi

MTP_ARGS=()
if [[ "${MTP}" == "on" ]]; then
  MTP_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${MTP_STEPS:-1}"
    --speculative-eagle-topk "${MTP_TOPK:-1}"
    --speculative-num-draft-tokens "${MTP_DRAFT_TOKENS:-2}"
  )
fi

echo "[HY4 MI355X FlexKV] model=${MODEL_PATH} mode=${PARALLEL_MODE} tp=${TP_SIZE} ep=${EP_SIZE} port=${PORT} kv=${KV_CACHE_DTYPE}"
echo "[HY4 MI355X FlexKV] config=${FLEXKV_CONFIG_FILE} dedup_indexer_group=${FLEXKV_DEDUP_INDEXER_GROUP}"
if [[ "${PARALLEL_MODE}" == "attn_cp" ]]; then
  echo "[HY4 MI355X FlexKV] cp_size=${CP_SIZE} cp_strategy=${CP_STRATEGY} (experimental)"
elif [[ "${PARALLEL_MODE}" == "attn_dp" ]]; then
  echo "[HY4 MI355X FlexKV] dp_size=${DP_SIZE}"
fi

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP_SIZE}" \
  --load-format safetensors \
  "${CHAT_ARGS[@]}" \
  --reasoning-parser hunyuan \
  --tool-call-parser hunyuan \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --context-length "${CONTEXT_LENGTH}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --disable-custom-all-reduce \
  --dsa-prefill-backend aiter \
  --dsa-decode-backend aiter \
  --moe-runner-backend triton \
  --disable-shared-experts-fusion \
  --model-loader-extra-config \
  "{\"enable_multithread_load\":true,\"num_threads\":${LOAD_THREADS}}" \
  --enable-flexkv \
  --flexkv-config-file "${FLEXKV_CONFIG_FILE}" \
  "${PARALLEL_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "$@"
