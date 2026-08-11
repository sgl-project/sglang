#!/usr/bin/env bash
set -euo pipefail

# RedNote dots3.note serving recipe for one 8-GPU node.
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the dots3.note checkpoint directory}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
DP_SIZE="${DP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-393216}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.87}"
CUDA_GRAPH_MIN_MEMORY_MIB="${CUDA_GRAPH_MIN_MEMORY_MIB:-120000}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"
SGLANG_BIN="${SGLANG_BIN:-sglang}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${SGLANG_BIN}" >/dev/null 2>&1; then
  echo "SGLang CLI not found: ${SGLANG_BIN}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to select the CUDA graph profile" >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR == 1 { min = $1 } $1 < min { min = $1 } END { print min }')"
if [[ ! "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Unable to determine GPU memory from nvidia-smi: ${GPU_MEMORY_MIB}" >&2
  exit 1
fi
if [[ ! "${CUDA_GRAPH_MIN_MEMORY_MIB}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_GRAPH_MIN_MEMORY_MIB must be an integer" >&2
  exit 1
fi

if (( GPU_MEMORY_MIB >= CUDA_GRAPH_MIN_MEMORY_MIB )); then
  DEFAULT_CUDA_GRAPH_MODE="decode"
  DEFAULT_CUDA_GRAPH_MAX_BS_DECODE=32
elif [[ "${LANGUAGE_ONLY:-0}" == "1" ]]; then
  DEFAULT_CUDA_GRAPH_MODE="decode"
  DEFAULT_CUDA_GRAPH_MAX_BS_DECODE=16
else
  DEFAULT_CUDA_GRAPH_MODE="disabled"
  DEFAULT_CUDA_GRAPH_MAX_BS_DECODE=16
fi
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-${DEFAULT_CUDA_GRAPH_MODE}}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-${DEFAULT_CUDA_GRAPH_MAX_BS_DECODE}}"

MODEL_CONFIG_PATH="${MODEL_PATH}/config.json"
if [[ ! -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "Model config not found: ${MODEL_CONFIG_PATH}" >&2
  exit 1
fi

CHECKPOINT_PRECISION="$("${PYTHON_BIN}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)
quant_config = config.get("quantization_config", config.get("quant_config"))
print("quantized" if quant_config is not None else "bf16")
' "${MODEL_CONFIG_PATH}")"

if [[ "${CHECKPOINT_PRECISION}" == "bf16" ]]; then
  MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-deep_gemm}"
  DEEPEP_DISPATCHER_OUTPUT_DTYPE="${DEEPEP_DISPATCHER_OUTPUT_DTYPE:-bf16}"
else
  MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-auto}"
  DEEPEP_DISPATCHER_OUTPUT_DTYPE="${DEEPEP_DISPATCHER_OUTPUT_DTYPE:-auto}"
fi

echo "Detected ${CHECKPOINT_PRECISION} checkpoint; MoE runner=${MOE_RUNNER_BACKEND}, DeepEP dispatcher dtype=${DEEPEP_DISPATCHER_OUTPUT_DTYPE}."

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-8192}"
export SGLANG_MAX_KV_CHUNK_CAPACITY="${SGLANG_MAX_KV_CHUNK_CAPACITY:-8192}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-1800}"

EXTRA_SERVER_ARGS=()
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--disable-radix-cache)
fi
if [[ "${LANGUAGE_ONLY:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--language-only)
fi

if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  CUDA_GRAPH_MODE="disabled"
fi
case "${CUDA_GRAPH_MODE}" in
  decode)
    DEEPEP_MODE="${DEEPEP_MODE:-auto}"
    CUDA_GRAPH_ARGS=(
      --cuda-graph-backend-decode full
      --cuda-graph-backend-prefill disabled
      --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
    )
    ;;
  disabled)
    DEEPEP_MODE="${DEEPEP_MODE:-normal}"
    CUDA_GRAPH_ARGS=(
      --cuda-graph-backend-decode disabled
      --cuda-graph-backend-prefill disabled
    )
    ;;
  *)
    echo "CUDA_GRAPH_MODE must be one of: decode, disabled" >&2
    exit 1
    ;;
esac
echo "GPU=${GPU_NAME} (${GPU_MEMORY_MIB} MiB minimum); decode CUDA graph mode=${CUDA_GRAPH_MODE}; prefill CUDA graph is disabled."

SPECULATIVE_ARGS=()
if [[ "${DISABLE_SPECULATIVE:-0}" != "1" ]]; then
  SPECULATIVE_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    --speculative-draft-model-path "${MODEL_PATH}"
    --speculative-draft-attention-backend fa3
  )
fi

if [[ "${DISABLE_DSA:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--json-model-override-args '{"index_topk":null}')
fi

exec "${SGLANG_BIN}" serve \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --enable-dp-attention \
  --dp-size "${DP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --ep-size "${EP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}" \
  --trust-remote-code \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.03}" \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --page-size 64 \
  --moe-dense-tp-size 1 \
  --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  --moe-a2a-backend deepep \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  --deepep-dispatcher-output-dtype "${DEEPEP_DISPATCHER_OUTPUT_DTYPE}" \
  --deepep-mode "${DEEPEP_MODE}" \
  --enable-nccl-nvls \
  --enable-multimodal \
  --enable-metrics \
  "${EXTRA_SERVER_ARGS[@]}"
