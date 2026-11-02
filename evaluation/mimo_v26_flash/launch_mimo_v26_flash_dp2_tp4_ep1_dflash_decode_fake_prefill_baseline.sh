#!/usr/bin/env bash
set -euo pipefail

# Standalone MiMo-V2.6-Flash fake-prefill decode launcher for gfx942:
# Default: convert checkpoint MXFP4 experts at load time to FP8 blockscale.
# Native MXFP4 A16W4/A8W4: set SGLANG_MXFP4_TO_FP8_128BLOCK=0 and
# AITER_CONFIG_FMOE=/path/to/flash_tp4_mxfp4_tuned_fmoe.csv. The CSV selects
# A16W4/A8W4 kernels per token count; a mixed recipe need not use A8W4 in decode.
# Requires AITER's gfx942 MXFP4 kernels (aiter PR #16) alongside Flash attention
# support. Recipes must match Flash TP4: model_dim=4096, inter_dim=512,
# expert=256, topk=8; Pro TP8 recipes have different shapes.
# Match the validated accuracy launcher's FlyDSL and fusion defaults. Fused
# MoE RMSNorm requires AITER's N=4096/6144 kernel and a rebuilt RMS module.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(dirname -- "${SCRIPT_DIR}")"
SGLANG_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Accept the older decode-launcher names while preferring baseline names.
if [[ -z "${MODEL_PATH:-}" && -n "${MODEL:-}" ]]; then
  export MODEL_PATH="${MODEL}"
fi
if [[ -z "${SERVER_HOST:-}" && -n "${HOST:-}" ]]; then
  export SERVER_HOST="${HOST}"
fi
if [[ -z "${SERVER_PORT:-}" && -n "${PORT:-}" ]]; then
  export SERVER_PORT="${PORT}"
fi
if [[ -z "${SPECULATIVE_DRAFT_MODEL_PATH:-}" && -n "${SPECULATIVE_DRAFT_MODEL:-}" ]]; then
  export SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL}"
fi

MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD/}"
export SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-${MODEL_PATH%/}/dflash}"
export DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-8}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-30001}"
if [[ -n "${AITER_ROOT:-}" ]]; then
  export PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

export SGLANG_MXFP4_TO_FP8_128BLOCK="${SGLANG_MXFP4_TO_FP8_128BLOCK:-1}"
export SGLANG_INT4_WEIGHT=0
case "${SGLANG_MXFP4_TO_FP8_128BLOCK}" in
  1)
    moe_weight_description="static FP8 blockscale MoE weights converted from MXFP4 at load time"
    ;;
  0)
    if [[ -z "${AITER_CONFIG_FMOE:-}" ]]; then
      echo "Native MXFP4 requires AITER_CONFIG_FMOE with Flash TP4 A16W4/A8W4 recipes" >&2
      exit 2
    fi
    if [[ "${DRY_RUN:-0}" != "1" && ! -r "${AITER_CONFIG_FMOE}" ]]; then
      echo "AITER_CONFIG_FMOE is not readable: ${AITER_CONFIG_FMOE}" >&2
      exit 2
    fi
    export AITER_CONFIG_FMOE
    # The fused MoE RMSNorm path produces per-1x128 FP8 activations, not the
    # BF16 input expected by the native A16W4/A8W4 MoE wrappers.
    export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
    moe_weight_description="native MXFP4 MoE weights; A16W4/A8W4 selected by AITER_CONFIG_FMOE"
    ;;
  *)
    echo "SGLANG_MXFP4_TO_FP8_128BLOCK must be 0 or 1" >&2
    exit 2
    ;;
esac
# MiMo's encoder otherwise defaults to a CUDA-only attention implementation.
export MM_ATTENTION_BACKEND=aiter_attn
# gfx942 CK LINEAR prefill has no single-token generation-health kernel.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_ENABLE_ASYNC_ASSERT=1
export SGLANG_FUSED_ASYNC_OOB_CHECKS="${SGLANG_FUSED_ASYNC_OOB_CHECKS:-1}"

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1

# Match the Flash accuracy baseline's communication and attention settings.
unset ROCM_QUICK_REDUCE_QUANTIZATION
export SGLANG_MIMO_MIXED_ROUTER=0
export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_AITER_UNIFIED_VERIFY=1
# Flash TP4 has two local draft KV heads. FlyDSL uses BF16 SHUFFLE-5D;
# setting SGLANG_AITER_PA_DECODE_IMPL=gluon restores the NHD draft fallback.
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export SGLANG_DFLASH_FUSED_COMPACT_KV="${SGLANG_DFLASH_FUSED_COMPACT_KV:-1}"
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL="${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
export SGLANG_AITER_DFLASH_SWA_IMPL="${SGLANG_AITER_DFLASH_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
# FlyDSL handles Qlen8 directly; retain qlen-1 expansion for the Gluon fallback.
export SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE=bf16
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
export SGLANG_TBO_MIM_SEQ_LEN="${SGLANG_TBO_MIM_SEQ_LEN:-2000}"

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=0
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0
# Fake-prefill benchmark's deterministic DFlash acceptance.
export SGLANG_SIMULATE_ACC_LEN=4
export SGLANG_SIMULATE_ACC_METHOD=match-expected

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-5000}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-5000}"
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS="${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS:-${MAX_RUNNING_REQUESTS}}"
DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-300}"

for name in MAX_RUNNING_REQUESTS TOKENIZER_WORKER_NUM DFLASH_BLOCK_SIZE \
  SGLANG_TBO_MIM_SEQ_LEN DECODE_LOG_INTERVAL WATCHDOG_TIMEOUT \
  SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, observed '${!name}'" >&2
    exit 2
  fi
done
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]] && ! [[ "${MAX_TOTAL_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOTAL_TOKENS must be a positive integer when set" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${EVALUATION_DIR}/logs/mimo_v26_flash_decode_fake_prefill_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-${SERVER_LOG_FILE:-server_dp2_tp4_ep1_dflash_decode_fake_prefill.log}}"
if [[ "${LOG_FILE}" = /* ]]; then
  LOG_DIR="$(dirname -- "${LOG_FILE}")"
  LOG_FILE="$(basename -- "${LOG_FILE}")"
fi

seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi

custom_all_reduce_args=()
custom_all_reduce_status=enabled
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  custom_all_reduce_args+=(--disable-custom-all-reduce)
  custom_all_reduce_status=disabled
fi

tbo_args=()
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" == "1" ]]; then
  tbo_args+=(--enable-two-batch-overlap)
fi

radix_cache_args=()
radix_cache_status=enabled
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  radix_cache_status=disabled
fi

decode_args=(
  --disaggregation-mode decode
  --disaggregation-transfer-backend fake
  --decode-log-interval "${DECODE_LOG_INTERVAL}"
  --watchdog-timeout "${WATCHDOG_TIMEOUT}"
)
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  decode_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

echo "MiMo-V2.6-Flash: ${moe_weight_description}"
echo "MoE recipe file: ${AITER_CONFIG_FMOE:-auto}"
echo "Configuration: model=${MODEL_PATH}, tp=4, dp=2, ep=1, max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}"
echo "Attention: target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, target-swa=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}, draft-swa=${SGLANG_AITER_DFLASH_SWA_IMPL}, qlen1=${SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1}, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}"
echo "Speculation: DFLASH, block=${DFLASH_BLOCK_SIZE}, draft=${SPECULATIVE_DRAFT_MODEL_PATH}, draft-attention=aiter/${SGLANG_AITER_DFLASH_SWA_IMPL}, draft-kv=bf16, draft-window=1024, acceptance=simulated"
echo "Optimizations: FlyPA prefill=${SGLANG_FLYPA_MIMO_PREFILL}; compact draft table=${SGLANG_DFLASH_FUSED_COMPACT_KV}; fused bounds checks=${SGLANG_FUSED_ASYNC_OOB_CHECKS}; fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}; fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}"
echo "Simulated acceptance: length=${SGLANG_SIMULATE_ACC_LEN}, method=${SGLANG_SIMULATE_ACC_METHOD}"
echo "Fake prefill: disaggregation-mode=decode, transfer-backend=fake"
echo "Decode: preallocate-reqs=${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS}, log-interval=${DECODE_LOG_INTERVAL}, watchdog=${WATCHDOG_TIMEOUT}, max-total-tokens=${MAX_TOTAL_TOKENS:-auto}"
echo "Quick Reduce: disabled; mixed-router=${SGLANG_MIMO_MIXED_ROUTER}; custom all-reduce=${custom_all_reduce_status}; radix cache=${radix_cache_status}; tbo=${ENABLE_TWO_BATCH_OVERLAP}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"

server_cmd=("${PYTHON_BIN}" -u -m sglang.launch_server \
  --mm-attention-backend "${MM_ATTENTION_BACKEND}" \
  --model-path "${MODEL_PATH}" \
  --tp-size 4 \
  --dp-size 2 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --moe-runner-backend aiter \
  --speculative-moe-a2a-backend none \
  --speculative-moe-runner-backend aiter \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}" \
  --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE}" \
  --speculative-draft-attention-backend aiter \
  --speculative-draft-kv-cache-dtype bf16 \
  --speculative-draft-window-size 1024 \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --attention-backend aiter \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --page-size 64 \
  "${tbo_args[@]}" \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
  "${decode_args[@]}" \
  "$@" \
)

printf 'Command:'
printf ' %q' "${server_cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p -- "${LOG_DIR}"
"${server_cmd[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
