#!/usr/bin/env bash
set -euo pipefail

# Eight GPUs: global MoE TP8, two attention TP4 groups, EP1, DFlash.
# Fake-prefill decode launcher for gfx942 with simulated acceptance mean 3.5.
# Default: convert checkpoint MXFP4 experts at load time to FP8 blockscale.
# Native MXFP4 A16W4/A8W4: set SGLANG_MXFP4_TO_FP8_128BLOCK=0 and
# AITER_CONFIG_FMOE=/path/to/flash_tp8_mxfp4_tuned_fmoe.csv. The CSV selects
# A16W4/A8W4 kernels per token count; a mixed recipe need not use A8W4 in decode.
# Requires AITER's gfx942 MXFP4 kernels (aiter PR #16) alongside Flash attention
# support. Recipes must match Flash MoE TP8: model_dim=4096, inter_dim=256,
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
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30001}"
export PYTHONPATH="${SGLANG_ROOT}/python:${AITER_ROOT:-/root/workspace/aiter}${PYTHONPATH:+:${PYTHONPATH}}"

export SGLANG_MXFP4_TO_FP8_128BLOCK="${SGLANG_MXFP4_TO_FP8_128BLOCK:-1}"
export SGLANG_INT4_WEIGHT=0
case "${SGLANG_MXFP4_TO_FP8_128BLOCK}" in
  1)
    moe_weight_description="static FP8 blockscale MoE weights converted from MXFP4 at load time"
    ;;
  0)
    if [[ -z "${AITER_CONFIG_FMOE:-}" ]]; then
      echo "Native MXFP4 requires AITER_CONFIG_FMOE with Flash MoE TP8 A16W4/A8W4 recipes" >&2
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
# Each attention TP4 group has two local draft KV heads. FlyDSL uses BF16 SHUFFLE-5D;
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
export CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-96}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
# DPA divides both limits by DP2: 16K prefill and 96 requests/group.
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-192}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-12345}"
# Match the DFlash accuracy baseline's reserve for draft KV and prefill scratch.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export KV_CACHE_DTYPE=bf16
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" != "0" ]]; then
  echo "Two-batch overlap is not supported by this MiMo DP-attention baseline" >&2
  exit 2
fi

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=0
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0
# Throughput-only simulation: sample bonus-inclusive lengths 3/4 equally.
# 3.5 is the expected mean; a finite run need not average exactly 3.5.
export SGLANG_SIMULATE_ACC_LEN=3.5
export SGLANG_SIMULATE_ACC_METHOD=match-expected

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-5000}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-5000}"
DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-300}"

for name in MAX_RUNNING_REQUESTS TOKENIZER_WORKER_NUM DFLASH_BLOCK_SIZE \
  CHUNKED_PREFILL_SIZE CUDA_GRAPH_MAX_BS_DECODE DECODE_LOG_INTERVAL WATCHDOG_TIMEOUT; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, observed '${!name}'" >&2
    exit 2
  fi
done
if (( MAX_RUNNING_REQUESTS < 2 || CHUNKED_PREFILL_SIZE < 2 )); then
  echo "MAX_RUNNING_REQUESTS and CHUNKED_PREFILL_SIZE must be at least 2 for DP2" >&2
  exit 2
fi
if (( DFLASH_BLOCK_SIZE < 4 )); then
  echo "DFLASH_BLOCK_SIZE must be at least 4 for simulated acceptance length 3.5" >&2
  exit 2
fi
# This env is per worker and is not divided by DP2 in the allocator.
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS="${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS:-$((MAX_RUNNING_REQUESTS / 2))}"
if ! [[ "${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS must be a positive integer" >&2
  exit 2
fi
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
export LOG_FILE="${LOG_FILE:-${SERVER_LOG_FILE:-server_dp2_tp8_ep1_dpa_dflash_decode_fake_prefill.log}}"
if [[ "${LOG_FILE}" = /* ]]; then
  LOG_DIR="$(dirname -- "${LOG_FILE}")"
  LOG_FILE="$(basename -- "${LOG_FILE}")"
fi

seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi

cuda_graph_args=(
  --cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
)
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
echo "Configuration: model=${MODEL_PATH}, global-tp=8, attention-dp=2 x tp=4, ep=1, dp-attention=enabled, dp-lm-head=enabled, max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}"
echo "Per-DP limits: requests=$((MAX_RUNNING_REQUESTS / 2)), chunked-prefill=$((CHUNKED_PREFILL_SIZE / 2))"
echo "Attention: target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, target-swa=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}, draft-swa=${SGLANG_AITER_DFLASH_SWA_IMPL}, qlen1=${SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1}, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-max-bs=${CUDA_GRAPH_MAX_BS_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}"
echo "Speculation: DFLASH, block=${DFLASH_BLOCK_SIZE}, draft=${SPECULATIVE_DRAFT_MODEL_PATH}, draft-attention=aiter/${SGLANG_AITER_DFLASH_SWA_IMPL}, draft-kv=bf16, draft-window=1024, acceptance=simulated"
echo "Optimizations: FlyPA prefill=${SGLANG_FLYPA_MIMO_PREFILL}; compact draft table=${SGLANG_DFLASH_FUSED_COMPACT_KV}; fused bounds checks=${SGLANG_FUSED_ASYNC_OOB_CHECKS}; fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}; fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}"
echo "Simulated acceptance: expected bonus-inclusive length=${SGLANG_SIMULATE_ACC_LEN}, method=${SGLANG_SIMULATE_ACC_METHOD} (throughput only)"
echo "Fake prefill: disaggregation-mode=decode, transfer-backend=fake"
echo "Decode: preallocate-reqs=${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS}, log-interval=${DECODE_LOG_INTERVAL}, watchdog=${WATCHDOG_TIMEOUT}, max-total-tokens=${MAX_TOTAL_TOKENS:-auto}"
echo "Quick Reduce: disabled; mixed-router=${SGLANG_MIMO_MIXED_ROUTER}; custom all-reduce=${custom_all_reduce_status}; radix cache=${radix_cache_status}; tbo=${ENABLE_TWO_BATCH_OVERLAP}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"

server_cmd=("${PYTHON_BIN}" -u -m sglang.launch_server \
  --mm-attention-backend "${MM_ATTENTION_BACKEND}" \
  --model-path "${MODEL_PATH}" \
  --skip-server-warmup \
  --tp-size 8 \
  --dp-size 2 \
  --ep-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
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
