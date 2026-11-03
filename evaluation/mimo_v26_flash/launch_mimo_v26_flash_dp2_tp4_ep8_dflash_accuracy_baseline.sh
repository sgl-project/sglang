#!/usr/bin/env bash
set -euo pipefail

# MiMo-V2.6: convert checkpoint MXFP4 experts once at load time to FP8
# 128x128 blockscale weights. Activations remain dynamically quantized.
# Global TP8 / EP8, DP2 with attention TP4, and optional DFlash block 8.
# ENABLE_DFLASH=1 (default) runs a TP4 draft in each DP group; 0 is target-only.
# Defaults: global chunk 32768 and request limit 192, or 16384 / 96 per DP.
# PERFORMANCE_MODE=1 uses fake routing for performance tests.
# PERFORMANCE_MODE=0 selects real routing and ordinary serving for accuracy.
# ENABLE_PREFILL_DELAYER=1 coordinates new prefill admission; set 0 for the prior baseline.
# Additional server options may be passed as positional arguments.
# DRY_RUN=1 prints the command
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD/}"
export ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
export SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-${MODEL_PATH%/}/dflash}"
export DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-8}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-30001}"
if [[ -n "${AITER_ROOT:-}" ]]; then
  export PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

export SGLANG_MXFP4_TO_FP8_128BLOCK=1
export SGLANG_INT4_WEIGHT=0
# MiMo's encoder otherwise defaults to a CUDA-only attention implementation.
export MM_ATTENTION_BACKEND=aiter_attn
# gfx942 CK LINEAR prefill has no single-token generation-health kernel.
# Use /health for process readiness; validate generation with real prompts.
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
# Baseline contract: never inherit the lossy Quick Reduce path or mixed-router
# optimization from the caller. Ablations must use a separate launcher.
unset ROCM_QUICK_REDUCE_QUANTIZATION
export SGLANG_MIMO_MIXED_ROUTER=0

# Validated native paged FlyPA full/SWA prefill. Set 0 for a CK comparison.
export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL=0

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_AITER_UNIFIED_VERIFY=1
# Flash TP4 has two local draft KV heads. FlyDSL uses BF16 SHUFFLE-5D
# draft KV; the Gluon fallback retains the existing NHD draft layout.
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export SGLANG_DFLASH_FUSED_COMPACT_KV="${SGLANG_DFLASH_FUSED_COMPACT_KV:-1}"
# Set SGLANG_AITER_PA_DECODE_IMPL=gluon to restore the Gluon target and NHD draft.
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL="${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
export SGLANG_AITER_DFLASH_SWA_IMPL="${SGLANG_AITER_DFLASH_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
# Retain qlen-1 expansion for the Gluon fallback. FlyDSL consumes Qlen8
# directly and uses native V128 for both Flash TP4 target KV pools.
export SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-192}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
# Optional reproducibility control; unset preserves SGLang's seed selection.
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export KV_CACHE_DTYPE=bf16
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export PERFORMANCE_MODE="${PERFORMANCE_MODE:-1}"
export ENABLE_PREFILL_DELAYER="${ENABLE_PREFILL_DELAYER:-1}"
# This is a count of scheduler negotiations, not a delay in milliseconds.
export PREFILL_DELAYER_MAX_DELAY_PASSES="${PREFILL_DELAYER_MAX_DELAY_PASSES:-100}"

for name in MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2
    exit 2
  fi
done
if [[ "${PERFORMANCE_MODE}" != "0" && "${PERFORMANCE_MODE}" != "1" ]]; then
  echo "PERFORMANCE_MODE must be 0 or 1" >&2
  exit 2
fi
if [[ "${ENABLE_DFLASH}" != "0" && "${ENABLE_DFLASH}" != "1" ]]; then
  echo "ENABLE_DFLASH must be 0 or 1" >&2
  exit 2
fi
if [[ "${ENABLE_PREFILL_DELAYER}" != "0" && "${ENABLE_PREFILL_DELAYER}" != "1" ]]; then
  echo "ENABLE_PREFILL_DELAYER must be 0 or 1" >&2
  exit 2
fi
if [[ "${ENABLE_PREFILL_DELAYER}" == "1" ]] && ! [[ "${PREFILL_DELAYER_MAX_DELAY_PASSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREFILL_DELAYER_MAX_DELAY_PASSES must be a positive integer" >&2
  exit 2
fi

if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer" >&2
  exit 2
fi
if [[ "${ENABLE_DFLASH}" == "1" ]] && ! [[ "${DFLASH_BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DFLASH_BLOCK_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

# MORI quantizes dispatch input. QKV fused quant also enables packed FP8 AG.
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
# Fuse selected-expert sigmoid weights and stable normalization; set 0 to disable.
export SGLANG_MIMO_FUSED_STABLE_ROUTER_WEIGHTS="${SGLANG_MIMO_FUSED_STABLE_ROUTER_WEIGHTS:-0}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=0
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0

# AUTO loads the bundled gfx942 EP8 tuning rules. Use MANUAL for a baseline.
export MORI_EP_LAUNCH_CONFIG_MODE="${MORI_EP_LAUNCH_CONFIG_MODE:-AUTO}"
export MORI_EP_TUNING_CONFIG="${MORI_EP_TUNING_CONFIG:-${SCRIPT_DIR}/gfx942_mimo_v26_flash_ep8_mori_a2a.json}"
if [[ "${MORI_EP_LAUNCH_CONFIG_MODE}" == "AUTO" && ! -r "${MORI_EP_TUNING_CONFIG}" ]]; then
  echo "MORI tuning config is not readable: ${MORI_EP_TUNING_CONFIG}" >&2
  exit 2
fi
export MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-ISOLATION}"
export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-0}"
export SGLANG_MORI_DISPATCH_DTYPE="${SGLANG_MORI_DISPATCH_DTYPE:-fp8}"
export SGLANG_MORI_COMBINE_DTYPE="${SGLANG_MORI_COMBINE_DTYPE:-bf16}"

# DP2 divides the global chunk and request limit between two workers.
# Bound decode by one token/request, or the DFlash verify block when enabled.
# Keep the conservative unscattered bound so graph padding fits the buffer.
effective_prefill_chunk=$((CHUNKED_PREFILL_SIZE / 2))
requested_max_running_per_dp=$((MAX_RUNNING_REQUESTS / 2))
decode_tokens_per_request=1
if [[ "${ENABLE_DFLASH}" == "1" ]]; then
  decode_tokens_per_request=${DFLASH_BLOCK_SIZE}
fi
verify_token_bound=$((MAX_RUNNING_REQUESTS * decode_tokens_per_request))
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a graph_sizes <<< "${CUDA_GRAPH_BS_DECODE}"
  for graph_bs in "${graph_sizes[@]}"; do
    if ! [[ "${graph_bs}" =~ ^[1-9][0-9]*$ ]]; then
      echo "CUDA_GRAPH_BS_DECODE must contain positive integers" >&2
      exit 2
    fi
    if (( graph_bs * decode_tokens_per_request > verify_token_bound )); then
      verify_token_bound=$((graph_bs * decode_tokens_per_request))
    fi
  done
fi
mori_capacity_default=${effective_prefill_chunk}
if (( verify_token_bound > mori_capacity_default )); then
  mori_capacity_default=${verify_token_bound}
fi
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-${mori_capacity_default}}"
if ! [[ "${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}" =~ ^[1-9][0-9]*$ ]] ||
   (( SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK < effective_prefill_chunk ||
      SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK < verify_token_bound )); then
  echo "MORI dispatch capacity must cover effective prefill chunk ${effective_prefill_chunk} and conservative verify bound ${verify_token_bound}" >&2
  exit 2
fi

performance_args=()
if [[ "${PERFORMANCE_MODE}" == "1" ]]; then
  performance_args+=(--ep-dispatch-algorithm fake)
fi

prefill_delayer_args=()
if [[ "${ENABLE_PREFILL_DELAYER}" == "1" ]]; then
  prefill_delayer_args=(
    --enable-prefill-delayer
    --prefill-delayer-max-delay-passes "${PREFILL_DELAYER_MAX_DELAY_PASSES}"
  )
fi

dflash_args=()
speculative_status=disabled
if [[ "${ENABLE_DFLASH}" == "1" ]]; then
  speculative_status=DFLASH
  dflash_args=(
    --speculative-moe-a2a-backend none
    --speculative-moe-runner-backend aiter
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}"
    --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE}"
    --speculative-draft-attention-backend aiter
    --speculative-draft-kv-cache-dtype bf16
    --speculative-draft-window-size 1024
  )
fi

# Real DFlash acceptance for accuracy validation.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${SGLANG_ROOT}/evaluation/logs/mimo_v26_flash_dp2_attn_tp4_ep8_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_dp2_attn_tp4_ep8_dflash.log}"

echo "MiMo-V2.6: static FP8 blockscale MoE weights converted from MXFP4 at load time"
echo "Attention: full target=${SGLANG_AITER_PA_DECODE_IMPL}; SWA target=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}"
echo "Optimizations: FlyPA prefill=${SGLANG_FLYPA_MIMO_PREFILL}; fused bounds checks=${SGLANG_FUSED_ASYNC_OOB_CHECKS}"

echo "Configuration: global-tp=8, attention-tp=4, dp=2, ep=8, moe-a2a=mori, max-running=${MAX_RUNNING_REQUESTS}, requested-max-running-per-dp=${requested_max_running_per_dp}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, effective-prefill-per-dp=${effective_prefill_chunk}, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-unset}, nccl-min-channels=${NCCL_MIN_NCHANNELS}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, speculative=${speculative_status}, aiter-ar-fusion=0, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, reasoning-parser=${REASONING_PARSER}, tbo=0, overlap=enabled"
echo "MORI: launch-config=${MORI_EP_LAUNCH_CONFIG_MODE}, dispatch=${SGLANG_MORI_DISPATCH_DTYPE}, combine=${SGLANG_MORI_COMBINE_DTYPE}, capacity-per-rank=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}, shmem=${MORI_SHMEM_MODE}, sdma=${MORI_ENABLE_SDMA}"
if [[ "${MORI_EP_LAUNCH_CONFIG_MODE}" == "AUTO" ]]; then
  echo "MORI tuning config: ${MORI_EP_TUNING_CONFIG}"
fi
echo "Performance mode: ${PERFORMANCE_MODE} (1=fake routing; 0=real routing; both use ordinary serving)"
echo "Prefill delayer: enabled=${ENABLE_PREFILL_DELAYER}, max-delay-passes=${PREFILL_DELAYER_MAX_DELAY_PASSES} (when enabled)"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"
if [[ "${ENABLE_DFLASH}" == "1" ]]; then
  echo "Speculation: DFLASH, block=${DFLASH_BLOCK_SIZE}, draft=${SPECULATIVE_DRAFT_MODEL_PATH}, draft-attention=aiter/${SGLANG_AITER_DFLASH_SWA_IMPL}, draft-kv=bf16, draft-window=1024, compact-kv=${SGLANG_DFLASH_FUSED_COMPACT_KV}, acceptance=real"
else
  echo "Speculation: disabled (target-only)"
fi


cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi
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

echo "Custom all-reduce: ${custom_all_reduce_status}"

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

radix_cache_args=()
radix_cache_status=enabled
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  radix_cache_status=disabled
fi

echo "Radix cache: ${radix_cache_status}"
echo "HIP non-greedy EAGLE verifier: ${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY}"

mm_attention_args=()
if [[ -n "${MM_ATTENTION_BACKEND:-}" ]]; then
  mm_attention_args+=(--mm-attention-backend "${MM_ATTENTION_BACKEND}")
fi

server_cmd=("${PYTHON_BIN}" -u -m sglang.launch_server \
  "${mm_attention_args[@]}" \
  --model-path "${MODEL_PATH}" \
  --tp-size 8 \
  --dp-size 2 \
  --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend mori \
  --deepep-mode normal \
  "${performance_args[@]}" \
  "${prefill_delayer_args[@]}" \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --moe-runner-backend aiter \
  "${dflash_args[@]}" \
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
  "${kv_cache_args[@]}" \
  --page-size 64 \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
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
