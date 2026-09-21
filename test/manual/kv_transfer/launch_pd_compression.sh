#!/usr/bin/env bash
# Single-GPU P/D + optional Prefill HiCache. See README_pd_compression.md.
set -euo pipefail

: "${ROLE:?Set ROLE to prefill or decode}"
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-8B model directory}"
: "${IB_DEVICE:?Set IB_DEVICE to the RDMA HCA visible in this container}"
case "$ROLE" in
  prefill) default_port=30000 ;;
  decode) default_port=30001 ;;
  *) echo "ROLE must be prefill or decode" >&2; exit 2 ;;
esac
export SGLANG_PD_KV_COMPRESSION="${COMPRESSION_MODE:-off}"
case "$SGLANG_PD_KV_COMPRESSION" in
  off|passthrough|lz4) ;;
  *) echo "COMPRESSION_MODE must be off, passthrough or lz4" >&2; exit 2 ;;
esac
export SGLANG_HICACHE_KV_COMPRESSION="${HICACHE_COMPRESSION:-off}"
case "$SGLANG_HICACHE_KV_COMPRESSION" in
  off|passthrough|lz4) ;;
  *) echo "HICACHE_COMPRESSION must be off, passthrough or lz4" >&2; exit 2 ;;
esac
export SGLANG_PD_KV_COMPRESSION_FORCE="${SGLANG_PD_KV_COMPRESSION_FORCE:-0}"
export SGLANG_PD_KV_COMPRESSION_VERIFY="${SGLANG_PD_KV_COMPRESSION_VERIFY:-1}"
if [[ "$SGLANG_PD_KV_COMPRESSION_FORCE" != 0 ]]; then
  if [[ "$SGLANG_PD_KV_COMPRESSION_FORCE" != 1 || "$SGLANG_PD_KV_COMPRESSION" != lz4 ||
        "$SGLANG_PD_KV_COMPRESSION_VERIFY" != 1 ]]; then
    echo "FORCE=1 is diagnostic only: lz4 and VERIFY=1 on both workers required" >&2; exit 2
  fi
  if [[ "${ENABLE_HICACHE:-0}" == 1 ]]; then
    if [[ "$ROLE" != prefill || "$SGLANG_HICACHE_KV_COMPRESSION" != lz4 ]]; then
      echo "FORCE allows only Prefill LZ4 L2" >&2; exit 2
    fi
  elif [[ "$SGLANG_HICACHE_KV_COMPRESSION" != off ]]; then
    echo "L2 compression requires HiCache" >&2; exit 2
  fi
fi
# Privileged Kubernetes containers can enumerate every GPU despite a 1-GPU
# resource request. Bind to the plugin allocation; never assume physical 0.
if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$NVIDIA_VISIBLE_DEVICES" =~ ^[0-9]+$ || "$NVIDIA_VISIBLE_DEVICES" == GPU-* ]]; then
    [[ "$NVIDIA_VISIBLE_DEVICES" != *,* ]] || { echo "Exactly one GPU is required" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
  elif [[ "$NVIDIA_VISIBLE_DEVICES" != all || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set an explicit single-GPU allocation; automatic GPU 0 selection is unsafe" >&2; exit 2
  fi
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Exactly one CUDA device is required" >&2; exit 2
fi
if [[ "$SGLANG_HICACHE_KV_COMPRESSION" != off ]] &&
   [[ "$ROLE" != prefill || "${ENABLE_HICACHE:-0}" != 1 ]]; then
  echo "Compressed L2 requires ROLE=prefill ENABLE_HICACHE=1" >&2; exit 2
fi
if [[ "$ROLE" == decode && "${ENABLE_HICACHE:-0}" == 1 ]]; then
  echo "This draft only attaches HiCache on Prefill" >&2; exit 2
fi
export SGLANG_KV_COMPRESSION_WORKSPACE_MB="${SGLANG_KV_COMPRESSION_WORKSPACE_MB:-512}"
export SGLANG_KV_COMPRESSION_TRACE_STORE="${SGLANG_KV_COMPRESSION_TRACE_STORE:-0}"
# Exact-operation fault injection is opt-in and disabled in normal runs.
export SGLANG_KV_COMPRESSION_TEST_FAULT="${SGLANG_KV_COMPRESSION_TEST_FAULT:-}"
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB="${SGLANG_DISAGG_STAGING_POOL_SIZE_MB:-512}"
# Keep worker counts fixed for comparisons; the feature also enforces these.
export SGLANG_DISAGGREGATION_QUEUE_SIZE=1
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=1
export SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=1
export SGLANG_DISAGG_STAGING_BUFFER=0
export SGLANG_RUST_SERVER=0

args=(
  -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --served-model-name qwen3-8b-pd-poc
  --host "${POD_IP:-0.0.0.0}"
  --port "${HTTP_PORT:-$default_port}"
  --disaggregation-mode "$ROLE"
  --disaggregation-transfer-backend mooncake
  --disaggregation-ib-device "$IB_DEVICE"
  --tp-size 1
  --dtype bfloat16
  --kv-cache-dtype auto
  --page-size 1
  --attention-backend flashinfer
  --chunked-prefill-size "${CHUNK_TOKENS:-1024}"
  --context-length "${CONTEXT_LENGTH:-16384}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-32768}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.75}"
  --disable-overlap-schedule
  --disable-cuda-graph
  --enable-metrics
)
if [[ "$ROLE" == prefill ]]; then
  args+=(--disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8998}")
else
  # Avoid the existing auto mode selecting a separate raw HiCache host pool.
  args+=(--disaggregation-decode-retraction-backup cpu_tensor)
fi
if [[ "${ENABLE_HICACHE:-0}" == 1 ]]; then
  args+=(--enable-hierarchical-cache
    --hicache-size "${HICACHE_SIZE_GB:-8}"
    --hicache-host-memory-mode cache
    --hicache-write-policy write_through
    --hicache-io-backend kernel
    --hicache-mem-layout layer_first)
else
  args+=(--disable-radix-cache)
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for name in SGLANG_PD_KV_COMPRESSION SGLANG_HICACHE_KV_COMPRESSION \
    CUDA_VISIBLE_DEVICES SGLANG_PD_KV_COMPRESSION_FORCE SGLANG_PD_KV_COMPRESSION_VERIFY SGLANG_KV_COMPRESSION_WORKSPACE_MB \
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND; do
    printf '%s=%q ' "$name" "${!name-}"
  done
  printf '%q ' "${PYTHON_BIN:-python3}" "${args[@]}"
  printf '\n'
  exit 0
fi
exec "${PYTHON_BIN:-python3}" "${args[@]}"
