#!/usr/bin/env bash
set -euo pipefail

# Target: standalone daemons fetch the registry, then an IPC client serves.
REPO_DIR="${REPO_DIR:-/root/sglang-weight-cache-tp}"
CONDA_ENV="${CONDA_ENV:-manifest}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.5-0.8B}"
DTYPE="${DTYPE:-bfloat16}"
TARGET_TP_SIZE="${TARGET_TP_SIZE:-2}"
TARGET_PP_SIZE="${TARGET_PP_SIZE:-1}"
TARGET_EP_SIZE="${TARGET_EP_SIZE:-1}"
TARGET_BASE_GPU_ID="${TARGET_BASE_GPU_ID:-1}"
TARGET_GPU_ID_STEP="${TARGET_GPU_ID_STEP:-1}"
TARGET_PORT="${TARGET_PORT:-8001}"
SOURCE_MANIFEST_IP="${SOURCE_MANIFEST_IP:-127.0.0.1}"
SOURCE_MANIFEST_PORT="${SOURCE_MANIFEST_PORT:-31999}"
KEEP_SERVING="${KEEP_SERVING:-true}"
LOG_DIR="${LOG_DIR:-/tmp/sglang-weight-manifest-server-e2e}"

export SGLANG_USE_MODELSCOPE="${SGLANG_USE_MODELSCOPE:-true}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

TARGET_DAEMON_PID=""
TARGET_ENGINE_PID=""
cleanup() {
  if [[ -n "${TARGET_ENGINE_PID}" ]]; then
    kill "${TARGET_ENGINE_PID}" 2>/dev/null || true
    wait "${TARGET_ENGINE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TARGET_DAEMON_PID}" ]]; then
    kill "${TARGET_DAEMON_PID}" 2>/dev/null || true
    wait "${TARGET_DAEMON_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

python -m sglang.srt.weight_cache.daemon \
  --model-path "${MODEL_PATH}" \
  --weight-heterogeneous-copy \
  --weight-heterogeneous-transfer-source-ip "${SOURCE_MANIFEST_IP}" \
  --weight-heterogeneous-transfer-source-port "${SOURCE_MANIFEST_PORT}" \
  --tp-size "${TARGET_TP_SIZE}" \
  --pp-size "${TARGET_PP_SIZE}" \
  --ep-size "${TARGET_EP_SIZE}" \
  --base-gpu-id "${TARGET_BASE_GPU_ID}" \
  --gpu-id-step "${TARGET_GPU_ID_STEP}" \
  --dtype "${DTYPE}" \
  >"${LOG_DIR}/target-daemon.log" 2>&1 &
TARGET_DAEMON_PID=$!

TARGET_WORLD_SIZE=$((TARGET_TP_SIZE * TARGET_PP_SIZE))
for ((rank = 0; rank < TARGET_WORLD_SIZE; rank++)); do
  gpu_id=$((TARGET_BASE_GPU_ID + rank * TARGET_GPU_ID_STEP))
  ready_file="/tmp/sglang_weight_cache_rank${gpu_id}.ready"
  until [[ -f "${ready_file}" ]]; do
    if ! kill -0 "${TARGET_DAEMON_PID}" 2>/dev/null; then
      echo "target daemon launcher exited; see ${LOG_DIR}/target-daemon.log" >&2
      exit 1
    fi
    sleep 1
  done
done

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size "${TARGET_TP_SIZE}" \
  --pp-size "${TARGET_PP_SIZE}" \
  --ep-size "${TARGET_EP_SIZE}" \
  --base-gpu-id "${TARGET_BASE_GPU_ID}" \
  --gpu-id-step "${TARGET_GPU_ID_STEP}" \
  --dtype "${DTYPE}" \
  --weight-cache-mode client \
  --host 0.0.0.0 \
  --port "${TARGET_PORT}" \
  >"${LOG_DIR}/target-engine.log" 2>&1 &
TARGET_ENGINE_PID=$!

until curl -fsS "http://127.0.0.1:${TARGET_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${TARGET_ENGINE_PID}" 2>/dev/null; then
    echo "target engine exited; see ${LOG_DIR}/target-engine.log" >&2
    exit 1
  fi
  sleep 1
done

echo "target ready: engine=http://127.0.0.1:${TARGET_PORT}"
curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}' \
  "http://127.0.0.1:${TARGET_PORT}/generate"
echo
echo "heterogeneous weight transfer E2E passed"
if [[ "${KEEP_SERVING}" == "true" ]]; then
  wait "${TARGET_ENGINE_PID}"
fi
