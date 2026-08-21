#!/usr/bin/env bash
set -euo pipefail

# Source: start standalone daemons with the manifest server, then an IPC client.
REPO_DIR="${REPO_DIR:-/root/sglang-weight-cache-tp}"
CONDA_ENV="${CONDA_ENV:-manifest}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.5-0.8B}"
DTYPE="${DTYPE:-bfloat16}"
SOURCE_TP_SIZE="${SOURCE_TP_SIZE:-1}"
SOURCE_PP_SIZE="${SOURCE_PP_SIZE:-1}"
SOURCE_EP_SIZE="${SOURCE_EP_SIZE:-1}"
SOURCE_BASE_GPU_ID="${SOURCE_BASE_GPU_ID:-0}"
SOURCE_GPU_ID_STEP="${SOURCE_GPU_ID_STEP:-1}"
SOURCE_PORT="${SOURCE_PORT:-8000}"
MANIFEST_SERVER_HOST="${MANIFEST_SERVER_HOST:-0.0.0.0}"
MANIFEST_SERVER_PORT="${MANIFEST_SERVER_PORT:-31999}"
LOG_DIR="${LOG_DIR:-/tmp/sglang-weight-manifest-server-e2e}"

export SGLANG_USE_MODELSCOPE="${SGLANG_USE_MODELSCOPE:-true}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

DAEMON_LAUNCHER_PID=""
ENGINE_PID=""
cleanup() {
  if [[ -n "${ENGINE_PID}" ]]; then
    kill "${ENGINE_PID}" 2>/dev/null || true
    wait "${ENGINE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${DAEMON_LAUNCHER_PID}" ]]; then
    kill "${DAEMON_LAUNCHER_PID}" 2>/dev/null || true
    wait "${DAEMON_LAUNCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

python -m sglang.srt.weight_cache.daemon \
  --model-path "${MODEL_PATH}" \
  --tp-size "${SOURCE_TP_SIZE}" \
  --pp-size "${SOURCE_PP_SIZE}" \
  --ep-size "${SOURCE_EP_SIZE}" \
  --base-gpu-id "${SOURCE_BASE_GPU_ID}" \
  --gpu-id-step "${SOURCE_GPU_ID_STEP}" \
  --dtype "${DTYPE}" \
  --enable-weight-heterogeneous-copy \
  --weight-heterogeneous-transfer-server-host "${MANIFEST_SERVER_HOST}" \
  --weight-heterogeneous-transfer-server-port "${MANIFEST_SERVER_PORT}" \
  >"${LOG_DIR}/source-daemon.log" 2>&1 &
DAEMON_LAUNCHER_PID=$!

SOURCE_WORLD_SIZE=$((SOURCE_TP_SIZE * SOURCE_PP_SIZE))
for ((rank = 0; rank < SOURCE_WORLD_SIZE; rank++)); do
  gpu_id=$((SOURCE_BASE_GPU_ID + rank * SOURCE_GPU_ID_STEP))
  ready_file="/tmp/sglang_weight_cache_rank${gpu_id}.ready"
  until [[ -f "${ready_file}" ]]; do
    if ! kill -0 "${DAEMON_LAUNCHER_PID}" 2>/dev/null; then
      echo "source daemon launcher exited; see ${LOG_DIR}/source-daemon.log" >&2
      exit 1
    fi
    sleep 1
  done
done

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size "${SOURCE_TP_SIZE}" \
  --pp-size "${SOURCE_PP_SIZE}" \
  --ep-size "${SOURCE_EP_SIZE}" \
  --base-gpu-id "${SOURCE_BASE_GPU_ID}" \
  --gpu-id-step "${SOURCE_GPU_ID_STEP}" \
  --dtype "${DTYPE}" \
  --weight-cache-mode client \
  --host 0.0.0.0 \
  --port "${SOURCE_PORT}" \
  >"${LOG_DIR}/source-engine.log" 2>&1 &
ENGINE_PID=$!

until curl -fsS "http://127.0.0.1:${SOURCE_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
    echo "source engine exited; see ${LOG_DIR}/source-engine.log" >&2
    exit 1
  fi
  sleep 1
done

echo "source ready: manifest=http://127.0.0.1:${MANIFEST_SERVER_PORT}"
wait "${DAEMON_LAUNCHER_PID}" || true
