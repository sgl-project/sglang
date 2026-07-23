#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:?usage: run_all.sh RESULTS_DIR}"
: "${MINWM_ROOT:?set MINWM_ROOT to the minWM main checkout}"
: "${MINWM_CHECKPOINT:?set MINWM_CHECKPOINT to the local model.pt}"
: "${MINWM_PRETRAINED_DIR:?set MINWM_PRETRAINED_DIR to the Wan2.2 donor directory}"
: "${MINWM_MODEL_DIR:?set MINWM_MODEL_DIR to the converted SGLang model directory}"

PORT="${MINWM_PORT:-30000}"
PROFILE="${MINWM_PARITY_PROFILE:-bitwise}"
ATTENTION_BACKEND="${MINWM_ATTENTION_BACKEND:-fa}"
ENABLE_TORCH_COMPILE="${MINWM_ENABLE_TORCH_COMPILE:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MINWM_PARITY_DETERMINISTIC="${MINWM_PARITY_DETERMINISTIC:-1}"
export MINWM_DETERMINISTIC_ATTENTION="${MINWM_DETERMINISTIC_ATTENTION:-true}"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}"
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D="${SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(cd "${RESULT_DIR}" && pwd)"
BASELINE_CONFIG_ARGS=()
if [[ -n "${MINWM_CONFIG:-}" ]]; then
  BASELINE_CONFIG_ARGS=(--config "${MINWM_CONFIG}")
fi

python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
  --minwm-root "${MINWM_ROOT}" \
  --checkpoint "${MINWM_CHECKPOINT}" \
  --pretrained-dir "${MINWM_PRETRAINED_DIR}" \
  --results "${RESULT_DIR}" \
  "${BASELINE_CONFIG_ARGS[@]}"

sglang serve \
  --model-path "${MINWM_MODEL_DIR}" \
  --pipeline-class-name MinWMCausalDMDPipeline \
  --attention-backend "${ATTENTION_BACKEND}" \
  --performance-mode speed \
  --enable-torch-compile "${ENABLE_TORCH_COMPILE}" \
  --warmup-mode off \
  --port "${PORT}" \
  >"${RESULT_DIR}/sglang-server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -200 "${RESULT_DIR}/sglang-server.log" >&2
    exit 1
  fi
  sleep 2
done
if [[ "${ready}" != 1 ]]; then
  tail -200 "${RESULT_DIR}/sglang-server.log" >&2
  echo "SGLang server did not become ready" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/run_sglang_api.py" \
  --results "${RESULT_DIR}" \
  --ws-url "ws://127.0.0.1:${PORT}/v1/realtime_video/generate"

cleanup
trap - EXIT INT TERM
python3 "${SCRIPT_DIR}/compare_results.py" \
  --results "${RESULT_DIR}" \
  --profile "${PROFILE}"

echo "Comparison complete. Open it with:"
echo "${SCRIPT_DIR}/play.sh ${RESULT_DIR}"
