#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"
: "${MINWM_MODEL_DIR:?set MINWM_MODEL_DIR}"
: "${MINWM_STATIC_TRANSFORMER:?set MINWM_STATIC_TRANSFORMER}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
CASE="${MINWM_THROUGHPUT_CASE:-00_forward_080_pottery_720p}"
PROFILE_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
LOCAL_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}/nsys-static-fp8"
NSYS_URL="${MINWM_NSYS_URL:-https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb}"
NSYS_ROOT="${LOCAL_ROOT}/nsight-systems"
NSYS_DEB="${LOCAL_ROOT}/nsight-systems-cli.deb"

[[ -f "${MINWM_MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${MINWM_STATIC_TRANSFORMER}/minwm_static_fp8_manifest.json" ]]
[[ -f "${CASES}" ]]
mkdir -p "${PROFILE_ROOT}" "${LOCAL_ROOT}" "${NSYS_ROOT}"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

if ! command -v nsys >/dev/null; then
  curl --fail --location --retry 3 --output "${NSYS_DEB}" "${NSYS_URL}"
  sha256sum "${NSYS_DEB}" | tee "${PROFILE_ROOT}/nsys-package-sha256.txt"
  dpkg-deb --extract "${NSYS_DEB}" "${NSYS_ROOT}"
  NSYS_BIN="$(find "${NSYS_ROOT}" -type f -name nsys -perm -111 -print -quit)"
  [[ -n "${NSYS_BIN}" ]]
  export PATH="$(dirname "${NSYS_BIN}"):${PATH}"
fi
nsys --version | tee "${PROFILE_ROOT}/nsys-version.txt"
nsys status -e | tee "${PROFILE_ROOT}/nsys-status.txt" || true

wait_for_server() {
  local launch_pid="$1" log_path="$2"
  for _ in $(seq 1 300); do
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    if ! kill -0 "${launch_pid}" 2>/dev/null; then
      tail -300 "${log_path}" >&2
      return 1
    fi
    sleep 2
  done
  tail -300 "${log_path}" >&2
  return 1
}

stop_server() {
  local launch_pid="$1"
  pkill -TERM -f "sglang serve --model-path ${MINWM_MODEL_DIR}.*--port 30000" \
    2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}

wait_for_chunk() {
  local log_path="$1" chunk_index="$2" client_pid="$3"
  local chunk_pattern="\"chunk_index\":${chunk_index},"
  for _ in $(seq 1 6000); do
    if grep -F "${chunk_pattern}" "${log_path}" 2>/dev/null \
      | grep -Fq '"event":"server.chunk_complete"'; then
      return 0
    fi
    if ! kill -0 "${client_pid}" 2>/dev/null; then
      echo "client exited before chunk ${chunk_index}" >&2
      return 1
    fi
    sleep 0.1
  done
  echo "timed out waiting for chunk ${chunk_index}" >&2
  return 1
}

cleanup_pid=""
cleanup() {
  if [[ -n "${cleanup_pid}" ]]; then
    stop_server "${cleanup_pid}"
  fi
}
trap cleanup EXIT INT TERM

for lane in bf16 static_fp8; do
  lane_dir="${LOCAL_ROOT}/${lane}"
  mkdir -p "${lane_dir}"
  session="minwm-${MINWM_RUN_ID}-${lane}"
  server_log="${lane_dir}/server.log"
  report="${lane_dir}/${lane}.nsys-rep"
  sqlite="${lane_dir}/${lane}.sqlite"
  transformer_args=()
  if [[ "${lane}" == "static_fp8" ]]; then
    transformer_args=(--transformer-path "${MINWM_STATIC_TRANSFORMER}")
  fi

  CUDA_VISIBLE_DEVICES=0 \
  MINWM_ATTENTION_IMPL=dense \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=false \
  MINWM_NATIVE_COMPONENTS= \
  nsys launch \
    --session-new="${session}" \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -- \
    sglang serve \
      --model-path "${MINWM_MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --attention-backend fa \
      --performance-mode speed \
      "${transformer_args[@]}" \
      --enable-torch-compile false \
      --warmup-mode off \
      --port 30000 \
      > "${server_log}" 2>&1 &
  cleanup_pid=$!
  wait_for_server "${cleanup_pid}" "${server_log}"

  # Throwaway request compiles/JITs kernels before capture.
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${CASES}" \
    --case "${CASE}" \
    --results "${lane_dir}/throwaway" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --output-prefix throwaway \
    --engine-name "minwm-${lane}-nsys-throwaway" \
    --warmup-runs 0 \
    > "${lane_dir}/throwaway-client.log" 2>&1

  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    --output "${lane_dir}/throughput.json" \
    --profile-name "${lane}" \
    --kv-cache-num-frames 45 \
    --cases "${CASES}" \
    --case "${CASE}" \
    --warmup-chunks 22 \
    --measured-chunks 6 \
    > "${lane_dir}/profile-client.log" 2>&1 &
  client_pid=$!
  # Chunk 20 leaves two additional warmup chunks for `nsys start` latency.
  wait_for_chunk "${server_log}" 19 "${client_pid}"
  nsys start \
    --session="${session}" \
    --output="${lane_dir}/${lane}" \
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --sample=none
  wait "${client_pid}"
  nsys stop --session="${session}"
  stop_server "${cleanup_pid}"
  cleanup_pid=""

  [[ -f "${report}" ]]
  nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum \
    "${report}" \
    > "${lane_dir}/stats.txt"
  nsys export \
    --type=sqlite \
    --output="${sqlite}" \
    --force-overwrite=true \
    "${report}"
  cp -r "${lane_dir}" "${PROFILE_ROOT}/"
done

echo "MINWM_STATIC_FP8_NSYS_COMPLETE results=${PROFILE_ROOT}"
