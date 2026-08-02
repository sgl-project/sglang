#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}"
CHECKPOINT="${WORK_ROOT}/checkpoint/model.pt"
PRETRAINED="${WORK_ROOT}/pretrained"
MODEL_DIR="${WORK_ROOT}/sglang-model"
PROFILE_ROOT="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}/nsys-sp12"
CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
MINWM_CONFIG="${MINWM_CONFIG_PATH:-/workspace/minWM/Wan21/configs/eval/wan22_5b_varlen_dmd.yaml}"
SP_DEGREES="${MINWM_SP_DEGREES:-1 2}"
NSYS_VAE_LANE="${MINWM_NSYS_VAE_LANE:-parity}"
NSYS_URL="${MINWM_NSYS_URL:-https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb}"
NSYS_ROOT="${WORK_ROOT}/nsight-systems"
NSYS_DEB="${WORK_ROOT}/nsight-systems-cli.deb"

[[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
[[ -f "${CASES}" ]]
[[ -f "${CHECKPOINT}" ]]
[[ -d "${PRETRAINED}" ]]
if ! [[ "${NSYS_VAE_LANE}" =~ ^(parity|parallel)$ ]]; then
  echo "MINWM_NSYS_VAE_LANE must be parity or parallel" >&2
  exit 2
fi
mkdir -p "${PROFILE_ROOT}" "${NSYS_ROOT}"

export MINWM_PARITY_DETERMINISTIC=1
export MINWM_DETERMINISTIC_ATTENTION=true
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
export SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MINWM_ROOT=/workspace/minWM
export MINWM_CHECKPOINT="${CHECKPOINT}"
export MINWM_PRETRAINED_DIR="${PRETRAINED}"
export MINWM_MODEL_DIR="${MODEL_DIR}"
export MINWM_CONFIG

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
  local log_path="$1"
  for _ in $(seq 1 300); do
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    sleep 2
  done
  tail -300 "${log_path}" >&2
  return 1
}

stop_server() {
  local launch_pid="$1"
  pkill -TERM -f "sglang serve --model-path ${MODEL_DIR}.*--port 30000" \
    2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}

analyze_metrics() {
  local sqlite_path="$1" output_path="$2"
  python3 - "${sqlite_path}" "${output_path}" <<'PY'
import json
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()

metric_names = {}
for metric_id in (3, 5, 18, 19):
    row = cur.execute(
        "SELECT metricName FROM TARGET_INFO_GPU_METRICS WHERE metricId=? LIMIT 1",
        (metric_id,),
    ).fetchone()
    metric_names[metric_id] = row[0] if row else f"metric_{metric_id}"

metrics = {}
for type_id, metric_id, mean_value, sample_count in cur.execute(
    """
    SELECT typeId, metricId, AVG(value), COUNT(*)
    FROM GPU_METRICS
    WHERE metricId IN (3, 5, 18, 19)
    GROUP BY typeId, metricId
    ORDER BY typeId, metricId
    """
):
    metrics.setdefault(str(type_id), {})[metric_names[metric_id]] = {
        "mean": mean_value,
        "samples": sample_count,
    }

devices = {}
for device_id, start, end in cur.execute(
    """
    SELECT deviceId, start, end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    ORDER BY deviceId, start
    """
):
    device = devices.setdefault(str(device_id), {"intervals": [], "kernel_count": 0})
    device["intervals"].append((start, end))
    device["kernel_count"] += 1

for device in devices.values():
    intervals = device.pop("intervals")
    if not intervals:
        continue
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy_ns = sum(end - start for start, end in merged)
    span_ns = intervals[-1][1] - intervals[0][0]
    device["kernel_busy_ms"] = busy_ns / 1e6
    device["kernel_span_ms"] = span_ns / 1e6
    device["kernel_busy_fraction"] = busy_ns / span_ns if span_ns else None

result = {"gpu_metrics": metrics, "kernel_activity": devices}
with open(sys.argv[2], "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

cleanup_pid=""
cleanup() {
  if [[ -n "${cleanup_pid}" ]]; then
    stop_server "${cleanup_pid}"
  fi
}
trap cleanup EXIT INT TERM

read -r -a sp_degrees <<< "${SP_DEGREES}"
for degree in "${sp_degrees[@]}"; do
  if ! [[ "${degree}" =~ ^(1|2)$ ]]; then
    echo "run_sp_nsys.sh only supports SP degree 1 or 2, got ${degree}" >&2
    exit 2
  fi
  lane="sp${degree}"
  session="minwm-${MINWM_RUN_ID}-${lane}"
  server_log="${PROFILE_ROOT}/${lane}-server.log"
  report="${PROFILE_ROOT}/${lane}.nsys-rep"
  sqlite="${PROFILE_ROOT}/${lane}.sqlite"
  rm -f "${report}" "${sqlite}"

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_VAE_LANE="${NSYS_VAE_LANE}" \
  nsys launch \
    --session-new="${session}" \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -- \
    sglang serve \
      --model-path "${MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --attention-backend fa \
      --performance-mode speed \
      --num-gpus "${degree}" \
      --tp-size 1 \
      --sp-degree "${degree}" \
      --ulysses-degree "${degree}" \
      --ring-degree 1 \
      --enable-cfg-parallel false \
      --enable-torch-compile false \
      --warmup-mode off \
      --port 30000 \
      > "${server_log}" 2>&1 &
  cleanup_pid=$!
  wait_for_server "${server_log}"

  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${CASES}" \
    --results "${PROFILE_ROOT}/${lane}-warmup" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --output-prefix warmup \
    --engine-name "sglang-minwm-${lane}-nsys-warmup-${NSYS_VAE_LANE}-vae" \
    --warmup-runs 0 \
    | tee "${PROFILE_ROOT}/${lane}-warmup-client.log"

  nsys start \
    --session="${session}" \
    --output="${PROFILE_ROOT}/${lane}" \
    --gpu-metrics-devices=all \
    --gpu-metrics-frequency=10000 \
    --sample=none
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${CASES}" \
    --results "${PROFILE_ROOT}/${lane}-measured" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --output-prefix measured \
    --engine-name "sglang-minwm-${lane}-nsys-${NSYS_VAE_LANE}-vae" \
    --warmup-runs 0 \
    | tee "${PROFILE_ROOT}/${lane}-measured-client.log"
  nsys stop --session="${session}"
  stop_server "${cleanup_pid}"
  cleanup_pid=""

  [[ -f "${report}" ]]
  nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum \
    "${report}" \
    > "${PROFILE_ROOT}/${lane}-stats.txt"
  nsys export \
    --type=sqlite \
    --output="${sqlite}" \
    --force-overwrite=true \
    "${report}"
  analyze_metrics "${sqlite}" "${PROFILE_ROOT}/${lane}-metrics.json" \
    | tee "${PROFILE_ROOT}/${lane}-metrics.log"
done

echo "MINWM_SP_NSYS_COMPLETE degrees=${sp_degrees[*]} vae_lane=${NSYS_VAE_LANE} results=${PROFILE_ROOT}"
