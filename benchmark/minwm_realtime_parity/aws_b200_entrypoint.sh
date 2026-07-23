#!/usr/bin/env bash
set -euo pipefail

: "${MINWM_BENCHMARK_MODE:?set MINWM_BENCHMARK_MODE}"
: "${MINWM_RUN_ID:?set MINWM_RUN_ID}"
: "${MINWM_CHECKPOINT_S3_EAST:?set MINWM_CHECKPOINT_S3_EAST}"
: "${MINWM_CHECKPOINT_VERSION_EAST:?set MINWM_CHECKPOINT_VERSION_EAST}"
: "${MINWM_CHECKPOINT_SOURCE_URI:?set MINWM_CHECKPOINT_SOURCE_URI}"
: "${MINWM_CHECKPOINT_SOURCE_VERSION:?set MINWM_CHECKPOINT_SOURCE_VERSION}"
: "${MINWM_CHECKPOINT_SOURCE_ETAG:?set MINWM_CHECKPOINT_SOURCE_ETAG}"
: "${MINWM_CHECKPOINT_BYTES:?set MINWM_CHECKPOINT_BYTES}"
: "${MINWM_CHECKPOINT_CRC64:?set MINWM_CHECKPOINT_CRC64}"
: "${MINWM_PRETRAINED_S3:?set MINWM_PRETRAINED_S3}"
: "${MINWM_RESULTS_ROOT:?set MINWM_RESULTS_ROOT}"

[[ "${MINWM_BENCHMARK_MODE}" == "smoke" \
  || "${MINWM_BENCHMARK_MODE}" == "full" \
  || "${MINWM_BENCHMARK_MODE}" == "profiles" \
  || "${MINWM_BENCHMARK_MODE}" == "long720p" \
  || "${MINWM_BENCHMARK_MODE}" == "triptych720p" \
  || "${MINWM_BENCHMARK_MODE}" == "spmatrix720p" ]] || {
  echo "unsupported MINWM_BENCHMARK_MODE=${MINWM_BENCHMARK_MODE}" >&2
  exit 2
}

WORK_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}"
CHECKPOINT="${WORK_ROOT}/checkpoint/model.pt"
PRETRAINED="${WORK_ROOT}/pretrained"
if [[ "${MINWM_STAGE_FROM_MOUNT:-0}" == "1" ]]; then
  : "${MINWM_CHECKPOINT_MOUNT_PATH:?set MINWM_CHECKPOINT_MOUNT_PATH}"
  : "${MINWM_PRETRAINED_MOUNT_PATH:?set MINWM_PRETRAINED_MOUNT_PATH}"
fi
MODEL_DIR="${WORK_ROOT}/sglang-model"
LOCAL_RESULTS="${WORK_ROOT}/results"
RESULTS="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
SCRIPT_DIR="/workspace/sglang/benchmark/minwm_realtime_parity"
MINWM_CONFIG="/workspace/minWM/Wan21/configs/eval/wan22_5b_varlen_dmd.yaml"
mkdir -p "${WORK_ROOT}/checkpoint" "${PRETRAINED}" "${LOCAL_RESULTS}" "${RESULTS}"

python3 --version | tee "${RESULTS}/python-version.txt"
python3 - <<'PY' | tee "${RESULTS}/runtime-before-install.json"
import json, platform
import torch
print(json.dumps({
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}, sort_keys=True))
PY
stage_started="$(date +%s)"
if [[ "${MINWM_STAGE_FROM_MOUNT:-0}" == "1" ]]; then
  echo "s3-csi-mount" | tee "${RESULTS}/staging-mode.txt"
  [[ -f "${MINWM_CHECKPOINT_MOUNT_PATH}" ]]
  [[ -d "${MINWM_PRETRAINED_MOUNT_PATH}/transformer" ]]
  cp "${MINWM_CHECKPOINT_MOUNT_PATH}" "${CHECKPOINT}"
  # Hugging Face safetensors are mmap'd and faulted in small ranges.  Reading
  # those mappings directly through mountpoint-s3 turns model load into many
  # remote range requests, so stage the immutable donor tree once on NVMe.
  cp -a "${MINWM_PRETRAINED_MOUNT_PATH}/." "${PRETRAINED}/"
  python3 - "${RESULTS}/checkpoint-head.json" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "ContentLength": int(os.environ["MINWM_CHECKPOINT_BYTES"]),
    "ChecksumCRC64NVME": os.environ["MINWM_CHECKPOINT_CRC64"],
    "VersionId": os.environ["MINWM_CHECKPOINT_VERSION_EAST"],
    "verification": "control-plane head-object verified before S3 CSI mount read",
}, indent=2, sort_keys=True) + "\n")
PY
else
  aws sts get-caller-identity | tee "${RESULTS}/aws-identity.json"
  checkpoint_bucket="${MINWM_CHECKPOINT_S3_EAST#s3://}"
  checkpoint_bucket="${checkpoint_bucket%%/*}"
  checkpoint_key="${MINWM_CHECKPOINT_S3_EAST#s3://${checkpoint_bucket}/}"
  aws s3api head-object --bucket "${checkpoint_bucket}" --key "${checkpoint_key}" \
    --version-id "${MINWM_CHECKPOINT_VERSION_EAST}" --checksum-mode ENABLED \
    > "${RESULTS}/checkpoint-head.json"
  python3 - "${RESULTS}/checkpoint-head.json" "${MINWM_CHECKPOINT_BYTES}" "${MINWM_CHECKPOINT_CRC64}" <<'PY'
import json, sys
head = json.load(open(sys.argv[1]))
assert head["ContentLength"] == int(sys.argv[2]), head
assert head["ChecksumCRC64NVME"] == sys.argv[3], head
PY
  aws s3api get-object --bucket "${checkpoint_bucket}" --key "${checkpoint_key}" \
    --version-id "${MINWM_CHECKPOINT_VERSION_EAST}" --checksum-mode ENABLED \
    "${CHECKPOINT}" > "${RESULTS}/checkpoint-get.json"
  aws s3 sync "${MINWM_PRETRAINED_S3%/}/" "${PRETRAINED}/" \
    --no-progress --only-show-errors
fi
stage_finished="$(date +%s)"
[[ "$(stat -c '%s' "${CHECKPOINT}")" == "${MINWM_CHECKPOINT_BYTES}" ]]
sha256sum "${CHECKPOINT}" | tee "${RESULTS}/checkpoint-sha256.txt"
find "${PRETRAINED}" -type f -printf '%s\n' \
  | awk -v elapsed="$((stage_finished - stage_started))" \
      '{bytes += $1; files += 1} END {printf "files=%d bytes=%.0f stage_seconds=%d\n", files, bytes, elapsed}' \
  | tee "${RESULTS}/staging-summary.txt"

if [[ "${MINWM_SKIP_UNUSED_GRPC_RUST:-1}" == "1" ]] && ! command -v cargo >/dev/null; then
  python3 - /workspace/sglang/python/pyproject.toml <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
block = '''[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
'''
if text.count(block) != 1:
    raise RuntimeError("expected exactly one SGLang gRPC Rust extension block")
path.write_text(text.replace(block, ""))
print("Skipped unused sglang.srt.grpc Rust extension for diffusion-only benchmark")
PY
fi
python3 -m pip install -e "/workspace/sglang/python[diffusion]" \
  --root-user-action=ignore
# The minWM training image pins peft==0.17.0, while this SGLang checkout pins
# transformers==5.12.1.  Merely leaving the old PEFT package installed makes
# diffusers detect and import it, which then fails on the removed HybridCache
# symbol before any model code runs.  Realtime inference does not load LoRA
# adapters, so remove that stale optional package instead of changing either
# side's model/runtime dependency set.
python3 -m pip uninstall -y peft
python3 -m pip install --force-reinstall --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 \
  'flashinfer-jit-cache==0.6.12+cu130'
python3 - <<'PY' | tee "${RESULTS}/runtime.json"
import importlib.util, json
import diffusers, torch, transformers
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "diffusers": diffusers.__version__,
    "peft_installed": importlib.util.find_spec("peft") is not None,
    "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0),
}, sort_keys=True))
PY

python3 /workspace/sglang/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py \
  --minwm-checkpoint "${CHECKPOINT}" \
  --donor-diffusers-dir "${PRETRAINED}" \
  --output-dir "${MODEL_DIR}" \
  --link-donor \
  --source-uri "${MINWM_CHECKPOINT_SOURCE_URI}" \
  --source-version-id "${MINWM_CHECKPOINT_SOURCE_VERSION}" \
  --source-etag "${MINWM_CHECKPOINT_SOURCE_ETAG}" \
  | tee "${RESULTS}/conversion.log"
cp "${MODEL_DIR}/minwm_conversion_manifest.json" "${RESULTS}/"

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

wait_for_server() {
  local server_pid="$1" log_path="$2"
  for _ in $(seq 1 300); do
    if curl --fail --silent http://127.0.0.1:30000/health >/dev/null; then
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -300 "${log_path}" >&2
      return 1
    fi
    sleep 2
  done
  tail -300 "${log_path}" >&2
  return 1
}

if [[ "${MINWM_BENCHMARK_MODE}" == "spmatrix720p" ]]; then
  SP_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
  # Keep completed lanes directly on the mounted result store. Spot retries can
  # then skip valid SP artifacts instead of discarding an entire matrix.
  SP_RESULTS="${MINWM_SP_RESULTS_DIR:-${RESULTS}/sp-matrix-720p}"
  SP_WARMUP_RUNS="${MINWM_SP_WARMUP_RUNS:-1}"
  ARTIFACT_HOLD_SECONDS="${MINWM_ARTIFACT_HOLD_SECONDS:-0}"
  if ! [[ "${SP_WARMUP_RUNS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_SP_WARMUP_RUNS must be a non-negative integer" >&2
    exit 2
  fi
  if ! [[ "${ARTIFACT_HOLD_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_ARTIFACT_HOLD_SECONDS must be a non-negative integer" >&2
    exit 2
  fi
  mkdir -p "${SP_RESULTS}"
  nvidia-smi -L | tee "${RESULTS}/sp-gpus.txt"
  nvidia-smi topo -m | tee "${RESULTS}/sp-topology.txt"

  sp_lane_complete() {
    local prefix="$1"
    python3 - "${SP_CASES}" "${SP_RESULTS}" "${prefix}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
prefix = sys.argv[3]
expected_shape = (
    int(manifest["contract"]["reference_pixel_frames"])
    + int(manifest["contract"]["generated_pixel_frames"]),
    int(manifest["contract"]["height"]),
    int(manifest["contract"]["width"]),
    3,
)
run_path = root / f"{prefix}_run.json"
if not run_path.is_file():
    raise SystemExit(1)
run = json.loads(run_path.read_text())
if {item["id"] for item in run["cases"]} != {
    item["id"] for item in manifest["cases"]
}:
    raise SystemExit(1)
for case in manifest["cases"]:
    case_dir = root / "cases" / case["id"]
    paths = [
        case_dir / f"{prefix}.npy",
        case_dir / f"{prefix}.mp4",
        case_dir / f"{prefix}.json",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise SystemExit(1)
    if np.load(paths[0], mmap_mode="r", allow_pickle=False).shape != expected_shape:
        raise SystemExit(1)
PY
  }

  run_sp_lane() {
    local degree="$1"
    local prefix="sp${degree}"
    local server_log="${SP_RESULTS}/${prefix}-server.log"
    local memory_log="${SP_RESULTS}/${prefix}-gpu-memory.csv"
    MINWM_ATTENTION_IMPL=packed \
    MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
    MINWM_NATIVE_COMPONENTS=text_encoder,vae \
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
    local server_pid=$!
    if ! wait_for_server "${server_pid}" "${server_log}"; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
      return 1
    fi
    (
      while kill -0 "${server_pid}" 2>/dev/null; do
        nvidia-smi --query-gpu=timestamp,index,memory.used \
          --format=csv,noheader,nounits || true
        sleep 1
      done
    ) > "${memory_log}" &
    local monitor_pid=$!
    set +e
    python3 "${SCRIPT_DIR}/run_sglang_api.py" \
      --cases "${SP_CASES}" \
      --results "${SP_RESULTS}" \
      --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
      --output-prefix "${prefix}" \
      --engine-name "sglang-minwm-${prefix}-ulysses" \
      --warmup-runs "${SP_WARMUP_RUNS}" \
      | tee "${SP_RESULTS}/${prefix}-client.log"
    local lane_status=${PIPESTATUS[0]}
    set -e
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    return "${lane_status}"
  }

  for degree in 1 2 4 8; do
    if sp_lane_complete "sp${degree}"; then
      echo "MINWM_SP_RESUME_SKIP degree=${degree}"
    else
      run_sp_lane "${degree}"
    fi
  done
  python3 "${SCRIPT_DIR}/compare_sp_matrix.py" \
    --cases "${SP_CASES}" \
    --results "${SP_RESULTS}" \
    --degrees 1,2,4,8 \
    | tee "${RESULTS}/sp-matrix-compare.log"
  cp "${SP_RESULTS}/sp_matrix_report.json" "${RESULTS}/"
  cp "${SP_RESULTS}/sp_matrix_report.md" "${RESULTS}/"
  touch "${RESULTS}/sp-matrix-artifacts-ready"
  echo "MINWM_B200_SP_MATRIX_COMPLETE results=${RESULTS}"
  if (( ARTIFACT_HOLD_SECONDS > 0 )); then
    echo "MINWM_ARTIFACT_HOLD seconds=${ARTIFACT_HOLD_SECONDS}"
    sleep "${ARTIFACT_HOLD_SECONDS}"
  fi
  exit 0
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "triptych720p" ]]; then
  TRIPTYCH_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
  TRIPTYCH_RESULTS="${LOCAL_RESULTS}/triptych-720p"
  TRIPTYCH_WARMUP_RUNS="${MINWM_TRIPTYCH_WARMUP_RUNS:-1}"
  ARTIFACT_HOLD_SECONDS="${MINWM_ARTIFACT_HOLD_SECONDS:-0}"
  if ! [[ "${TRIPTYCH_WARMUP_RUNS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_TRIPTYCH_WARMUP_RUNS must be a non-negative integer" >&2
    exit 2
  fi
  if ! [[ "${ARTIFACT_HOLD_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_ARTIFACT_HOLD_SECONDS must be a non-negative integer" >&2
    exit 2
  fi
  mkdir -p "${TRIPTYCH_RESULTS}"
  python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
    --cases "${TRIPTYCH_CASES}" \
    --minwm-root /workspace/minWM \
    --checkpoint "${CHECKPOINT}" \
    --pretrained-dir "${PRETRAINED}" \
    --config "${MINWM_CONFIG}" \
    --results "${TRIPTYCH_RESULTS}" \
    --warmup-runs "${TRIPTYCH_WARMUP_RUNS}" \
    | tee "${RESULTS}/triptych-baseline.log"

  run_triptych_lane() {
    local lane="$1" attention_impl="$2" packed_deterministic="$3"
    local native_components="$4" torch_compile="$5" segment_compile="$6"
    local parity_deterministic="$7"
    local server_log="${RESULTS}/triptych-${lane}-server.log"
    local memory_log="${RESULTS}/triptych-${lane}-gpu-memory.csv"
    MINWM_ATTENTION_IMPL="${attention_impl}" \
    MINWM_PACKED_ATTENTION_DETERMINISTIC="${packed_deterministic}" \
    MINWM_NATIVE_COMPONENTS="${native_components}" \
    MINWM_SEGMENT_COMPILE="${segment_compile}" \
    MINWM_PARITY_DETERMINISTIC="${parity_deterministic}" \
    MINWM_DETERMINISTIC_ATTENTION="${parity_deterministic}" \
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${parity_deterministic}" \
    sglang serve \
      --model-path "${MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --attention-backend fa \
      --performance-mode speed \
      --enable-torch-compile "${torch_compile}" \
      --warmup-mode off \
      --port 30000 \
      > "${server_log}" 2>&1 &
    local server_pid=$!
    if ! wait_for_server "${server_pid}" "${server_log}"; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
      return 1
    fi
    (
      while kill -0 "${server_pid}" 2>/dev/null; do
        nvidia-smi --query-gpu=timestamp,memory.used \
          --format=csv,noheader,nounits || true
        sleep 1
      done
    ) > "${memory_log}" &
    local monitor_pid=$!
    set +e
    python3 "${SCRIPT_DIR}/run_sglang_api.py" \
      --cases "${TRIPTYCH_CASES}" \
      --results "${TRIPTYCH_RESULTS}" \
      --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
      --output-prefix "sglang_${lane}" \
      --engine-name "sglang-minwm-${lane}" \
      --warmup-runs "${TRIPTYCH_WARMUP_RUNS}" \
      | tee "${RESULTS}/triptych-${lane}-client.log"
    local lane_status=${PIPESTATUS[0]}
    set -e
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    return "${lane_status}"
  }

  run_triptych_lane bitwise packed true text_encoder,vae false true 1
  # Whole-DiT compilation subsumes minWM's small dynamic segment compiles. The
  # nested compilers add graph breaks/recompiles, so the speed lane leaves those
  # helpers eager and lets module-level Inductor own the transformer scope.
  run_triptych_lane optimized dense false "" true false 0

  python3 "${SCRIPT_DIR}/compare_triptych.py" \
    --cases "${TRIPTYCH_CASES}" \
    --results "${TRIPTYCH_RESULTS}" \
    | tee "${RESULTS}/triptych-compare.log"
  cp -r "${TRIPTYCH_RESULTS}" "${RESULTS}/"
  cp "${TRIPTYCH_RESULTS}/triptych_report.json" "${RESULTS}/"
  cp "${TRIPTYCH_RESULTS}/triptych_report.md" "${RESULTS}/"
  touch "${RESULTS}/triptych-artifacts-ready"
  echo "MINWM_TRIPTYCH720P_COMPLETE results=${RESULTS}"
  if (( ARTIFACT_HOLD_SECONDS > 0 )); then
    echo "MINWM_ARTIFACT_HOLD seconds=${ARTIFACT_HOLD_SECONDS}"
    sleep "${ARTIFACT_HOLD_SECONDS}"
  fi
  exit 0
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "long720p" ]]; then
  LONG_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_5s.json}"
  LONG_ENABLE_TORCH_COMPILE="${MINWM_LONG_ENABLE_TORCH_COMPILE:-false}"
  if [[ "${LONG_ENABLE_TORCH_COMPILE}" != "true" && "${LONG_ENABLE_TORCH_COMPILE}" != "false" ]]; then
    echo "MINWM_LONG_ENABLE_TORCH_COMPILE must be true or false" >&2
    exit 2
  fi
  if [[ "${LONG_ENABLE_TORCH_COMPILE}" == "true" ]]; then
    LONG_PROFILE=compile
  else
    LONG_PROFILE=eager
  fi
  LONG_RESULTS="${LOCAL_RESULTS}/720p-5s-${LONG_PROFILE}"
  mkdir -p "${LONG_RESULTS}"
  python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
    --cases "${LONG_CASES}" \
    --minwm-root /workspace/minWM \
    --checkpoint "${CHECKPOINT}" \
    --pretrained-dir "${PRETRAINED}" \
    --config "${MINWM_CONFIG}" \
    --results "${LONG_RESULTS}" \
    | tee "${RESULTS}/720p-baseline.log"

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --attention-backend fa \
    --performance-mode speed \
    --enable-torch-compile "${LONG_ENABLE_TORCH_COMPILE}" \
    --warmup-mode off \
    --port 30000 \
    > "${RESULTS}/720p-sglang-server.log" 2>&1 &
  SERVER_PID=$!
  (
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
      nvidia-smi --query-gpu=timestamp,memory.used \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "${RESULTS}/720p-gpu-memory.csv" &
  GPU_MONITOR_PID=$!
  cleanup_long_server() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  }
  trap cleanup_long_server EXIT INT TERM
  for _ in $(seq 1 300); do
    curl --fail --silent http://127.0.0.1:30000/health >/dev/null && break
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -300 "${RESULTS}/720p-sglang-server.log" >&2
      exit 1
    fi
    sleep 2
  done
  curl --fail --silent http://127.0.0.1:30000/health >/dev/null
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${LONG_CASES}" \
    --results "${LONG_RESULTS}" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    | tee "${RESULTS}/720p-sglang-client.log"
  cleanup_long_server
  trap - EXIT INT TERM

  set +e
  python3 "${SCRIPT_DIR}/compare_results.py" \
    --cases "${LONG_CASES}" \
    --results "${LONG_RESULTS}" \
    --profile bitwise \
    | tee "${RESULTS}/720p-bitwise-compare.log"
  bitwise_status=${PIPESTATUS[0]}
  set -e
  cp "${LONG_RESULTS}/report.json" "${RESULTS}/720p-bitwise-report.json"
  numeric_status=0
  if (( bitwise_status != 0 )); then
    set +e
    python3 "${SCRIPT_DIR}/compare_results.py" \
      --cases "${LONG_CASES}" \
      --results "${LONG_RESULTS}" \
      --profile bf16_backend_candidate \
      | tee "${RESULTS}/720p-numeric-compare.log"
    numeric_status=${PIPESTATUS[0]}
    set -e
    cp "${LONG_RESULTS}/report.json" "${RESULTS}/720p-numeric-report.json"
  fi

  python3 - "${LONG_RESULTS}" "${RESULTS}/720p-gpu-memory.csv" \
    "${RESULTS}/720p-summary.json" "${bitwise_status}" "${numeric_status}" \
    "${LONG_ENABLE_TORCH_COMPILE}" <<'PY'
import json, re, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
run = json.loads((root / "sglang_run.json").read_text())
cases = []
all_scheduler_ms = []
all_client_ms = []
all_frames = 0
for case in run["cases"]:
    steady = [
        item for item in case["chunk_stats"] if int(item["chunk_index"]) > 0
    ]
    scheduler_ms = [float(item["scheduler_forward_ms"]) for item in steady]
    frames = sum(int(item["num_frames"]) for item in steady)
    client_ms = [
        float(value)
        for value in case["client_timing"]["steady_payload_interarrival_ms"]
    ]
    cases.append({
        "id": case["id"],
        "ttff_ms": case["client_timing"]["init_send_start_to_first_payload_complete_ms"],
        "steady_scheduler_fps": frames / (sum(scheduler_ms) / 1000),
        "steady_client_fps": frames / (sum(client_ms) / 1000),
        "steady_chunk_p50_ms": statistics.median(client_ms),
        "steady_chunk_count": len(steady),
    })
    all_scheduler_ms.extend(scheduler_ms)
    all_client_ms.extend(client_ms)
    all_frames += frames
memory_values = []
for line in Path(sys.argv[2]).read_text().splitlines():
    match = re.search(r",\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
    if match:
        memory_values.append(float(match.group(1)))
summary = {
    "contract": {
        "size": "1248x704",
        "fps": 24,
        "frames_per_case": 129,
        "seconds_per_case": 129 / 24,
        "chunks_per_case": 8,
        "action_type": "primitive_token_residual",
        "action_output_format": "primitive_float",
        "kv_semantics": "minWM unbounded; bounded request preallocates 33 latent frames",
        "torch_compile": sys.argv[6].lower() == "true",
    },
    "bitwise_status": int(sys.argv[4]),
    "numeric_status": int(sys.argv[5]),
    "case_results": cases,
    "aggregate_steady_scheduler_fps": all_frames / (sum(all_scheduler_ms) / 1000),
    "aggregate_steady_client_fps": all_frames / (sum(all_client_ms) / 1000),
    "aggregate_steady_chunk_p50_ms": statistics.median(all_client_ms),
    "peak_gpu_memory_mb": max(memory_values) if memory_values else None,
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  cp -r "${LONG_RESULTS}" "${RESULTS}/"
  touch "${RESULTS}/720p-artifacts-ready"
  echo "MINWM_B200_LONG720P_COMPLETE results=${RESULTS} bitwise_status=${bitwise_status} numeric_status=${numeric_status}"
  ARTIFACT_HOLD_SECONDS="${MINWM_ARTIFACT_HOLD_SECONDS:-0}"
  if ! [[ "${ARTIFACT_HOLD_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_ARTIFACT_HOLD_SECONDS must be a non-negative integer" >&2
    exit 2
  fi
  if (( ARTIFACT_HOLD_SECONDS > 0 )); then
    echo "MINWM_ARTIFACT_HOLD seconds=${ARTIFACT_HOLD_SECONDS}"
    sleep "${ARTIFACT_HOLD_SECONDS}"
  fi
  if (( bitwise_status != 0 && numeric_status != 0 )); then
    exit "${numeric_status}"
  fi
  exit 0
fi

if [[ "${MINWM_BENCHMARK_MODE}" != "profiles" ]]; then
SMOKE_RESULTS="${LOCAL_RESULTS}/smoke"
mkdir -p "${SMOKE_RESULTS}"
python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
  --minwm-root /workspace/minWM \
  --checkpoint "${CHECKPOINT}" \
  --pretrained-dir "${PRETRAINED}" \
  --config "${MINWM_CONFIG}" \
  --results "${SMOKE_RESULTS}" \
  --case 00_forward_pottery \
  | tee "${RESULTS}/baseline-smoke.log"

MINWM_ATTENTION_IMPL=packed \
MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
MINWM_NATIVE_COMPONENTS=text_encoder,vae \
sglang serve \
  --model-path "${MODEL_DIR}" \
  --pipeline-class-name MinWMCausalDMDPipeline \
  --attention-backend fa \
  --performance-mode speed \
  --enable-torch-compile false \
  --warmup-mode off \
  --port 30000 \
  > "${RESULTS}/sglang-server.log" 2>&1 &
SERVER_PID=$!
cleanup_server() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup_server EXIT INT TERM
for _ in $(seq 1 300); do
  curl --fail --silent http://127.0.0.1:30000/health >/dev/null && break
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -300 "${RESULTS}/sglang-server.log" >&2
    exit 1
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:30000/health >/dev/null
python3 "${SCRIPT_DIR}/run_sglang_api.py" \
  --results "${SMOKE_RESULTS}" \
  --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
  --case 00_forward_pottery \
  | tee "${RESULTS}/sglang-smoke.log"
cleanup_server
trap - EXIT INT TERM

python3 - "${SMOKE_RESULTS}" "${RESULTS}/smoke-summary.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]) / "cases" / "00_forward_pottery"
baseline = np.load(root / "baseline.npy", allow_pickle=False)
candidate = np.load(root / "sglang.npy", allow_pickle=False)
error = np.abs(baseline.astype(np.int16) - candidate.astype(np.int16))
summary = {
    "shape": list(baseline.shape),
    "bitwise_equal": bool(np.array_equal(baseline, candidate)),
    "max_abs": int(error.max(initial=0)),
    "changed_value_fraction": float(np.count_nonzero(error) / error.size),
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
# mountpoint-s3 does not support preserving POSIX directory timestamps.
cp -r "${SMOKE_RESULTS}" "${RESULTS}/"
if [[ "${MINWM_BENCHMARK_MODE}" == "smoke" ]]; then
  echo "MINWM_B200_SMOKE_COMPLETE results=${RESULTS}"
  exit 0
fi

FULL_RESULTS="${LOCAL_RESULTS}/ten-case"
set +e
MINWM_ATTENTION_IMPL=packed \
MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  bash "${SCRIPT_DIR}/run_all.sh" "${FULL_RESULTS}" \
  > >(tee "${RESULTS}/ten-case.log") 2>&1
bitwise_status=$?
set -e
cp -r "${FULL_RESULTS}" "${RESULTS}/"
if [[ -f "${FULL_RESULTS}/report.json" ]]; then
  cp "${FULL_RESULTS}/report.json" "${RESULTS}/ten-case-bitwise-report.json"
fi

numeric_status=0
if (( bitwise_status != 0 )); then
  set +e
  python3 "${SCRIPT_DIR}/compare_results.py" \
    --results "${FULL_RESULTS}" \
    --profile bf16_backend_candidate \
    | tee "${RESULTS}/ten-case-numeric-compare.log"
  numeric_status=${PIPESTATUS[0]}
  set -e
  cp "${FULL_RESULTS}/report.json" "${RESULTS}/ten-case-numeric-report.json"
fi
else
  bitwise_status=0
  numeric_status=0
fi

run_throughput_profile() {
  local profile="$1" attention_impl="$2" packed_deterministic="$3"
  local native_components="$4" torch_compile="$5" kv_frames="$6"
  local profile_dir="${LOCAL_RESULTS}/throughput/${profile}"
  local profile_results="${RESULTS}/throughput/${profile}"
  mkdir -p "${profile_dir}" "${profile_results}"
  MINWM_ATTENTION_IMPL="${attention_impl}" \
  MINWM_PACKED_ATTENTION_DETERMINISTIC="${packed_deterministic}" \
  MINWM_NATIVE_COMPONENTS="${native_components}" \
  sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --attention-backend fa \
    --performance-mode speed \
    --enable-torch-compile "${torch_compile}" \
    --warmup-mode off \
    --port 30000 \
    > "${profile_results}/server.log" 2>&1 &
  local profile_server_pid=$!
  if ! wait_for_server "${profile_server_pid}" "${profile_results}/server.log"; then
    kill "${profile_server_pid}" 2>/dev/null || true
    wait "${profile_server_pid}" 2>/dev/null || true
    return 1
  fi
  set +e
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    --output "${profile_dir}/throughput.json" \
    --profile-name "${profile}" \
    --kv-cache-num-frames "${kv_frames}" \
    > >(tee "${profile_results}/client.log") 2>&1
  local profile_status=$?
  set -e
  kill "${profile_server_pid}" 2>/dev/null || true
  wait "${profile_server_pid}" 2>/dev/null || true
  if [[ -f "${profile_dir}/throughput.json" ]]; then
    cp "${profile_dir}/throughput.json" "${profile_results}/"
  fi
  return "${profile_status}"
}

profiles=(
  "exact-packed-det-kv128 packed true text_encoder,vae false 128"
  "exact-packed-det-kv45 packed true text_encoder,vae false 45"
  "packed-nondeterministic-kv45 packed false text_encoder,vae false 45"
  "lingbot-style-dense-native-kv45 dense false text_encoder,vae false 45"
  "dense-optimized-components-kv45 dense false '' false 45"
  "dense-optimized-compile-kv45 dense false '' true 45"
)
profile_failures=()
for spec in "${profiles[@]}"; do
  read -r profile attention_impl packed_deterministic native_components torch_compile kv_frames <<< "${spec}"
  if [[ "${native_components}" == "''" ]]; then
    native_components=""
  fi
  if ! run_throughput_profile \
    "${profile}" "${attention_impl}" "${packed_deterministic}" \
    "${native_components}" "${torch_compile}" "${kv_frames}"; then
    profile_failures+=("${profile}")
  fi
done

python3 - "${RESULTS}/throughput" "${RESULTS}/throughput-summary.json" \
  "${profile_failures[*]}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
profiles = {}
for path in sorted(root.glob("*/throughput.json")):
    data = json.loads(path.read_text())
    profiles[data["profile_name"]] = data
summary = {
    "profiles": profiles,
    "failed_profiles": sys.argv[3].split() if sys.argv[3] else [],
}
exact_name = "exact-packed-det-kv45"
comparisons = {}
if exact_name in profiles:
    exact = profiles[exact_name]
    for name, value in profiles.items():
        if name == exact_name:
            continue
        metric_deltas = {}
        for label, getter in {
            "scheduler_fps": lambda item: item["server"].get(
                "scheduler_forward_fps_ratio_of_sums"
            ),
            "client_fps": lambda item: item["client"][
                "steady_received_fps_ratio_of_sums"
            ],
        }.items():
            exact_fps = getter(exact)
            candidate_fps = getter(value)
            if exact_fps and candidate_fps:
                metric_deltas[label] = {
                    "exact": exact_fps,
                    "candidate": candidate_fps,
                    "candidate_speedup_over_exact_pct": 100
                    * (candidate_fps / exact_fps - 1),
                    "exact_throughput_loss_vs_candidate_pct": 100
                    * (1 - exact_fps / candidate_fps),
                }
        comparisons[f"{name}_vs_{exact_name}"] = metric_deltas
summary["comparisons"] = comparisons
isolated_pairs = {
    "deterministic_packed_tax": (
        "exact-packed-det-kv45",
        "packed-nondeterministic-kv45",
    ),
    "packed_vs_lingbot_style_dense": (
        "exact-packed-det-kv45",
        "lingbot-style-dense-native-kv45",
    ),
    "native_component_tax": (
        "lingbot-style-dense-native-kv45",
        "dense-optimized-components-kv45",
    ),
    "whole_compile_effect": (
        "dense-optimized-components-kv45",
        "dense-optimized-compile-kv45",
    ),
    "kv128_vs_kv45_effect": (
        "exact-packed-det-kv128",
        "exact-packed-det-kv45",
    ),
}
isolated = {}
for label, (reference_name, candidate_name) in isolated_pairs.items():
    if reference_name not in profiles or candidate_name not in profiles:
        continue
    reference = profiles[reference_name]
    candidate = profiles[candidate_name]
    isolated[label] = {}
    for metric, getter in {
        "scheduler_fps": lambda item: item["server"].get(
            "scheduler_forward_fps_ratio_of_sums"
        ),
        "client_fps": lambda item: item["client"][
            "steady_received_fps_ratio_of_sums"
        ],
    }.items():
        reference_fps = getter(reference)
        candidate_fps = getter(candidate)
        if reference_fps and candidate_fps:
            isolated[label][metric] = {
                "reference_profile": reference_name,
                "candidate_profile": candidate_name,
                "reference": reference_fps,
                "candidate": candidate_fps,
                "candidate_speedup_pct": 100
                * (candidate_fps / reference_fps - 1),
                "reference_throughput_loss_vs_candidate_pct": 100
                * (1 - reference_fps / candidate_fps),
            }
summary["isolated_comparisons"] = isolated
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    name: {
        "scheduler_fps": value["server"].get("scheduler_forward_fps_ratio_of_sums"),
        "client_fps": value["client"]["steady_received_fps_ratio_of_sums"],
    }
    for name, value in profiles.items()
}, indent=2, sort_keys=True))
PY

echo "MINWM_B200_FULL_COMPLETE results=${RESULTS} bitwise_status=${bitwise_status} numeric_status=${numeric_status} profile_failures=${profile_failures[*]}"
if (( bitwise_status != 0 && numeric_status != 0 )); then
  exit "${numeric_status}"
fi
if (( ${#profile_failures[@]} != 0 )); then
  exit 1
fi
