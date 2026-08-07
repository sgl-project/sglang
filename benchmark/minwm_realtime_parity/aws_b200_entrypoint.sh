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
  || "${MINWM_BENCHMARK_MODE}" == "setup_only" \
  || "${MINWM_BENCHMARK_MODE}" == "full" \
  || "${MINWM_BENCHMARK_MODE}" == "profiles" \
  || "${MINWM_BENCHMARK_MODE}" == "long720p" \
  || "${MINWM_BENCHMARK_MODE}" == "triptych720p" \
  || "${MINWM_BENCHMARK_MODE}" == "spmatrix720p" \
  || "${MINWM_BENCHMARK_MODE}" == "cudagraphmatrix" \
  || "${MINWM_BENCHMARK_MODE}" == "bounded5s" \
  || "${MINWM_BENCHMARK_MODE}" == "calibratedfp8" \
  || "${MINWM_BENCHMARK_MODE}" == "nvfp4" \
  || "${MINWM_BENCHMARK_MODE}" == "dragon60s" ]] || {
  echo "unsupported MINWM_BENCHMARK_MODE=${MINWM_BENCHMARK_MODE}" >&2
  exit 2
}

WORK_ROOT="/work/minwm-realtime/${MINWM_RUN_ID}"
REUSE_INPUT_ROOT=""
if [[ -n "${MINWM_REUSE_INPUT_RUN_ID:-}" ]]; then
  [[ "${MINWM_REUSE_INPUT_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "MINWM_REUSE_INPUT_RUN_ID contains unsupported characters" >&2
    exit 2
  }
  [[ "${MINWM_REUSE_INPUT_RUN_ID}" != "${MINWM_RUN_ID}" ]] || {
    echo "MINWM_REUSE_INPUT_RUN_ID must differ from MINWM_RUN_ID" >&2
    exit 2
  }
  REUSE_INPUT_ROOT="/work/minwm-realtime/${MINWM_REUSE_INPUT_RUN_ID}"
  CHECKPOINT="${REUSE_INPUT_ROOT}/checkpoint/model.pt"
  PRETRAINED="${REUSE_INPUT_ROOT}/pretrained"
else
  CHECKPOINT="${WORK_ROOT}/checkpoint/model.pt"
  PRETRAINED="${WORK_ROOT}/pretrained"
fi
if [[ "${MINWM_STAGE_FROM_MOUNT:-0}" == "1" ]]; then
  : "${MINWM_CHECKPOINT_MOUNT_PATH:?set MINWM_CHECKPOINT_MOUNT_PATH}"
  : "${MINWM_PRETRAINED_MOUNT_PATH:?set MINWM_PRETRAINED_MOUNT_PATH}"
fi
if [[ -n "${MINWM_REUSE_MODEL_RUN_ID:-}" ]]; then
  [[ "${MINWM_REUSE_MODEL_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "MINWM_REUSE_MODEL_RUN_ID contains unsupported characters" >&2
    exit 2
  }
  [[ "${MINWM_REUSE_MODEL_RUN_ID}" != "${MINWM_RUN_ID}" ]] || {
    echo "MINWM_REUSE_MODEL_RUN_ID must differ from MINWM_RUN_ID" >&2
    exit 2
  }
  MODEL_DIR="/work/minwm-realtime/${MINWM_REUSE_MODEL_RUN_ID}/sglang-model"
else
  MODEL_DIR="${WORK_ROOT}/sglang-model"
fi
LOCAL_RESULTS="${WORK_ROOT}/results"
RESULTS="${MINWM_RESULTS_ROOT%/}/${MINWM_RUN_ID}"
SCRIPT_DIR="/workspace/sglang/benchmark/minwm_realtime_parity"
MINWM_CONFIG="${MINWM_CONFIG_PATH:-/workspace/minWM/Wan21/configs/eval/wan22_5b_varlen_dmd.yaml}"
mkdir -p "${LOCAL_RESULTS}" "${RESULTS}"
if [[ -z "${REUSE_INPUT_ROOT}" ]]; then
  mkdir -p "${WORK_ROOT}/checkpoint" "${PRETRAINED}"
fi

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
if [[ -n "${REUSE_INPUT_ROOT}" ]]; then
  echo "reused-local-stage:${MINWM_REUSE_INPUT_RUN_ID}" \
    | tee "${RESULTS}/staging-mode.txt"
  [[ -f "${CHECKPOINT}" ]]
  [[ -d "${PRETRAINED}/transformer" ]]
  python3 - "${RESULTS}/checkpoint-head.json" <<'PY'
import json, os, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "ContentLength": int(os.environ["MINWM_CHECKPOINT_BYTES"]),
    "ChecksumCRC64NVME": os.environ["MINWM_CHECKPOINT_CRC64"],
    "VersionId": os.environ["MINWM_CHECKPOINT_VERSION_EAST"],
    "verification": "reused immutable local stage verified against pinned metadata",
}, indent=2, sort_keys=True) + "\n")
PY
elif [[ "${MINWM_STAGE_FROM_MOUNT:-0}" == "1" ]]; then
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

if [[ "${MINWM_SKIP_INSTALL:-0}" != "1" ]]; then
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
  if [[ "${MINWM_PRESERVE_IMAGE_GPU_STACK:-0}" == "1" ]]; then
    # SM120 images carry a matched Torch/FA4/CUTLASS stack. Resolving the full
    # diffusion extra can downgrade that stack to packages without a usable
    # varlen FA4 kernel, so install SGLang and only its Python runtime deps.
    python3 -m pip install -e /workspace/sglang/python \
      --no-deps --root-user-action=ignore
    python3 -m pip install \
      'fastapi==0.141.1' \
      'IPython==9.16.0' \
      'pyzmq==27.1.0' \
      --root-user-action=ignore
    python3 -m pip install \
      'compressed-tensors' \
      'diffusers==0.37.0' \
      'gguf' \
      'partial-json-parser' \
      'transformers==5.12.1' \
      --root-user-action=ignore
    python3 -m pip install --no-deps \
      'addict==2.4.0' \
      'msgspec==0.21.1' \
      'orjson==3.11.9' \
      'prometheus-client==0.26.0' \
      'pybase64==1.4.3' \
      'python-multipart==0.0.32' \
      'sgl-deep-gemm==0.1.4' \
      'setproctitle==1.3.7' \
      'uvicorn==0.52.0' \
      'uvloop==0.22.1' \
      'watchfiles==1.2.0' \
      'websockets==17.0.1' \
      'zstandard==0.25.0' \
      --root-user-action=ignore
    if [[ -n "${MINWM_SGLANG_KERNEL_WHEEL_PATH:-}" ]]; then
      [[ -f "${MINWM_SGLANG_KERNEL_WHEEL_PATH}" ]]
      kernel_wheel_install_dir="$(mktemp -d)"
      kernel_wheel_install_path="${kernel_wheel_install_dir}/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl"
      cp "${MINWM_SGLANG_KERNEL_WHEEL_PATH}" "${kernel_wheel_install_path}"
      python3 -m pip install --no-deps \
        "${kernel_wheel_install_path}" \
        --root-user-action=ignore
    elif [[ "${MINWM_BUILD_SGLANG_KERNEL_FROM_SOURCE:-1}" == "1" ]]; then
      # The SM120 image carries a newer Torch ABI than the released
      # sglang-kernel wheel. Build the exact checkout against the image's
      # Torch/CUDA stack so the FP8 ops are both SM120-capable and ABI-safe.
      python3 -m pip install \
        'cmake>=3.31' \
        ninja \
        'nvidia-cuda-cccl==13.0.85' \
        'nvidia-cuda-crt==13.0.88' \
        'nvidia-cuda-nvcc==13.0.88' \
        'nvidia-nvvm==13.0.88' \
        scikit-build-core \
        uv \
        --root-user-action=ignore
      export CUDA_HOME=/opt/minwm/venv/lib/python3.12/site-packages/nvidia/cu13
      export CUDACXX="${CUDA_HOME}/bin/nvcc"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      if [[ ! -e "${CUDA_HOME}/lib64" ]]; then
        ln -s lib "${CUDA_HOME}/lib64"
      fi
      if [[ ! -e "${CUDA_HOME}/lib/libcudart.so" ]]; then
        ln -s libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
      fi
      nvcc --version
      kernel_wheel_dir="$(mktemp -d)"
      CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${CUDACXX} -DCUDA_VERSION=13.0 -DCUDA_nvrtc_LIBRARY=${CUDA_HOME}/lib/libnvrtc.so.13 -DENABLE_BELOW_SM90=OFF -DSGL_KERNEL_ENABLE_FA3=OFF -DSGL_KERNEL_COMPILE_THREADS=2" \
        CMAKE_BUILD_PARALLEL_LEVEL=8 \
        MAX_JOBS=8 \
        python3 -m pip wheel /workspace/sglang/sgl-kernel \
          --no-build-isolation --no-deps --wheel-dir "${kernel_wheel_dir}"
      kernel_wheel_path="$(find "${kernel_wheel_dir}" -maxdepth 1 -type f -name '*.whl' -print -quit)"
      [[ -n "${kernel_wheel_path}" ]]
      python3 -m pip install --no-deps \
        "${kernel_wheel_path}" \
        --root-user-action=ignore
      if [[ -n "${MINWM_SGLANG_KERNEL_WHEEL_OUTPUT_PATH:-}" ]]; then
        mkdir -p "${MINWM_SGLANG_KERNEL_WHEEL_OUTPUT_PATH%/*}"
        cp "${kernel_wheel_path}" "${MINWM_SGLANG_KERNEL_WHEEL_OUTPUT_PATH}"
      fi
    else
      python3 -m pip install --no-deps \
        'sglang-kernel==0.4.4' \
        --root-user-action=ignore
    fi
  else
    python3 -m pip install -e "/workspace/sglang/python[diffusion]" \
      --root-user-action=ignore
  fi
  # The minWM training image pins peft==0.17.0, while this SGLang checkout pins
  # transformers==5.12.1.  Merely leaving the old PEFT package installed makes
  # diffusers detect and import it, which then fails on the removed HybridCache
  # symbol before any model code runs.  Realtime inference does not load LoRA
  # adapters, so remove that stale optional package instead of changing either
  # side's model/runtime dependency set.
  python3 -m pip uninstall -y peft
  if [[ "${MINWM_PRESERVE_IMAGE_GPU_STACK:-0}" != "1" ]]; then
    python3 -m pip install --force-reinstall --no-deps \
      --index-url https://flashinfer.ai/whl/cu130 \
      'flashinfer-jit-cache==0.6.12+cu130'
  fi
else
  echo "Skipped dependency installation for reused in-Pod benchmark environment"
fi
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

if [[ -n "${MINWM_REUSE_MODEL_RUN_ID:-}" ]]; then
  [[ -f "${MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors.index.json" ]]
  [[ -f "${MODEL_DIR}/minwm_conversion_manifest.json" ]]
  echo "reused-local-model:${MINWM_REUSE_MODEL_RUN_ID}" \
    | tee "${RESULTS}/conversion.log"
else
  convert_args=(
    --rope-position-mode "${MINWM_CONVERT_ROPE_POSITION_MODE:-absolute}"
    --rope-max-frame-gap "${MINWM_CONVERT_ROPE_MAX_FRAME_GAP:-1}"
  )
  if [[ "${MINWM_CONVERT_PROMPT_FIRST_FRAME_PIN_ENABLED:-false}" == "true" ]]; then
    convert_args+=(--prompt-first-frame-pin-enabled)
  fi
  python3 /workspace/sglang/python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py \
    --minwm-checkpoint "${CHECKPOINT}" \
    --donor-diffusers-dir "${PRETRAINED}" \
    --output-dir "${MODEL_DIR}" \
    --link-donor \
    --source-uri "${MINWM_CHECKPOINT_SOURCE_URI}" \
    --source-version-id "${MINWM_CHECKPOINT_SOURCE_VERSION}" \
    --source-etag "${MINWM_CHECKPOINT_SOURCE_ETAG}" \
    --local-attn-size "${MINWM_CONVERT_LOCAL_ATTN_SIZE:--1}" \
    --sink-size "${MINWM_CONVERT_SINK_SIZE:-0}" \
    --sliding-window-num-frames "${MINWM_CONVERT_WINDOW_SIZE:-128}" \
    "${convert_args[@]}" \
    | tee "${RESULTS}/conversion.log"
fi
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

if [[ "${MINWM_BENCHMARK_MODE}" == "setup_only" ]]; then
  python3 - <<'PY'
import json
import os

print(json.dumps({
    "checkpoint": os.environ["MINWM_CHECKPOINT"],
    "config": os.environ["MINWM_CONFIG"],
    "model_dir": os.environ["MINWM_MODEL_DIR"],
    "pretrained_dir": os.environ["MINWM_PRETRAINED_DIR"],
    "status": "ready",
}, sort_keys=True))
PY
  echo "MINWM_SETUP_ONLY_COMPLETE run_id=${MINWM_RUN_ID}"
  exit 0
fi

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

if [[ "${MINWM_BENCHMARK_MODE}" == "dragon60s" ]]; then
  DRAGON_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_dragon_ride_60s_832x480.json}"
  DRAGON_RESULTS="${RESULTS}/dragon-ride-60s-bitwise"
  DRAGON_SINK_SIZE="${MINWM_SINK_SIZE:-4}"
  DRAGON_PIPELINE_CLASS_NAME="${MINWM_PIPELINE_CLASS_NAME:-MinWMCausalDMDPipeline}"
  ARTIFACT_HOLD_SECONDS="${MINWM_ARTIFACT_HOLD_SECONDS:-0}"
  if ! [[ "${ARTIFACT_HOLD_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_ARTIFACT_HOLD_SECONDS must be a non-negative integer" >&2
    exit 2
  fi
  [[ -f "${MINWM_CONFIG}" ]]
  if [[ -n "${MINWM_CONFIG_SHA256:-}" ]]; then
    echo "${MINWM_CONFIG_SHA256}  ${MINWM_CONFIG}" | sha256sum --check -
  fi
  python3 - "${MODEL_DIR}/transformer/config.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
assert int(config["local_attn_size"]) == -1, config
print(json.dumps({
    "converted_local_attn_size": config["local_attn_size"],
    "converted_sink_size": config["sink_size"],
    "converted_sliding_window_num_frames": config["sliding_window_num_frames"],
}, sort_keys=True))
PY
  mkdir -p "${DRAGON_RESULTS}"
  if python3 - "${DRAGON_CASES}" "${DRAGON_RESULTS}" \
    "${MINWM_GIT_REF:-}" "${MINWM_CONFIG_SHA256:-}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
run_path = root / "baseline_run.json"
if not run_path.is_file():
    raise SystemExit(1)
run = json.loads(run_path.read_text())
if sys.argv[3] and run["minwm_git_sha"] != sys.argv[3]:
    raise SystemExit(1)
if sys.argv[4] and run["config_sha256"] != sys.argv[4]:
    raise SystemExit(1)
expected_shape = (
    int(manifest["contract"]["reference_pixel_frames"])
    + int(manifest["contract"]["generated_pixel_frames"]),
    int(manifest["contract"]["height"]),
    int(manifest["contract"]["width"]),
    3,
)
for case in manifest["cases"]:
    case_dir = root / "cases" / case["id"]
    required = [
        case_dir / "baseline.npy",
        case_dir / "baseline.mp4",
        case_dir / "baseline.json",
        case_dir / "baseline_latents.pt",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise SystemExit(1)
    if np.load(required[0], mmap_mode="r", allow_pickle=False).shape != expected_shape:
        raise SystemExit(1)
print("MINWM_DRAGON60_RESUME baseline=valid")
PY
  then
    echo "Skipping completed baseline after retry"
  else
    python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
      --cases "${DRAGON_CASES}" \
      --minwm-root /workspace/minWM \
      --checkpoint "${CHECKPOINT}" \
      --pretrained-dir "${PRETRAINED}" \
      --config "${MINWM_CONFIG}" \
      --results "${DRAGON_RESULTS}" \
      | tee "${RESULTS}/dragon60-baseline.log"
  fi

  MINWM_ATTENTION_IMPL=packed \
  MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
  MINWM_NATIVE_COMPONENTS=text_encoder,vae \
  MINWM_SEGMENT_COMPILE=true \
  sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name "${DRAGON_PIPELINE_CLASS_NAME}" \
    --attention-backend fa \
    --performance-mode speed \
    --enable-torch-compile false \
    --warmup-mode off \
    --port 30000 \
    > "${RESULTS}/dragon60-sglang-server.log" 2>&1 &
  SERVER_PID=$!
  (
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
      nvidia-smi --query-gpu=timestamp,memory.used \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "${RESULTS}/dragon60-gpu-memory.csv" &
  GPU_MONITOR_PID=$!
  cleanup_dragon_server() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  }
  trap cleanup_dragon_server EXIT INT TERM
  if ! wait_for_server "${SERVER_PID}" "${RESULTS}/dragon60-sglang-server.log"; then
    exit 1
  fi
  python3 "${SCRIPT_DIR}/run_sglang_api.py" \
    --cases "${DRAGON_CASES}" \
    --results "${DRAGON_RESULTS}" \
    --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate \
    --sink-size "${DRAGON_SINK_SIZE}" \
    | tee "${RESULTS}/dragon60-sglang-client.log"
  cleanup_dragon_server
  trap - EXIT INT TERM

  set +e
  python3 "${SCRIPT_DIR}/compare_results.py" \
    --cases "${DRAGON_CASES}" \
    --results "${DRAGON_RESULTS}" \
    --profile bitwise \
    | tee "${RESULTS}/dragon60-bitwise-compare.log"
  bitwise_status=${PIPESTATUS[0]}
  set -e
  cp "${DRAGON_RESULTS}/report.json" "${RESULTS}/dragon60-bitwise-report.json"
  if (( bitwise_status != 0 )); then
    set +e
    python3 "${SCRIPT_DIR}/compare_results.py" \
      --cases "${DRAGON_CASES}" \
      --results "${DRAGON_RESULTS}" \
      --profile bf16_backend_candidate \
      | tee "${RESULTS}/dragon60-numeric-compare.log"
    numeric_status=${PIPESTATUS[0]}
    set -e
    cp "${DRAGON_RESULTS}/report.json" "${RESULTS}/dragon60-numeric-report.json"
  else
    numeric_status=0
  fi

  python3 - "${DRAGON_RESULTS}" "${RESULTS}/dragon60-summary.json" \
    "${bitwise_status}" "${numeric_status}" "${DRAGON_SINK_SIZE}" \
    "${DRAGON_PIPELINE_CLASS_NAME}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline = json.loads((root / "baseline_run.json").read_text())
sglang = json.loads((root / "sglang_run.json").read_text())
exact = json.loads((Path(root).parent / "dragon60-bitwise-report.json").read_text())
assert baseline["sink_size"] == int(sys.argv[5]), baseline
case = sglang["cases"][0]
steady = [
    item for item in case["chunk_stats"] if int(item["chunk_index"]) > 0
]
scheduler_ms = [float(item["scheduler_forward_ms"]) for item in steady]
client_ms = [
    float(value)
    for value in case["client_timing"]["steady_payload_interarrival_ms"]
]
frames = sum(int(item["num_frames"]) for item in steady)
summary = {
    "contract": {
        "generated_seconds": 60.0,
        "fps": 24,
        "generated_pixel_frames": 1440,
        "reference_pixel_frames": 1,
        "chunks": 90,
        "baseline_local_attn_size": baseline["local_attn_size"],
        "baseline_configured_window_size": baseline["window_size"],
        "baseline_sink_size": baseline["sink_size"],
        "sglang_runtime_window": case["request"].get(
            "realtime_causal_kv_cache_num_frames"
        ),
        "sglang_runtime_sink": case["request"].get(
            "realtime_causal_sink_size"
        ),
        "sglang_pipeline_class": sys.argv[6],
        "effective_cache_semantics": (
            "unbounded local_attn_size=-1; YAML window_size is not consumed "
            "by minWM main causal inference"
        ),
    },
    "baseline_git_sha": baseline["minwm_git_sha"],
    "baseline_config_sha256": baseline["config_sha256"],
    "bitwise_status": int(sys.argv[3]),
    "numeric_status": int(sys.argv[4]),
    "report_summary": exact["summary"],
    "steady_scheduler_fps": (
        frames / (sum(scheduler_ms) / 1000) if scheduler_ms else None
    ),
    "steady_client_fps": (
        frames / (sum(client_ms) / 1000) if client_ms else None
    ),
    "steady_chunk_p50_ms": statistics.median(client_ms) if client_ms else None,
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  touch "${RESULTS}/dragon60-artifacts-ready"
  echo "MINWM_DRAGON60_COMPLETE results=${RESULTS} bitwise_status=${bitwise_status}"
  if (( ARTIFACT_HOLD_SECONDS > 0 )); then
    echo "MINWM_ARTIFACT_HOLD seconds=${ARTIFACT_HOLD_SECONDS}"
    sleep "${ARTIFACT_HOLD_SECONDS}"
  fi
  exit "${bitwise_status}"
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "bounded5s" ]]; then
  BOUNDED_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_action_control_kv_roll_832x480.json}"
  BOUNDED_LOCAL_ATTN_SIZE="${MINWM_LOCAL_ATTN_SIZE:-18}"
  BOUNDED_SINK_SIZE="${MINWM_SINK_SIZE:-9}"
  BOUNDED_RESULTS="${LOCAL_RESULTS}/bounded-window-parity-5s"
  mkdir -p "${BOUNDED_RESULTS}"
  export MINWM_CASES_PATH="${BOUNDED_CASES}"
  export MINWM_LOCAL_ATTN_SIZE="${BOUNDED_LOCAL_ATTN_SIZE}"
  export MINWM_SINK_SIZE="${BOUNDED_SINK_SIZE}"
  export MINWM_PARITY_PROFILE=bitwise
  export MINWM_ENABLE_TORCH_COMPILE=false
  export MINWM_ATTENTION_IMPL=packed
  export MINWM_PACKED_ATTENTION_DETERMINISTIC=true
  export MINWM_NATIVE_COMPONENTS=text_encoder,vae
  export MINWM_PARITY_DUMP_DIR="${BOUNDED_RESULTS}/parity-dumps"

  set +e
  bash "${SCRIPT_DIR}/run_all.sh" "${BOUNDED_RESULTS}" \
    | tee "${RESULTS}/bounded-window-run.log"
  bitwise_status=${PIPESTATUS[0]}
  set -e
  cp "${BOUNDED_RESULTS}/report.json" "${RESULTS}/bounded-bitwise-report.json"

  numeric_status=0
  if (( bitwise_status != 0 )); then
    set +e
    python3 "${SCRIPT_DIR}/compare_results.py" \
      --cases "${BOUNDED_CASES}" \
      --results "${BOUNDED_RESULTS}" \
      --profile bf16_backend_candidate \
      | tee "${RESULTS}/bounded-numeric-compare.log"
    numeric_status=${PIPESTATUS[0]}
    set -e
    cp "${BOUNDED_RESULTS}/report.json" "${RESULTS}/bounded-numeric-report.json"
  fi

  python3 - "${BOUNDED_RESULTS}" "${RESULTS}/bounded-summary.json" \
    "${bitwise_status}" "${numeric_status}" \
    "${BOUNDED_LOCAL_ATTN_SIZE}" "${BOUNDED_SINK_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline = json.loads((root / "baseline_run.json").read_text())
sglang = json.loads((root / "sglang_run.json").read_text())
report = json.loads((root / "report.json").read_text())
request = sglang["cases"][0]["request"]
summary = {
    "contract": {
        "local_attn_size": int(sys.argv[5]),
        "sink_size": int(sys.argv[6]),
        "window_includes_sink_and_current_chunk": True,
        "baseline_local_attn_size": baseline["local_attn_size"],
        "baseline_sink_size": baseline["sink_size"],
        "sglang_runtime_window": request["realtime_causal_kv_cache_num_frames"],
        "sglang_runtime_sink": request["realtime_causal_sink_size"],
    },
    "bitwise_status": int(sys.argv[3]),
    "numeric_status": int(sys.argv[4]),
    "report_summary": report["summary"],
    "baseline_git_sha": baseline["minwm_git_sha"],
    "checkpoint_size": baseline["checkpoint_size"],
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  cp -r "${BOUNDED_RESULTS}" "${RESULTS}/"
  touch "${RESULTS}/bounded-window-artifacts-ready"
  echo "MINWM_BOUNDED5S_COMPLETE results=${RESULTS} bitwise_status=${bitwise_status} numeric_status=${numeric_status}"
  if (( bitwise_status != 0 && numeric_status != 0 )); then
    exit "${numeric_status}"
  fi
  exit 0
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "calibratedfp8" ]]; then
  python3 -m pytest -q \
    /workspace/sglang/python/sglang/multimodal_gen/test/unit/test_minwm_static_fp8_transformer.py
  if [[ -n "${MINWM_REUSE_STATIC_FP8_RUN_ID:-}" ]]; then
    [[ "${MINWM_REUSE_STATIC_FP8_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]
    STATIC_FP8_TRANSFORMER="/work/minwm-realtime/${MINWM_REUSE_STATIC_FP8_RUN_ID}/static-fp8-transformer"
    [[ -f "${STATIC_FP8_TRANSFORMER}/minwm_static_fp8_manifest.json" ]]
    echo "reused-static-fp8:${MINWM_REUSE_STATIC_FP8_RUN_ID}" \
      | tee "${RESULTS}/static-fp8-calibration.log"
  else
    CALIBRATION_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
    CALIBRATION_RESULTS="${LOCAL_RESULTS}/static-fp8-calibration"
    CALIBRATION_PATH="${CALIBRATION_RESULTS}/static-fp8-calibration.json"
    STATIC_FP8_TRANSFORMER="${WORK_ROOT}/static-fp8-transformer"
    mkdir -p "${CALIBRATION_RESULTS}"
    python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
      --cases "${CALIBRATION_CASES}" \
      --minwm-root /workspace/minWM \
      --checkpoint "${CHECKPOINT}" \
      --pretrained-dir "${PRETRAINED}" \
      --config "${MINWM_CONFIG}" \
      --results "${CALIBRATION_RESULTS}" \
      --fp8-calibration-output "${CALIBRATION_PATH}" \
      | tee "${RESULTS}/static-fp8-calibration.log"
    python3 -m sglang.multimodal_gen.tools.build_minwm_static_fp8_transformer \
      --input-dir "${MODEL_DIR}/transformer" \
      --calibration "${CALIBRATION_PATH}" \
      --output-dir "${STATIC_FP8_TRANSFORMER}" \
      --activation-margin "${MINWM_STATIC_FP8_MARGIN:-1.0}" \
      --module-scope "${MINWM_STATIC_FP8_SCOPE:-all}" \
      | tee "${RESULTS}/static-fp8-conversion.log"
    cp "${CALIBRATION_PATH}" "${RESULTS}/"
    cp "${CALIBRATION_RESULTS}/baseline_run.json" "${RESULTS}/static-fp8-calibration-baseline.json"
  fi
  cp "${STATIC_FP8_TRANSFORMER}/minwm_static_fp8_manifest.json" "${RESULTS}/"
  export MINWM_PROFILE_TRANSFORMER_PATH="${STATIC_FP8_TRANSFORMER}"
  export MINWM_PROFILE_QUANTIZATION_LABEL="${MINWM_PROFILE_QUANTIZATION_LABEL:-static_fp8}"
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "nvfp4" ]]; then
  python3 -m pytest -q \
    /workspace/sglang/python/sglang/multimodal_gen/test/unit/test_minwm_nvfp4_transformer.py
  NVFP4_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
  NVFP4_RESULTS="${LOCAL_RESULTS}/nvfp4-calibration"
  NVFP4_DUMP_DIR="${NVFP4_RESULTS}/parity-dumps"
  NVFP4_TRANSFORMER="${WORK_ROOT}/nvfp4-transformer"
  mkdir -p "${NVFP4_RESULTS}"
  MINWM_PARITY_DUMP_DIR="${NVFP4_DUMP_DIR}" \
    python3 "${SCRIPT_DIR}/run_minwm_baseline.py" \
      --cases "${NVFP4_CASES}" \
      --minwm-root /workspace/minWM \
      --checkpoint "${CHECKPOINT}" \
      --pretrained-dir "${PRETRAINED}" \
      --config "${MINWM_CONFIG}" \
      --results "${NVFP4_RESULTS}" \
      | tee "${RESULTS}/nvfp4-calibration.log"
  python3 -m sglang.multimodal_gen.tools.build_minwm_nvfp4_transformer \
    --input-dir "${MODEL_DIR}/transformer" \
    --output-dir "${NVFP4_TRANSFORMER}" \
    --minwm-root /workspace/minWM \
    --checkpoint "${CHECKPOINT}" \
    --pretrained-dir "${PRETRAINED}" \
    --config "${MINWM_CONFIG}" \
    --calibration-forward "${NVFP4_DUMP_DIR}/baseline/forward_000.pt" \
    | tee "${RESULTS}/nvfp4-conversion.log"
  cp "${NVFP4_RESULTS}/baseline_run.json" "${RESULTS}/nvfp4-calibration-baseline.json"
  cp "${NVFP4_TRANSFORMER}/minwm_nvfp4_manifest.json" "${RESULTS}/"
  export MINWM_PROFILE_TRANSFORMER_PATH="${NVFP4_TRANSFORMER}"
  export MINWM_PROFILE_QUANTIZATION_LABEL="${MINWM_PROFILE_QUANTIZATION_LABEL:-nvfp4}"
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "cudagraphmatrix" ]]; then
  CG_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
  CG_CASE="${MINWM_CUDA_GRAPH_CASE:-00_forward_080_pottery_720p}"
  CG_RESULTS="${RESULTS}/cuda-graph-matrix"
  CG_DEGREES="${MINWM_CUDA_GRAPH_SP_DEGREES:-1 2}"
  CG_WARMUP_CHUNKS="${MINWM_CUDA_GRAPH_WARMUP_CHUNKS:-10}"
  CG_MEASURED_CHUNKS="${MINWM_CUDA_GRAPH_MEASURED_CHUNKS:-30}"
  CG_WINDOW="${MINWM_CUDA_GRAPH_KV_FRAMES:-20}"
  mkdir -p "${CG_RESULTS}"
  nvidia-smi -L | tee "${CG_RESULTS}/gpus.txt"
  nvidia-smi topo -m | tee "${CG_RESULTS}/topology.txt"

  python3 -m pytest -q \
    /workspace/sglang/python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
    -k 'cuda_graph or cache_plan_is_shared_across_layers_and_reused_for_recompute or accepts_supported_ulysses_sequence_parallelism'
  python3 -m pytest -q \
    /workspace/sglang/python/sglang/multimodal_gen/test/unit/test_server_args.py \
    -k speed_mode_cuda_graph_does_not_enable_torch_compile

  python3 - "${MODEL_DIR}/transformer/config.json" "${CG_WINDOW}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
assert config["rope_position_mode"] == "block_relative", config
assert int(config["local_attn_size"]) == int(sys.argv[2]), config
assert int(config["sliding_window_num_frames"]) == int(sys.argv[2]), config
print(json.dumps({
    key: config[key]
    for key in (
        "local_attn_size",
        "sink_size",
        "sliding_window_num_frames",
        "rope_position_mode",
        "rope_max_frame_gap",
    )
}, sort_keys=True))
PY

  run_cuda_graph_lane() {
    local degree="$1" enabled="$2"
    local suffix="eager"
    if [[ "${enabled}" == "true" ]]; then
      suffix="cuda-graph"
    fi
    local profile="sp${degree}-${suffix}"
    local profile_dir="${CG_RESULTS}/${profile}"
    mkdir -p "${profile_dir}"
    MINWM_ATTENTION_IMPL="${MINWM_ATTENTION_IMPL:-packed}" \
    MINWM_PACKED_ATTENTION_DETERMINISTIC=true \
    MINWM_NATIVE_COMPONENTS=text_encoder,vae \
    MINWM_VAE_LANE=parity \
    sglang serve \
      --model-path "${MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --vae-config.use-parallel-decode true \
      --vae-config.parallel-decode-mode auto \
      --attention-backend "${MINWM_SERVER_ATTENTION_BACKEND:-fa}" \
      --performance-mode speed \
      --num-gpus "${degree}" \
      --tp-size 1 \
      --sp-degree "${degree}" \
      --ulysses-degree "${degree}" \
      --ring-degree 1 \
      --enable-cfg-parallel false \
      --enable-torch-compile false \
      --enable-cuda-graph "${enabled}" \
      --warmup-mode off \
      --port 30000 \
      > "${profile_dir}/server.log" 2>&1 &
    local server_pid=$!
    if ! wait_for_server "${server_pid}" "${profile_dir}/server.log"; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
      return 1
    fi
    (
      while kill -0 "${server_pid}" 2>/dev/null; do
        nvidia-smi \
          --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
          --format=csv,noheader,nounits || true
        sleep 1
      done
    ) > "${profile_dir}/gpu.csv" &
    local monitor_pid=$!
    set +e
    python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
      --cases "${CG_CASES}" \
      --case "${CG_CASE}" \
      --output "${profile_dir}/throughput.json" \
      --profile-name "${profile}" \
      --warmup-chunks "${CG_WARMUP_CHUNKS}" \
      --measured-chunks "${CG_MEASURED_CHUNKS}" \
      --kv-cache-num-frames "${CG_WINDOW}" \
      > >(tee "${profile_dir}/client.log") 2>&1
    local lane_status=$?
    set -e
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    if (( lane_status != 0 )); then
      tail -300 "${profile_dir}/server.log" >&2
    fi
    return "${lane_status}"
  }

  read -r -a cg_degrees <<< "${CG_DEGREES}"
  for degree in "${cg_degrees[@]}"; do
    if ! [[ "${degree}" =~ ^(1|2)$ ]]; then
      echo "unsupported CUDA graph SP degree: ${degree}" >&2
      exit 2
    fi
    run_cuda_graph_lane "${degree}" false
    run_cuda_graph_lane "${degree}" true
  done

  python3 - "${CG_RESULTS}" <<'PY' | tee "${CG_RESULTS}/summary.log"
import base64
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
profiles = {
    path.parent.name: json.load(open(path))
    for path in sorted(root.glob("*/throughput.json"))
}
summary = {"profiles": profiles, "comparisons": {}}


def denoise_metrics(profile_name, warmup_chunks):
    values = []
    log_path = root / profile_name / "server.log"
    for line in log_path.read_text().splitlines():
        if "realtime_trace {" not in line:
            continue
        try:
            event = json.loads(line[line.index("{"):])
        except (ValueError, json.JSONDecodeError):
            continue
        if (
            event.get("event") == "server.model_denoise_complete"
            and event.get("component") == "minwm_denoising"
            and "source" not in event
            and "error" not in event
            and int(event.get("chunk_index", -1)) >= warmup_chunks
        ):
            values.append(float(event["duration_ms"]))
    if not values:
        return None
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "latent_fps_ratio_of_sums": 4 * len(values) / (sum(values) / 1000.0),
        "pixel_fps_ratio_of_sums": 16 * len(values) / (sum(values) / 1000.0),
    }


def sampled_pixel_error(eager, graph):
    eager_samples = eager["measured_payload_samples_base64"]
    graph_samples = graph["measured_payload_samples_base64"]
    if eager_samples.keys() != graph_samples.keys():
        raise AssertionError("eager/CUDA graph sampled payload keys differ")
    absolute_error_sum = 0
    squared_error_sum = 0
    equal_count = 0
    max_error = 0
    count = 0
    for key in eager_samples:
        eager_bytes = base64.b64decode(eager_samples[key])
        graph_bytes = base64.b64decode(graph_samples[key])
        if len(eager_bytes) != len(graph_bytes):
            raise AssertionError(f"sampled payload length differs for {key}")
        for eager_value, graph_value in zip(eager_bytes, graph_bytes):
            error = abs(eager_value - graph_value)
            absolute_error_sum += error
            squared_error_sum += error * error
            equal_count += int(error == 0)
            max_error = max(max_error, error)
            count += 1
    mse = squared_error_sum / count
    return {
        "sample_count": count,
        "mae_u8": absolute_error_sum / count,
        "rmse_u8": math.sqrt(mse),
        "psnr_db": None if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
        "max_abs_u8": max_error,
        "exact_fraction": equal_count / count,
    }


for profile_name, profile in profiles.items():
    profile["dit_denoise"] = denoise_metrics(
        profile_name, int(profile["warmup_chunks"])
    )
for degree in (1, 2):
    eager_name = f"sp{degree}-eager"
    graph_name = f"sp{degree}-cuda-graph"
    if eager_name not in profiles or graph_name not in profiles:
        continue
    eager = profiles[eager_name]
    graph = profiles[graph_name]
    comparison = {}
    comparison["measured_payload_sha256"] = {
        "eager": eager["measured_payload_sha256"],
        "cuda_graph": graph["measured_payload_sha256"],
        "equal": eager["measured_payload_sha256"]
        == graph["measured_payload_sha256"],
    }
    comparison["sampled_pixel_error"] = sampled_pixel_error(eager, graph)
    for name, getter in {
        "scheduler_fps": lambda item: item["server"]["scheduler_forward_fps_ratio_of_sums"],
        "client_fps": lambda item: item["client"]["steady_received_fps_ratio_of_sums"],
        "scheduler_p50_ms": lambda item: item["server"]["scheduler_forward_ms"]["p50"],
        "dit_denoise_pixel_fps": lambda item: item["dit_denoise"]["pixel_fps_ratio_of_sums"],
        "dit_denoise_p50_ms": lambda item: item["dit_denoise"]["p50_ms"],
    }.items():
        baseline = getter(eager)
        candidate = getter(graph)
        comparison[name] = {
            "eager": baseline,
            "cuda_graph": candidate,
            "speedup": baseline / candidate if name.endswith("_ms") else candidate / baseline,
        }
    summary["comparisons"][f"sp{degree}"] = comparison
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["comparisons"], indent=2, sort_keys=True))
parity_failures = {
    degree: comparison["sampled_pixel_error"]
    for degree, comparison in summary["comparisons"].items()
    if not comparison["measured_payload_sha256"]["equal"]
    or comparison["sampled_pixel_error"]["max_abs_u8"] != 0
}
if parity_failures:
    raise AssertionError(
        f"MinWM CUDA graph payload parity failed: {parity_failures}"
    )
PY
  echo "MINWM_CUDA_GRAPH_MATRIX_COMPLETE results=${CG_RESULTS}"
  exit 0
fi

if [[ "${MINWM_BENCHMARK_MODE}" == "spmatrix720p" ]]; then
  SP_CASES="${MINWM_CASES_PATH:-${SCRIPT_DIR}/cases_720p_compile_smoke.json}"
  # Keep completed lanes directly on the mounted result store. Spot retries can
  # then skip valid SP artifacts instead of discarding an entire matrix.
  SP_RESULTS="${MINWM_SP_RESULTS_DIR:-${RESULTS}/sp-matrix-720p}"
  SP_WARMUP_RUNS="${MINWM_SP_WARMUP_RUNS:-1}"
  SP_DEGREES="${MINWM_SP_DEGREES:-1 2 4 8}"
  SP_VAE_LANE="${MINWM_SP_VAE_LANE:-parity}"
  SP_VAE_PARALLEL_DECODE="${MINWM_SP_VAE_PARALLEL_DECODE:-true}"
  ARTIFACT_HOLD_SECONDS="${MINWM_ARTIFACT_HOLD_SECONDS:-0}"
  if ! [[ "${SP_WARMUP_RUNS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_SP_WARMUP_RUNS must be a non-negative integer" >&2
    exit 2
  fi
  if ! [[ "${ARTIFACT_HOLD_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "MINWM_ARTIFACT_HOLD_SECONDS must be a non-negative integer" >&2
    exit 2
  fi
  if ! [[ "${SP_VAE_LANE}" =~ ^(parity|parallel)$ ]]; then
    echo "MINWM_SP_VAE_LANE must be parity or parallel" >&2
    exit 2
  fi
  if ! [[ "${SP_VAE_PARALLEL_DECODE}" =~ ^(true|false)$ ]]; then
    echo "MINWM_SP_VAE_PARALLEL_DECODE must be true or false" >&2
    exit 2
  fi
  mkdir -p "${SP_RESULTS}"
  SP_RESULTS_S3=""
  if [[ -n "${MINWM_RESULTS_S3_URI:-}" ]]; then
    SP_RESULTS_S3="${MINWM_RESULTS_S3_URI%/}/${MINWM_RUN_ID}"
    aws s3 sync "${SP_RESULTS_S3}/" "${RESULTS}/" \
      --no-progress --only-show-errors
  fi
  nvidia-smi -L | tee "${RESULTS}/sp-gpus.txt"
  nvidia-smi topo -m | tee "${RESULTS}/sp-topology.txt"

  sync_sp_results() {
    [[ -n "${SP_RESULTS_S3}" ]] || return 0
    aws s3 sync "${RESULTS}/" "${SP_RESULTS_S3}/" \
      --no-progress --only-show-errors
  }

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
    MINWM_VAE_LANE="${SP_VAE_LANE}" \
    CUDA_LAUNCH_BLOCKING="${MINWM_CUDA_LAUNCH_BLOCKING:-0}" \
    sglang serve \
      --model-path "${MODEL_DIR}" \
      --pipeline-class-name MinWMCausalDMDPipeline \
      --vae-config.use-parallel-decode "${SP_VAE_PARALLEL_DECODE}" \
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
      --engine-name "sglang-minwm-${prefix}-ulysses-${SP_VAE_LANE}-vae-parallel-${SP_VAE_PARALLEL_DECODE}" \
      --warmup-runs "${SP_WARMUP_RUNS}" \
      | tee "${SP_RESULTS}/${prefix}-client.log"
    local lane_status=${PIPESTATUS[0]}
    set -e
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    if (( lane_status == 0 )); then
      sync_sp_results
    else
      echo "MINWM_SP_LANE_FAILED degree=${degree}; server log follows" >&2
      tail -300 "${server_log}" >&2
    fi
    return "${lane_status}"
  }

  read -r -a sp_degrees <<< "${SP_DEGREES}"
  for degree in "${sp_degrees[@]}"; do
    if ! [[ "${degree}" =~ ^(1|2|4|8)$ ]]; then
      echo "MINWM_SP_DEGREES contains unsupported degree: ${degree}" >&2
      exit 2
    fi
    if sp_lane_complete "sp${degree}"; then
      echo "MINWM_SP_RESUME_SKIP degree=${degree}"
    else
      run_sp_lane "${degree}"
    fi
  done
  if [[ "${sp_degrees[*]}" != "1 2 4 8" ]]; then
    echo "MINWM_SP_PARTIAL_COMPLETE degrees=${sp_degrees[*]} vae_lane=${SP_VAE_LANE} vae_parallel_decode=${SP_VAE_PARALLEL_DECODE} results=${SP_RESULTS}"
    sync_sp_results
    exit 0
  fi
  python3 "${SCRIPT_DIR}/compare_sp_matrix.py" \
    --cases "${SP_CASES}" \
    --results "${SP_RESULTS}" \
    --degrees 1,2,4,8 \
    | tee "${RESULTS}/sp-matrix-compare.log"
  cp "${SP_RESULTS}/sp_matrix_report.json" "${RESULTS}/"
  cp "${SP_RESULTS}/sp_matrix_report.md" "${RESULTS}/"
  touch "${RESULTS}/sp-matrix-artifacts-ready"
  sync_sp_results
  echo "MINWM_B200_SP_MATRIX_COMPLETE vae_lane=${SP_VAE_LANE} vae_parallel_decode=${SP_VAE_PARALLEL_DECODE} results=${RESULTS}"
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

if [[ "${MINWM_BENCHMARK_MODE}" != "profiles" \
  && "${MINWM_BENCHMARK_MODE}" != "calibratedfp8" \
  && "${MINWM_BENCHMARK_MODE}" != "nvfp4" ]]; then
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
  local quantization_args=()
  local transformer_args=()
  local client_args=(
    --output "${profile_dir}/throughput.json"
    --profile-name "${profile}"
    --kv-cache-num-frames "${kv_frames}"
  )
  if [[ -n "${MINWM_PROFILE_QUANTIZATION:-}" ]]; then
    quantization_args=(--quantization "${MINWM_PROFILE_QUANTIZATION}")
  fi
  if [[ -n "${MINWM_PROFILE_TRANSFORMER_PATH:-}" ]]; then
    transformer_args=(--transformer-path "${MINWM_PROFILE_TRANSFORMER_PATH}")
  fi
  if [[ -n "${MINWM_THROUGHPUT_CASES_PATH:-}" ]]; then
    [[ -f "${MINWM_THROUGHPUT_CASES_PATH}" ]]
    client_args+=(--cases "${MINWM_THROUGHPUT_CASES_PATH}")
  fi
  if [[ -n "${MINWM_THROUGHPUT_CASE:-}" ]]; then
    client_args+=(--case "${MINWM_THROUGHPUT_CASE}")
  fi
  if [[ -n "${MINWM_THROUGHPUT_WARMUP_CHUNKS:-}" ]]; then
    client_args+=(--warmup-chunks "${MINWM_THROUGHPUT_WARMUP_CHUNKS}")
  fi
  if [[ -n "${MINWM_THROUGHPUT_MEASURED_CHUNKS:-}" ]]; then
    client_args+=(--measured-chunks "${MINWM_THROUGHPUT_MEASURED_CHUNKS}")
  fi
  mkdir -p "${profile_dir}" "${profile_results}"
  MINWM_ATTENTION_IMPL="${attention_impl}" \
  MINWM_PACKED_ATTENTION_DETERMINISTIC="${packed_deterministic}" \
  MINWM_NATIVE_COMPONENTS="${native_components}" \
  sglang serve \
    --model-path "${MODEL_DIR}" \
    --pipeline-class-name MinWMCausalDMDPipeline \
    --attention-backend fa \
    --performance-mode speed \
    "${quantization_args[@]}" \
    "${transformer_args[@]}" \
    --enable-torch-compile "${torch_compile}" \
    --warmup-mode off \
    --port 30000 \
    > "${profile_results}/server.log" 2>&1 &
  local profile_server_pid=$!
  (
    while kill -0 "${profile_server_pid}" 2>/dev/null; do
      nvidia-smi \
        --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits || true
      sleep 1
    done
  ) > "${profile_results}/gpu-memory.csv" &
  local profile_gpu_monitor_pid=$!
  if ! wait_for_server "${profile_server_pid}" "${profile_results}/server.log"; then
    kill "${profile_server_pid}" 2>/dev/null || true
    wait "${profile_server_pid}" 2>/dev/null || true
    kill "${profile_gpu_monitor_pid}" 2>/dev/null || true
    wait "${profile_gpu_monitor_pid}" 2>/dev/null || true
    return 1
  fi
  set +e
  python3 "${SCRIPT_DIR}/benchmark_realtime_throughput.py" \
    "${client_args[@]}" \
    > >(tee "${profile_results}/client.log") 2>&1
  local profile_status=$?
  set -e
  kill "${profile_server_pid}" 2>/dev/null || true
  wait "${profile_server_pid}" 2>/dev/null || true
  kill "${profile_gpu_monitor_pid}" 2>/dev/null || true
  wait "${profile_gpu_monitor_pid}" 2>/dev/null || true
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
)
if [[ "${MINWM_INCLUDE_COMPILE_PROFILE:-true}" == "true" ]]; then
  profiles+=("dense-optimized-compile-kv45 dense false '' true 45")
fi
if [[ -n "${MINWM_THROUGHPUT_PROFILES:-}" ]]; then
  selected_profiles=()
  for spec in "${profiles[@]}"; do
    read -r profile _ <<< "${spec}"
    if [[ ",${MINWM_THROUGHPUT_PROFILES}," == *",${profile},"* ]]; then
      selected_profiles+=("${spec}")
    fi
  done
  if (( ${#selected_profiles[@]} == 0 )); then
    echo "No throughput profiles matched MINWM_THROUGHPUT_PROFILES=${MINWM_THROUGHPUT_PROFILES}" >&2
    exit 1
  fi
  profiles=("${selected_profiles[@]}")
fi
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
  "${profile_failures[*]}" \
  "${MINWM_PROFILE_QUANTIZATION_LABEL:-${MINWM_PROFILE_QUANTIZATION:-}}" \
  "${SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND:-}" <<'PY'
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
    "quantization": sys.argv[4] or None,
    "nvfp4_backend": sys.argv[5] or None,
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
