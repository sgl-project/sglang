#!/usr/bin/env bash
set -Eeuo pipefail

# Remote SANA-WM validation and benchmark runner.
#
# Modes:
#   quick              unit + 1-GPU CI smoke + component accuracy
#   gt                 generate local consistency GT frames using the CI path
#   benchmark          online serving benchmark + offline throughput
#   all                quick + perf baseline + benchmark
#   serve-benchmark    online serving benchmark only
#   offline-benchmark  offline throughput only
#   action-benchmark   SANA-WM action/camera manifest stress test
#   perf-baseline      CI perf baseline generation only
#   full-quality       one native-size sglang generate run

MODE="${1:-all}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SANA_WM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${SANA_WM_PYTHON_BIN:-python3}"

MODEL_PATH="${SANA_WM_MODEL_PATH:-Efficient-Large-Model/SANA-WM_bidirectional}"
PIPELINE_CLASS="${SANA_WM_PIPELINE_CLASS:-SanaWMTwoStagePipeline}"
PORT="${SANA_WM_PORT:-30000}"
BASE_URL="${SANA_WM_BASE_URL:-http://127.0.0.1:${PORT}}"
CUDA_DEVICES="${SANA_WM_CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${SANA_WM_NUM_GPUS:-1}"
OUTPUT_ROOT="${SANA_WM_OUTPUT_ROOT:-/workspace/outputs/sana_wm}"
ASSET_DIR="${SANA_WM_ASSET_DIR:-/workspace/Sana/asset/sana_wm}"
IMAGE_DIR="${SANA_WM_IMAGE_DIR:-${ASSET_DIR}}"
IMAGE_PATH="${SANA_WM_IMAGE_PATH:-${ASSET_DIR}/demo_0.png}"
PROMPT_FILE="${SANA_WM_PROMPT_FILE:-${ASSET_DIR}/demo_0.txt}"
INTRINSICS_PATH="${SANA_WM_INTRINSICS_PATH:-${ASSET_DIR}/demo_0_intrinsics.npy}"
CAMERA_TO_WORLD_PATH="${SANA_WM_CAMERA_TO_WORLD_PATH:-}"

WIDTH="${SANA_WM_WIDTH:-640}"
HEIGHT="${SANA_WM_HEIGHT:-384}"
NUM_FRAMES="${SANA_WM_NUM_FRAMES:-17}"
FPS="${SANA_WM_FPS:-16}"
STEPS="${SANA_WM_STEPS:-12}"
GUIDANCE_SCALE="${SANA_WM_GUIDANCE_SCALE:-4.5}"
SERVING_NUM_PROMPTS="${SANA_WM_SERVING_NUM_PROMPTS:-64}"
ACTION_NUM_REQUESTS="${SANA_WM_ACTION_NUM_REQUESTS:-64}"
CONCURRENCY="${SANA_WM_CONCURRENCY:-2}"
REQUEST_RATE="${SANA_WM_REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${SANA_WM_WARMUP_REQUESTS:-1}"
SLO_SCALE="${SANA_WM_SLO_SCALE:-3.0}"
POLL_TIMEOUT="${SANA_WM_POLL_TIMEOUT:-1800}"
OFFLINE_NUM_PROMPTS="${SANA_WM_OFFLINE_NUM_PROMPTS:-16}"
OFFLINE_BATCH_SIZE="${SANA_WM_OFFLINE_BATCH_SIZE:-1}"

FULL_WIDTH="${SANA_WM_FULL_WIDTH:-1280}"
FULL_HEIGHT="${SANA_WM_FULL_HEIGHT:-704}"
FULL_NUM_FRAMES="${SANA_WM_FULL_NUM_FRAMES:-49}"
FULL_STEPS="${SANA_WM_FULL_STEPS:-20}"

CLIENT="${REPO_ROOT}/python/sglang/multimodal_gen/test/scripts/sana_wm_batch_client.py"
LOG_DIR="${OUTPUT_ROOT}/logs"
MANIFEST="${OUTPUT_ROOT}/sana_wm_action_manifest.jsonl"
ACTION_RESULTS="${OUTPUT_ROOT}/sana_wm_action_results.jsonl"
ACTION_SUMMARY="${OUTPUT_ROOT}/sana_wm_action_summary.json"
ACTION_OUTPUT_DIR="${OUTPUT_ROOT}/action_batch_outputs"
GT_OUTPUT_DIR="${SANA_WM_GT_OUTPUT_DIR:-${OUTPUT_ROOT}/consistency_gt}"
SERVER_LOG="${LOG_DIR}/sana_wm_server.log"
SERVER_PID=""

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${ACTION_OUTPUT_DIR}"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_unit_tests() {
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" -m unittest -v \
    sglang.multimodal_gen.test.unit.test_sana_wm_pipeline_config \
    sglang.multimodal_gen.test.unit.test_server_args.TestOffloadDefaults.test_auto_multi_gpu_sana_wm_prefers_fsdp_and_cfg_parallel \
    sglang.multimodal_gen.test.unit.test_server_args.TestModelIdResolution.test_sana_wm_model_path_resolves_registry \
    2>&1 | tee "${LOG_DIR}/unit.log"
}

run_ci_smoke() {
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" -m pytest -s python/sglang/multimodal_gen/test/server/test_server_1_gpu.py \
    -k sana_wm_ti2v \
    2>&1 | tee "${LOG_DIR}/ci_smoke.log"
}

run_component_accuracy() {
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" -m pytest -s python/sglang/multimodal_gen/test/server/test_component_accuracy_1_gpu.py \
    -k sana_wm_ti2v \
    2>&1 | tee "${LOG_DIR}/component_accuracy.log"
}

run_consistency_gt() {
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" python/sglang/multimodal_gen/test/scripts/gen_diffusion_ci_outputs.py \
    --suite 1-gpu \
    --case-ids sana_wm_ti2v \
    --out-dir "${GT_OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/consistency_gt.log"
}

run_perf_baseline() {
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" python/sglang/multimodal_gen/test/scripts/gen_perf_baselines.py \
    --case sana_wm_ti2v \
    --out "${OUTPUT_ROOT}/sana_wm_perf_baselines.json" \
    --timeout "${POLL_TIMEOUT}" \
    2>&1 | tee "${LOG_DIR}/perf_baseline.log"
}

start_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    return
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  sglang serve \
    --model-type diffusion \
    --model-path "${MODEL_PATH}" \
    --pipeline-class-name "${PIPELINE_CLASS}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --num-gpus "${NUM_GPUS}" \
    --input-save-path "${OUTPUT_ROOT}/server_inputs" \
    --output-path "${OUTPUT_ROOT}/server_outputs" \
    ${SANA_WM_EXTRA_SERVE_ARGS:-} \
    > "${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"

  "${PYTHON_BIN}" "${CLIENT}" wait-health --base-url "${BASE_URL}" --timeout "${POLL_TIMEOUT}"
}

run_serving_vbench() {
  local dataset_args=()
  if [[ -n "${SANA_WM_DATASET_PATH:-}" ]]; then
    dataset_args+=(--dataset-path "${SANA_WM_DATASET_PATH}")
  fi

  "${PYTHON_BIN}" -m sglang.multimodal_gen.benchmarks.bench_serving \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --dataset vbench \
    --task image-to-video \
    "${dataset_args[@]}" \
    --num-prompts "${SERVING_NUM_PROMPTS}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --num-frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --num-inference-steps "${STEPS}" \
    --max-concurrency "${CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --slo \
    --slo-scale "${SLO_SCALE}" \
    --output-file "${OUTPUT_ROOT}/vbench_i2v_serving_metrics.json" \
    2>&1 | tee "${LOG_DIR}/vbench_i2v_serving.log"
}

run_action_manifest() {
  local intrinsics_args=()
  if [[ -f "${INTRINSICS_PATH}" ]]; then
    intrinsics_args+=(--intrinsics-path "${INTRINSICS_PATH}")
  fi
  if [[ -n "${CAMERA_TO_WORLD_PATH}" && -f "${CAMERA_TO_WORLD_PATH}" ]]; then
    intrinsics_args+=(--camera-to-world-path "${CAMERA_TO_WORLD_PATH}")
  fi

  "${PYTHON_BIN}" "${CLIENT}" build-manifest \
    --out "${MANIFEST}" \
    --model "${MODEL_PATH}" \
    --num-requests "${ACTION_NUM_REQUESTS}" \
    --image-dir "${IMAGE_DIR}" \
    --image-path "${IMAGE_PATH}" \
    --prompt-file "${PROMPT_FILE}" \
    --output-dir "${ACTION_OUTPUT_DIR}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --num-frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --num-inference-steps "${STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    "${intrinsics_args[@]}"

  "${PYTHON_BIN}" "${CLIENT}" run-manifest \
    --manifest "${MANIFEST}" \
    --base-url "${BASE_URL}" \
    --concurrency "${CONCURRENCY}" \
    --poll-timeout "${POLL_TIMEOUT}" \
    --results "${ACTION_RESULTS}" \
    --summary "${ACTION_SUMMARY}" \
    --fail-on-error \
    2>&1 | tee "${LOG_DIR}/action_manifest.log"

  "${PYTHON_BIN}" "${CLIENT}" validate-videos \
    --results "${ACTION_RESULTS}" \
    --output-dir "${ACTION_OUTPUT_DIR}" \
    --expected-width "${WIDTH}" \
    --expected-height "${HEIGHT}" \
    --expected-frames "${NUM_FRAMES}" \
    --expected-fps "${FPS}" \
    --summary "${OUTPUT_ROOT}/action_video_validation.json"
}

run_offline_benchmark() {
  local random_config_file="${OUTPUT_ROOT}/offline_random_config.json"
  SANA_WM_IMAGE_PATH_FOR_CONFIG="${IMAGE_PATH}" \
  SANA_WM_INTRINSICS_PATH_FOR_CONFIG="${INTRINSICS_PATH}" \
  SANA_WM_OFFLINE_OUTPUT_DIR="${OUTPUT_ROOT}/offline_outputs" \
  "${PYTHON_BIN}" - <<'PY' > "${random_config_file}"
import json
import os

image_path = os.environ["SANA_WM_IMAGE_PATH_FOR_CONFIG"]
intrinsics_path = os.environ.get("SANA_WM_INTRINSICS_PATH_FOR_CONFIG", "")
output_dir = os.environ["SANA_WM_OFFLINE_OUTPUT_DIR"]
actions = ["w-16", "s-16", "a-16", "d-16", "jw-8,w-8", "lw-8,w-8"]
profiles = []
for action in actions:
    kwargs = {
        "action": action,
        "translation_speed": 0.05,
        "rotation_speed_deg": 1.2,
        "pitch_limit_deg": 85,
    }
    if intrinsics_path:
        kwargs["intrinsics_path"] = intrinsics_path
    profiles.append(
        {
            "weight": 1,
            "image_path": image_path,
            "output_path": output_dir,
            "diffusers_kwargs": kwargs,
        }
    )
print(json.dumps(profiles))
PY

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${PYTHON_BIN}" -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \
    --model-path "${MODEL_PATH}" \
    --pipeline-class-name "${PIPELINE_CLASS}" \
    --dataset random \
    --num-prompts "${OFFLINE_NUM_PROMPTS}" \
    --batch-size "${OFFLINE_BATCH_SIZE}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --num-frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --num-inference-steps "${STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --random-request-config "$(cat "${random_config_file}")" \
    --output-file "${OUTPUT_ROOT}/offline_action_throughput.jsonl" \
    2>&1 | tee "${LOG_DIR}/offline_action_throughput.log"
}

run_full_quality() {
  mkdir -p "${OUTPUT_ROOT}/full_quality"
  local kwargs_file="${OUTPUT_ROOT}/full_quality_diffusers_kwargs.json"
  SANA_WM_INTRINSICS_PATH_FOR_CONFIG="${INTRINSICS_PATH}" \
  SANA_WM_CAMERA_TO_WORLD_PATH_FOR_CONFIG="${CAMERA_TO_WORLD_PATH}" \
  "${PYTHON_BIN}" - <<'PY' > "${kwargs_file}"
import json
import os

kwargs = {}
intrinsics_path = os.environ.get("SANA_WM_INTRINSICS_PATH_FOR_CONFIG", "")
camera_path = os.environ.get("SANA_WM_CAMERA_TO_WORLD_PATH_FOR_CONFIG", "")
if intrinsics_path:
    kwargs["intrinsics_path"] = intrinsics_path
if camera_path:
    kwargs["camera_to_world_path"] = camera_path
else:
    kwargs.update(
        {
            "action": "w-32,jw-16,w-16",
            "translation_speed": 0.05,
            "rotation_speed_deg": 1.2,
            "pitch_limit_deg": 85,
        }
    )
print(json.dumps(kwargs))
PY

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  sglang generate \
    --model-path "${MODEL_PATH}" \
    --pipeline-class-name "${PIPELINE_CLASS}" \
    --image-path "${IMAGE_PATH}" \
    --prompt-path "${PROMPT_FILE}" \
    --height "${FULL_HEIGHT}" \
    --width "${FULL_WIDTH}" \
    --num-frames "${FULL_NUM_FRAMES}" \
    --fps "${FPS}" \
    --num-inference-steps "${FULL_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --negative-prompt "" \
    --diffusers-kwargs "$(cat "${kwargs_file}")" \
    --output-file-path "${OUTPUT_ROOT}/full_quality/full_quality_${FULL_NUM_FRAMES}f.mp4" \
    --perf-dump-path "${OUTPUT_ROOT}/full_quality_perf.json" \
    2>&1 | tee "${LOG_DIR}/full_quality.log"
}

run_quick() {
  run_unit_tests
  run_ci_smoke
  run_component_accuracy
}

run_benchmark() {
  start_server
  run_serving_vbench
  cleanup
  SERVER_PID=""
  run_offline_benchmark
}

case "${MODE}" in
  quick)
    run_quick
    ;;
  gt)
    run_consistency_gt
    ;;
  benchmark)
    run_benchmark
    ;;
  all)
    run_quick
    run_perf_baseline
    run_benchmark
    ;;
  serve-benchmark)
    start_server
    run_serving_vbench
    ;;
  offline-benchmark)
    run_offline_benchmark
    ;;
  action-benchmark)
    start_server
    run_action_manifest
    ;;
  perf-baseline)
    run_perf_baseline
    ;;
  full-quality)
    run_full_quality
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

echo "SANA-WM ${MODE} completed. Outputs: ${OUTPUT_ROOT}"
