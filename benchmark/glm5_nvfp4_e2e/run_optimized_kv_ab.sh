#!/usr/bin/env bash
# One-round, fair-capacity GLM-5 TP8 FP8-vs-NVFP4 KV-cache E2E validation.
# Run this script inside the SGLang enroot image on one exclusive 8-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the GLM-5 checkpoint}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH must point to the benchmark dataset}"
SGL_KERNEL_OVERLAY="${SGL_KERNEL_OVERLAY:?SGL_KERNEL_OVERLAY must point to the staged sgl_kernel package}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR must name a persistent result directory}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="http://${HOST}:${PORT}"
COMMON_CAPACITY="${COMMON_CAPACITY:-497984}"
RUN_VARIANTS="${RUN_VARIANTS:-fp8 nvfp4}"
RUN_CASES="${RUN_CASES:-all}"
RUN_NATIVE_GATES="${RUN_NATIVE_GATES:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
export REPO_ROOT MODEL_PATH DATASET_PATH SGL_KERNEL_OVERLAY RESULT_DIR

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SGL_KERNEL_OVERLAY}:${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/glm5-optimized-ab-triton}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-/tmp/glm5-optimized-ab-sglang}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/tmp/glm5-optimized-ab-deepgemm}"

mkdir -p \
  "${RESULT_DIR}/environment" \
  "${RESULT_DIR}/server_logs" \
  "${RESULT_DIR}/server_info" \
  "${RESULT_DIR}/bench_logs" \
  "${RESULT_DIR}/raw" \
  "${RESULT_DIR}/telemetry" \
  "${TRITON_CACHE_DIR}" \
  "${SGLANG_CACHE_DIR}" \
  "${SGLANG_DG_CACHE_DIR}"

exec > >(tee -a "${RESULT_DIR}/pipeline.log") 2>&1

SERVER_PID=""
TELEMETRY_PID=""

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID}" ]] && kill -0 "${TELEMETRY_PID}" 2>/dev/null; then
    kill -TERM "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

stop_server() {
  stop_telemetry
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  local server_pid="${SERVER_PID}"
  SERVER_PID=""
  # The setsid leader can exit while TP workers remain in its process group.
  # Always signal/check the whole group instead of returning when the leader
  # itself is already gone.
  kill -TERM -- "-${server_pid}" 2>/dev/null || kill -TERM "${server_pid}" 2>/dev/null || true
  for _ in $(seq 1 90); do
    if ! kill -0 -- "-${server_pid}" 2>/dev/null \
      && ! kill -0 "${server_pid}" 2>/dev/null; then
      wait "${server_pid}" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  kill -KILL -- "-${server_pid}" 2>/dev/null || kill -KILL "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}

trap 'rc=$?; stop_server; exit "$rc"' EXIT INT TERM

capture_environment() {
  {
    echo "captured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_root=${REPO_ROOT}"
    echo "model_path=${MODEL_PATH}"
    echo "dataset_path=${DATASET_PATH}"
    echo "sgl_kernel_overlay=${SGL_KERNEL_OVERLAY}"
    echo "common_capacity=${COMMON_CAPACITY}"
    echo "git_head=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "git_branch=$(git -C "${REPO_ROOT}" branch --show-current)"
    echo "container_cuda_home=${CUDA_HOME}"
  } >"${RESULT_DIR}/environment/manifest.txt"
  git -C "${REPO_ROOT}" status --short --branch >"${RESULT_DIR}/environment/git_status.txt"
  git -C "${REPO_ROOT}" diff --binary HEAD >"${RESULT_DIR}/environment/tracked_worktree.patch"
  cp "${REPO_ROOT}/benchmark/glm5_nvfp4_e2e/run_optimized_kv_ab.sh" \
    "${RESULT_DIR}/environment/run_optimized_kv_ab.sh"
  cp "${REPO_ROOT}/benchmark/glm5_nvfp4_e2e/summarize_optimized_kv_ab.py" \
    "${RESULT_DIR}/environment/summarize_optimized_kv_ab.py"
  sha256sum \
    "${REPO_ROOT}/python/sglang/srt/layers/attention/dsa/nvfp4_k_cache.py" \
    "${REPO_ROOT}/python/sglang/srt/layers/attention/dsv4/nvfp4_k_cache.py" \
    "${RESULT_DIR}/environment/run_optimized_kv_ab.sh" \
    "${RESULT_DIR}/environment/summarize_optimized_kv_ab.py" \
    "${SGL_KERNEL_OVERLAY}/sgl_kernel/flashmla_ops.abi3.so" \
    "${SGL_KERNEL_OVERLAY}/sgl_kernel/flash_ops.abi3.so" \
    "${MODEL_PATH}/config.json" \
    "${MODEL_PATH}/model.safetensors.index.json" \
    >"${RESULT_DIR}/environment/sha256.txt"
  nvidia-smi -L >"${RESULT_DIR}/environment/nvidia_smi_l.txt"
  nvidia-smi topo -m >"${RESULT_DIR}/environment/nvidia_topology.txt"
  nvcc --version >"${RESULT_DIR}/environment/nvcc.txt"
  python3 - <<'PY' >"${RESULT_DIR}/environment/python_runtime.txt"
import json
import os
from pathlib import Path

import torch
import sglang
import sgl_kernel
import sgl_kernel.flash_mla

repo = Path(os.environ["REPO_ROOT"]).resolve()
overlay = Path(os.environ["SGL_KERNEL_OVERLAY"]).resolve()
model = Path(os.environ["MODEL_PATH"]).resolve()
config = json.loads((model / "config.json").read_text())
devices = [
    {
        "name": torch.cuda.get_device_name(i),
        "capability": torch.cuda.get_device_capability(i),
    }
    for i in range(torch.cuda.device_count())
]
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "sglang": sglang.__file__,
    "sgl_kernel": sgl_kernel.__file__,
    "flashmla_module": sgl_kernel.flash_mla.__file__,
    "devices": devices,
    "has_nvfp4_op": hasattr(torch.ops.sgl_kernel, "fwd_kvcache_mla_nvfp4"),
    "nvfp4_op_schema": str(torch.ops.sgl_kernel.fwd_kvcache_mla_nvfp4.default._schema),
}, sort_keys=True))
assert Path(sglang.__file__).resolve().is_relative_to(repo / "python")
assert Path(sgl_kernel.__file__).resolve().is_relative_to(overlay)
assert sgl_kernel.flash_mla._flashmla_import_error is None
assert torch.cuda.device_count() == 8
assert all(item["capability"] == (9, 0) for item in devices)
assert hasattr(torch.ops.sgl_kernel, "fwd_kvcache_mla_nvfp4")
assert config["architectures"][0] == "GlmMoeDsaForCausalLM"
assert config["index_topk"] == 2048
assert config["kv_lora_rank"] == 512
assert config["qk_rope_head_dim"] == 64
assert config["num_attention_heads"] % 8 == 0
PY
}

run_native_gates() {
  local dsa_log="${RESULT_DIR}/environment/test_dsa_nvfp4_k_cache.log"
  local flashmla_log="${RESULT_DIR}/environment/test_flashmla_nvfp4.log"
  CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
    "${REPO_ROOT}/python/sglang/jit_kernel/tests/test_dsa_nvfp4_k_cache.py" \
    2>&1 | tee "${dsa_log}"
  grep -Eq '[1-9][0-9]* passed' "${dsa_log}"
  CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
    "${REPO_ROOT}/sgl-kernel/tests/test_flashmla_nvfp4.py" \
    2>&1 | tee "${flashmla_log}"
  grep -Eq '[1-9][0-9]* passed' "${flashmla_log}"
}

start_telemetry() {
  local variant="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv -l 1 \
    >"${RESULT_DIR}/telemetry/${variant}.csv" 2>&1 &
  TELEMETRY_PID=$!
}

start_server() {
  local variant="$1"
  local server_log="${RESULT_DIR}/server_logs/${variant}.log"
  local -a kv_args
  if [[ "${variant}" == "fp8" ]]; then
    kv_args=(--kv-cache-dtype fp8_e4m3)
  elif [[ "${variant}" == "nvfp4" ]]; then
    kv_args=(--kv-cache-dtype fp4_e2m1 --fp4-kv-cache-recipe nvfp4)
  else
    echo "unknown variant: ${variant}" >&2
    return 2
  fi

  local -a command=(
    python3 -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --served-model-name GLM-5-FP8
    --quantization fp8
    --dtype bfloat16
    --tp-size 8
    --trust-remote-code
    --attention-backend dsa
    --dsa-prefill-backend flashmla_sparse
    --dsa-decode-backend flashmla_kv
    --page-size 64
    --chunked-prefill-size 8192
    --max-prefill-tokens 16384
    --context-length 131072
    --mem-fraction-static 0.85
    --max-total-tokens "${COMMON_CAPACITY}"
    --max-running-requests 32
    --disable-radix-cache
    --cuda-graph-max-bs 32
    --cuda-graph-bs 1 2 4 8 16 32
    --random-seed 42
    --watchdog-timeout 3600
    --disable-custom-all-reduce
    --disable-flashinfer-autotune
    --moe-a2a-backend none
    --ep-size 1
    --reasoning-parser glm45
    --tool-call-parser glm47
    --host "${HOST}"
    --port "${PORT}"
    --log-level info
    "${kv_args[@]}"
  )

  printf '%q ' "${command[@]}" >"${RESULT_DIR}/server_info/${variant}_command.txt"
  printf '\n' >>"${RESULT_DIR}/server_info/${variant}_command.txt"
  start_telemetry "${variant}"
  setsid "${command[@]}" >"${server_log}" 2>&1 &
  SERVER_PID=$!

  local start_seconds="${SECONDS}"
  while ! curl --fail --silent --max-time 30 "${BASE_URL}/health_generate" >/dev/null 2>&1 \
    || ! grep -Fq "The server is fired up and ready to roll!" "${server_log}"; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "${variant} server exited before readiness" >&2
      tail -n 240 "${server_log}" >&2 || true
      return 1
    fi
    if (( SECONDS - start_seconds >= 1800 )); then
      echo "${variant} server readiness timeout" >&2
      tail -n 240 "${server_log}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[server] ${variant} ready in $((SECONDS - start_seconds)) seconds"
  curl --fail --silent "${BASE_URL}/server_info" >"${RESULT_DIR}/server_info/${variant}.json"

  grep -Fq "dsa_prefill_backend='flashmla_sparse'" "${server_log}"
  grep -Fq "dsa_decode_backend='flashmla_kv'" "${server_log}"
  grep -Fq "Capture target decode CUDA graph end." "${server_log}"
  grep -Fq "The server is fired up and ready to roll!" "${server_log}"
  if [[ "${variant}" == "fp8" ]]; then
    grep -Fq "kv_cache_dtype='fp8_e4m3'" "${server_log}"
  else
    grep -Fq "kv_cache_dtype='fp4_e2m1'" "${server_log}"
    grep -Fq "fp4_kv_cache_recipe='nvfp4'" "${server_log}"
    grep -Fq "Initialized DSA NVFP4 global scales for 78/78 local layers." "${server_log}"
  fi
}

check_server_log() {
  local variant="$1"
  local server_log="${RESULT_DIR}/server_logs/${variant}.log"
  if grep -Ein \
    'Traceback \(most recent call last\)|CUDA error|illegal memory|misaligned address|out of memory|detected NaN' \
    "${server_log}"; then
    echo "fatal pattern found in ${variant} server log" >&2
    return 1
  fi
}

run_case() {
  local variant="$1"
  local case_name="$2"
  local input_len="$3"
  local output_len="$4"
  local concurrency="$5"
  local prompts="$6"
  local output_file="${RESULT_DIR}/raw/${variant}__${case_name}.jsonl"
  local bench_log="${RESULT_DIR}/bench_logs/${variant}__${case_name}.log"

  echo "[bench] variant=${variant} case=${case_name} il=${input_len} ol=${output_len} c=${concurrency} n=${prompts}"
  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --host "${HOST}" \
    --port "${PORT}" \
    --model GLM-5-FP8 \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random \
    --dataset-path "${DATASET_PATH}" \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --random-range-ratio 1 \
    --num-prompts "${prompts}" \
    --request-rate inf \
    --max-concurrency "${concurrency}" \
    --warmup-requests "${concurrency}" \
    --tokenize-prompt \
    --temperature 0 \
    --seed 42 \
    --ready-check-timeout-sec 30 \
    --output-details \
    --disable-tqdm \
    --tag "${variant}__${case_name}" \
    --output-file "${output_file}" \
    2>&1 | tee "${bench_log}"
}

case_enabled() {
  local case_name="$1"
  [[ "${RUN_CASES}" == "all" || ",${RUN_CASES}," == *",${case_name},"* ]]
}

run_variant() {
  local variant="$1"
  echo "[variant] ${variant} start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_server "${variant}"

  # Prefill-focused: B1 latency, chunked 32K latency, and saturated C12 input throughput.
  case_enabled prefill_8k_b1 && run_case "${variant}" prefill_8k_b1 8192 32 1 3
  case_enabled prefill_32k_b1 && run_case "${variant}" prefill_32k_b1 32768 32 1 3
  case_enabled prefill_32k_c12 && run_case "${variant}" prefill_32k_c12 32768 32 12 12

  # Decode-focused: single-request TPOT, long-output feasible C12, and high-BS C32.
  case_enabled decode_32k_b1 && run_case "${variant}" decode_32k_b1 32768 512 1 3
  case_enabled decode_32k_c12_long && run_case "${variant}" decode_32k_c12_long 32768 2048 12 12
  case_enabled decode_8k_c32 && run_case "${variant}" decode_8k_c32 8192 512 32 32

  check_server_log "${variant}"
  stop_server
  echo "[variant] ${variant} complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

capture_environment
if [[ "${RUN_NATIVE_GATES}" == "1" ]]; then
  run_native_gates
fi
for variant in ${RUN_VARIANTS}; do
  run_variant "${variant}"
done
if [[ "${RUN_SUMMARY}" == "1" ]]; then
  python3 "${REPO_ROOT}/benchmark/glm5_nvfp4_e2e/summarize_optimized_kv_ab.py" \
    --result-dir "${RESULT_DIR}" \
    --expected-capacity "${COMMON_CAPACITY}"
fi
echo "[pipeline] PASS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
