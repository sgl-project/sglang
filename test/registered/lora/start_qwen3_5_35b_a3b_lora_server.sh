#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}/sglang/python:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
LORA_REPO="${LORA_REPO:-opherlie/lora-test-case-Qwen3.5-35B-A3B}"
LORA_NAME="${LORA_NAME:-my_lora}"
PRELOAD_LORA="${PRELOAD_LORA:-1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.6}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
LORA_BACKEND="${LORA_BACKEND:-triton}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-triton}"
LOG_LEVEL="${LOG_LEVEL:-info}"
UNIT_TEST_LORA_FLAGS="${UNIT_TEST_LORA_FLAGS:-1}"

# The probability guard is expensive, so normal SGLang keeps it opt-in. This
# repro script enables it because it is meant to catch bad probs before CUDA
# turns torch.multinomial into a device-side assert.
export SGLANG_SAMPLER_PROBS_GUARD="${SGLANG_SAMPLER_PROBS_GUARD:-1}"
export SGLANG_SAMPLER_PROBS_GUARD_SYNC="${SGLANG_SAMPLER_PROBS_GUARD_SYNC:-0}"

if [[ "${PRELOAD_LORA}" == "1" && -z "${LORA_PATH:-}" ]]; then
  LORA_PATH="$(
    LORA_REPO="${LORA_REPO}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

print(snapshot_download(os.environ["LORA_REPO"], repo_type="dataset"))
PY
  )"
fi

SERVER_ARGS=(
  python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --trust-remote-code
  --enable-lora
  --max-lora-rank "${MAX_LORA_RANK}"
  --lora-backend "${LORA_BACKEND}"
  --moe-runner-backend "${MOE_RUNNER_BACKEND}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --log-level "${LOG_LEVEL}"
)

if [[ "${PRELOAD_LORA}" == "1" ]]; then
  SERVER_ARGS+=(--lora-paths "${LORA_NAME}=${LORA_PATH}")
else
  # shellcheck disable=SC2206
  LORA_TARGET_MODULES_ARRAY=(${LORA_TARGET_MODULES})
  SERVER_ARGS+=(--lora-target-modules "${LORA_TARGET_MODULES_ARRAY[@]}")
fi

if [[ -n "${EP_SIZE}" ]]; then
  SERVER_ARGS+=(--ep-size "${EP_SIZE}")
fi

if [[ "${UNIT_TEST_LORA_FLAGS}" == "1" ]]; then
  SERVER_ARGS+=(
    --experts-shared-outer-loras
    --lora-use-virtual-experts
    --disable-shared-experts-fusion
  )
fi

if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SERVER_ARGS_ARRAY=(${EXTRA_SERVER_ARGS})
  SERVER_ARGS+=("${EXTRA_SERVER_ARGS_ARRAY[@]}")
fi

echo "Starting SGLang server at http://${HOST}:${PORT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "LORA_NAME=${LORA_NAME}"
echo "PRELOAD_LORA=${PRELOAD_LORA}"
if [[ "${PRELOAD_LORA}" == "1" ]]; then
  echo "LORA_PATH=${LORA_PATH}"
else
  echo "LORA_TARGET_MODULES=${LORA_TARGET_MODULES}"
fi
echo "SGLANG_SAMPLER_PROBS_GUARD=${SGLANG_SAMPLER_PROBS_GUARD}"
exec "${SERVER_ARGS[@]}"
