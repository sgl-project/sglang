#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="${HCU_CI_REPO_ROOT:-/sglang-checkout}"
readonly MODEL_PATH="${HCU_CI_SMOKE_MODEL:?HCU_CI_SMOKE_MODEL is required}"
readonly TARGET_SHA="${HCU_CI_TARGET_SHA:?HCU_CI_TARGET_SHA is required}"
readonly PORT="${HCU_CI_PORT:-31000}"
readonly LOG_DIR="${HCU_CI_LOG_DIR:-${REPO_ROOT}/.hcu-ci-logs}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/smoke.log") 2>&1

[[ -d "${MODEL_PATH}" ]] || { echo "Model path not found: ${MODEL_PATH}"; exit 1; }
[[ "${PORT}" =~ ^[0-9]+$ ]] || { echo "Invalid port: ${PORT}"; exit 1; }

cd "${REPO_ROOT}"
git config --global --add safe.directory "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${TARGET_SHA}" ]] || {
  echo "Checkout SHA does not match ${TARGET_SHA}"
  exit 1
}

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
python3 - <<'PY'
import sgl_kernel
import tabulate

print(f"Using image-provided sgl_kernel from {sgl_kernel.__file__}")
print(f"Using image-provided tabulate from {tabulate.__file__}")
PY

python3 test/run_suite.py \
  --hw hcu \
  --suite stage-a-test-1-hcu \
  --timeout-per-file 300 \
  2>&1 | tee "${LOG_DIR}/pytest.log"

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tp-size 1 \
  --page-size 64 \
  --attention-backend triton \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --trust-remote-code \
  >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID="$!"

HCU_CI_BASE_URL="http://127.0.0.1:${PORT}" \
HCU_CI_SERVER_PID="${SERVER_PID}" \
python3 - <<'PY' 2>&1 | tee "${LOG_DIR}/request.log"
import json
import os
import time
import urllib.request

base_url = os.environ["HCU_CI_BASE_URL"]
server_pid = int(os.environ["HCU_CI_SERVER_PID"])
deadline = time.monotonic() + 600

while True:
    try:
        os.kill(server_pid, 0)
    except ProcessLookupError as exc:
        raise RuntimeError("SGLang server exited before becoming ready") from exc

    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
            models = json.load(response)
        break
    except (OSError, ValueError):
        if time.monotonic() >= deadline:
            raise RuntimeError("SGLang server did not become ready within 600 seconds")
        time.sleep(5)

model_id = models["data"][0]["id"]
payload = json.dumps(
    {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with one short greeting."}],
        "temperature": 0,
        "max_tokens": 16,
    }
).encode()
request = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)

text = result["choices"][0]["message"].get("content", "")
if not text.strip():
    raise RuntimeError(f"Chat completion returned no content: {result}")
print(json.dumps(result, indent=2))
PY

echo "HCU PR smoke completed successfully."
