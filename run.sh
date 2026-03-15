#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-180}"

PORT_BASELINE="${PORT_BASELINE:-31000}"
PORT_FA3_OK="${PORT_FA3_OK:-31002}"
PORT_FA3_BAD="${PORT_FA3_BAD:-31001}"

LOG_BASELINE="/tmp/sglang_18550_baseline.log"
LOG_FA3_OK="/tmp/sglang_18550_fa3_ok.log"
LOG_FA3_BAD="/tmp/sglang_18550_fa3_bad.log"

cleanup_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -ti :"${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      kill -9 ${pids} >/dev/null 2>&1 || true
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k -n tcp "${port}" >/dev/null 2>&1 || true
  fi
}

wait_health() {
  local port="$1"
  local timeout="$2"
  local i
  for i in $(seq 1 "${timeout}"); do
    if curl -sf "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_server() {
  local port="$1"
  local log_file="$2"
  shift 2
  cleanup_port "${port}"

  PYTHONPATH=python "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tp 1 \
    --host "${HOST}" \
    --port "${port}" \
    --mem-fraction-static 0.3 \
    --max-running-requests 8 \
    "$@" >"${log_file}" 2>&1 &

  echo $!
}

smoke_request() {
  local port="$1"
  curl -s "http://${HOST}:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":16,\"temperature\":0.0}"
}

echo "[1/4] Run unit tests for #18550 compatibility checks"
PYTHONPATH=python "${PYTHON_BIN}" -m unittest discover -v \
  -s test/registered/core \
  -p "test_server_args.py" \
  -k Fa3KvCacheDtypeCompatibility

echo "[2/4] Run baseline real-inference smoke test"
pid_baseline="$(start_server "${PORT_BASELINE}" "${LOG_BASELINE}")"
if ! wait_health "${PORT_BASELINE}" "${SERVER_TIMEOUT}"; then
  echo "Baseline server failed to start. Log tail:"
  tail -n 120 "${LOG_BASELINE}" || true
  kill -9 "${pid_baseline}" >/dev/null 2>&1 || true
  exit 1
fi
baseline_resp="$(smoke_request "${PORT_BASELINE}")"
echo "Baseline response: ${baseline_resp}"
kill -9 "${pid_baseline}" >/dev/null 2>&1 || true

echo "[3/4] Run positive fa3+fp8_e4m3 real-inference smoke test"
pid_fa3_ok="$(start_server "${PORT_FA3_OK}" "${LOG_FA3_OK}" --attention-backend fa3 --kv-cache-dtype fp8_e4m3)"
if ! wait_health "${PORT_FA3_OK}" "${SERVER_TIMEOUT}"; then
  echo "FA3+fp8_e4m3 server failed to start. Log tail:"
  tail -n 120 "${LOG_FA3_OK}" || true
  kill -9 "${pid_fa3_ok}" >/dev/null 2>&1 || true
  exit 1
fi
fa3_ok_resp="$(smoke_request "${PORT_FA3_OK}")"
echo "FA3+fp8_e4m3 response: ${fa3_ok_resp}"
kill -9 "${pid_fa3_ok}" >/dev/null 2>&1 || true

echo "[4/4] Verify incompatible fa3+fp8_e5m2 is rejected"
cleanup_port "${PORT_FA3_BAD}"
set +e
PYTHONPATH=python "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp 1 \
  --host "${HOST}" \
  --port "${PORT_FA3_BAD}" \
  --attention-backend fa3 \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.3 \
  >"${LOG_FA3_BAD}" 2>&1
exit_code=$?
set -e

if [[ "${exit_code}" -eq 0 ]]; then
  echo "Expected fa3+fp8_e5m2 to fail, but server started successfully."
  exit 1
fi

if ! grep -q "FlashAttention3 does not support" "${LOG_FA3_BAD}"; then
  echo "Expected validation error not found in log. Log tail:"
  tail -n 120 "${LOG_FA3_BAD}" || true
  exit 1
fi

echo "All checks passed for #18550."

