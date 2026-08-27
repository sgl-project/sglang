set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Base-7B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
SERVER_LOG="${SERVER_LOG:-dream_server.log}"

export PYTHONPATH="python${PYTHONPATH:+:${PYTHONPATH}}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --tp-size 1 \
  --pp-size 1 \
  --dllm-algorithm Dream \
  --no-dllm-fdfo \
  --disable-radix-cache \
  --disable-prefill-cuda-graph \
  --disable-decode-cuda-graph \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >"${SERVER_LOG}" 2>&1 &
server_pid=$!

echo "Dream server PID: ${server_pid}"
echo "Server log: ${SERVER_LOG}"

"${PYTHON_BIN}" test_dream_online.py \
  --base-url "http://${HOST}:${PORT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --startup-timeout "${STARTUP_TIMEOUT}"
