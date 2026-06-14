#!/usr/bin/env bash
# Launch an SGLang server, wait for it to come up, run the boundary sweep once,
# then shut the server down. Edit MODEL / PORT / SERVER_ARGS as needed.
#
# Usage: bash run.sh [extra bench_boundary.py args...]
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-30000}"
LABEL="${LABEL:-default}"
# e.g. SERVER_ARGS="--disable-radix-cache" or "--chunked-prefill-size 4096"
SERVER_ARGS="${SERVER_ARGS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching server: $MODEL on :$PORT $SERVER_ARGS"
python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" $SERVER_ARGS &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait for readiness
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
done

python3 "$HERE/bench_boundary.py" \
  --model "$MODEL" --port "$PORT" --server-config-label "$LABEL" "$@"
