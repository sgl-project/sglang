#!/usr/bin/env bash
# 3-prompt smoke check for the DSV4-NPU fused compressor path.
# Reads SGLANG_DSV4_NPU_FUSED_COMPRESSOR from the environment (set to 0 or 1
# at the caller). Assumes a running sglang server on $HOST:$PORT or launches
# one if SGLANG_LAUNCH_CMD is set.
#
# Usage:
#   SGLANG_DSV4_NPU_FUSED_COMPRESSOR=0 bench_sanity.sh
#   SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1 bench_sanity.sh
set -euo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

flag="${SGLANG_DSV4_NPU_FUSED_COMPRESSOR:-0}"
echo "=== Sanity run (SGLANG_DSV4_NPU_FUSED_COMPRESSOR=$flag) ==="

# Launch server if requested.
if [[ -n "${SGLANG_LAUNCH_CMD:-}" ]]; then
    echo "Launching server: $SGLANG_LAUNCH_CMD"
    eval "$SGLANG_LAUNCH_CMD" &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
    # Wait for readiness — sglang prints `Ready to roll!` when /generate is up.
    for i in {1..120}; do
        if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then break; fi
        sleep 5
    done
fi

ask() {
    local prompt="$1"
    curl -sS "http://$HOST:$PORT/generate" \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"${prompt//\"/\\\"}\",\"sampling_params\":{\"max_new_tokens\":64,\"temperature\":0}}" \
        | python -c "import json,sys; print(json.load(sys.stdin)['text'])"
}

echo "--- Q1: Paris ---"
ask "What is the capital of France?"
echo "--- Q2: 56 multiplication ---"
ask "What is 7 times 8?"
echo "--- Q3: Janet apples ---"
ask "Janet has 12 apples. She gives 5 to John and eats 2. How many does she have left?"
