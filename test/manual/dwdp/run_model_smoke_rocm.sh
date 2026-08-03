#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-R1-0528-MXFP4-th}"
DWDP_SIZE="${DWDP_SIZE:-4}"
BACKENDS="${BACKENDS:-vmm ipc}"
BASE_PORT="${BASE_PORT:-31000}"
RESULT_DIR="${RESULT_DIR:-/results}"

export SGLANG_USE_AITER=1
export GPU_ARCHS="${GPU_ARCHS:-gfx950}"
mkdir -p "$RESULT_DIR"

backend_index=0
for backend in $BACKENDS; do
    port=$((BASE_PORT + backend_index * 100))
    log_file="${RESULT_DIR}/model-smoke-${backend}.log"
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp-size "$DWDP_SIZE" \
        --dwdp-size "$DWDP_SIZE" \
        --dwdp-weight-backend "$backend" \
        --host 127.0.0.1 \
        --port "$port" \
        --trust-remote-code \
        --watchdog-timeout 1800 \
        --mem-fraction-static 0.80 \
        --max-running-requests 8 \
        --chunked-prefill-size 8192 \
        --disable-radix-cache \
        --attention-backend aiter \
        >"$log_file" 2>&1 &
    server_pid=$!

    cleanup_server() {
        if kill -0 "$server_pid" 2>/dev/null; then
            SERVER_PID="$server_pid" python3 - <<'PY'
import os

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(os.environ["SERVER_PID"]), wait_timeout=30)
PY
            wait "$server_pid" || true
        fi
    }
    wait_vram_clean() {
        local attempt
        for attempt in $(seq 1 6); do
            if python3 scripts/ci/amd/check_dwdp_rocm_vram.py \
                --expected-gpus "${EXPECTED_GPU_COUNT:-8}" \
                --max-used-gib "${MAX_USED_VRAM_GIB:-4}" \
                --max-skew-gib "${MAX_VRAM_SKEW_GIB:-2}"; then
                return 0
            fi
            sleep 10
        done
        echo "VRAM did not return to the model-smoke baseline after ${backend}" >&2
        return 1
    }
    trap cleanup_server EXIT

    ready=0
    for _ in $(seq 1 360); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "DWDP ${backend} server exited during startup" >&2
            wait "$server_pid"
        fi
        sleep 5
    done
    if [[ "$ready" != 1 ]]; then
        echo "DWDP ${backend} server did not become healthy" >&2
        exit 1
    fi

    response=$(
        curl -fsS "http://127.0.0.1:${port}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${MODEL_PATH}\",
                \"prompt\": \"The capital of France is\",
                \"max_tokens\": 16,
                \"temperature\": 0
            }"
    )
    RESPONSE="$response" python3 - <<'PY'
import json
import os

response = json.loads(os.environ["RESPONSE"])
text = response["choices"][0]["text"]
if not text.strip():
    raise SystemExit("empty completion")
print(text)
PY

    cleanup_server
    wait_vram_clean
    trap - EXIT
    backend_index=$((backend_index + 1))
done
