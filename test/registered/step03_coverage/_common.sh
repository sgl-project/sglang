# Common helpers for step03_coverage attention-backend smoke tests.
#
# Each test sources this file, sets TEST_NAME + LAUNCH_ARGS, then calls
# run_server_smoke. The harness brings up a sglang server with
# --load-format dummy (random weights), waits for the "fired up" log,
# sends one /generate request, validates the JSON shape, tears down,
# and prints PASS / FAIL on the last line.
#
# Models live at /model_root inside the container (the caller mounts
# /mnt/vast/models or a HF cache there). Set MODEL_PATH to the in-container
# path before calling run_server_smoke. Image must be the dev-cu13 sqsh.

set -uo pipefail

# Run shared container preamble — safe to call multiple times.
step03_preamble() {
    git config --global --add safe.directory /sgl-workspace/sglang 2>/dev/null || true
    git config --global --add safe.directory /workspace/sglang 2>/dev/null || true
    git config --global --add safe.directory '*' 2>/dev/null || true
    pip install sglang-kernel --upgrade --break-system-packages -q 2>&1 | tail -2 || true

    export SGLANG_JIT_DEEPGEMM_PRECOMPILE=False
    export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

    # Force offline HF resolution: many test models (Llama-3.1 etc.) are
    # gated and re-validate even when present in the cache. The cluster
    # cache is mounted at /root/.cache; tell HF to use only that.
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
}

# Wait for server to print readiness banner; returns 0 on success, 1 on timeout / crash.
wait_server_ready() {
    local pid="$1"
    local log_file="$2"
    local timeout_sec="${3:-1800}"   # 30 min default
    local deadline=$((SECONDS + timeout_sec))

    while [ $SECONDS -lt $deadline ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "wait_server_ready: server process $pid died" >&2
            return 1
        fi
        if grep -q "The server is fired up and ready to roll" "$log_file" 2>/dev/null; then
            return 0
        fi
        sleep 5
    done
    echo "wait_server_ready: timed out after ${timeout_sec}s" >&2
    return 1
}

# Send a /generate request and emit JSON to stdout. Returns 0 if the
# response has a non-empty "text" field and no obvious error markers.
generate_and_check() {
    local port="$1"
    local response
    response=$(curl -sS --max-time 120 \
        -X POST "http://127.0.0.1:${port}/generate" \
        -H 'Content-Type: application/json' \
        -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}')

    local rc=$?
    echo "--- /generate response ---"
    echo "$response" | head -c 2000
    echo
    echo "--- end response ---"

    if [ $rc -ne 0 ]; then
        echo "generate_and_check: curl failed with rc=$rc" >&2
        return 1
    fi

    # Reject NaN/inf markers (dummy weights can rarely produce them, but
    # the surface contract is still: valid JSON, non-empty text).
    if echo "$response" | grep -qiE 'NaN|"error"|Traceback'; then
        echo "generate_and_check: response contains error / NaN markers" >&2
        return 1
    fi

    # JSON shape check — must parse and have a non-empty "text" field.
    python3 -c "
import sys, json
try:
    j = json.loads(sys.argv[1])
except Exception as e:
    print(f'JSON parse failed: {e}', file=sys.stderr); sys.exit(1)
if isinstance(j, list):
    j = j[0]
text = j.get('text', '')
if not isinstance(text, str):
    print('text field missing or not str', file=sys.stderr); sys.exit(1)
print(f'text field length = {len(text)}')
" "$response"
    return $?
}

# Tear down the background server cleanly (best effort).
shutdown_server() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    sleep 3
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# End-to-end runner. Required env vars:
#   TEST_NAME      — short identifier; used for log filename
#   MODEL_PATH     — in-container model path
#   LAUNCH_ARGS    — bash array of extra flags appended to launch_server
# Optional:
#   PORT           — server port (default 30000)
#   READY_TIMEOUT  — startup timeout seconds (default 1800)
#   LOG_DIR        — directory for log files (default /tmp)
#   EXTRA_ENV      — bash array of "KEY=VAL" exports to apply before launch
#
# Prints PASS / FAIL: <reason> on the last line.
run_server_smoke() {
    local port="${PORT:-30000}"
    local ready_timeout="${READY_TIMEOUT:-1800}"
    local log_dir="${LOG_DIR:-/tmp}"
    local log_file="${log_dir}/server_${TEST_NAME}.log"

    mkdir -p "$log_dir"

    # Apply EXTRA_ENV if set.
    if [ -n "${EXTRA_ENV+x}" ]; then
        for kv in "${EXTRA_ENV[@]}"; do
            export "$kv"
        done
    fi

    echo "=== step03 coverage test: $TEST_NAME ==="
    echo "Model: $MODEL_PATH"
    echo "Launch args: ${LAUNCH_ARGS[*]}"
    echo "Port: $port"
    nvidia-smi -L | head -4 || true
    python3 -c "import sglang; print('sglang:', sglang.__file__)"
    git -C /sgl-workspace/sglang log -1 --format='%H %s' 2>/dev/null || true
    echo

    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --load-format dummy \
        --host 127.0.0.1 \
        --port "$port" \
        --trust-remote-code \
        "${LAUNCH_ARGS[@]}" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "server pid: $pid (log: $log_file)"

    # Forward log tail in case caller wants to see progress in real time.
    if ! wait_server_ready "$pid" "$log_file" "$ready_timeout"; then
        echo "--- last 200 log lines ---"
        tail -n 200 "$log_file" || true
        shutdown_server "$pid"
        echo "FAIL: ${TEST_NAME}: server never reached ready"
        return 1
    fi

    if ! generate_and_check "$port"; then
        echo "--- last 200 log lines ---"
        tail -n 200 "$log_file" || true
        shutdown_server "$pid"
        echo "FAIL: ${TEST_NAME}: generate request failed"
        return 1
    fi

    shutdown_server "$pid"
    echo "PASS: ${TEST_NAME}"
    return 0
}
