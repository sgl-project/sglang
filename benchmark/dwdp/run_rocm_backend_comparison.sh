#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-R1-0528-MXFP4-th}"
TP_SIZE="${TP_SIZE:-4}"
BACKENDS="${BACKENDS:-dep ipc}"
INPUT_LENGTHS="${INPUT_LENGTHS:-4096 8192 16384 32768}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-$((32768 * TP_SIZE))}"
BASE_PORT="${BASE_PORT:-32000}"
RESULT_DIR="${RESULT_DIR:-/results/dwdp-benchmark}"

export SGLANG_USE_AITER=1
export GPU_ARCHS="${GPU_ARCHS:-gfx950}"
mkdir -p "$RESULT_DIR"

case_index=0
for backend in $BACKENDS; do
    port=$((BASE_PORT + case_index * 100))
    server_log="${RESULT_DIR}/${backend}-server.log"
    common_args=(
        --model-path "$MODEL_PATH"
        --tp-size "$TP_SIZE"
        --host 127.0.0.1
        --port "$port"
        --trust-remote-code
        --watchdog-timeout 1800
        --mem-fraction-static 0.80
        --max-running-requests "$MAX_CONCURRENCY"
        --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
        --disable-cuda-graph
        --disable-radix-cache
        --attention-backend aiter
        --log-level warning
    )
    if [[ "$backend" == "dep" ]]; then
        backend_args=(
            --dp-size "$TP_SIZE"
            --enable-dp-attention
            --ep-size "$TP_SIZE"
            --moe-dense-tp-size 1
        )
    else
        backend_args=(
            --dwdp-size "$TP_SIZE"
            --dwdp-weight-backend "$backend"
        )
    fi

    python3 -m sglang.launch_server \
        "${common_args[@]}" \
        "${backend_args[@]}" \
        >"$server_log" 2>&1 &
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
        echo "VRAM did not return to the benchmark baseline after ${backend}" >&2
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
            echo "${backend} server exited during startup" >&2
            wait "$server_pid"
        fi
        sleep 5
    done
    [[ "$ready" == 1 ]] || {
        echo "${backend} server did not become healthy" >&2
        exit 1
    }

    for input_len in $INPUT_LENGTHS; do
        output="${RESULT_DIR}/${backend}-isl${input_len}.jsonl"
        python3 -m sglang.benchmark.serving \
            --backend sglang \
            --base-url "http://127.0.0.1:${port}" \
            --model "$MODEL_PATH" \
            --dataset-name random \
            --num-prompts "$NUM_PROMPTS" \
            --random-input-len "$input_len" \
            --random-output-len 1 \
            --random-range-ratio 1.0 \
            --request-rate inf \
            --max-concurrency "$MAX_CONCURRENCY" \
            --output-file "$output"
    done

    cleanup_server
    wait_vram_clean
    trap - EXIT
    case_index=$((case_index + 1))
done

python3 benchmark/dwdp/summarize_rocm_backend_comparison.py "$RESULT_DIR"
