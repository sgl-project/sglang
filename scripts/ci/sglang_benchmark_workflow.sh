#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT_FILE="${SGLANG_CI_PORT_STATE_FILE:-${WORKFLOW_ROOT}/.sglang_ci_server_port}"

TYPE=${1:-launch}
model_name=${2:-Qwen3.5-397B-A17B}
model_path=${3:-/models/Qwen3.5-397B-A17B}
TP=${4:-8}
TIMEOUT=${5:-60}
BASE_PORT="${SGLANG_CI_BASE_PORT:-${6:-9000}}"

# Returns 0 if something accepts TCP connections on 127.0.0.1:port, else 1.
tcp_port_in_use() {
    local port=$1
    timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null
}

# Prints the first free port starting at start_port (inclusive), up to max_port.
pick_free_port() {
    local p=${1:?start port required}
    local max=${2:-65535}
    while (( p <= max )); do
        if ! tcp_port_in_use "$p"; then
            echo "$p"
            return 0
        fi
        ((p++)) || true
    done
    echo "error: no free TCP port found from ${1} to ${max}" >&2
    return 1
}

resolve_server_port() {
    if [[ -n "${SGLANG_SERVER_PORT:-}" ]]; then
        echo "${SGLANG_SERVER_PORT}"
        return 0
    fi
    if [[ -f "$PORT_FILE" ]]; then
        tr -d '[:space:]' <"$PORT_FILE"
        return 0
    fi
    echo "${BASE_PORT}"
}

export SGLANG_TORCH_PROFILER_DIR=./
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

echo "PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"

echo "Detect TYPE ${TYPE}"
echo "Detect model_name: ${model_name}"
echo "Detect model_path ${model_path}"
echo "Detect TP ${TP}"
echo "Detect TIMEOUT ${TIMEOUT}"
echo "Detect BASE_PORT (launch starting point / fallback) ${BASE_PORT}"
echo "Detect PORT_FILE ${PORT_FILE}"

if [[ "${TYPE}" == "launch" ]]; then
    PORT="$(pick_free_port "${BASE_PORT}")"
    echo "${PORT}" >"${PORT_FILE}"
    echo "Using server port ${PORT} (written to ${PORT_FILE})"

    echo
    echo "========== LAUNCHING SERVER ========"
    if [[ "${model_name}" == "Qwen3.5-397B-A17B" ]] || [[ "${model_name}" == "Qwen3.5-397B-A17B-FP8" ]]; then
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export SGLANG_USE_AITER=1
        export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
        export OPTFLAG="w8a8_gemm,moe"
        echo "********** AOT Prebuild aiter kernel start ... **********"
        cd /aiter
        python3 op_tests/test_gemma_rms_norm.py
        echo "********** AOT Prebuild aiter kernel finished ... **********"
        cd /sglang-checkout
        launch_log="sglang_launch_${model_name}_TP${TP}.log"
        nohup python -m sglang.launch_server \
            --model ${model_path} \
            --model-path "${model_path}" \
            --port "${PORT}" \
            --tp-size ${TP} \
            --mem-fraction-static 0.8 \
            --context-length 262144 \
            --reasoning-parser qwen3 \
            --attention-backend triton \
            --disable-radix-cache \
            --cuda-graph-max-bs 64 \
            --watchdog-timeout 1200 \
            >"${launch_log}" 2>&1 &
        sglang_pid=$!
        disown "${sglang_pid}" 2>/dev/null || true
        echo "launch_server PID ${sglang_pid}, log ${launch_log}"
    else
        echo "Unknown model_name: ${model_name}"
        exit 1
    fi

    echo
    echo "========== WAITING FOR SERVER TO BE READY ========"
    max_retries=${TIMEOUT}
    retry_interval=60
    for ((i=1; i<=max_retries; i++)); do
        if curl -s "http://localhost:${PORT}/v1/completions" -o /dev/null; then
            echo "SGLang server is up."
            break
        fi
        echo "Waiting for SGLang server to be ready... ($i/$max_retries)"
        sleep $retry_interval
    done

    if ! curl -s "http://localhost:${PORT}/v1/completions" -o /dev/null; then
        echo "SGLang server did not start after $((max_retries * retry_interval)) seconds."
        kill $sglang_pid
        exit 1
    fi

    echo
    echo "========== TESTING SERVER ========"
    echo "Testing server with test image"
    curl --request POST \
         --url "http://localhost:${PORT}/v1/chat/completions" \
         --header "Content-Type: application/json" \
         --data '{
                    "model": "'"${model_path}"'",
                    "messages": [
                    {
                        "role": "user",
                        "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail."
                        }
                        ]
                    }
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 100
                }'

elif [[ "${TYPE}" == "evaluation" ]]; then
    PORT="$(resolve_server_port)"
    echo "Connecting to SGLang server on port ${PORT}"

    echo
    echo "========== STARTING MODEL EVALUATION =========="
    python3 benchmark/mmmu/bench_sglang.py \
        --port "${PORT}" \
        --concurrency 64 \
        --max-new-tokens 512 \
        2>&1 | tee vision_model_evaluation_${model_name}_TP${TP}.log

elif [[ "${TYPE}" == "benchmark" ]]; then
    PORT="$(resolve_server_port)"
    echo "Connecting to SGLang server on port ${PORT}"

    echo
    echo "========== STARTING PERFORMANCE BENCHMARK =========="
    input_tokens=8000
    output_tokens=500
    num_prompts=32
    max_concurrency=1
    dataset_name="random"

    echo "bench model: ${model_name}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"
    echo "dataset-name: ${dataset_name}"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --model ${model_path} \
        --dataset-name ${dataset_name} \
        --host localhost \
        --port "${PORT}" \
        --num-prompts ${num_prompts} \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
        --max-concurrency ${max_concurrency} \
        2>&1 | tee performance_benchmark_${model_name}_TP${TP}.log

else
    echo "Unknown TYPE: ${TYPE}"
    echo "Usage: $0 {launch|evaluation|benchmark} [model_name] [model_path] [TP] [TIMEOUT] [base_port]"
    echo ""
    echo "  Positional:"
    echo "    TIMEOUT     — for launch only: max retries (interval 60s) while waiting for /v1/completions (default: 60)."
    echo "    base_port   — where launch scans for a free port; if busy, tries base_port+1, ... (default: 9000)."
    echo ""
    echo "  Port selection:"
    echo "    launch      — picks first free TCP port from base_port upward; writes it to PORT_FILE (see below)."
    echo "    evaluation,"
    echo "    benchmark  — use SGLANG_SERVER_PORT if set, else first line of PORT_FILE from a prior launch,"
    echo "                  else base_port (default 9000)."
    echo ""
    echo "  Environment:"
    echo "    SGLANG_CI_BASE_PORT       — same as positional base_port when 6th arg omitted (default 9000)."
    echo "    SGLANG_SERVER_PORT        — force client port for evaluation/benchmark."
    echo "    SGLANG_CI_PORT_STATE_FILE — override path for the port state file (default: repo_root/.sglang_ci_server_port)."
    exit 1
fi

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo
    echo "========== SGLANG BENCHMARK ${TYPE} COMPLETED SUCCESSFULLY =========="
else
    echo
    echo "========== SGLANG BENCHMARK ${TYPE} FAILED WITH EXIT CODE $exit_code =========="
    exit $exit_code
fi
