#!/bin/bash

set -euo pipefail

TYPE=${1:-launch}
model_name=${2:-Qwen3-VL-235B}
model_path=${3:-/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/}
TP=${4:-8}
EP=${5:-8}

export SGLANG_TORCH_PROFILER_DIR=./
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

if [[ "${TYPE}" == "launch" ]]; then
    echo
    echo "========== LAUNCHING SERVER ========"
    if [[ "${model_name}" == "Qwen3-VL-235B" ]]; then
        python3 -m sglang.launch_server \
            --model-path "${model_path}" \
            --host localhost \
            --port 9000 \
            --tp-size "${TP}" \
            --ep-size "${EP}" \
            --trust-remote-code \
            --chunked-prefill-size 32768 \
            --mem-fraction-static 0.6 \
            --disable-radix-cache \
            --max-prefill-tokens 32768 \
            --cuda-graph-max-bs 128 &
        sglang_pid=$!
    elif [[ "${model_name}" == "Qwen3-next" ]]; then
        export SGLANG_USE_AITER=1
        python3 -m sglang.launch_server \
            --model-path "${model_path}" \
            --host localhost \
            --port 9000 \
            --tp-size ${TP} \
            --ep-size ${EP} \
            --trust-remote-code \
            --chunked-prefill-size 32768 \
            --mem-fraction-static 0.85 \
            --disable-radix-cache \
            --max-prefill-tokens 32768 \
            --cuda-graph-max-bs 256 \
            --page-size 64 \
            --attention-backend triton \
            --max-running-requests 128 &
        sglang_pid=$!
    else
        echo "Unknown model_name: ${model_name}"
        exit 1
    fi

    echo
    echo "========== WAITING FOR SERVER TO BE READY ========"
    max_retries=60
    retry_interval=60
    for ((i=1; i<=max_retries; i++)); do
        if curl -s http://localhost:9000/v1/completions -o /dev/null; then
            echo "SGLang server is up."
            break
        fi
        echo "Waiting for SGLang server to be ready... ($i/$max_retries)"
        sleep $retry_interval
    done

    if ! curl -s http://localhost:9000/v1/completions -o /dev/null; then
        echo "SGLang server did not start after $((max_retries * retry_interval)) seconds."
        kill $sglang_pid
        exit 1
    fi

    echo
    echo "========== TESTING SERVER ========"
    echo "Downloading test image"
    wget https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png
    echo "Testing server with test image"
    curl --request POST \
        --url "http://localhost:9000/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --data '{
            "model": "${model}",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": "dog.png"
                    }
                    },
                    {
                    "type": "text",
                    "text": "请简要描述图片是什么内容？"
                    }
                ]
                }
            ],
            "temperature": 0.0,
            "top_p": 0.0001,
            "top_k": 1,
            "max_tokens": 100
        }'


elif [[ "${TYPE}" == "evaluation" ]]; then
    echo
    echo "========== STARTING MODEL EVALUATION =========="
    python3 benchmark/mmmu/bench_sglang.py \
        --port 9000 \
        --concurrency 16 \
        | tee vision_model_evaluation_${model_name}_TP${TP}_EP${EP}.log

elif [[ "${TYPE}" == "performance" ]]; then
    echo
    echo "========== STARTING PERFORMANCE BENCHMARK =========="
    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --dataset-name image \
        --image-count 1 \
        --image-resolution 800x800 \
        --random-input-len 1000 \
        --random-output-len 2000 \
        --max-concurrency 64 \
        --num-prompts 192 \
        | tee performance_benchmark_${model_name}_TP${TP}_EP${EP}.log

else
    echo "Unknown TYPE: ${TYPE}"
    echo "Usage: $0 {launch|evaluation|performance} [model_name] [model_path] [TP] [EP]"
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