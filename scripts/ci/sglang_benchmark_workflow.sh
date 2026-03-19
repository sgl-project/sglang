#!/bin/bash

set -euo pipefail

TYPE=${1:-launch}
model_name=${2:-Qwen3.5-397B-A17B}
model_path=${3:-/models/Qwen3.5-397B-A17B}
TP=${4:-8}
TIMEOUT=${5:-60}

export SGLANG_TORCH_PROFILER_DIR=./
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

echo "PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"

echo "Detect TYPE ${TYPE}"
echo "Detect model_name: ${model_name}"
echo "Detect model_path ${model_path}"
echo "Detect TP ${TP}"
echo "Detect TIMEOUT ${TIMEOUT}"

if [[ "${TYPE}" == "launch" ]]; then
    echo
    echo "========== LAUNCHING SERVER ========"
    if [[ "${model_name}" == "Qwen3.5-397B-A17B" ]] || [[ "${model_name}" == "Qwen3.5-397B-A17B-FP8" ]]; then
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export SGLANG_USE_AITER=1
        export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

        echo "********** AOT Prebuild aiter kernel start ... **********"
        cd /aiter
        python3 op_tests/test_gemma_rms_norm.py
        echo "********** AOT Prebuild aiter kernel finished ... **********"
        cd /sglang-checkout
        python -m sglang.launch_server \
            --model-path "${model_path}" \
            --port 9000 \
            --tp-size ${TP} \
            --mem-fraction-static 0.8 \
            --context-length 262144 \
            --reasoning-parser qwen3 \
            --attention-backend triton \
            --disable-radix-cache \
            --cuda-graph-max-bs 64 \
            --watchdog-timeout 1200 &
        sglang_pid=$!
    else
        echo "Unknown model_name: ${model_name}"
        exit 1
    fi

    echo
    echo "========== WAITING FOR SERVER TO BE READY ========"
    max_retries=${TIMEOUT}
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
    echo "Testing server with test image"
    curl --request POST \
         --url "http://localhost:9000/v1/chat/completions" \
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
    echo
    echo "========== STARTING MODEL EVALUATION =========="
    python3 benchmark/mmmu/bench_sglang.py \
        --port 9000 \
        --concurrency 64 \
        --max-new-tokens 512 \
        | tee vision_model_evaluation_${model_name}_TP${TP}.log

elif [[ "${TYPE}" == "benchmark" ]]; then
    echo
    echo "========== STARTING PERFORMANCE BENCHMARK =========="
    model="${model_path}"
    input_tokens=8000
    output_tokens=500
    num_prompts=32
    max_concurrency=1
    dataset_name="random"

    echo "bench model: ${model}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"
    echo "dataset-name: ${dataset_name}"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --model ${model} \
        --dataset-name ${dataset_name} \
        --host localhost \
        --port 9000 \
        --num-prompts ${num_prompts} \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
        --max-concurrency ${max_concurrency} \
        2>&1 | tee performance_benchmark_${model_name}_TP${TP}.log

else
    echo "Unknown TYPE: ${TYPE}"
    echo "Usage: $0 {launch|evaluation|benchmark} [model_name] [model_path] [TP] [TIMEOUT]"
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
