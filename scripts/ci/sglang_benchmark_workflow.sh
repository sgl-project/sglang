#!/bin/bash

set -euo pipefail

TYPE=${1:-launch}
model_name=${2:-Qwen3-VL-235B}
model_path=${3:-/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/}
TP=${4:-8}
EP=${5:-8}
TIMEOUT=${6:-60}

export SGLANG_TORCH_PROFILER_DIR=./
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1
echo "GPU_ARCHS: ${GPU_ARCHS}"
echo "PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"

echo "Dectect TYPE ${TYPE}"
echo "Detect model_name: ${model_name}"
echo "Detect model_path ${model_path}"
echo "Detect TP ${TP}"
echo "Detect EP ${EP}"
echo "Detect TIMEOUT ${TIMEOUT}"


if [[ "${TYPE}" == "launch" ]]; then
    echo
    echo "========== LAUNCHING SERVER ========"
    if [[ "${model_name}" == "Qwen3-VL-235B" ]]; then
        export SGLANG_USE_AITER=1
        echo "********** AOT Prebuild aiter kernel start ... **********"
        cd /aiter
        python3 op_tests/test_rope.py
        python3 op_tests/test_layernorm2d.py
        python3 op_tests/test_rmsnorm2d.py
        python3 op_tests/test_rmsnorm2dFusedAddQuant.py
        python3 op_tests/test_trtllm_all_reduce_fusion.py
        echo "********** AOT Prebuild aiter kernel finished ... **********"
        python3 -m sglang.launch_server \
            --model-path "${model_path}" \
            --host localhost \
            --port 9000 \
            --tp-size "${TP}" \
            --trust-remote-code \
            --chunked-prefill-size 32768 \
            --mem-fraction-static 0.90 \
            --disable-radix-cache \
            --max-prefill-tokens 32768 \
            --cuda-graph-max-bs 128 \
            --page-size 1024 \
            --mm-attention-backend aiter_attn \
            --mm-enable-dp-encoder \
            --enable-aiter-allreduce-fusion \
            --kv-cache-dtype fp8_e4m3 \
            --mm-processor-kwargs '{"max_pixels": 1638400, "min_pixels": 740}' \
            --watchdog-timeout 1200 &
        sglang_pid=$!
    elif [[ "${model_name}" == "Qwen3-next" ]]; then
        export SGLANG_USE_AITER=1
        export SGLANG_ROCM_USE_AITER_PA_ASM_PRESHUFFLE_LAYOUT=0
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
            --page-size 1024 \
            --attention-backend triton \
            --max-running-requests 128 \
            --kv-cache-dtype fp8_e4m3 \
            --watchdog-timeout 1200 &
        sglang_pid=$!
    elif [[ "${model_name}" == "Qwen3-Omni" ]]; then
        echo "Qwen3-Omni-Server Launch"
        export SGLANG_USE_CUDA_IPC_TRANSPORT=1
        export SGLANG_VLM_CACHE_SIZE_MB=8192
        export SGLANG_USE_AITER=1
        export USE_PA=1
        export SGLANG_ROCM_USE_AITER_PA_ASM_PRESHUFFLE_LAYOUT=0
        export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
        export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=0
        export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
        echo "********** AOT Prebuild aiter kernel start ... **********"
        cd /aiter
        python3 op_tests/test_rope.py
        python3 op_tests/test_layernorm2d.py
        python3 op_tests/test_rmsnorm2d.py
        python3 op_tests/test_rmsnorm2dFusedAddQuant.py
        python3 op_tests/test_trtllm_all_reduce_fusion.py
        echo "********** AOT Prebuild aiter kernel finished ... **********"
        python3 -m sglang.launch_server \
            --model-path "${model_path}" \
            --host localhost \
            --port 9000 \
            --tp-size ${TP} \
            --trust-remote-code \
            --mm-attention-backend "aiter_attn"\
            --chunked-prefill-size 32768 \
            --mem-fraction-static 0.85 \
            --disable-radix-cache \
            --max-prefill-tokens 32768 \
            --cuda-graph-max-bs 8 \
            --page-size 1024  \
            --mm-enable-dp-encoder \
            --kv-cache-dtype fp8_e4m3 \
            --enable-aiter-allreduce-fusion \
            --max-running-requests 128 \
            --watchdog-timeout 1200 &
        sglang_pid=$!
    elif [[ "${model_name}" == "Qwen3-235B-A22B-Instruct-2507-FP8-Dynamic" ]]; then
        export SGLANG_USE_AITER=1
        export AITER_ROPE_FUSED_QKNORM=1
        export SGLANG_ROCM_USE_AITER_PA_ASM_PRESHUFFLE_LAYOUT=0
        export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
        export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
        python3 -m sglang.launch_server \
            --model-path "${model_path}" \
            --host localhost \
            --port 9000 \
            --tp-size ${TP} \
            --disable-radix-cache \
            --trust-remote-code \
            --max-prefill-tokens 65536 \
            --context-length 65536 \
            --kv-cache-dtype fp8_e4m3 \
            --page-size 1024 \
            --max-running-requests 512 \
            --chunked-prefill-size 65536 \
            --mem-fraction-static 0.9 \
            --mm-attention-backend aiter_attn \
            --enable-aiter-allreduce-fusion \
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
            "model": "${model_path}",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png"
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
    if [[ "${model_name}" == "Qwen3-235B-A22B-Instruct-2507-FP8-Dynamic" ]]; then
        python3 benchmark/gsm8k/bench_sglang.py \
            --port 9000 \
            --num-questions 2000 \
            | tee text_model_evaluation_${model_name}_TP${TP}_EP${EP}.log
    else
        python3 benchmark/mmmu/bench_sglang.py \
            --port 9000 \
            --concurrency 16 \
            | tee vision_model_evaluation_${model_name}_TP${TP}_EP${EP}.log
    fi

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