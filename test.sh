#!/bin/bash
echo "开始运行第一个测试（带 --max-concurrency 1）..."
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name random-image \
    --num-prompts 500 \
    --random-image-num-images 3 \
    --random-image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512 \
    --max-concurrency 1

echo "等待5分钟..."
sleep 300

echo "开始运行第二个测试（不带 --max-concurrency 1）..."
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name random-image \
    --num-prompts 500 \
    --random-image-num-images 3 \
    --random-image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512