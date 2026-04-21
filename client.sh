#!/usr/bin/env bash
# Benchmark client for Qwen3.5-397B-A17B-FP8.
#
# Decode-heavy workload (short prompts, short outputs) specifically stresses
# the masked GEMM path where most expert slots are empty.  The original
# parameters are preserved; a second target is added for memory profiling.
#
# Usage:
#   bash client.sh            — throughput benchmark (original params)
#   bash client.sh --profile  — single-prompt pass for peak-memory measurement

if [[ "$1" == "--profile" ]]; then
  # Single prompt, measure peak GPU memory allocated during decode.
  python3 -m sglang.bench_serving \
    --backend sglang \
    --model  /workspace/models/Qwen3.5-397B-A17B-FP8 \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 40 \
    --num-prompts 1 \
    --max-concurrency 1 \
    --request-rate 1
else
  # Throughput benchmark — same params as before.
  python3 -m sglang.bench_serving \
    --backend sglang \
    --model  /workspace/models/Qwen3.5-397B-A17B-FP8 \
    --dataset-name random \
    --random-input-len 10000 \
    --random-output-len 40 \
    --num-prompts 100 \
    --max-concurrency 50 \
    --request-rate 2
fi
