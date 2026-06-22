#!/usr/bin/env bash
set -euo pipefail

CLIENT_HOST=${CLIENT_HOST:-127.0.0.1}
PORT=${PORT:-30000}
BENCH_LOG=${BENCH_LOG:-/tmp/sglang_dsv4_benchmark.log}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-96}
INPUT_LEN=${INPUT_LEN:-8000}
OUTPUT_LEN=${OUTPUT_LEN:-1000}
NUM_PROMPTS=${NUM_PROMPTS:-96}

echo "Running benchmark, results -> ${BENCH_LOG}"
python3 -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host "${CLIENT_HOST}" \
    --port "${PORT}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-requests 0 \
    2>&1 | tee "${BENCH_LOG}"
