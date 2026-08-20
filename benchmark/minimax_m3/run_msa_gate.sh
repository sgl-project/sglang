#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:?set BASE_URL, for example http://127.0.0.1:30000}"
MODEL="${MODEL:?set MODEL to the served MiniMax-M3 model id/path}"
LABEL="${LABEL:?set LABEL to external or flashinfer}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a unique result directory}"
LONGBENCH_SUBSET="${LONGBENCH_SUBSET:?set LONGBENCH_SUBSET to the frozen JSON subset}"
NUM_THREADS="${NUM_THREADS:-32}"

mkdir -p "${OUTPUT_DIR}"

python benchmark/minimax_m3/record_fixed_parity.py \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --label "${LABEL}" \
  --output "${OUTPUT_DIR}/fixed_parity.json"

python -m sglang.test.run_eval \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --eval-name gpqa \
  --num-examples 198 \
  --num-threads "${NUM_THREADS}" \
  --max-tokens 8192 \
  --temperature 0 \
  | tee "${OUTPUT_DIR}/gpqa.log"
cp "/tmp/gpqa_${MODEL//\//_}.json" "${OUTPUT_DIR}/gpqa.json"

python -m sglang.test.run_eval \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --eval-name longbench_v2 \
  --dataset-path "${LONGBENCH_SUBSET}" \
  --num-examples 100 \
  --num-threads "${NUM_THREADS}" \
  --max-tokens 4096 \
  --temperature 0 \
  | tee "${OUTPUT_DIR}/longbench_v2.log"
cp "/tmp/longbench_v2_${MODEL//\//_}.json" "${OUTPUT_DIR}/longbench_v2.json"
cp "${LONGBENCH_SUBSET}.manifest.json" "${OUTPUT_DIR}/longbench_v2_subset_manifest.json"

for concurrency in 1 8 32 128; do
  python -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --dataset-name random \
    --num-prompts 256 \
    --random-input-len 8192 \
    --random-output-len 1024 \
    --random-range-ratio 1 \
    --request-rate inf \
    --max-concurrency "${concurrency}" \
    --seed 20260819 \
    --flush-cache \
    --output-file "${OUTPUT_DIR}/serving_c${concurrency}.jsonl"
done
