#!/usr/bin/env bash
# bench_serving sweep — gsp, 4096 ISL / 512 OSL, ~55% prefix cache hit
set -euo pipefail

PORT=30000

# ISL = sys + q = 2253 + 1843 = 4096; cache_hit = 2253/4096 ≈ 0.550
SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512
NUM_PROMPTS=$(( 5 * 64 ))
NUM_GROUPS=1

# Per-concurrency seeds, mirroring the reference sweep pattern.
declare -A SEEDS=( [16]=213 [32]=431 [64]=31234 )

# for CONCURRENCY in 16 32 64; do
for CONCURRENCY in 64; do
  SEED="${SEEDS[$CONCURRENCY]}"
  echo ">>> concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS} groups=${NUM_GROUPS}x${NUM_PROMPTS} seed=${SEED}"

  python3 -m sglang.bench_serving \
    --backend sglang \
    --port "${PORT}" \
    --seed "${SEED}" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "${NUM_GROUPS}" \
    --gsp-prompts-per-group "${NUM_PROMPTS}" \
    --gsp-system-prompt-len "${SYS_LEN}" \
    --gsp-question-len "${Q_LEN}" \
    --gsp-output-len "${OUT_LEN}" \
    --gsp-range-ratio 1.0 \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --output-file "gsp_isl4096_osl512_c${CONCURRENCY}.jsonl" \
    --output-details
done
