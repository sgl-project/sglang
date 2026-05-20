#!/usr/bin/env bash
# bench_serving sweep — native_nsa baseline (DeepSeek-V3.2 FP8, single-instance)
#
# Mirrors development/benchmark.sh exactly except for the output file tagging
# so a downstream two-column report can pair the same workload (ISL, OSL,
# prefix-cache hit, concurrency, seed) across the native_nsa baseline and the
# double_sparsity run without diff churn on every column.
#
# Pair with development/benchmark.sh (set MODE=double_sparsity there) for the
# eventual side-by-side report. The two columns must agree on:
#   { GPU id, TP size, page size, radix-cache setting, concurrency, seed,
#     dataset shape (ISL/OSL/cache-hit), model revision }
#
# Server must already be running on PORT (default 30000) with **NSA enabled
# and Double Sparsity disabled** — see development/serve_native_nsa.sh for
# the exact server invocation.

set -euo pipefail

PORT="${PORT:-30000}"
MODE="${MODE:-native_nsa}"

# ISL = sys + q = 2253 + 1843 = 4096; cache_hit = 2253/4096 ≈ 0.550
SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512
NUM_PROMPTS=$(( 5 * 64 ))
NUM_GROUPS=1

declare -A SEEDS=( [16]=213 [32]=431 [64]=31234 )

# Default sweep matches development/benchmark.sh: conc 64 only. Override by
# setting CONCURRENCIES, e.g. CONCURRENCIES="16 32 64".
CONCURRENCIES="${CONCURRENCIES:-64}"

for CONCURRENCY in ${CONCURRENCIES}; do
  SEED="${SEEDS[$CONCURRENCY]}"
  OUTPUT_FILE="${MODE}_gsp_isl4096_osl512_c${CONCURRENCY}.jsonl"
  echo ">>> mode=${MODE} concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS} groups=${NUM_GROUPS}x${NUM_PROMPTS} seed=${SEED} output=${OUTPUT_FILE}"

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
    --output-file "${OUTPUT_FILE}" \
    --output-details
done
