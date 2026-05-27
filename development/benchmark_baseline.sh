#!/usr/bin/env bash
# bench_serving sweep — native_nsa baseline (DeepSeek-V3.2 FP8, single-instance)
#
# Mirrors development/benchmark.sh exactly except for the output file tagging
# (MODE=native_nsa) so the downstream two-column report (development/
# benchmark_compare.py) can pair the same workload (ISL, OSL, prefix-cache
# hit, concurrency, seed) across the native_nsa baseline and the
# double_sparsity run without diff churn on every column.
#
# Pair with development/benchmark.sh (MODE=double_sparsity) for the
# eventual side-by-side report. The two columns must agree on:
#   { GPU id, TP size, page size, radix-cache setting, concurrency, seed,
#     dataset shape (ISL/OSL/cache-hit), model revision }
#
# Server must already be running on PORT (default 30000) with **NSA enabled
# and Double Sparsity disabled** — see development/serve_native_nsa.sh for
# the exact server invocation.
#
# AC-9 / AC-11 spec: default sweep is concurrency 16 / 32 / 64 (was conc=64
# only). Each run writes its JSONL into development/results/ and emits a
# `.meta.json` sidecar with commit SHA, server args, seed,
# chunked-prefill setting, and timestamp so the AC-11 comparator can
# verify both columns share the same operating point.

set -euo pipefail

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MODE="${MODE:-native_nsa}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/development/results}"
mkdir -p "${RESULTS_DIR}"

# ISL = sys + q = 2253 + 1843 = 4096; cache_hit = 2253/4096 ≈ 0.550
SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512
NUM_PROMPTS=$(( 5 * 64 ))
NUM_GROUPS=1

declare -A SEEDS=( [16]=213 [32]=431 [64]=31234 )

# Default sweep: AC-9 plan-locked conc 16 / 32 / 64 (matches benchmark.sh).
CONCURRENCIES="${CONCURRENCIES:-16 32 64}"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

for CONCURRENCY in ${CONCURRENCIES}; do
  SEED="${SEEDS[$CONCURRENCY]}"
  OUTPUT_FILE="${RESULTS_DIR}/${MODE}_gsp_isl4096_osl512_c${CONCURRENCY}.jsonl"
  META_FILE="${OUTPUT_FILE}.meta.json"
  echo ">>> mode=${MODE} concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS} groups=${NUM_GROUPS}x${NUM_PROMPTS} seed=${SEED} output=${OUTPUT_FILE}"

  SERVER_ARGS_JSON="$(curl -s --max-time 5 "http://${HOST}:${PORT}/get_server_info" || echo '{}')"
  TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

  python3 - <<PYEOF > "${META_FILE}"
import json
print(json.dumps({
    "commit_sha": "${COMMIT_SHA}",
    "mode": "${MODE}",
    "concurrency": ${CONCURRENCY},
    "seed": ${SEED},
    "num_prompts": ${NUM_PROMPTS},
    "isl_total_tokens": $(( SYS_LEN + Q_LEN )),
    "osl_tokens": ${OUT_LEN},
    "timestamp_utc": "${TIMESTAMP_UTC}",
    "server_args": ${SERVER_ARGS_JSON},
}, indent=2))
PYEOF
  echo "    sidecar          = ${META_FILE}"
done
