#!/usr/bin/env bash
# bench_serving sweep — gsp, 4096 ISL / 512 OSL, ~55% prefix cache hit
#
# MODE=double_sparsity (default) tags output files for the DS column of the
# two-column comparison report. Pair with development/benchmark_baseline.sh
# (MODE=native_nsa) on the same hardware and workload to populate both
# columns. Outputs land in development/results/ next to the comparator
# (benchmark_compare.py) and AC-9/AC-11 artifact dir.
#
# Server must already be running on PORT (default 30000) with Double
# Sparsity enabled — see development/serve_double_sparsity.sh for the
# exact server invocation.
#
# Per AC-8 / AC-9 the locked sweep is concurrency 16 / 32 / 64 (was conc=64
# only). Override CONCURRENCIES to narrow if you only need a quick smoke.
# Each run emits a `.meta.json` sidecar alongside the JSONL with the
# commit SHA, server args (from /get_server_info), seed,
# chunked-prefill setting, and timestamp so the AC-11 comparator can
# verify both columns share the same operating point.
set -euo pipefail

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MODE="${MODE:-double_sparsity}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/development/results}"
mkdir -p "${RESULTS_DIR}"

# ISL = sys + q = 2253 + 1843 = 4096; cache_hit = 2253/4096 ≈ 0.550
SYS_LEN=2253
Q_LEN=1843
OUT_LEN=512
NUM_PROMPTS=$(( 5 * 64 ))
NUM_GROUPS=1

# Per-concurrency seeds, mirroring the reference sweep pattern.
declare -A SEEDS=( [16]=213 [32]=431 [64]=31234 )

# Default sweep: AC-8 / AC-9 plan-locked conc 16 / 32 / 64.
# Override CONCURRENCIES (space-separated) for a narrower sweep.
CONCURRENCIES="${CONCURRENCIES:-16 32 64}"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

for CONCURRENCY in ${CONCURRENCIES}; do
  SEED="${SEEDS[$CONCURRENCY]}"
  OUTPUT_FILE="${RESULTS_DIR}/${MODE}_gsp_isl4096_osl512_c${CONCURRENCY}.jsonl"
  META_FILE="${OUTPUT_FILE}.meta.json"
  echo ">>> mode=${MODE} concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS} groups=${NUM_GROUPS}x${NUM_PROMPTS} seed=${SEED} output=${OUTPUT_FILE}"

  # Capture server args for the AC-11 comparator (best-effort; absent on
  # CI / non-server environments — the .meta.json field will be null).
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

  # Pass /get_server_info JSON as env-var DATA, not Python source. The
  # Round 23 heredoc spliced it as source code and crashed on JSON
  # `true`/`false`/`null` (not valid Python identifiers).
  COMMIT_SHA="${COMMIT_SHA}" \
  MODE="${MODE}" \
  CONCURRENCY="${CONCURRENCY}" \
  SEED="${SEED}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  ISL_TOTAL_TOKENS="$(( SYS_LEN + Q_LEN ))" \
  OSL_TOKENS="${OUT_LEN}" \
  TIMESTAMP_UTC="${TIMESTAMP_UTC}" \
  SERVER_ARGS_JSON="${SERVER_ARGS_JSON}" \
  TRIAL_ID="${TRIAL_ID:-1}" \
  WARMUP_REQUESTS="${WARMUP_REQUESTS:-}" \
  MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-}" \
  python3 "$(dirname "$0")/_bench_meta_writer.py" > "${META_FILE}"
  echo "    sidecar          = ${META_FILE}"
done
