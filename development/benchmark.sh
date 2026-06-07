#!/usr/bin/env bash
# bench_serving sweep — gsp, 4096 ISL / 512 OSL, ~55% prefix cache hit
#
# Canonical client workload for zai-org/GLM-5.1 (FP8) per development/CLIENT_SLOS.md
# (rebased 2026-06-07): 4096 ISL, 512 OSL, max-concurrency 64 / min 16, ~55% prefix
# cache hit. The model is resolved server-side, so this driver is model-agnostic —
# point PORT at a GLM-5.1 (FP8) server (DS or native-DSA). bench_serving now emits
# the new per-request decode-throughput SLO metric `*_decode_throughput_tps`
# (output_tokens / (e2e - ttft)); the >= 30 tok/s floor is read from that field.
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
# Per-epoch prompt count. bench_serving runs FULL epochs of NUM_PROMPTS before
# re-checking the measurement window, so one epoch at this 4096-ISL shape can
# take ~15 min at conc 16. The plan-locked AC-8/AC-9/AC-11 default is 5*64=320;
# a TIER-1 smoke can override NUM_PROMPTS (e.g. 64) so each epoch is short and
# the shortened-window smoke actually finishes quickly. Keep it >= the largest
# concurrency so a smoke epoch still saturates the requested concurrency.
NUM_PROMPTS="${NUM_PROMPTS:-$(( 5 * 64 ))}"
NUM_GROUPS=1

# Per-concurrency seeds, mirroring the reference sweep pattern.
declare -A SEEDS=( [16]=213 [32]=431 [64]=31234 )

# Default sweep: AC-8 / AC-9 plan-locked conc 16 / 32 / 64.
# Override CONCURRENCIES (space-separated) for a narrower sweep.
CONCURRENCIES="${CONCURRENCIES:-16 32 64}"

# AC-11 spec: >= 3 independent trials per concurrency, fixed seed family,
# 120s warmup, 600s measurement window. TRIALS=3 is the minimum the
# comparator accepts; operators can override for additional headroom.
TRIALS="${TRIALS:-3}"
WARMUP_SECONDS="${WARMUP_SECONDS:-120}"
MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

for CONCURRENCY in ${CONCURRENCIES}; do
  SEED="${SEEDS[$CONCURRENCY]}"
  for TRIAL_ID in $(seq 1 "${TRIALS}"); do
    OUTPUT_FILE="${RESULTS_DIR}/${MODE}_gsp_isl4096_osl512_c${CONCURRENCY}_t${TRIAL_ID}.jsonl"
    META_FILE="${OUTPUT_FILE}.meta.json"
    echo ">>> mode=${MODE} concurrency=${CONCURRENCY} trial=${TRIAL_ID}/${TRIALS} num_prompts=${NUM_PROMPTS} seed=${SEED} output=${OUTPUT_FILE}"

    # Capture server args for the AC-11 comparator (best-effort; absent on
    # CI / non-server environments — the .meta.json field will be null).
    SERVER_ARGS_JSON="$(curl -s --max-time 5 "http://${HOST}:${PORT}/get_server_info" || echo '{}')"
    TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    python3 -m sglang.bench_serving \
      --backend sglang \
      --host "${HOST}" \
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
      --warmup-seconds "${WARMUP_SECONDS}" \
      --measurement-window-seconds "${MEASUREMENT_WINDOW_S}" \
      --output-file "${OUTPUT_FILE}" \
      --output-details

    # Refuse the run if the observed JSONL `duration` is below
    # MEASUREMENT_WINDOW_S — guards against bench_serving bailing out
    # early before the time-based loop met its threshold.
    OBSERVED_DURATION="$(python3 -c "
import json,sys
with open('${OUTPUT_FILE}') as fh:
    last = [json.loads(l) for l in fh if l.strip()][-1]
print(last.get('duration', 0.0))
")"
    if python3 -c "import sys; sys.exit(0 if float('${OBSERVED_DURATION}') >= float('${MEASUREMENT_WINDOW_S}') else 1)"; then
      :
    else
      echo "FATAL: ${OUTPUT_FILE} duration=${OBSERVED_DURATION}s < MEASUREMENT_WINDOW_S=${MEASUREMENT_WINDOW_S}s — refusing to publish AC-11 artifact." >&2
      exit 1
    fi

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
    TRIAL_ID="${TRIAL_ID}" \
    WARMUP_SECONDS="${WARMUP_SECONDS}" \
    MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S}" \
    python3 "$(dirname "$0")/_bench_meta_writer.py" > "${META_FILE}"
    echo "    sidecar          = ${META_FILE}"
  done
done
