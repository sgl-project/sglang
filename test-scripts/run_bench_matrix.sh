#!/usr/bin/env bash
# Orchestrate the full bench matrix:
#   modes:          dense, sparse
#   input_len:      16384, 32768, 65536, 102400  (16k / 32k / 64k / 100k)
#   output_len:     1024, 2048, 4096, 8192        ( 1k /  2k /  4k / 8k)
#
# Total = 2 * 4 * 4 = 32 runs.
#
# Server reuse: one server is launched per (input_len, mode) and reused
# across all output_lens (output_len is a client-side bench param only),
# saving one ~700s startup per extra output_len. Modes iterate inside each
# input_len so dense/sparse for the same workload run back-to-back.
#
# Concurrency / num_prompts are auto-scaled by bench_multi.sh based on
# total context length (longer ctx -> smaller conc).
#
# Override any cell by exporting env vars before calling this script:
#   VERIFY=1              only run the first (sparse,16k,1k) cell with tiny N for smoke
#   MODES="dense"         restrict modes
#   INPUT_LENS="16384"    restrict input lens
#   OUTPUT_LENS="1024"    restrict output lens
#   MAX_CONC=8            override concurrency
#   NUM_PROMPTS=16        override num prompts (VERIFY sets this to 8)
#
# Run in background:
#   nohup bash test-scripts/run_bench_matrix.sh > /tmp/matrix.log 2>&1 &
# pkill -9 -f "sglang"
set -euo pipefail
cd "$(dirname "$0")/.."
RESULTS_DIR="${RESULTS_DIR:-/tmp/glm_bench_matrix}"
mkdir -p "$RESULTS_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_MULTI="$SCRIPT_DIR/bench_multi.sh"

# default matrix
MODES="${MODES:-dense sparse}"
INPUT_LENS="${INPUT_LENS:-16384 32768 65536 102400}"
OUTPUT_LENS="${OUTPUT_LENS:-1024 2048 4096 8192}"

if [[ "${VERIFY:-0}" == "1" ]]; then
  echo "=== VERIFY mode: only (sparse, 16k, 1k) with tiny NUM_PROMPTS ==="
  MODES="sparse"
  INPUT_LENS="16384"
  OUTPUT_LENS="1024"
  export NUM_PROMPTS=8
  export MAX_CONC=4
fi

MATRIX_LOG="$RESULTS_DIR/matrix.log"
# NOTE: do NOT use `exec > >(tee ...)` here — process substitution is torn
# down immediately under a non-interactive nohup background shell (exit 137).
# Callers should redirect stdout/stderr to a file instead.

echo "=========================================================="
echo " bench_matrix start: $(date '+%F %T')"
echo "   modes:       $MODES"
echo "   input_lens:  $INPUT_LENS"
echo "   output_lens: $OUTPUT_LENS"
echo "   results_dir: $RESULTS_DIR"
echo "=========================================================="

TOTAL=0; DONE=0; FAIL=0
for MODE in $MODES; do
  for IL in $INPUT_LENS; do
    for OL in $OUTPUT_LENS; do
      TOTAL=$((TOTAL+1))
    done
  done
done
echo "[matrix] total runs: $TOTAL"

# One server per (input_len, mode); inner sweep over output_lens reuses it.
# To keep dense/sparse comparable per input_len, iterate modes inside input_len.
for IL in $INPUT_LENS; do
  for MODE in $MODES; do
    # collect only output_lens that are NOT already in results.csv
    PENDING=""
    for OL in $OUTPUT_LENS; do
      if [[ "${SKIP_DONE:-1}" == "1" && -f "$RESULTS_DIR/results.csv" ]] \
         && grep -q ",$MODE,$IL,$OL," "$RESULTS_DIR/results.csv"; then
        echo "[matrix] SKIP mode=$MODE in=$IL out=$OL — already in results.csv"
        DONE=$((DONE+1))
        continue
      fi
      PENDING="$PENDING $OL"
    done
    PENDING="${PENDING# }"

    if [[ -z "$PENDING" ]]; then
      echo "[matrix] all output_lens done for mode=$MODE in=$IL — skipping server launch"
      continue
    fi

    NPEND=$(wc -w <<< "$PENDING")
    echo ""
    echo "=========================================================="
    echo "[matrix] server: mode=$MODE in=$IL outs=[$PENDING]  @ $(date '+%F %T')"
    echo "         ($((DONE+1))..$((DONE+NPEND)) / $TOTAL)"
    echo "=========================================================="

    # pass through NUM_PROMPTS / MAX_CONC only if set (e.g. VERIFY mode)
    EXTRA_ENV=()
    [[ -n "${NUM_PROMPTS:-}" ]] && EXTRA_ENV+=("NUM_PROMPTS=$NUM_PROMPTS")
    [[ -n "${MAX_CONC:-}"    ]] && EXTRA_ENV+=("MAX_CONC=$MAX_CONC")
    if env MODE="$MODE" INPUT_LEN="$IL" OUTPUT_LENS="$PENDING" \
          RESULTS_DIR="$RESULTS_DIR" \
          "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}" \
          bash "$BENCH_MULTI"; then
      echo "[matrix] OK mode=$MODE in=$IL ($NPEND cells)"
      DONE=$((DONE+NPEND))
    else
      FAIL=$((FAIL+NPEND))
      DONE=$((DONE+NPEND))
      echo "[matrix] FAILED mode=$MODE in=$IL — continuing"
    fi
    # safety cool-down + ensure GPU memory freed
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 15
  done
done

echo ""
echo "=========================================================="
echo " bench_matrix DONE: total=$TOTAL ok=$((TOTAL-FAIL)) fail=$FAIL"
echo "   summary: $RESULTS_DIR/results.csv"
echo "=========================================================="
column -t -s, "$RESULTS_DIR/results.csv" 2>/dev/null || cat "$RESULTS_DIR/results.csv"
