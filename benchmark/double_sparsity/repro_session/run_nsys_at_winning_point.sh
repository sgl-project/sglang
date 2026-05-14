#!/usr/bin/env bash
# Capture nsys profiles of DS-off vs DS-on at a fixed workload point.
# Intended to be run AFTER the 128K sweep identifies the winning
# concurrency. Default CTX=131072, CONC=8, OUTPUT_LEN=128 (short to
# keep the trace size manageable while still covering many decode steps
# per request).
#
# Usage:
#   CONC=8 bash benchmark/double_sparsity/repro_session/run_nsys_at_winning_point.sh \
#       /path/to/calib.json
#
# Output:
#   /workspace/nsys_reports/ds_native_off_${CTX}_c${CONC}.nsys-rep
#   /workspace/nsys_reports/ds_native_on_${CTX}_c${CONC}.nsys-rep
#
# Diff via `benchmark/double_sparsity/repro_session/compare_nsys.py`
# (already exists, just point it at the new pair of reports).

set -euo pipefail

CALIB="${1:?usage: $0 <path-to-calibration.json>}"
CTX="${CTX:-131072}"
CONC="${CONC:-8}"
N_REQ="${N_REQUESTS:-${CONC}}"
OUT_LEN="${OUTPUT_LEN:-128}"
OUT_DIR="/workspace/nsys_reports"

mkdir -p "$OUT_DIR"

# nsys capture window: cover model load (~3-5 min) + workload (~1-2 min).
# Use --capture-range cudaProfilerApi to bound the trace to the workload
# only; bench_decode.py doesn't currently emit cudaProfilerApi markers
# so we just record the whole run with --capture-range none and accept
# the slightly larger trace. (Future: have bench_decode emit markers.)
NSYS_BASE_ARGS=(
  --trace cuda,nvtx,osrt
  --gpuctxsw=true
  --force-overwrite=true
  --stats=false
)

# Smaller capture: 30s of decode-time samples is enough to characterize
# the kernel mix. We use --duration to bound trace size and start
# nsys AFTER the server is up.
# bench_decode.py launches the server inline, so the simplest path is
# to wrap the whole bench invocation in nsys.

echo "nsys leg 1: DS off"
nsys profile "${NSYS_BASE_ARGS[@]}" \
  -o "$OUT_DIR/ds_native_off_${CTX}_c${CONC}" \
  -- python3 benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_off \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --context-len "$CTX" \
    --output-len "$OUT_LEN" \
    --n-requests "$N_REQ" \
    --concurrency "$CONC" \
    --tp-size 8 \
    --mem-fraction-static 0.85 \
    --max-running-requests "$CONC" \
    --output-json "$OUT_DIR/off_${CTX}_c${CONC}.json"

echo "nsys leg 2: DS on (native)"
nsys profile "${NSYS_BASE_ARGS[@]}" \
  -o "$OUT_DIR/ds_native_on_${CTX}_c${CONC}" \
  -- python3 benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_on \
    --calibration "$CALIB" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --context-len "$CTX" \
    --output-len "$OUT_LEN" \
    --n-requests "$N_REQ" \
    --concurrency "$CONC" \
    --tp-size 8 \
    --mem-fraction-static 0.85 \
    --max-running-requests "$CONC" \
    --token-budget 512 --recent-tokens 64 --sink-tokens 4 --min-seq-len 4096 \
    --max-selected-per-request 8192 --block-t 1024 --k-block 64 \
    --output-json "$OUT_DIR/on_${CTX}_c${CONC}.json"

echo
echo "Reports written:"
echo "  $OUT_DIR/ds_native_off_${CTX}_c${CONC}.nsys-rep"
echo "  $OUT_DIR/ds_native_on_${CTX}_c${CONC}.nsys-rep"
echo
echo "Diff via:"
echo "  # edit compare_nsys.py paths to point at the new pair, then:"
echo "  PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/compare_nsys.py"
