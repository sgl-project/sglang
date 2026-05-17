#!/usr/bin/env bash
# Capture nsys profiles of DS-off vs DS-on at the actual winning bench point.
#
# Defaults match the 70B/TP=8/128K conc=32 both-gates-pass point
# documented in benchmark/double_sparsity/DESIGN.md:
#   CONC=32, OUTPUT_LEN=64 (short trace; the headline bench uses
#   output_len=256/512), TOKEN_BUDGET=8192, MAX_SELECTED=16384,
#   BLOCK_T=1024, K_BLOCK=64, RECENT=64, SINK=4, MIN_SEQ_LEN=4096.
#
# Override via env, e.g. to reproduce a conc=16 search-left run:
#   CONC=16 TOKEN_BUDGET=2048 MAX_SELECTED=8192 \
#       bash benchmark/double_sparsity/repro_session/run_nsys_at_winning_point.sh \
#       /path/to/calib.json
#
# Output:
#   /workspace/nsys_reports/ds_native_off_${CTX}_c${CONC}.nsys-rep
#   /workspace/nsys_reports/ds_native_on_${CTX}_c${CONC}.nsys-rep
#
# Diff via:
#   python3 benchmark/double_sparsity/repro_session/compare_nsys.py \
#       /workspace/nsys_reports/ds_native_off_${CTX}_c${CONC}.nsys-rep \
#       /workspace/nsys_reports/ds_native_on_${CTX}_c${CONC}.nsys-rep

set -euo pipefail

CALIB="${1:?usage: $0 <path-to-calibration.json>}"
CTX="${CTX:-131072}"
CONC="${CONC:-32}"
N_REQ="${N_REQUESTS:-${CONC}}"
OUT_LEN="${OUTPUT_LEN:-64}"
TOKEN_BUDGET="${TOKEN_BUDGET:-8192}"
MAX_SELECTED="${MAX_SELECTED:-16384}"
BLOCK_T="${BLOCK_T:-1024}"
K_BLOCK="${K_BLOCK:-64}"
RECENT="${RECENT:-64}"
SINK="${SINK:-4}"
MIN_SEQ_LEN="${MIN_SEQ_LEN:-4096}"
OUT_DIR="${OUT_DIR:-/workspace/nsys_reports}"

mkdir -p "$OUT_DIR"

NSYS_BASE_ARGS=(
  --trace cuda,nvtx,osrt
  --gpuctxsw=true
  --force-overwrite=true
  --stats=false
)

echo "nsys leg 1: DS off"
nsys profile "${NSYS_BASE_ARGS[@]}" \
  -o "$OUT_DIR/ds_native_off_${CTX}_c${CONC}" \
  -- env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 PYTHONPATH=python \
    python3 benchmark/double_sparsity/bench_decode.py \
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
  -- env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 PYTHONPATH=python \
    python3 benchmark/double_sparsity/bench_decode.py \
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
    --token-budget "$TOKEN_BUDGET" \
    --recent-tokens "$RECENT" --sink-tokens "$SINK" \
    --min-seq-len "$MIN_SEQ_LEN" \
    --max-selected-per-request "$MAX_SELECTED" \
    --block-t "$BLOCK_T" --k-block "$K_BLOCK" \
    --output-json "$OUT_DIR/on_${CTX}_c${CONC}.json"

echo
echo "Reports written:"
echo "  $OUT_DIR/ds_native_off_${CTX}_c${CONC}.nsys-rep"
echo "  $OUT_DIR/ds_native_on_${CTX}_c${CONC}.nsys-rep"
echo
echo "Diff via:"
echo "  python3 benchmark/double_sparsity/repro_session/compare_nsys.py \\"
echo "    $OUT_DIR/ds_native_off_${CTX}_c${CONC}.nsys-rep \\"
echo "    $OUT_DIR/ds_native_on_${CTX}_c${CONC}.nsys-rep"
