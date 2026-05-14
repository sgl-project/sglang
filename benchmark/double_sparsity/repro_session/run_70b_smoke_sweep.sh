#!/usr/bin/env bash
# Quick end-to-end validation of the native sparse-decode dispatch.
# Short context, small concurrency sweep, small n_requests — should
# complete in ~10 minutes after model load on 8xH200.
#
# Run from repo root:
#   bash benchmark/double_sparsity/repro_session/run_70b_smoke_sweep.sh
set -euo pipefail

CALIB="${CALIB:-/workspace/calib_llama_3_1_70b_wikitext_s32.json}"
CTX="${CTX:-16384}"
CONC="${CONCURRENCIES:-1,4}"
N_REQ="${N_REQUESTS:-4}"
OUT_LEN="${OUTPUT_LEN:-128}"
OUT_DIR="${OUT_DIR:-./bench_70b_smoke_${CTX}}"

mkdir -p "$OUT_DIR"
echo "Smoke sweep: CTX=$CTX  concurrencies=$CONC  n_req=$N_REQ  out_len=$OUT_LEN"

COMMON=(
  --model meta-llama/Llama-3.1-70B-Instruct
  --context-len "$CTX"
  --output-len "$OUT_LEN"
  --n-requests "$N_REQ"
  --concurrency "$CONC"
  --tp-size 8
  --mem-fraction-static 0.85
  --max-running-requests 16
)

echo
echo "=== Leg 1: DS off ==="
time SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=python python3 benchmark/double_sparsity/bench_decode.py \
  --config branch_ds_off \
  --output-json "$OUT_DIR/off.json" \
  --server-log "$OUT_DIR/server_off.log" \
  "${COMMON[@]}"

echo
echo "=== Leg 2: DS on (native) ==="
time SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=python python3 benchmark/double_sparsity/bench_decode.py \
  --config branch_ds_on \
  --calibration "$CALIB" \
  --output-json "$OUT_DIR/on.json" \
  --server-log "$OUT_DIR/server_on.log" \
  --token-budget 512 \
  --recent-tokens 64 \
  --sink-tokens 4 \
  --min-seq-len 4096 \
  --max-selected-per-request 8192 \
  --block-t 1024 \
  --k-block 64 \
  "${COMMON[@]}"

echo
echo "=== compare ==="
PYTHONPATH=python python3 benchmark/double_sparsity/compare.py \
  --branch-off "$OUT_DIR/off.json" \
  --branch-on "$OUT_DIR/on.json" || true
