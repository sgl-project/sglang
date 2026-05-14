#!/usr/bin/env bash
# 70B/TP=8 long-context concurrency-sweep driver — matches the v2 plan
# (post-pivot from single-request latency to long-context throughput).
#
# Sweeps concurrency in {1, 2, 4, 8, 16} against the SAME server (one
# model load per leg), against both DS-off and DS-on. Outputs a per-leg
# JSON with `concurrencies` and `results` arrays; `compare.py` produces
# the per-concurrency table and picks the best-speedup point.
#
# Usage:
#   CTX=65536 bash benchmark/double_sparsity/run_70b_sweep.sh /path/to/calib.json
#   CTX=131072 bash benchmark/double_sparsity/run_70b_sweep.sh /path/to/calib.json
#
# Required env / args:
#   $1 — path to a wikitext-calibrated double-sparsity JSON
#   CTX — bench context length (default 65536)
#   OUT_DIR — where to write JSONs (default ./bench_70b_sweep_${CTX})
#   CONCURRENCIES — CSV (default 1,2,4,8,16)
#   N_REQUESTS — sample size per concurrency (default 8)
#   OUTPUT_LEN — generated tokens per request (default 512 for iteration,
#                bump to 1024 for the final report)

set -euo pipefail

CALIB="${1:?usage: $0 <path-to-calibration.json>}"
CTX="${CTX:-65536}"
CONC="${CONCURRENCIES:-1,2,4,8,16}"
N_REQ="${N_REQUESTS:-8}"
OUT_LEN="${OUTPUT_LEN:-512}"
OUT_DIR="${OUT_DIR:-./bench_70b_sweep_${CTX}}"

mkdir -p "$OUT_DIR"
echo "Sweep: CTX=$CTX  concurrencies=$CONC  n_req=$N_REQ  out_len=$OUT_LEN"
echo "Calibration: $CALIB"
echo "Output dir: $OUT_DIR"

COMMON_ARGS=(
  --model meta-llama/Llama-3.1-70B-Instruct
  --context-len "$CTX"
  --output-len "$OUT_LEN"
  --n-requests "$N_REQ"
  --concurrency "$CONC"
  --tp-size 8
  --mem-fraction-static 0.85
  --max-running-requests 16
  --niah
  --niah-n-samples 5
  --niah-context-tokens "$CTX"
)

echo
echo "=== Leg 1: DS off ==="
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=python python3 benchmark/double_sparsity/bench_decode.py \
  --config branch_ds_off \
  --output-json "$OUT_DIR/branch_ds_off.json" \
  --server-log "$OUT_DIR/server_off.log" \
  "${COMMON_ARGS[@]}"

echo
echo "=== Leg 2: DS on (native sparse decode) ==="
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=python python3 benchmark/double_sparsity/bench_decode.py \
  --config branch_ds_on \
  --calibration "$CALIB" \
  --output-json "$OUT_DIR/branch_ds_on.json" \
  --server-log "$OUT_DIR/server_on.log" \
  --token-budget 512 \
  --recent-tokens 64 \
  --sink-tokens 4 \
  --min-seq-len 4096 \
  --max-selected-per-request 8192 \
  --block-t 1024 \
  --k-block 64 \
  "${COMMON_ARGS[@]}"

echo
echo "=== compare ==="
PYTHONPATH=python python3 benchmark/double_sparsity/compare.py \
  --branch-off "$OUT_DIR/branch_ds_off.json" \
  --branch-on "$OUT_DIR/branch_ds_on.json"
