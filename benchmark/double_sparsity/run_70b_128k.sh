#!/usr/bin/env bash
#
# 70B TP=8 H200 visible-win driver — discovery (two-way) run.
#
# Step 1: generate real (wikitext) calibration with --device-map auto.
# Step 2: 3x DS-off baseline at 128K with NIAH on.
# Step 3: 3x DS-on at 128K (block_t=2048, k_block=64) with NIAH on.
# Step 4: compare the median run of each. Exits non-zero if
#         VISIBLE_WIN=FAIL or quality_guard=FAIL.
#
# Stops on first error. Calibration is regenerated only if absent —
# delete the JSON to force a fresh calibration.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-70B-Instruct}"
CTX="${CTX:-131072}"
OUT_LEN="${OUT_LEN:-1024}"
N_REQ="${N_REQ:-4}"
BLOCK_T="${BLOCK_T:-2048}"
K_BLOCK="${K_BLOCK:-64}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1024}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MAX_RUN="${MAX_RUN:-4}"
NIAH_N="${NIAH_N:-5}"
CALIB="${CALIB:-${WORKSPACE}/calib_llama_3_1_70b_wikitext_s32.json}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Step 1: calibration ==="
if [[ ! -f "${CALIB}" ]]; then
  python3 scripts/double_sparsity/calibrate.py \
      --model "${MODEL}" \
      --output "${CALIB}" \
      --heavy-channels 32 \
      --n-samples 64 --seq-len 4096 \
      --dataset wikitext --dataset-subset wikitext-2-raw-v1 \
      --device-map auto
else
  echo "  calibration JSON already present: ${CALIB}"
fi

echo "=== Step 2: DS-off baseline (3 repeats) ==="
for i in 1 2 3; do
  echo "  --- DS-off repeat ${i}/3 ---"
  python3 benchmark/double_sparsity/bench_decode.py \
      --config branch_ds_off --tp-size 8 \
      --model "${MODEL}" \
      --context-len "${CTX}" --output-len "${OUT_LEN}" \
      --n-requests "${N_REQ}" --concurrency 1 \
      --mem-fraction-static "${MEM_FRAC}" --max-running-requests "${MAX_RUN}" \
      --niah --niah-context-tokens "${CTX}" --niah-n-samples "${NIAH_N}" \
      --output-json "${WORKSPACE}/70b_128k_off_${i}.json"
done

echo "=== Step 3: DS-on (3 repeats, block_t=${BLOCK_T} k_block=${K_BLOCK}) ==="
for i in 1 2 3; do
  echo "  --- DS-on repeat ${i}/3 ---"
  python3 benchmark/double_sparsity/bench_decode.py \
      --config branch_ds_on --tp-size 8 \
      --model "${MODEL}" \
      --calibration "${CALIB}" \
      --context-len "${CTX}" --output-len "${OUT_LEN}" \
      --n-requests "${N_REQ}" --concurrency 1 \
      --block-t "${BLOCK_T}" --k-block "${K_BLOCK}" --token-budget "${TOKEN_BUDGET}" \
      --mem-fraction-static "${MEM_FRAC}" --max-running-requests "${MAX_RUN}" \
      --niah --niah-context-tokens "${CTX}" --niah-n-samples "${NIAH_N}" \
      --output-json "${WORKSPACE}/70b_128k_on_${i}.json"
done

echo "=== Step 4: pick median-decode JSON of each leg and compare ==="
# Median selector: sort by decode_tok_per_s and pick the middle one.
pick_median() {
  python3 - <<PY
import glob, json, sys
files = sorted(glob.glob("${WORKSPACE}/70b_128k_${1}_*.json"))
ranked = sorted(files, key=lambda p: json.load(open(p))["decode_tok_per_s"])
print(ranked[len(ranked)//2])
PY
}

OFF_MED="$(pick_median off)"
ON_MED="$(pick_median on)"
echo "  median DS-off: ${OFF_MED}"
echo "  median DS-on:  ${ON_MED}"

python3 benchmark/double_sparsity/compare.py \
    --main "${OFF_MED}" \
    --branch-off "${OFF_MED}" \
    --branch-on "${ON_MED}"
