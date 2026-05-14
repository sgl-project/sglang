#!/usr/bin/env bash
# Single-trial 70B TP=8 128K DS-off vs DS-on bench.
# CTX=129792 so server_ctx_len = 131072 = exactly 64*2048
# (keeps DS merge num_blocks*k_block <= 4096 with block_t=2048 k_block=64).
set -euo pipefail

WORKSPACE=/workspace
MODEL=meta-llama/Llama-3.1-70B-Instruct
CTX=129792
OUT_LEN=1024
N_REQ=4
BLOCK_T=2048
K_BLOCK=64
TOKEN_BUDGET=1024
MEM_FRAC=0.85
MAX_RUN=4
NIAH_N=5
CALIB=${WORKSPACE}/calib_llama_3_1_70b_wikitext_s32.json

cd /workspace/sglang

echo "=== Step 1: DS-off baseline (1 run) ==="
python3 benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_off --tp-size 8 \
    --model "${MODEL}" \
    --context-len "${CTX}" --output-len "${OUT_LEN}" \
    --n-requests "${N_REQ}" --concurrency 1 \
    --mem-fraction-static "${MEM_FRAC}" --max-running-requests "${MAX_RUN}" \
    --niah --niah-context-tokens "${CTX}" --niah-n-samples "${NIAH_N}" \
    --output-json "${WORKSPACE}/70b_128k_off_1.json"

echo "=== Step 2: DS-on (1 run, block_t=${BLOCK_T} k_block=${K_BLOCK}) ==="
python3 benchmark/double_sparsity/bench_decode.py \
    --config branch_ds_on --tp-size 8 \
    --model "${MODEL}" \
    --calibration "${CALIB}" \
    --context-len "${CTX}" --output-len "${OUT_LEN}" \
    --n-requests "${N_REQ}" --concurrency 1 \
    --block-t "${BLOCK_T}" --k-block "${K_BLOCK}" --token-budget "${TOKEN_BUDGET}" \
    --mem-fraction-static "${MEM_FRAC}" --max-running-requests "${MAX_RUN}" \
    --niah --niah-context-tokens "${CTX}" --niah-n-samples "${NIAH_N}" \
    --output-json "${WORKSPACE}/70b_128k_on_1.json"

echo "=== Step 3: compare ==="
python3 benchmark/double_sparsity/compare.py \
    --main "${WORKSPACE}/70b_128k_off_1.json" \
    --branch-off "${WORKSPACE}/70b_128k_off_1.json" \
    --branch-on "${WORKSPACE}/70b_128k_on_1.json"
