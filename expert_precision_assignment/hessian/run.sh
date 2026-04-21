#!/bin/bash
# Per-expert Hessian sensitivity via DP=N torchrun.
#
# Every rank holds a full BF16 model replica on its own GPU.
# Calibration samples split round-robin across ranks; per-expert
# scalars AllReduce'd at the end.
#
# Examples:
#   bash run.sh                                 # default: DP=8, N=8 × 512 (probe)
#   NSAMPLES=128 SEQLEN=512 bash run.sh         # statistical run
#   GPUS=0,1,2,3 NSAMPLES=16 bash run.sh        # DP=4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
INT4_CKPT="${INT4_CKPT:-/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NSAMPLES="${NSAMPLES:-8}"
SEQLEN="${SEQLEN:-512}"
SEED="${SEED:-42}"
GROUP_SIZE="${GROUP_SIZE:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-2}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
MASTER_PORT="${MASTER_PORT:-29503}"

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NGPUS=${#GPU_ARR[@]}

echo "=== Hessian per-expert sensitivity ==="
echo "  model:       ${MODEL_PATH}"
echo "  int4_ckpt:   ${INT4_CKPT}"
echo "  GPUs:        ${GPUS}  (DP=${NGPUS})"
echo "  nsamples:    ${NSAMPLES}  (${NSAMPLES} / ${NGPUS} per rank)"
echo "  seqlen:      ${SEQLEN}"
echo "  chunk_size:  ${CHUNK_SIZE} layers"
echo "  out_dir:     ${OUT_DIR}"
echo ""

mkdir -p "${OUT_DIR}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate sglang

CUDA_VISIBLE_DEVICES="${GPUS}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --standalone \
    --nproc-per-node="${NGPUS}" \
    --master-port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/hessian_score.py" \
    --model_path "${MODEL_PATH}" \
    --int4_checkpoint "${INT4_CKPT}" \
    --nsamples "${NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --group_size "${GROUP_SIZE}" \
    --chunk_size "${CHUNK_SIZE}" \
    --out_dir "${OUT_DIR}"

echo ""
echo "Done! Results: ${OUT_DIR}/hessian_scores.json"
