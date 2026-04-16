#!/bin/bash
# Per-expert INT4 sensitivity sweep for Qwen3-30B-A3B.
#
# Divides 128 experts across GPUs.  Each GPU runs an independent process
# that propagates through all layers and sweeps its expert slice.
# Safetensors mmap shares model pages across processes.
#
# Examples:
#   GPU=4 bash run.sh                   # single GPU, all 128 experts
#   GPUS=0,1,2,3 bash run.sh            # 4 GPUs, 32 experts each
#   GPUS=0,1,2,3,4,5,6,7 bash run.sh   # 8 GPUs, 16 experts each
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
INT4_CKPT="${INT4_CKPT:-/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441}"

# GPU selection: GPUS takes precedence, falls back to single GPU
GPUS="${GPUS:-${GPU:-4}}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-42}"
GROUP_SIZE="${GROUP_SIZE:-128}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:--1}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
NGPUS=${#GPU_ARR[@]}

echo "=== Per-expert sensitivity sweep ==="
echo "  model:      ${MODEL_PATH}"
echo "  int4_ckpt:  ${INT4_CKPT}"
echo "  GPUs:       ${GPUS} (${NGPUS} worker(s))"
echo "  nsamples:   ${NSAMPLES}"
echo "  seqlen:     ${SEQLEN}"
echo "  out_dir:    ${OUT_DIR}"
echo "  layers:     ${START_LAYER}..${END_LAYER}"
echo ""

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
    --model_path "${MODEL_PATH}"
    --int4_checkpoint "${INT4_CKPT}"
    --nsamples "${NSAMPLES}"
    --seqlen "${SEQLEN}"
    --seed "${SEED}"
    --group_size "${GROUP_SIZE}"
    --out_dir "${OUT_DIR}"
    --start_layer "${START_LAYER}"
    --end_layer "${END_LAYER}"
)

# Activate conda env once (works for both single and multi-GPU)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate sglang

if [[ "${NGPUS}" -eq 1 ]]; then
    # Single GPU — sweep all experts in one process
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" \
    python "${SCRIPT_DIR}/sensitivity.py" \
        "${COMMON_ARGS[@]}" --gpu 0
else
    # Multi-GPU — one process per GPU, disjoint expert ranges
    EXPERTS_PER_GPU=$(( NUM_EXPERTS / NGPUS ))
    PIDS=()

    for (( i=0; i<NGPUS; i++ )); do
        E_START=$(( i * EXPERTS_PER_GPU ))
        E_END=$(( (i + 1) * EXPERTS_PER_GPU ))
        GPU_ID="${GPU_ARR[$i]}"

        echo "  rank ${i}: GPU ${GPU_ID}, experts [${E_START}, ${E_END})"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python "${SCRIPT_DIR}/sensitivity.py" \
            "${COMMON_ARGS[@]}" \
            --gpu 0 \
            --expert_start "${E_START}" --expert_end "${E_END}" \
            --rank "${i}" \
            > "${OUT_DIR}/rank${i}.log" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "Waiting for ${NGPUS} workers (logs in ${OUT_DIR}/rank*.log) ..."

    FAILED=0
    for (( i=0; i<NGPUS; i++ )); do
        if ! wait "${PIDS[$i]}"; then
            echo "ERROR: rank ${i} (GPU ${GPU_ARR[$i]}) failed — see ${OUT_DIR}/rank${i}.log"
            FAILED=$((FAILED + 1))
        else
            echo "  rank ${i} done"
        fi
    done

    if [[ "${FAILED}" -gt 0 ]]; then
        echo "FATAL: ${FAILED} worker(s) failed"
        exit 1
    fi

    # Merge partial results
    echo ""
    echo "Merging ${NGPUS} partial results ..."
    python "${SCRIPT_DIR}/sensitivity.py" \
        --merge --out_dir "${OUT_DIR}"
fi

echo ""
echo "Done! Results: ${OUT_DIR}/summary.json"
