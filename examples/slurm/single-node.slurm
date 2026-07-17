#!/usr/bin/env bash
#SBATCH --job-name=sglang-serve
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=your_account

# --------------------------------------------------------------------------
# SGLang single-node serving example.
#
# Usage:
#   sbatch examples/slurm/single-node.slurm
#
# Customize the model, port, and GPU count via environment variables:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct GPUS=2 sbatch examples/slurm/single-node.slurm
# --------------------------------------------------------------------------

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-30000}"
GPUS="${GPUS:-1}"
HOST="${HOST:-0.0.0.0}"
HUGGINGFACE_CACHE="${HUGGINGFACE_CACHE:-$HOME/.cache/huggingface}"

echo "Starting SGLang server..."
echo "  Model: ${MODEL}"
echo "  GPUs:  ${GPUS}"
echo "  Host:  ${HOST}:${PORT}"
echo

if command -v apptainer &>/dev/null; then
    CONTAINER_EXEC="apptainer exec --nv --bind ${HUGGINGFACE_CACHE}:/root/.cache/huggingface"
elif command -v singularity &>/dev/null; then
    CONTAINER_EXEC="singularity exec --nv --bind ${HUGGINGFACE_CACHE}:/root/.cache/huggingface"
elif command -v enroot &>/dev/null; then
    CONTAINER_EXEC="enroot start --mount ${HUGGINGFACE_CACHE}:/root/.cache/huggingface"
fi

if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    echo "Using container image: ${CONTAINER_IMAGE}"
    CONTAINER_CMD="${CONTAINER_EXEC} ${CONTAINER_IMAGE}"
else
    echo "Running directly via Python (no container detected)."
    if ! python3 -c "import sglang" 2>/dev/null; then
        pip install sglang
    fi
    CONTAINER_CMD=""
fi

python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${GPUS}" \
    --trust-remote-code
