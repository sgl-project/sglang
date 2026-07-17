#!/usr/bin/env bash
#SBATCH --job-name=sglang-deepseek
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --account=your_account

# --------------------------------------------------------------------------
# SGLang multi-node serving example for DeepSeek models.
#
# Adapts the multi-node launch pattern from:
#   benchmark/deepseek_v3/README.md
#
# Usage:
#   sbatch examples/slurm/multi-node-deepseek.slurm
#
# This script uses srun to launch one task per node.
# SGLang's built-in multi-GPU support handles intra-node distribution.
# --------------------------------------------------------------------------

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V3}"
PORT="${PORT:-30000}"
NODES="${SLURM_NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HOST="${HOST:-0.0.0.0}"
HUGGINGFACE_CACHE="${HUGGINGFACE_CACHE:-$HOME/.cache/huggingface}"

NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# Set dist-init-addr to the hostname of the first node.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
DIST_INIT_ADDR="${HEAD_NODE}:5000"

echo "=== SGLang Multi-Node DeepSeek Serving ==="
echo "  Model:          ${MODEL}"
echo "  Nodes:          ${NODES}"
echo "  GPUs per node:  ${GPUS_PER_NODE}"
echo "  Total GPUs:     $((NODES * GPUS_PER_NODE))"
echo "  Head node:      ${HEAD_NODE}"
echo "  Dist addr:      ${DIST_INIT_ADDR}"
echo "  Serving on:     ${HOST}:${PORT}"
echo

if command -v apptainer &>/dev/null; then
    CONTAINER_LAUNCHER="apptainer exec --nv --bind ${HUGGINGFACE_CACHE}:/root/.cache/huggingface"
elif command -v singularity &>/dev/null; then
    CONTAINER_LAUNCHER="singularity exec --nv --bind ${HUGGINGFACE_CACHE}:/root/.cache/huggingface"
elif command -v enroot &>/dev/null; then
    CONTAINER_LAUNCHER="enroot start --mount ${HUGGINGFACE_CACHE}:/root/.cache/huggingface"
else
    CONTAINER_LAUNCHER=""
fi

if [[ -n "${CONTAINER_IMAGE:-}" && -n "${CONTAINER_LAUNCHER:-}" ]]; then
    echo "Using container image: ${CONTAINER_IMAGE}"
    RUN_PREFIX="${CONTAINER_LAUNCHER} ${CONTAINER_IMAGE}"
else
    echo "Running directly (no container). Install sglang first if needed."
    RUN_PREFIX=""
fi

# SGLang uses torch.distributed internally, so each task only needs the
# node rank from Slurm.  srun maps each Slurm task to a single node.
srun --ntasks="${NODES}" --nodes="${NODES}" \
    --gpus-per-task="${GPUS_PER_NODE}" \
    bash -c "
        set -euo pipefail

        echo \"Node: \$(hostname), SLURM_PROCID=${SLURM_PROCID}\"

        export NCCL_DEBUG=${NCCL_DEBUG}

        ${RUN_PREFIX} python3 -m sglang.launch_server \
            --model-path ${MODEL} \
            --tp $((GPUS_PER_NODE * NODES)) \
            --dist-init-addr ${DIST_INIT_ADDR} \
            --nnodes ${NODES} \
            --node-rank ${SLURM_PROCID} \
            --host ${HOST} \
            --port ${PORT} \
            --trust-remote-code
    "
