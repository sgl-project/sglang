#!/usr/bin/env bash
# Serve MiniMax-H3 on four V100s. fl2va is text and keyframes; ref2va is
# reference image, video, and audio. One process loads one partition.
# Setup and use guideline is in docs/v100/MiniMax-H3.md.
#
# Usage (from the repo root):
#   bash scripts/serve_minimax_h3_v100.sh            # fl2va: t2va and keyframes
#   bash scripts/serve_minimax_h3_v100.sh ref2va     # reference image/video/audio
#
# Env overrides:
#   H3_MODEL=/path/to/MiniMax-H3          (default: $HOME/models/MiniMax-H3)
#   H3_GPUS=0,1,2,3
#   H3_PORT=30010
#   SGLANG_V100_VENV=/path/to/venv
set -euo pipefail

VARIANT="${1:-fl2va}"
case "$VARIANT" in
  fl2va) PARTITION=FL2VA ;;
  ref2va) PARTITION=Ref2VA ;;
  *) echo "variant must be fl2va or ref2va" >&2; exit 1 ;;
esac

VENV="${SGLANG_V100_VENV:-$HOME/sglang-v100-venv}"
[[ -x "$VENV/bin/python" ]] || VENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-$VENV}}"
[[ -x "$VENV/bin/sglang" ]] || { echo "no sglang in $VENV; set SGLANG_V100_VENV" >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${H3_MODEL:-$HOME/models/MiniMax-H3}"
[[ -d "$MODEL/$PARTITION" ]] || { echo "no $PARTITION partition at $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${H3_GPUS:-0,1,2,3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
for _v in 14 13 12; do
  if [[ -x "/usr/bin/g++-$_v" ]] && ls /usr/libexec/gcc/*/$_v/cc1plus >/dev/null 2>&1; then
    export CC="/usr/bin/gcc-$_v" CXX="/usr/bin/g++-$_v" CUDAHOSTCXX="/usr/bin/g++-$_v"
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-$_v"
    break
  fi
done
export TORCH_CUDA_ARCH_LIST=7.0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# NCCL_P2P_LEVEL stays unset: NCCL picks NVLink where it exists and PCIe P2P
# behind a PCIe switch; forcing NVL routes PCIe-only cards through host memory.
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
exec "$VENV/bin/sglang" serve \
  --model-path "$MODEL" \
  --model-variant "$VARIANT" \
  --num-gpus 4 \
  --tp-size 4 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --performance-mode speed \
  --quantization v100_w4a16_awq \
  --attention-backend tilelang_fa_v100 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --enable-torch-compile false \
  --warmup-mode off \
  --host "${H3_HOST:-0.0.0.0}" \
  --port "${H3_PORT:-30010}"
