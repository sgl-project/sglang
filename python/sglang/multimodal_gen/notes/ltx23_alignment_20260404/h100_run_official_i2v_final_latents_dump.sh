#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LTX23_PROMPT="${LTX23_PROMPT:-A beautiful sunset over the ocean}"
export LTX23_IMAGE_PATH="${LTX23_IMAGE_PATH:-/tmp/ltx23_i2v_input_sunset.png}"
export LTX23_OFFICIAL_FINAL_LATENTS="${LTX23_OFFICIAL_FINAL_LATENTS:-/tmp/ltx23_official_i2v_final.pt}"
export LTX23_STREAMING_PREFETCH_COUNT="${LTX23_STREAMING_PREFETCH_COUNT:-1}"
export LTX23_OFFICIAL_VENV="${LTX23_OFFICIAL_VENV:-/tmp/ltx23_official_venv}"

if [ -d /tmp/LTX-2 ]; then
  LTX_REPO_ROOT=/tmp/LTX-2
elif [ -d /tmp/LTX-2-official ]; then
  LTX_REPO_ROOT=/tmp/LTX-2-official
else
  echo "LTX repo not found under /tmp" >&2
  exit 1
fi

. "$LTX23_OFFICIAL_VENV/bin/activate"
export PYTHONPATH="$LTX_REPO_ROOT/packages/ltx-core/src:$LTX_REPO_ROOT/packages/ltx-pipelines/src"

python /sgl-workspace/sglang/python/sglang/multimodal_gen/notes/ltx23_alignment_20260404/official_i2v_final_latents_dump.py
