#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKTREE="${WORKTREE:-/sgl-workspace/sglang}"
export LTX23_OFFICIAL_FINAL_LATENTS="${LTX23_OFFICIAL_FINAL_LATENTS:-/tmp/ltx23_official_i2v_final.pt}"
export LTX23_OFFICIAL_DECODE_DUMP="${LTX23_OFFICIAL_DECODE_DUMP:-/tmp/ltx23_official_i2v_decode.pt}"
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

python "$WORKTREE/python/sglang/multimodal_gen/notes/ltx23_alignment_20260404/official_i2v_final_decode_dump.py"
