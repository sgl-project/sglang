#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORKTREE="${WORKTREE:-/sgl-workspace/sglang}"
OUT_DIR="${LTX23_SGLANG_OUT_DIR:-/tmp/ltx23_sglang_i2v_injected}"
PROMPT="${LTX23_PROMPT:-A beautiful sunset over the ocean}"
IMAGE_PATH="${LTX23_IMAGE_PATH:-/tmp/ltx23_i2v_input_sunset.png}"
INJECT_PATH="${LTX23_INJECT_LATENTS_PATH:-/tmp/ltx23_official_i2v_final.pt}"
export PYTHONPATH="$WORKTREE/python:${PYTHONPATH:-}"
export SGLANG_DIFFUSION_LTX2_POST_DENOISE_INJECT_PATH="$INJECT_PATH"

cd "$WORKTREE"
git rev-parse --short HEAD
mkdir -p "$OUT_DIR"

python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
  --model-path Lightricks/LTX-2.3 \
  --prompt "$PROMPT" \
  --image-path "$IMAGE_PATH" \
  --output-path "$OUT_DIR" \
  --output-file-name sglang_i2v_injected.mp4 \
  --save-output
