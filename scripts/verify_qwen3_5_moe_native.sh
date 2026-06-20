#!/bin/bash
# Verify the sglang-native Qwen3.5 MoE model loads weights and forwards
# a few tokens.  Compares the first generated token against mlx_lm
# generate for the same prompt.

set -euo pipefail

SGLANG_PYTHON=${SGLANG_PYTHON:-/Users/toufupi/miniconda3/envs/sglang/bin/python}
SGLANG_ROOT=${SGLANG_ROOT:-/Users/toufupi/project/SGLang/sglang}
MODEL_PATH=${MODEL_PATH:-/Users/toufupi/models/Qwen3.6-35B-A3B-UD-MLX-4bit}
PROMPT=${PROMPT:-"The capital of France is"}
MAX_TOKENS=${MAX_TOKENS:-10}

export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"

echo "=== Step 1: forward pass through sglang-native model ==="
"${SGLANG_PYTHON}" - <<EOF
import mlx.core as mx
from sglang.srt.hardware_backend.mlx.models.qwen3_5_moe import load, TextModelArgs

print("Loading sglang-native Qwen3.5 MoE...")
model = load("${MODEL_PATH}")
print("Loaded.")

prompt_ids = [9707, 13, 8377, 13, 359, 1181, 374, 298]  # "The capital of France is"
inputs = mx.array([prompt_ids], dtype=mx.int32)
out = model(inputs)
mx.eval(out)
top_id = int(mx.argmax(out[0, -1]))
print(f"sglang top-1 token id for prompt: {top_id}")
EOF

echo ""
echo "=== Step 2: mlx_lm baseline (same prompt) ==="
"${SGLANG_PYTHON}" - <<EOF
import mlx.core as mx
from mlx_lm import load as mlx_load, generate

print("Loading mlx_lm Qwen3.5 MoE...")
model, tok = mlx_load("${MODEL_PATH}")
out = generate(model, tok, prompt="${PROMPT}", max_tokens=${MAX_TOKENS}, verbose=False)
print(f"mlx_lm output: {out!r}")
EOF
