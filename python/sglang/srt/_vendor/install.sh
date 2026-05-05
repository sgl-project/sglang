#!/usr/bin/env bash
# Install the vendored DeepGEMM wheel that this branch needs for the
# FP4 acts + MXF4 kind + mega_moe_pre_dispatch path. Run once after
# `pip install -e python/`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL="$(ls "$HERE"/deep_gemm-*.whl | head -1)"
if [[ -z "$WHEEL" ]]; then
    echo "No deep_gemm wheel found in $HERE" >&2
    exit 1
fi

echo "Installing $WHEEL"
pip install "$WHEEL" --force-reinstall --no-deps

python - <<'PY'
import deep_gemm
need = ("mega_moe_pre_dispatch", "fp8_fp4_mega_moe",
        "transform_weights_for_mega_moe", "get_symm_buffer_for_mega_moe")
missing = [s for s in need if not hasattr(deep_gemm, s)]
if missing:
    raise SystemExit(f"deep_gemm install incomplete; missing: {missing}")
print("deep_gemm install OK; all required symbols present")
PY
