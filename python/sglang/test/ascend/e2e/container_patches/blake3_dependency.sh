#!/bin/bash
# PATCH_SET: blake3_dependency
# PATCH_VERSION: 1.0
# TARGET_IMAGE: cann9.1.0-a3-B010
# REMOVE_AFTER: 2026-12-31
# SCOPE: pip
#
# The image predates the `blake3>=1.0.0` dependency added to
# python/pyproject_npu.toml. With INSTALL_SGLANG_FROM_SOURCE the pod only
# prepends the checkout to PYTHONPATH, so sglang is imported from source while
# the image's site-packages is left untouched. The server then dies at import
# time on `sglang.srt.multimodal.cache.digest` -> `import blake3`.
#
# Remove this patch set once a released image ships blake3 (docker/npu.Dockerfile)
# and the NPU workflows use it.

set -euo pipefail

if python -c "import blake3" >/dev/null 2>&1; then
    echo "blake3 already installed, skip pip."
    exit 0
fi

echo "Install blake3"
python -m pip install --no-cache-dir "blake3>=1.0.0"

python - <<'PY'
import blake3

print(f"blake3 {blake3.__version__} installed")
PY
