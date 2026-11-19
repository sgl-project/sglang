#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_FOR_SGLANG:-python}"

# The Nightly hook runs before it configures PYTHONPATH. Match its later choice
# of checkout versus image code, without importing torch or initializing NPU.
if [ "$#" -gt 0 ]; then
    package_root="$1"
elif [ "${SGLANG_USE_IMAGE:-}" = "true" ] || [ "${SGLANG_USE_IMAGE:-}" = "True" ]; then
    package_root="$("${python_bin}" -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("sglang").origin).parent)')"
else
    package_root="/root/sglang/python/sglang"
fi

"${python_bin}" "${patch_dir}/install.py" --package-root "${package_root}"
# Nightly applies this archive before sourcing the SDK. This process only reads
# metadata; actual loaded-library identity is captured again on the failure.
"${python_bin}" "${package_root}/srt/mem_cache/pool_host/npu_pin_diagnostics.py" --environment
