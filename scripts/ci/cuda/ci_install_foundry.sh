#!/bin/bash
# CUDA CI install plus the optional dependency Foundry (runner_config 1-gpu-large-foundry),
# for --cuda-graph-persistence.
set -euxo pipefail

# Sourced, not run: keeps its venv and uv settings (PIP_CMD, UV_SYSTEM_PYTHON) for the steps below.
# shellcheck source=scripts/ci/cuda/ci_install_dependency.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ci_install_dependency.sh" "$@"

# Foundry's C++ build needs Boost filesystem/json with CMake configs and CMake >= 4.0.
FOUNDRY_APT_PACKAGES=(libboost-filesystem-dev libboost-json-dev)
MISSING_FOUNDRY_APT_PACKAGES=()
for pkg in "${FOUNDRY_APT_PACKAGES[@]}"; do
    is_apt_package_installed "$pkg" || MISSING_FOUNDRY_APT_PACKAGES+=("$pkg")
done
if [ ${#MISSING_FOUNDRY_APT_PACKAGES[@]} -gt 0 ]; then
    apt-get update || true
    apt-get install -y --no-install-recommends "${MISSING_FOUNDRY_APT_PACKAGES[@]}"
fi
$PIP_CMD install "cmake>=4.0" ninja "setuptools>=80" wheel $PIP_INSTALL_SUFFIX

# The requirement of the sglang[foundry] extra, read from pyproject.toml so the pin lives in one
# place. It is installed on its own rather than through `python[...,foundry]` because foundry.ops
# is a torch extension: it must build against the torch installed above (--no-build-isolation),
# while sglang itself builds in isolation. FOUNDRY_GIT_REF overrides the ref for a test run.
FOUNDRY_SPEC=$(grep -Po -m1 '"\Kfoundry @ [^"]+' python/pyproject.toml)
if [ -n "${FOUNDRY_GIT_REF:-}" ]; then
    FOUNDRY_SPEC="${FOUNDRY_SPEC%@*}@${FOUNDRY_GIT_REF}"
fi
$PIP_CMD install --no-build-isolation --no-deps "${FOUNDRY_SPEC}" $PIP_INSTALL_SUFFIX

python3 -c '
from sglang.srt.utils.foundry_adapter import FoundryAdapter
adapter = FoundryAdapter.create(True, mode="save")
assert adapter.enabled
import foundry.ops
print("foundry installed:", foundry.ops.__file__)
'
