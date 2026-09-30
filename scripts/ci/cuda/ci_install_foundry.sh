#!/bin/bash
# CUDA CI install plus the external Foundry plugin (runner_config 1-gpu-large-foundry).
# FOUNDRY_GIT_REF selects the branch, tag or commit of https://github.com/foundry-org/foundry.
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

# foundry.ops is a torch extension: build it against the torch installed above and keep that torch.
$PIP_CMD install --no-build-isolation --no-deps \
    "git+https://github.com/foundry-org/foundry.git@${FOUNDRY_GIT_REF:-sglang-registry}" $PIP_INSTALL_SUFFIX

python3 -c '
from importlib.metadata import entry_points
assert any(e.name == "foundry" for e in entry_points(group="sglang.srt.plugins")), "foundry entry point missing"
import foundry.ops
print("foundry plugin installed:", foundry.ops.__file__)
'
