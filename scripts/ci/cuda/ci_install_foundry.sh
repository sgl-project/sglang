#!/bin/bash
# CUDA CI install plus the optional dependency Foundry (PyPI foundry-core, import name foundry;
# runner_config 1-gpu-large-foundry), for --cuda-graph-persistence.
#
# Default: install the sglang[foundry] requirement from python/pyproject.toml
# ("foundry-core>=...") with pip, keeping the CI torch (--no-deps).
# Pre-release testing:
#   FOUNDRY_WHEEL    path or URL of a built foundry-core wheel (e.g. from tools/release/build_wheel.sh); installed
#                    as is, with --no-deps (it must match the CI torch). Takes precedence over the variables below.
# or set FOUNDRY_GIT_URL and/or FOUNDRY_GIT_REF to build from a repository instead:
#   FOUNDRY_GIT_URL  repository URL, without "git+" and "@ref"
#                    (default: https://github.com/foundry-org/foundry.git)
#   FOUNDRY_GIT_REF  branch, tag or commit (default: main)
set -euxo pipefail

# Sourced, not run: keeps its venv and uv settings (PIP_CMD, UV_SYSTEM_PYTHON) for the steps below.
# shellcheck source=scripts/ci/cuda/ci_install_dependency.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ci_install_dependency.sh" "$@"

if [ -n "${FOUNDRY_WHEEL:-}" ]; then
    echo "Installing foundry-core from the wheel ${FOUNDRY_WHEEL}"
    $PIP_CMD install --no-deps "${FOUNDRY_WHEEL}" $PIP_INSTALL_SUFFIX
elif [ -n "${FOUNDRY_GIT_URL:-}" ] || [ -n "${FOUNDRY_GIT_REF:-}" ]; then
    # Source build: Foundry's C++ build needs Boost filesystem/json with CMake configs and
    # CMake >= 4.0, and its torch extension must build against the torch installed above
    # (--no-build-isolation).
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
    FOUNDRY_URL="${FOUNDRY_GIT_URL:-https://github.com/foundry-org/foundry.git}"
    FOUNDRY_REF="${FOUNDRY_GIT_REF:-main}"
    echo "Installing foundry-core from ${FOUNDRY_URL}@${FOUNDRY_REF}"
    $PIP_CMD install --no-build-isolation --no-deps "foundry-core @ git+${FOUNDRY_URL}@${FOUNDRY_REF}" $PIP_INSTALL_SUFFIX
else
    # The requirement of the sglang[foundry] extra, read from pyproject.toml so the pin lives in one place.
    FOUNDRY_SPEC=$(grep -Po -m1 '"\Kfoundry-core[^"]*' python/pyproject.toml)
    echo "Installing ${FOUNDRY_SPEC}"
    $PIP_CMD install --no-deps "${FOUNDRY_SPEC}" $PIP_INSTALL_SUFFIX
fi

python3 -c '
from sglang.srt.utils.foundry_adapter import FoundryAdapter
adapter = FoundryAdapter.create(True, mode="save")
assert adapter.enabled
import foundry.ops
print("foundry-core installed:", foundry.ops.__file__)
'
