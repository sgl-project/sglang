#!/bin/bash
# CUDA CI install plus the optional dependency Foundry (PyPI foundry-core, import name foundry;
# runner_config 1-gpu-large-foundry), for --cuda-graph-persistence.
#
# Installs the sglang[foundry] requirement from python/pyproject.toml ("foundry-core>=...")
# with pip, keeping the CI torch (--no-deps: the wheel pins the torch it was built against,
# which is the torch this CI installs).
set -euxo pipefail

# Sourced, not run: keeps its venv and uv settings (PIP_CMD, UV_SYSTEM_PYTHON) for the steps below.
# shellcheck source=scripts/ci/cuda/ci_install_dependency.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ci_install_dependency.sh" "$@"

# The requirement of the sglang[foundry] extra, read from pyproject.toml so the pin lives in one place.
FOUNDRY_SPEC=$(grep -Po -m1 '"\Kfoundry-core[^"]*' python/pyproject.toml)
echo "Installing ${FOUNDRY_SPEC}"
$PIP_CMD install --no-deps "${FOUNDRY_SPEC}" $PIP_INSTALL_SUFFIX

python3 -c '
from sglang.srt.utils.foundry_adapter import FoundryAdapter
adapter = FoundryAdapter.create(True, mode="save")
assert adapter.enabled
import foundry.ops
print("foundry-core installed:", foundry.ops.__file__)
'
