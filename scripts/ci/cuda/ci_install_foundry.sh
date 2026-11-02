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
# Foundry publishes one package index per torch/CUDA pair (whl/<cuda>/torch<A.B>/); the installed
# torch selects it. With the torch the PyPI wheel was built for this resolves to the PyPI wheel.
FOUNDRY_INDEX=$(python3 - <<'PY'
import torch
mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
print(f"https://foundry-org.github.io/foundry/whl/cu{torch.version.cuda.replace('.', '')}/torch{mm}/")
PY
)
echo "Installing ${FOUNDRY_SPEC} (extra index ${FOUNDRY_INDEX})"
$PIP_CMD install --no-deps "${FOUNDRY_SPEC}" --extra-index-url "${FOUNDRY_INDEX}" $PIP_INSTALL_SUFFIX

python3 -c '
from sglang.srt.utils.foundry_adapter import FoundryAdapter
adapter = FoundryAdapter.create(True, mode="save")
assert adapter.enabled
import foundry.ops
print("foundry-core installed:", foundry.ops.__file__)
'
