#!/bin/bash
# CUDA CI install plus MSCCL++ built from source, for runner configs that run
# the MSCCL++ communicator tests. Referenced from scripts/ci/runner_configs.yml.
set -euo pipefail

INSTALL_MSCCLPP=1 exec bash "$(dirname "${BASH_SOURCE[0]}")/ci_install_dependency.sh" "$@"
