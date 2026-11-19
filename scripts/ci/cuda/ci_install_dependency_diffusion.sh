#!/bin/bash
# CUDA CI install for the diffusion stages (kind_installs in
# scripts/ci/runner_configs.yml). An install names a script and takes no
# arguments, so this runs ci_install_dependency.sh in its diffusion mode.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/ci_install_dependency.sh" diffusion
