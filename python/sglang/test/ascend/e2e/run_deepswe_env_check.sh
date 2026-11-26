#!/bin/bash
# Lightweight DeepSWE eval-environment check.
#
# Validates, on the current (amd64) container and without any model service or
# k8s pods, that the eval venv can be built and that evalscope + datacurve-pier
# import. This exercises the same run_evalscope.sh path used inside the eval
# pod (Miniconda Python >=3.12 provisioning + pip installs), so environment
# problems surface in minutes instead of after a full multi-node deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${DEEPSWE_ENV_CHECK_DIR:-/tmp/deepswe_env_check}"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

export EVALSCOPE_WITH_PIER=1
export EVALSCOPE_PIP_MIRROR="${EVALSCOPE_PIP_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"

echo "===== DeepSWE eval env check (no model service) ====="
echo "host arch=$(uname -m) python=$(python3 -V 2>&1 || true)"

# run_evalscope.sh creates test_env_evalscope in the current directory.
( cd "${WORK_DIR}" && bash "${SCRIPT_DIR}/run_evalscope.sh" )

VENV_PYTHON="${WORK_DIR}/test_env_evalscope/bin/python"
if [ ! -x "${VENV_PYTHON}" ]; then
    echo "ERROR: eval venv was not created at ${VENV_PYTHON}" >&2
    exit 1
fi

"${VENV_PYTHON}" - <<'PY'
import sys
import evalscope
import pier

print(
    "EVALSCOPE_ENV_CHECK_OK",
    "evalscope", getattr(evalscope, "__version__", "unknown"),
    "pier", getattr(pier, "__version__", "unknown"),
    "python", sys.version.split()[0],
)
PY
