#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PIP_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/pip
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope
pip_mirror_source="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Bounds on key dependencies to prevent the pip resolver from degrading to ancient, incompatible versions (e.g. aiohttp 1.0.5).
EVALSCOPE_CONSTRAINTS=(
    "aiohttp>=3.11,<4"
    "httpx>=0.28,<1"
)

# Upper bound (seconds) for the pip install step. If any other component triggers
# resolver backtracking and the install hangs, fail fast instead of timing out the
# whole job.
EVALSCOPE_INSTALL_TIMEOUT=1800

if [ -d "${PYTHON_ENV_FOR_EVALSCOPE}" ]; then
    echo "Virtual env ${PYTHON_ENV_FOR_EVALSCOPE} already exists, skip installation."
    exit 0
fi

echo "===== Install evalscope in virtual env - Begin ====="
python -m venv ${PYTHON_ENV_FOR_EVALSCOPE}

if [ ! -d "${EVALSCOPE_SOURCE_PATH}" ]; then
    echo "The evalscope source does not exist: ${EVALSCOPE_SOURCE_PATH}."
    echo "Install evalscope online."
    ${PIP_FOR_EVALSCOPE} install -U pip -i ${pip_mirror_source}
    timeout ${EVALSCOPE_INSTALL_TIMEOUT} ${PIP_FOR_EVALSCOPE} install evalscope "${EVALSCOPE_CONSTRAINTS[@]}" -i ${pip_mirror_source} || {
        echo "ERROR: evalscope install timed out after ${EVALSCOPE_INSTALL_TIMEOUT}s."
        exit 1
    }
else
    echo "Install evalscope from local source: ${EVALSCOPE_SOURCE_PATH}"
    ${PIP_FOR_EVALSCOPE} install -U pip -i ${pip_mirror_source}
    timeout ${EVALSCOPE_INSTALL_TIMEOUT} ${PIP_FOR_EVALSCOPE} install -e ${EVALSCOPE_SOURCE_PATH} "${EVALSCOPE_CONSTRAINTS[@]}" -i ${pip_mirror_source} || {
        echo "ERROR: evalscope install timed out after ${EVALSCOPE_INSTALL_TIMEOUT}s."
        exit 1
    }
fi
echo "===== Install evalscope in virtual env - End ====="