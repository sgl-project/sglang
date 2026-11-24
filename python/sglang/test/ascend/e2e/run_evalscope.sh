#!/bin/bash

set -e

# Reuse the system python (e.g. pre-installed in the image); no venv is created.
PYTHON_FOR_EVALSCOPE="${PYTHON_FOR_EVALSCOPE:-python}"
PIP_FOR_EVALSCOPE="${PYTHON_FOR_EVALSCOPE} -m pip"
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope
pip_mirror_source="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Bound key deps so the resolver cannot fall back to ancient versions.
EVALSCOPE_CONSTRAINTS=(
    "aiohttp>=3.11,<4"
    "httpx>=0.28,<1"
)

# Fail fast when the pip install hangs instead of timing out the whole job.
EVALSCOPE_INSTALL_TIMEOUT=1200

# DeepSWE (evalscope's deep_swe) runs each task through Pier, which evalscope
# ships as the datacurve-pier package (import name pier, Python>=3.12).
PIER_REQUIREMENT="datacurve-pier>=0.3.0"

if ! ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1; then
    echo "===== Install evalscope in the system python - Begin ====="

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
    echo "===== Install evalscope in the system python - End ====="
fi

# Install Pier on demand: images that already ship it never touch pip.
if ! ${PYTHON_FOR_EVALSCOPE} -c "import pier" >/dev/null 2>&1; then
    echo "===== Install datacurve-pier in the system python - Begin ====="
    timeout ${EVALSCOPE_INSTALL_TIMEOUT} ${PIP_FOR_EVALSCOPE} install "${PIER_REQUIREMENT}" -i ${pip_mirror_source} || {
        echo "ERROR: datacurve-pier install timed out after ${EVALSCOPE_INSTALL_TIMEOUT}s."
        exit 1
    }
    if ! ${PYTHON_FOR_EVALSCOPE} -c "import pier" >/dev/null 2>&1; then
        echo "ERROR: datacurve-pier is installed but 'pier' is still not importable (Python>=3.12 required)."
        exit 1
    fi
    echo "===== Install datacurve-pier in the system python - End ====="
fi
