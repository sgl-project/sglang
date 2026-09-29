#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PYTHON_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/python
PIP_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/pip
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope
pip_mirror_source="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Bound key deps so the resolver cannot fall back to ancient versions.
EVALSCOPE_CONSTRAINTS=(
    "aiohttp>=3.11,<4"
    "httpx>=0.28,<1"
)

# Fail fast when the pip install hangs instead of timing out the whole job.
EVALSCOPE_INSTALL_TIMEOUT=1200

# Highest priority: reuse a system-wide evalscope (e.g. pre-installed in the
# image); create ${PYTHON_FOR_EVALSCOPE} with --system-site-packages when needed.
if python -c "import evalscope" >/dev/null 2>&1; then
    echo "evalscope found in the system python, reuse it instead of installing."
    if ! ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1; then
        python -m venv --system-site-packages ${PYTHON_ENV_FOR_EVALSCOPE}
    fi
    # Only skip the install when the env really imports evalscope.
    if ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1; then
        exit 0
    fi
    echo "The env cannot provide evalscope, install it instead."
fi

# Otherwise reuse the virtual env when it already imports evalscope.
if [ -x "${PYTHON_FOR_EVALSCOPE}" ] && ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1; then
    echo "evalscope already exists in ${PYTHON_ENV_FOR_EVALSCOPE}, skip installation."
    exit 0
fi

python -m venv ${PYTHON_ENV_FOR_EVALSCOPE}

echo "===== Install evalscope in virtual env - Begin ====="

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
