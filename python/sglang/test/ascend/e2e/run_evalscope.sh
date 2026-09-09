#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PIP_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/pip
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope
# Try mirrors in order: some runners get 403-blocked by a specific mirror,
# so fall back to alternates and finally the official PyPI.
pip_mirror_sources=(
    "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.org/simple"
)

pip_install_with_fallback() {
    for idx in "${pip_mirror_sources[@]}"; do
        echo "Trying pip index: ${idx}"
        if ${PIP_FOR_EVALSCOPE} install --retries 3 --timeout 60 "$@" -i "${idx}"; then
            return 0
        fi
        echo "WARN: pip install failed with index ${idx}, trying next mirror..."
    done
    return 1
}

if [ -d "${PYTHON_ENV_FOR_EVALSCOPE}" ]; then
    echo "Virtual env ${PYTHON_ENV_FOR_EVALSCOPE} already exists, skip installation."
    exit 0
fi

echo "===== Install evalscope in virtual env - Begin ====="
python -m venv ${PYTHON_ENV_FOR_EVALSCOPE}

if [ ! -d "${EVALSCOPE_SOURCE_PATH}" ]; then
    echo "The evalscope source does not exist: ${EVALSCOPE_SOURCE_PATH}."
    echo "Install evalscope online."
    pip_install_with_fallback -U pip || echo "WARN: pip self-upgrade failed, continue with existing pip"
    pip_install_with_fallback evalscope
else
    echo "Install evalscope from local source: ${EVALSCOPE_SOURCE_PATH}"
    pip_install_with_fallback -U pip || echo "WARN: pip self-upgrade failed, continue with existing pip"
    pip_install_with_fallback -e ${EVALSCOPE_SOURCE_PATH}
fi
echo "===== Install evalscope in virtual env - End ====="
