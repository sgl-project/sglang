#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PIP_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/pip
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope
pip_mirror_source="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

if [ -d "${PYTHON_ENV_FOR_EVALSCOPE}" ]; then
    echo "Virtual env ${PYTHON_ENV_FOR_EVALSCOPE} already exists, skip installation."
    exit 0
fi

echo "===== Install evalscope in virtual env - Begin ====="
python -m venv ${PYTHON_ENV_FOR_EVALSCOPE}

if [ ! -d "${EVALSCOPE_SOURCE_PATH}" ]; then
    echo "The evalscope source does not exist: ${EVALSCOPE_SOURCE_PATH}."
    echo "Install evalscope online."
    # The bundled pip is sufficient; make self-upgrade non-fatal because some
    # pypi mirrors intermittently return 403 for the latest pip wheel.
    ${PIP_FOR_EVALSCOPE} install -U pip -i ${pip_mirror_source} \
        || echo "WARN: pip self-upgrade failed, continue with existing pip"
    ${PIP_FOR_EVALSCOPE} install --retries 5 --timeout 60 evalscope -i ${pip_mirror_source}
else
    echo "Install evalscope from local source: ${EVALSCOPE_SOURCE_PATH}"
    ${PIP_FOR_EVALSCOPE} install -U pip -i ${pip_mirror_source} \
        || echo "WARN: pip self-upgrade failed, continue with existing pip"
    ${PIP_FOR_EVALSCOPE} install --retries 5 --timeout 60 -e ${EVALSCOPE_SOURCE_PATH} -i ${pip_mirror_source}
fi
echo "===== Install evalscope in virtual env - End ====="
