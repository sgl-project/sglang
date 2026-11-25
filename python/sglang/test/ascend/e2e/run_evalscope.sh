#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PYTHON_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/python
PIP_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/pip
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope
pip_mirror_source="${EVALSCOPE_PIP_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"

# Agent-style datasets (deep_swe) are driven by datacurve-pier, which sits in
# the same isolated venv as evalscope so its openai 3.x cannot disturb the
# system environment (SGLang pins openai==2.6.1).
EVALSCOPE_WITH_PIER="${EVALSCOPE_WITH_PIER:-0}"

# Bound key deps so the resolver cannot fall back to ancient versions.
EVALSCOPE_CONSTRAINTS=(
    "aiohttp>=3.11,<4"
    "httpx>=0.28,<1"
)

# Fail fast when the pip install hangs instead of timing out the whole job.
EVALSCOPE_INSTALL_TIMEOUT=1200

# The venv is usable only when it provides everything this caller needs.
evalscope_env_ready() {
    [ -x "${PYTHON_FOR_EVALSCOPE}" ] || return 1
    ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1 || return 1
    if [ "${EVALSCOPE_WITH_PIER}" = "1" ]; then
        ${PYTHON_FOR_EVALSCOPE} -c "import pier" >/dev/null 2>&1 || return 1
    fi
    return 0
}

install_evalscope() {
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
}

install_pier() {
    if [ "${EVALSCOPE_WITH_PIER}" != "1" ]; then
        return 0
    fi
    if ${PYTHON_FOR_EVALSCOPE} -c "import pier" >/dev/null 2>&1; then
        return 0
    fi
    echo "===== Install datacurve-pier in virtual env - Begin ====="
    timeout ${EVALSCOPE_INSTALL_TIMEOUT} ${PIP_FOR_EVALSCOPE} install "datacurve-pier>=0.3.0" -i ${pip_mirror_source} || {
        echo "ERROR: datacurve-pier install timed out after ${EVALSCOPE_INSTALL_TIMEOUT}s."
        exit 1
    }
    ${PYTHON_FOR_EVALSCOPE} -c "import pier" >/dev/null 2>&1 || {
        echo "ERROR: datacurve-pier installed but 'import pier' still fails."
        exit 1
    }
    echo "===== Install datacurve-pier in virtual env - End ====="
}

log_env_versions() {
    ${PYTHON_FOR_EVALSCOPE} -c "import sys, evalscope; print('evalscope', getattr(evalscope, '__version__', 'unknown'), 'python', sys.version.split()[0])" 2>/dev/null || true
    if [ "${EVALSCOPE_WITH_PIER}" = "1" ]; then
        ${PYTHON_FOR_EVALSCOPE} -c "import pier; print('pier', getattr(pier, '__version__', 'unknown'))" 2>/dev/null || true
    fi
}

# Highest priority: reuse a system-wide evalscope (e.g. pre-installed in the
# image); create ${PYTHON_FOR_EVALSCOPE} with --system-site-packages when needed.
if python -c "import evalscope" >/dev/null 2>&1; then
    echo "evalscope found in the system python, reuse it instead of installing."
    if [ ! -x "${PYTHON_FOR_EVALSCOPE}" ]; then
        python -m venv --system-site-packages ${PYTHON_ENV_FOR_EVALSCOPE}
    fi
    if evalscope_env_ready; then
        log_env_versions
        exit 0
    fi
    install_pier
    if evalscope_env_ready; then
        log_env_versions
        exit 0
    fi
    echo "The env still cannot provide evalscope, install it instead."
fi

# Otherwise reuse the virtual env when it already provides what is needed.
if evalscope_env_ready; then
    echo "evalscope already exists in ${PYTHON_ENV_FOR_EVALSCOPE}, skip installation."
    log_env_versions
    exit 0
fi

[ -x "${PYTHON_FOR_EVALSCOPE}" ] || python -m venv ${PYTHON_ENV_FOR_EVALSCOPE}

if ! ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1; then
    install_evalscope
fi

install_pier
log_env_versions
