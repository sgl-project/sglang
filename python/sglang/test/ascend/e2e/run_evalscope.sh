#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PYTHON_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/python
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
        if timeout ${EVALSCOPE_INSTALL_TIMEOUT} ${PIP_FOR_EVALSCOPE} install --retries 3 --timeout 60 "$@" -i "${idx}"; then
            return 0
        fi
        echo "WARN: pip install failed with index ${idx}, trying next mirror..."
    done
    echo "ERROR: pip install failed on all mirror sources."
    return 1
}

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
    pip_install_with_fallback -U pip || echo "WARN: pip self-upgrade failed, continue with existing pip"
    pip_install_with_fallback evalscope "${EVALSCOPE_CONSTRAINTS[@]}"
else
    echo "Install evalscope from local source: ${EVALSCOPE_SOURCE_PATH}"
    pip_install_with_fallback -U pip || echo "WARN: pip self-upgrade failed, continue with existing pip"
    pip_install_with_fallback -e ${EVALSCOPE_SOURCE_PATH} "${EVALSCOPE_CONSTRAINTS[@]}"
fi
echo "===== Install evalscope in virtual env - End ====="
