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

# datacurve-pier requires Python >= 3.12. The base images we can pull ship an
# older interpreter, so provision a matching one when needed.
EVALSCOPE_MIN_PYTHON="${EVALSCOPE_MIN_PYTHON:-3.12}"
# Miniconda installer / conda channel (Tsinghua mirror; the pip mirror above is
# already served from there).
CONDA_INSTALLER_URL="${CONDA_INSTALLER_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-$(uname -m).sh}"
CONDA_CHANNEL="${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
CONDA_PREFIX="${CONDA_PREFIX:-/opt/py312}"
CONDA_HOME="${CONDA_HOME:-/opt/miniconda3}"

# Bound key deps so the resolver cannot fall back to ancient versions.
EVALSCOPE_CONSTRAINTS=(
    "aiohttp>=3.11,<4"
    "httpx>=0.28,<1"
)

# Fail fast when the pip install hangs instead of timing out the whole job.
EVALSCOPE_INSTALL_TIMEOUT=1200

py_ge_min() {
    "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= tuple(int(x) for x in '${EVALSCOPE_MIN_PYTHON}'.split('.')) else 1)" >/dev/null 2>&1
}

find_python_ge_min() {
    # Prefer a >=min interpreter that already provides evalscope so the venv
    # can reuse it; otherwise fall back to the first >=min interpreter.
    local first=""
    for candidate in python3.13 python3.12 python3 python; do
        command -v "${candidate}" >/dev/null 2>&1 || continue
        py_ge_min "${candidate}" || continue
        if [ -z "${first}" ]; then
            first="${candidate}"
        fi
        if "${candidate}" -c "import evalscope" >/dev/null 2>&1; then
            echo "${candidate}"
            return 0
        fi
    done
    if [ -n "${first}" ]; then
        echo "${first}"
        return 0
    fi
    return 1
}

install_python_via_conda() {
    if [ -x "${CONDA_PREFIX}/bin/python" ] && py_ge_min "${CONDA_PREFIX}/bin/python"; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
    fi
    local installer=/tmp/miniconda.sh
    # Diagnostics must go to stderr: this function's stdout is captured as the
    # interpreter path by the caller.
    echo "No Python >= ${EVALSCOPE_MIN_PYTHON} found; installing Miniconda from ${CONDA_INSTALLER_URL}" >&2
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --connect-timeout 15 -o "${installer}" "${CONDA_INSTALLER_URL}" >&2 || return 1
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${installer}" "${CONDA_INSTALLER_URL}" >&2 || return 1
    else
        echo "Neither curl nor wget is available to download Miniconda" >&2
        return 1
    fi
    bash "${installer}" -b -p "${CONDA_HOME}" >/dev/null 2>&1 || return 1
    "${CONDA_HOME}/bin/conda" create -y --override-channels -c "${CONDA_CHANNEL}" \
        -p "${CONDA_PREFIX}" "python=${EVALSCOPE_MIN_PYTHON}" >/dev/null 2>&1 || return 1
    if [ -x "${CONDA_PREFIX}/bin/python" ] && py_ge_min "${CONDA_PREFIX}/bin/python"; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
    fi
    return 1
}

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

# Pick the interpreter used to create the venv. Only the pier path needs the
# newer interpreter; other accuracy callers keep using the system python.
BASE_PYTHON="python"
if [ "${EVALSCOPE_WITH_PIER}" = "1" ]; then
    if BASE_PYTHON="$(find_python_ge_min)"; then
        echo "Using ${BASE_PYTHON} (>= ${EVALSCOPE_MIN_PYTHON})"
    elif BASE_PYTHON="$(install_python_via_conda)"; then
        echo "Using conda python ${BASE_PYTHON} (>= ${EVALSCOPE_MIN_PYTHON})"
    else
        echo "ERROR: datacurve-pier requires Python >= ${EVALSCOPE_MIN_PYTHON} and none could be provisioned." >&2
        exit 1
    fi
    if ! printf '%s' "${BASE_PYTHON}" | grep -Eq '^[^[:space:]]+$'; then
        echo "ERROR: failed to resolve a single Python interpreter (got '${BASE_PYTHON}')" >&2
        exit 1
    fi
fi

# Reuse evalscope already provided by the chosen interpreter (e.g. baked into
# the sglang image) through a --system-site-packages venv, and only install the
# extra pier package. This avoids reinstalling evalscope and its dependency
# tree, which is a large chunk of the setup time.
REUSE_SYSTEM_EVALSCOPE=0
if "${BASE_PYTHON}" -c "import evalscope" >/dev/null 2>&1; then
    REUSE_SYSTEM_EVALSCOPE=1
    echo "evalscope found in ${BASE_PYTHON}; reusing it via --system-site-packages."
fi

if [ "${REUSE_SYSTEM_EVALSCOPE}" = "1" ] && [ ! -x "${PYTHON_FOR_EVALSCOPE}" ]; then
    "${BASE_PYTHON}" -m venv --system-site-packages ${PYTHON_ENV_FOR_EVALSCOPE}
fi

# Reuse the existing venv when it already provides what is needed.
if evalscope_env_ready; then
    echo "evalscope env already ready in ${PYTHON_ENV_FOR_EVALSCOPE}, skip installation."
    log_env_versions
    exit 0
fi

if [ ! -x "${PYTHON_FOR_EVALSCOPE}" ]; then
    "${BASE_PYTHON}" -m venv ${PYTHON_ENV_FOR_EVALSCOPE}
fi

if ! ${PYTHON_FOR_EVALSCOPE} -c "import evalscope" >/dev/null 2>&1; then
    install_evalscope
fi

install_pier
log_env_versions
