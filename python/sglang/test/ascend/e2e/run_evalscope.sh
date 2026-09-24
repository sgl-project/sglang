#!/bin/bash

set -e

PYTHON_ENV_FOR_EVALSCOPE=test_env_evalscope
PYTHON_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/python
PIP_FOR_EVALSCOPE=${PYTHON_ENV_FOR_EVALSCOPE}/bin/pip
EVALSCOPE_SOURCE_PATH=/root/.cache/.cache/evalscope

# Bound key deps so the resolver cannot fall back to ancient versions.
EVALSCOPE_CONSTRAINTS=(
    "aiohttp>=3.11,<4"
    "httpx>=0.28,<1"
)

# Budget for a single pip install attempt; the install itself is retried below.
EVALSCOPE_INSTALL_TIMEOUT=1500

# Overall budget for the whole install, retries and mirror fallback included.
EVALSCOPE_INSTALL_TOTAL_TIMEOUT=2700

# Every round walks the whole mirror list; the list is walked this many times.
EVALSCOPE_INSTALL_RETRIES=3
EVALSCOPE_RETRY_DELAY=15

# In-cluster PyPI cache first (no external network), then public mirrors.
# The fallback is serial on purpose: with --extra-index-url every mirror stays
# in the candidate set, so one slow mirror would still stall the whole install.
PIP_MIRRORS=(
    "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
    "https://mirrors.huaweicloud.com/repository/pypi/simple/"
    "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    "https://mirrors.aliyun.com/pypi/simple/"
)

# The image sets PIP_CACHE_DIR=/tmp/pip-cache, which dies with the pod, so every
# job re-downloads the whole dependency set. /root/.cache is the shared PVC.
export PIP_CACHE_DIR=/root/.cache/pip

# Epoch seconds; the install is capped by this one budget.
install_deadline=$(date +%s)

install_budget_left() {
    echo $((EVALSCOPE_INSTALL_TOTAL_TIMEOUT - $(date +%s) + install_deadline))
}

# 000 means curl got no response at all (DNS/connect/timeout).
mirror_reachable() {
    command -v curl >/dev/null 2>&1 || return 0
    [ "$(curl -s -o /dev/null --max-time 5 -w '%{http_code}' "$1")" != "000" ]
}

# --timeout/--retries bound a single HTTP request, timeout(1) bounds the attempt.
run_pip_install() {
    local attempt_timeout="$1"
    local mirror="$2"
    shift 2
    local opts=("--index-url" "${mirror}" "--timeout" "30" "--retries" "3")
    if [[ "${mirror}" == http://* ]]; then
        opts+=("--trusted-host" "$(echo "${mirror}" | cut -d/ -f3 | cut -d: -f1)")
    fi
    timeout "${attempt_timeout}" "${PIP_FOR_EVALSCOPE}" install "$@" "${opts[@]}"
}

# Try each mirror in turn, and repeat the whole list EVALSCOPE_INSTALL_RETRIES
# times. pip is idempotent, so a retry after a timeout resumes where it stopped.
# $1 is the per-attempt budget, which is additionally capped by whatever is left
# of EVALSCOPE_INSTALL_TOTAL_TIMEOUT.
pip_install_with_fallback() {
    local per_attempt_timeout="$1"
    shift
    local attempt mirror remaining attempt_timeout
    for attempt in $(seq 1 "${EVALSCOPE_INSTALL_RETRIES}"); do
        for mirror in "${PIP_MIRRORS[@]}"; do
            remaining=$(install_budget_left)
            if [ "${remaining}" -le 0 ]; then
                echo "ERROR: install exceeded the ${EVALSCOPE_INSTALL_TOTAL_TIMEOUT}s total budget."
                return 1
            fi
            if ! mirror_reachable "${mirror}"; then
                echo "WARNING: ${mirror} is unreachable, skipping."
                continue
            fi
            attempt_timeout=${per_attempt_timeout}
            if [ "${remaining}" -lt "${attempt_timeout}" ]; then
                attempt_timeout=${remaining}
            fi
            echo "pip install (attempt ${attempt}/${EVALSCOPE_INSTALL_RETRIES}, ${attempt_timeout}s) via ${mirror}"
            if run_pip_install "${attempt_timeout}" "${mirror}" "$@"; then
                return 0
            fi
            echo "WARNING: pip install via ${mirror} failed, trying the next mirror."
        done
        if [ "${attempt}" -lt "${EVALSCOPE_INSTALL_RETRIES}" ]; then
            echo "WARNING: all mirrors failed, retrying in ${EVALSCOPE_RETRY_DELAY}s."
            sleep "${EVALSCOPE_RETRY_DELAY}"
        fi
    done
    return 1
}

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
    EVALSCOPE_PIP_TARGET=("evalscope")
else
    echo "Install evalscope from local source: ${EVALSCOPE_SOURCE_PATH}"
    EVALSCOPE_PIP_TARGET=("-e" "${EVALSCOPE_SOURCE_PATH}")
fi

if ! pip_install_with_fallback "${EVALSCOPE_INSTALL_TIMEOUT}" "${EVALSCOPE_PIP_TARGET[@]}" "${EVALSCOPE_CONSTRAINTS[@]}"; then
    echo "ERROR: evalscope install failed within the ${EVALSCOPE_INSTALL_TOTAL_TIMEOUT}s budget (${EVALSCOPE_INSTALL_RETRIES} rounds over ${#PIP_MIRRORS[@]} mirrors)."
    exit 1
fi
echo "===== Install evalscope in virtual env - End ====="
