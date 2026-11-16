#!/bin/bash
# Install flashinfer-jit-cache with caching and retry logic (flashinfer.ai can have transient DNS issues).
# The jit-cache is 1.2+ GB across its wheels, so we skip the download entirely if already installed.
#
# Required environment (caller must export or set):
#   UNINSTALL_JIT_CACHE          — literal true/false (skip download when false)
#   FLASHINFER_PYTHON_REQUIRED   — e.g. from python/pyproject.toml (flashinfer_python)
#   CU_VERSION                   — e.g. cu130
#   PIP_CMD                      — e.g. "pip" or "uv pip"
#   PIP_INSTALL_SUFFIX           — extra pip args for this runner
set -euxo pipefail

: "${UNINSTALL_JIT_CACHE:?must be set}"
: "${FLASHINFER_PYTHON_REQUIRED:?must be set}"
: "${CU_VERSION:?must be set}"
: "${PIP_CMD:?must be set}"

FLASHINFER_JIT_CACHE_INSTALLED=false
if [ "$UNINSTALL_JIT_CACHE" = false ]; then
    FLASHINFER_JIT_CACHE_INSTALLED=true
    echo "flashinfer-jit-cache already at correct version, skipping download"
fi

if [ "$FLASHINFER_JIT_CACHE_INSTALLED" = false ]; then
    FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer-wheels"
    mkdir -p "${FLASHINFER_CACHE_DIR}"

    # The per-arch wheel names are unclaimed on PyPI, so resolve the whole set
    # from disk. The glob also matches the single pre-0.7.0 wheel.
    FLASHINFER_WHEEL_GLOB="${FLASHINFER_CACHE_DIR}/flashinfer_jit_cache*-${FLASHINFER_PYTHON_REQUIRED}+${CU_VERSION}-*.whl"

    set -- $FLASHINFER_WHEEL_GLOB
    if [ -f "$1" ]; then
        echo "Found $# cached flashinfer wheel(s)"
        if $PIP_CMD install --no-index "$@" $PIP_INSTALL_SUFFIX; then
            FLASHINFER_JIT_CACHE_INSTALLED=true
            echo "Successfully installed flashinfer-jit-cache from cache"
        else
            echo "Failed to install from cache, will try downloading..."
            rm -f "$@"
        fi
    fi

    if [ "$FLASHINFER_JIT_CACHE_INSTALLED" = false ]; then
        for i in {1..5}; do
            # Download wheel to cache directory (use pip directly as uv pip doesn't support download)
            if timeout 600 pip download "flashinfer-jit-cache==${FLASHINFER_PYTHON_REQUIRED}" \
                --index-url "https://flashinfer.ai/whl/${CU_VERSION}" \
                -d "${FLASHINFER_CACHE_DIR}"; then

                set -- $FLASHINFER_WHEEL_GLOB
                if [ -f "$1" ]; then
                    if $PIP_CMD install --no-index "$@" $PIP_INSTALL_SUFFIX; then
                        FLASHINFER_JIT_CACHE_INSTALLED=true
                        echo "Successfully downloaded and installed flashinfer-jit-cache"
                        break
                    fi
                else
                    echo "Warning: Download succeeded but wheel file not found"
                fi
            fi
            echo "Attempt $i to download flashinfer-jit-cache failed, retrying in 10 seconds..."
            sleep 10
        done
    fi
fi

if [ "$FLASHINFER_JIT_CACHE_INSTALLED" = false ]; then
    echo "ERROR: Failed to install flashinfer-jit-cache after 5 attempts"
    exit 1
fi

# A resolve can succeed and still register no provider, which degrades silently.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! bash "${SCRIPT_DIR}/ci_check_flashinfer_jit_cache.sh"; then
    echo "ERROR: flashinfer-jit-cache installed but serves no cubins"
    exit 1
fi
