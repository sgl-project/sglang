#!/bin/bash
# Install flashinfer-jit-cache with caching and retry logic (flashinfer.ai can have transient DNS issues).
# The jit-cache wheel is 1.2+ GB, so we skip the download entirely if already installed.
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

    FLASHINFER_WHEEL_PATTERN="flashinfer_jit_cache-${FLASHINFER_PYTHON_REQUIRED}*.whl"
    CACHED_WHEEL=$(find "${FLASHINFER_CACHE_DIR}" -name "${FLASHINFER_WHEEL_PATTERN}" -type f 2>/dev/null | head -n 1)

    if [ -n "$CACHED_WHEEL" ] && [ -f "$CACHED_WHEEL" ]; then
        echo "Found cached flashinfer wheel: $CACHED_WHEEL"
        if $PIP_CMD install "$CACHED_WHEEL" $PIP_INSTALL_SUFFIX; then
            FLASHINFER_JIT_CACHE_INSTALLED=true
            echo "Successfully installed flashinfer-jit-cache from cache"
        else
            echo "Failed to install from cache, will try downloading..."
            rm -f "$CACHED_WHEEL"
        fi
    fi

    if [ "$FLASHINFER_JIT_CACHE_INSTALLED" = false ]; then
        for i in {1..5}; do
            # Download wheel to cache directory (use pip directly as uv pip doesn't support download)
            if timeout 600 pip download "flashinfer-jit-cache==${FLASHINFER_PYTHON_REQUIRED}" \
                --index-url "https://flashinfer.ai/whl/${CU_VERSION}" \
                -d "${FLASHINFER_CACHE_DIR}"; then

                CACHED_WHEEL=$(find "${FLASHINFER_CACHE_DIR}" -name "${FLASHINFER_WHEEL_PATTERN}" -type f 2>/dev/null | head -n 1)
                if [ -n "$CACHED_WHEEL" ] && [ -f "$CACHED_WHEEL" ]; then
                    if $PIP_CMD install "$CACHED_WHEEL" $PIP_INSTALL_SUFFIX; then
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
