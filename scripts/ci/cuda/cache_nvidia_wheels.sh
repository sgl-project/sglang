#!/bin/bash
# Cache and pre-install nvidia wheels that torch pins.
#
# pypi.nvidia.com returns Cache-Control: no-store, so pip re-downloads
# cudnn (~707 MB) and nvshmem (~125 MB) on every CI run. This script
# caches the wheels locally and installs them so that the subsequent
# `pip install -e "python[dev]"` sees "Requirement already satisfied".
#
# Integrity: uses `unzip -t` to detect partial/corrupt downloads.
#
# Usage: source scripts/ci/cuda/cache_nvidia_wheels.sh

NVIDIA_WHEEL_CACHE="/root/.cache/nvidia-wheels"
mkdir -p "$NVIDIA_WHEEL_CACHE"

download_wheel() {
    local url="$1"
    local whl="$2"
    local partial="${whl}.partial"

    if [ -f "$whl" ] && unzip -tq "$whl" &>/dev/null; then
        return 0
    fi

    rm -f "$whl"
    curl --fail --location \
        --retry 5 \
        --retry-delay 2 \
        --retry-all-errors \
        --connect-timeout 10 \
        --max-time 1800 \
        --continue-at - \
        -o "$partial" \
        "$url"
    unzip -tq "$partial" &>/dev/null
    mv "$partial" "$whl"
}

for url in \
    "https://pypi.nvidia.com/nvidia-cudnn-cu12/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl" \
    "https://pypi.nvidia.com/nvidia-nvshmem-cu12/nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"; do
    whl="$NVIDIA_WHEEL_CACHE/$(basename "$url")"
    download_wheel "$url" "$whl"
done

pip install --no-deps "$NVIDIA_WHEEL_CACHE"/nvidia_cudnn_cu12-*.whl \
    "$NVIDIA_WHEEL_CACHE"/nvidia_nvshmem_cu12-*.whl 2>/dev/null || true
