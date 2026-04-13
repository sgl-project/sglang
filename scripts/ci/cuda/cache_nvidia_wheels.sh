#!/bin/bash
# Cache and pre-install nvidia wheels that torch pins.
#
# pypi.nvidia.com returns Cache-Control: no-store, so pip re-downloads
# ~2 GB of NVIDIA wheels on every CI run. This script:
# 1. Caches cudnn + nvshmem wheels locally and pre-installs them
# 2. Points pip at a local wheel directory via PIP_FIND_LINKS so that
#    all NVIDIA torch dependencies (cublas, cufft, nvrtc, etc.) are
#    installed from local files instead of re-downloading.
#
# Pre-cache the wheels on the host at /opt/ci-cache/nvidia-pip-wheels/
# (mounted as /root/.cache/nvidia-pip-wheels inside containers).
# See the 5090 ops guide post-reboot checklist for how to populate this.
#
# Integrity: uses `unzip -t` to detect partial/corrupt downloads.
#
# Usage: source scripts/ci/cuda/cache_nvidia_wheels.sh

NVIDIA_WHEEL_CACHE="/root/.cache/nvidia-wheels"
NVIDIA_PIP_WHEELS="/root/.cache/nvidia-pip-wheels"
mkdir -p "$NVIDIA_WHEEL_CACHE"

for url in \
    "https://pypi.nvidia.com/nvidia-cudnn-cu12/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl" \
    "https://pypi.nvidia.com/nvidia-nvshmem-cu12/nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"; do
    whl="$NVIDIA_WHEEL_CACHE/$(basename "$url")"
    [ -f "$whl" ] && unzip -tq "$whl" &>/dev/null || curl -fL -o "$whl" "$url"
done

pip install --no-deps "$NVIDIA_WHEEL_CACHE"/nvidia_cudnn_cu12-*.whl \
    "$NVIDIA_WHEEL_CACHE"/nvidia_nvshmem_cu12-*.whl 2>/dev/null || true

# If pre-cached NVIDIA pip wheels exist, tell pip to check there first.
# This avoids re-downloading ~2 GB of cublas/cufft/nvrtc/etc. every run
# (pypi.nvidia.com sends Cache-Control: no-store).
if [ -d "$NVIDIA_PIP_WHEELS" ] && ls "$NVIDIA_PIP_WHEELS"/*.whl &>/dev/null; then
    export PIP_FIND_LINKS="${PIP_FIND_LINKS:+$PIP_FIND_LINKS }$NVIDIA_PIP_WHEELS"
fi
