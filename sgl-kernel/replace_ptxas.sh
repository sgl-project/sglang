#!/usr/bin/env bash
# https://github.com/Dao-AILab/flash-attention/blob/7ff1b621112ba8b538e2fc6a316f2a6b6f22e518/hopper/setup.py#L404
set -ex

if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_VERSION>"
    exit 1
fi

CUDA_VERSION=$1

if awk "BEGIN {exit !($CUDA_VERSION >= 12.6 && $CUDA_VERSION < 12.8)}"; then
    NVCC_ARCHIVE_VERSION="12.8.93"
    NVCC_ARCHIVE_NAME="cuda_nvcc-linux-x86_64-${NVCC_ARCHIVE_VERSION}-archive"
    NVCC_ARCHIVE_TAR="${NVCC_ARCHIVE_NAME}.tar.xz"
    NVCC_ARCHIVE_URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/${NVCC_ARCHIVE_TAR}"
    wget "$NVCC_ARCHIVE_URL"
    tar -xf "$NVCC_ARCHIVE_TAR"
    mkdir -p /usr/local/cuda/bin
    cp "${NVCC_ARCHIVE_NAME}/bin/ptxas" /usr/local/cuda/bin/
fi
