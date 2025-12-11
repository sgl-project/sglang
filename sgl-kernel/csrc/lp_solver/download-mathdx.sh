#!/bin/bash

# $1: Target directory (e.g., build directory) where 'mathdx' folder will be created.

TARGET_BASE=${1:-.}
TARGET_DIR=$TARGET_BASE/mathdx

# Detect CUDA version
if command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN=nvcc
elif [ -f /usr/local/cuda/bin/nvcc ]; then
    NVCC_BIN=/usr/local/cuda/bin/nvcc
else
    echo "nvcc not found, assuming CUDA < 12.9"
    CUDA_VERSION="12.0"
fi

if [ -n "$NVCC_BIN" ]; then
    CUDA_VERSION=$($NVCC_BIN --version | grep -oP 'release \K[0-9]+\.[0-9]+')
fi

# Set the appropriate version of nvidia-mathdx
# Simple comparison using sort
if [ "$(printf '%s\n' "12.9" "$CUDA_VERSION" | sort -V | head -n1)" = "12.9" ]; then
     VERSION="25.06.0"
else
     VERSION="25.01.1"
fi

echo "Downloading MathDx $VERSION for CUDA $CUDA_VERSION"

FILENAME="nvidia-mathdx-$VERSION.tar.gz"
if [[ ! -f "$FILENAME" ]]; then
    wget -q "https://developer.download.nvidia.com/compute/cuSOLVERDx/redist/cuSOLVERDx/$FILENAME"
else
    echo "Files already exist - skipping download"
fi

echo "Extracting $FILENAME to $TARGET_DIR"
rm -rf $TARGET_DIR
# Extract temporarily
tar -xf "$FILENAME"

# Move the inner directory to target
# Structure is usually nvidia-mathdx-VERSION/nvidia/mathdx/VERSION_MAJOR_MINOR
# E.g. nvidia-mathdx-25.01.1/nvidia/mathdx/25.01
MAJOR_MINOR=${VERSION%.*}
SOURCE_DIR="nvidia-mathdx-$VERSION/nvidia/mathdx/$MAJOR_MINOR"

if [ -d "$SOURCE_DIR" ]; then
    mkdir -p $(dirname $TARGET_DIR)
    mv "$SOURCE_DIR" "$TARGET_DIR"
else
    echo "Error: Expected source directory $SOURCE_DIR not found after extraction."
    ls -R "nvidia-mathdx-$VERSION"
    exit 1
fi

rm -rf "nvidia-mathdx-$VERSION"
