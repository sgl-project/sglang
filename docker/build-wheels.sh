#!/bin/bash
# Build Triton and triton_kernels wheels in a Docker container
# This ensures glibc compatibility with the target image.
#
# Usage:
#   ./docker/build-wheels.sh
#
# Output:
#   docker/wheels/triton-*.whl
#   docker/wheels/triton_kernels-*.whl

set -euo pipefail

MAX_JOBS="${MAX_JOBS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRITON_SRC="${REPO_ROOT}/../triton"
TRITON_KERNELS_SRC="${TRITON_SRC}/python/triton_kernels"
WHEELS_DIR="${SCRIPT_DIR}/wheels"
BASE_IMAGE="${BASE_IMAGE:-public.ecr.aws/droidcraft/sglang:base}"

echo "=== Building Triton Wheels (in container) ==="
echo "Triton source:   ${TRITON_SRC}"
echo "Triton kernels:  ${TRITON_KERNELS_SRC}"
echo "Output dir:      ${WHEELS_DIR}"
echo "Build image:     ${BASE_IMAGE}"
echo "Max jobs:        ${MAX_JOBS}"
echo ""

# Create wheels directory
mkdir -p "${WHEELS_DIR}"
rm -f "${WHEELS_DIR}"/*.whl

# Build wheels inside container for glibc compatibility
docker run --rm \
    -v "${TRITON_SRC}:/src/triton-src:ro" \
    -v "${WHEELS_DIR}:/wheels" \
    -e MAX_JOBS="${MAX_JOBS}" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
    -e MAKEFLAGS="-j${MAX_JOBS}" \
    "${BASE_IMAGE}" \
    bash -c '
set -ex

# Install build dependencies
apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build build-essential git zlib1g-dev rsync

pip install build wheel setuptools "pybind11>=2.13.1"

# Copy source to writable location (build needs to write .egg-info)
# Exclude build directory to avoid CMake cache conflicts
rsync -a --exclude="build" --exclude="*.egg-info" /src/triton-src/ /src/triton/

# Build triton_kernels wheel (pure Python, fast)
echo "Building triton_kernels wheel..."
cd /src/triton/python/triton_kernels
python -m build --wheel --outdir /wheels

# Build triton wheel (C++ compilation, slow)
echo "Building triton wheel..."
cd /src/triton
python -m build --wheel --outdir /wheels

echo "Done!"
ls -la /wheels/*.whl
'

echo ""
echo "=== Wheels built successfully ==="
ls -la "${WHEELS_DIR}"/*.whl
