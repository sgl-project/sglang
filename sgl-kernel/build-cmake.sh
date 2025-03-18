#!/bin/bash
set -ex

# Parse arguments.
PYTHON_VERSION=${1:-"3.9"}
CUDA_VERSION=${2:-"12.1"}
PARALLEL_JOBS=${3:-$(nproc)}

# Print build configuration.
echo "Build configuration:"
echo "  Python version: $PYTHON_VERSION"
echo "  CUDA version: $CUDA_VERSION"
echo "  Parallel jobs: $PARALLEL_JOBS"

# Set environment variables.
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX"
export SGL_KERNEL_ENABLE_BF16=1
export SGL_KERNEL_ENABLE_FP8=1
export CMAKE_BUILD_PARALLEL_LEVEL=$PARALLEL_JOBS

# Enable SM90A for CUDA 12+.
if (( ${CUDA_VERSION%.*} >= 12 )); then
    export SGL_KERNEL_ENABLE_SM90A=1
else
    export SGL_KERNEL_ENABLE_SM90A=0
fi

# Check if uv is installed.
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing..."
    pip install -U uv
fi
start=$(date +%s)

# Build the wheel.
uv build \
  --wheel \
  -Cbuild-dir=build \
  . \
  --verbose \
  --color=always

# Calculate build time.
end=$(date +%s)
build_time=$((end - start))
build_minutes=$((build_time / 60))
build_seconds=$((build_time % 60))

echo "Build completed successfully with $PARALLEL_JOBS parallel jobs."
echo "Build time: ${build_time} seconds (${build_minutes} minutes ${build_seconds} seconds)."
echo "Wheel file is in dist/ directory."
