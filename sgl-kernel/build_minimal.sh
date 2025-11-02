#!/bin/bash
# Minimal build script for sgl-kernel (only compile needed architectures)

# Clean previous build artifacts
rm -rf _skbuild/ build/ *.egg-info

# Configure build options
export CMAKE_BUILD_PARALLEL_LEVEL=2          # Reduce parallelism to avoid OOM
export SGL_KERNEL_COMPILE_THREADS=4          # NVCC thread count

# Only compile for B200 (SM100) architecture, skip others
# B200 is compute_100, no need for SM80/89/90
export ENABLE_BELOW_SM90=OFF                 # Skip older architectures (A100/V100 etc.)
export SGL_KERNEL_ENABLE_FA3=OFF             # Skip Flash-Attention 3 (SM90a)
export SGL_KERNEL_ENABLE_SM90A=OFF           # Skip SM90A

# Only keep SM100
export TORCH_CUDA_ARCH_LIST="10.0"

# Start compilation
pip install -e . --no-build-isolation -v 2>&1 | tee build.log

echo ""
echo "Build completed! Check build.log for details."
