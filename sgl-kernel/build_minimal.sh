#!/bin/bash
# Minimal build script for sgl-kernel (only compile needed architectures)

# 清理之前的编译
rm -rf _skbuild/ build/ *.egg-info

# 设置编译选项
export CMAKE_BUILD_PARALLEL_LEVEL=2          # 减少并行度，避免 OOM
export SGL_KERNEL_COMPILE_THREADS=4          # NVCC 线程数

# 只编译 B200 (SM100) 的架构，跳过其他
# 你的 B200 是 compute_100，不需要 SM80/89/90
export ENABLE_BELOW_SM90=OFF                 # 跳过 A100/V100 等旧架构
export SGL_KERNEL_ENABLE_FA3=OFF             # 跳过 Flash-Attention 3 (SM90a)
export SGL_KERNEL_ENABLE_SM90A=OFF           # 跳过 SM90A

# 只保留 SM100
export TORCH_CUDA_ARCH_LIST="10.0"

# 开始编译
pip install -e . --no-build-isolation -v 2>&1 | tee build.log

echo ""
echo "Build completed! Check build.log for details."

