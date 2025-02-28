# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import multiprocessing
import os
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()


if "bdist_wheel" in sys.argv and "--plat-name" not in sys.argv:
    sys.argv.extend(["--plat-name", "manylinux2014_x86_64"])


def _get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


def _get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernels"
cutlass_default = root / "3rdparty" / "cutlass"
cutlass = Path(os.environ.get("CUSTOM_CUTLASS_SRC_DIR", default=cutlass_default))
flashinfer = root / "3rdparty" / "flashinfer"
turbomind = root / "3rdparty" / "turbomind"
include_dirs = [
    cutlass.resolve() / "include",
    cutlass.resolve() / "tools" / "util" / "include",
    root / "src" / "sgl-kernel" / "include",
    root / "src" / "sgl-kernel" / "csrc",
    flashinfer.resolve() / "include",
    flashinfer.resolve() / "include" / "gemm",
    flashinfer.resolve() / "csrc",
    "cublas",
    turbomind.resolve(),
    turbomind.resolve() / "src",
]

nvcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-std=c++17",
    "-use_fast_math",
    "-DFLASHINFER_ENABLE_F16",
    "-Xcompiler=-Wconversion",
    "-Xcompiler=-fno-strict-aliasing",
]
nvcc_flags_fp8 = [
    "-DFLASHINFER_ENABLE_FP8",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
    "-DFLASHINFER_ENABLE_FP8_E5M2",
]

sources = [
    "src/sgl-kernel/torch_extension.cc",
    "src/sgl-kernel/csrc/trt_reduce_internal.cu",
    "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
    "src/sgl-kernel/csrc/moe_align_kernel.cu",
    "src/sgl-kernel/csrc/int8_gemm_kernel.cu",
    "src/sgl-kernel/csrc/fp8_gemm_kernel.cu",
    "src/sgl-kernel/csrc/fp8_blockwise_gemm_kernel.cu",
    "src/sgl-kernel/csrc/lightning_attention_decode_kernel.cu",
    "src/sgl-kernel/csrc/fused_add_rms_norm_kernel.cu",
    "src/sgl-kernel/csrc/eagle_utils.cu",
    "src/sgl-kernel/csrc/speculative_sampling.cu",
    "src/sgl-kernel/csrc/per_token_group_quant_fp8.cu",
    "src/sgl-kernel/csrc/cublas_grouped_gemm.cu",
    "3rdparty/flashinfer/csrc/activation.cu",
    "3rdparty/flashinfer/csrc/bmm_fp8.cu",
    "3rdparty/flashinfer/csrc/norm.cu",
    "3rdparty/flashinfer/csrc/sampling.cu",
    "3rdparty/flashinfer/csrc/renorm.cu",
    "3rdparty/flashinfer/csrc/rope.cu",
]

enable_bf16 = os.getenv("SGL_KERNEL_ENABLE_BF16", "0") == "1"
enable_fp8 = os.getenv("SGL_KERNEL_ENABLE_FP8", "0") == "1"
enable_sm90a = os.getenv("SGL_KERNEL_ENABLE_SM90A", "0") == "1"
cuda_version = _get_cuda_version()
sm_version = _get_device_sm()

if torch.cuda.is_available():
    if cuda_version >= (12, 0) and sm_version >= 90:
        nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    if sm_version >= 90:
        nvcc_flags.extend(nvcc_flags_fp8)
    if sm_version >= 80:
        nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")
else:
    # compilation environment without GPU
    if enable_sm90a:
        nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
    if enable_fp8:
        nvcc_flags.extend(nvcc_flags_fp8)
    if enable_bf16:
        nvcc_flags.append("-DFLASHINFER_ENABLE_BF16")

for flag in [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass

cxx_flags = ["-O3"]
libraries = ["c10", "torch", "torch_python", "cuda", "cublas"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.ops._kernels",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=True,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True, max_jobs=multiprocessing.cpu_count()
        )
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
