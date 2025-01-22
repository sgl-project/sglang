from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()


def get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


def update_wheel_platform_tag():
    wheel_dir = Path("dist")
    if wheel_dir.exists() and wheel_dir.is_dir():
        old_wheel = next(wheel_dir.glob("*.whl"))
        new_wheel = wheel_dir / old_wheel.name.replace(
            "linux_x86_64", "manylinux2014_x86_64"
        )
        old_wheel.rename(new_wheel)


def get_cuda_version():
    if torch.version.cuda:
        return tuple(map(int, torch.version.cuda.split(".")))
    return (0, 0)


def get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


cuda_version = get_cuda_version()
sm_version = get_device_sm()

cutlass = root / "3rdparty" / "cutlass"
flashinfer = root / "3rdparty" / "flashinfer"
include_dirs = [
    cutlass.resolve() / "include",
    cutlass.resolve() / "tools" / "util" / "include",
    root / "src" / "sgl-kernel" / "csrc",
    flashinfer.resolve() / "include",
    flashinfer.resolve() / "include" / "gemm",
    flashinfer.resolve() / "csrc",
]

nvcc_flags = [
    "-DNDEBUG",
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
    "-DFLASHINFER_ENABLE_BF16",
]

if cuda_version >= (12, 0) and sm_version >= 90:
    nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")

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
libraries = ["c10", "torch", "torch_python", "cuda"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]
ext_modules = [
    CUDAExtension(
        name="sgl_kernel.ops._kernels",
        sources=[
            "src/sgl-kernel/csrc/trt_reduce_internal.cu",
            "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
            "src/sgl-kernel/csrc/moe_align_kernel.cu",
            "src/sgl-kernel/csrc/int8_gemm_kernel.cu",
            "src/sgl-kernel/csrc/fp8_gemm_kernel.cu",
            "src/sgl-kernel/csrc/sampling_scaling_penalties.cu",
            "src/sgl-kernel/csrc/sgl_kernel_ops.cu",
            "src/sgl-kernel/csrc/rotary_embedding.cu",
            "3rdparty/flashinfer/csrc/activation.cu",
            "3rdparty/flashinfer/csrc/bmm_fp8.cu",
            "3rdparty/flashinfer/csrc/group_gemm.cu",
            "3rdparty/flashinfer/csrc/group_gemm_sm90.cu",
            "3rdparty/flashinfer/csrc/norm.cu",
            "3rdparty/flashinfer/csrc/sampling.cu",
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="sgl-kernel",
    version=get_version(),
    packages=find_packages(),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

update_wheel_platform_tag()
