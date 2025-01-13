from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
import multiprocessing

root = Path(__file__).parent.resolve()

# 添加调试模式控制
debug_build = os.environ.get('DEBUG_BUILD', '0').lower() in ('1', 'true', 'yes', 'on')
print(f"Debug build: {'enabled' if debug_build else 'disabled'}")

def get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


def update_wheel_platform_tag():
    wheel_dir = Path("dist")
    old_wheel = next(wheel_dir.glob("*.whl"))
    new_wheel = wheel_dir / old_wheel.name.replace(
        "linux_x86_64", "manylinux2014_x86_64"
    )
    old_wheel.rename(new_wheel)


cutlass = root / "3rdparty" / "cutlass"
nlohmann = root / "3rdparty" / "nlohmann"

include_dirs = [
    cutlass.resolve() / "include",
    cutlass.resolve() / "tools" / "util" / "include",
    root / "src" / "sgl-kernel" / "csrc",
    nlohmann.resolve(),
]

# nvcc_flags = [
#     "-O3",
#     "-Xcompiler",
#     "-fPIC",
#     "-gencode=arch=compute_75,code=sm_75",
#     "-gencode=arch=compute_80,code=sm_80",
#     "-gencode=arch=compute_89,code=sm_89",
#     "-gencode=arch=compute_90,code=sm_90",
#     "-U__CUDA_NO_HALF_OPERATORS__",
#     "-U__CUDA_NO_HALF2_OPERATORS__",
# ]
nvcc_flags = [
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_89,code=sm_89",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

# 如果是调试模式，添加调试标志
if debug_build:
    nvcc_flags.extend([
        "-DSGL_DEBUG_BUILD",
    ])
cxx_flags = ["-O3"]
if debug_build:
    cxx_flags.extend(["-DSGL_DEBUG_BUILD"])

libraries = ["c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib"]
ext_modules = [
    CUDAExtension(
        name="sgl_kernel.ops._kernels",
        sources=[
            "src/sgl-kernel/csrc/trt_reduce_internal.cu",
            "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
            "src/sgl-kernel/csrc/moe_align_kernel.cu",
            # "src/sgl-kernel/csrc/int8_gemm_kernel.cu",
            "src/sgl-kernel/csrc/fp8_gemm_kernel.cu",
            "src/sgl-kernel/csrc/sgl_kernel_ops.cu",
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

def set_parallel_jobs():
    if sys.platform == 'win32':
        num_cores = int(os.environ.get('NUMBER_OF_PROCESSORS', 4))
    else:
        num_cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    
    # 限制并行度为核心数的1/4或更少
    num_jobs = max(1, num_cores // 2)
    os.environ['MAX_JOBS'] = str(num_jobs)
    
    # 设置CUDA编译的并行任务数
    os.environ['CUDA_NVCC_THREADS'] = str(num_jobs)
    return num_jobs
set_parallel_jobs()
setup(
    name="sgl-kernel",
    version=get_version(),
    packages=["sgl_kernel"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

update_wheel_platform_tag()
