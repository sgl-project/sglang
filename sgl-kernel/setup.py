import glob
import os
import shutil

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_base_version():
    with open("pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


base_version = get_base_version()
cuda_version = os.environ.get("CUDA_VERSION", "").replace(".", "")
final_version = f"{base_version}+cu{cuda_version}" if cuda_version else base_version


def rename_wheel_with_cuda_version(dist_dir="dist"):
    if not cuda_version:
        return
    wheel_files = glob.glob(f"{dist_dir}/*.whl")
    for wheel_file in wheel_files:
        base_name = os.path.basename(wheel_file)
        if "+cu" not in base_name:
            name_parts = base_name.split("-")
            name_parts[1] += f"+cu{cuda_version}"
            new_name = "-".join(name_parts)
            new_path = os.path.join(dist_dir, new_name)
            shutil.move(wheel_file, new_path)


setup(
    name="sgl-kernel",
    version=final_version,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            "sgl_kernel.ops.warp_reduce_cuda",
            [
                "src/sgl-kernel/csrc/warp_reduce.cc",
                "src/sgl-kernel/csrc/warp_reduce_kernel.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-Xcompiler",
                    "-fPIC",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
                "cxx": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

rename_wheel_with_cuda_version()
