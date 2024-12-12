import os
import shutil
import zipfile
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()


def get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


def rename_wheel():
    if not os.environ.get("CUDA_VERSION"):
        return
    cuda_version = os.environ["CUDA_VERSION"].replace(".", "")
    base_version = get_version()

    wheel_dir = Path("dist")
    old_wheel = next(wheel_dir.glob("*.whl"))
    tmp_dir = wheel_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(old_wheel, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    old_info = tmp_dir / f"sgl_kernel-{base_version}.dist-info"
    new_info = tmp_dir / f"sgl_kernel-{base_version}.post0+cu{cuda_version}.dist-info"
    old_info.rename(new_info)

    platform = "manylinux2014_x86_64"
    new_wheel = wheel_dir / old_wheel.name.replace("linux_x86_64", platform)
    new_wheel = wheel_dir / new_wheel.name.replace(
        base_version, f"{base_version}.post0+cu{cuda_version}"
    )

    with zipfile.ZipFile(new_wheel, "w", zipfile.ZIP_DEFLATED) as new_zip:
        for file_path in tmp_dir.rglob("*"):
            if file_path.is_file():
                new_zip.write(file_path, file_path.relative_to(tmp_dir))

    old_wheel.unlink()
    shutil.rmtree(tmp_dir)


def update_wheel_platform_tag():
    wheel_dir = Path("dist")
    old_wheel = next(wheel_dir.glob("*.whl"))
    new_wheel = wheel_dir / old_wheel.name.replace(
        "linux_x86_64", "manylinux2014_x86_64"
    )
    old_wheel.rename(new_wheel)


setup(
    name="sgl-kernel",
    version=get_version(),
    packages=["sgl_kernel"],
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

update_wheel_platform_tag()
