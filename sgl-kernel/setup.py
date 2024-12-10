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
    new_info = tmp_dir / f"sgl_kernel-{base_version}+cu{cuda_version}.dist-info"
    old_info.rename(new_info)

    new_wheel = (
        wheel_dir
        / f"sgl_kernel-{base_version}+cu{cuda_version}-{old_wheel.name.split('-', 2)[-1]}"
    )
    with zipfile.ZipFile(new_wheel, "w", zipfile.ZIP_DEFLATED) as new_zip:
        for file_path in tmp_dir.rglob("*"):
            if file_path.is_file():
                new_zip.write(file_path, file_path.relative_to(tmp_dir))

    old_wheel.unlink()
    shutil.rmtree(tmp_dir)


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
                "nvcc": ["-O3", "-Xcompiler", "-fPIC"],
                "cxx": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

rename_wheel()
