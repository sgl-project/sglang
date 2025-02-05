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


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernels"
include_dirs = [
    root / "src" / "sgl-kernel" / "include",
    root / "src" / "sgl-kernel" / "csrc",
]

sources = [
    "src/sgl-kernel/torch_extension_rocm.cc",
    "src/sgl-kernel/csrc/moe_align_kernel.cu",
]

cxx_flags = ["-O3"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-D__HIP_PLATFORM_AMD__=1",
    "--amdgpu-target=gfx942",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "nvcc": hipcc_flags,
                "cxx": cxx_flags,
            },
            libraries=libraries,
            extra_link_args=extra_link_args,
            py_limited_api=True,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True, max_jobs=multiprocessing.cpu_count()
        )
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires=["torch"],
)
