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

import os
import shutil
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension

root = Path(__file__).parent.resolve()

if "bdist_wheel" in sys.argv and "--plat-name" not in sys.argv:
    sys.argv.extend(["--plat-name", "manylinux2014_x86_64"])


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = []

sources = [
    "csrc/cpu/activation.cpp",
    "csrc/cpu/bmm.cpp",
    "csrc/cpu/decode.cpp",
    "csrc/cpu/extend.cpp",
    "csrc/cpu/gemm.cpp",
    "csrc/cpu/gemm_int8.cpp",
    "csrc/cpu/moe.cpp",
    "csrc/cpu/moe_int8.cpp",
    "csrc/cpu/norm.cpp",
    "csrc/cpu/qkv_proj.cpp",
    "csrc/cpu/topk.cpp",
    "csrc/cpu/interface.cpp",
    "csrc/cpu/shm.cpp",
    "csrc/cpu/torch_extension_cpu.cpp",
]

extra_compile_args = {
    "cxx": [
        "-O3",
        "-Wno-unknown-pragmas",
        "-march=native",
        "-fopenmp",
    ]
}
libraries = ["c10", "torch", "torch_python"]
cmdclass = {
    "build_ext": BuildExtension.with_options(use_ninja=True),
}
Extension = CppExtension

extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

ext_modules = [
    Extension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=True,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
