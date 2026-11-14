# Copyright 2026 SGLang Team. All Rights Reserved.
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
import platform
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent.resolve()


_BUILD_REQUIRES = [
    "setuptools>=80",
    "mlx==0.32.4",
    "nanobind==3.1.0",
    "cmake>=3.25",
    "ninja",
]


def _ensure_toolchain():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("setup_metal.py only supports macOS (Apple Silicon).")
    if shutil.which("c++") is None or shutil.which("xcrun") is None:
        raise SystemExit(
            "Apple toolchain not found. Install the Xcode Command Line Tools "
            "with `xcode-select --install` (or a full Xcode install) and retry."
        )
    compiler = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if compiler.returncode:
        raise SystemExit(
            "Apple Metal shader compiler unavailable. Install full Xcode; if its "
            "Metal component is missing, run `xcodebuild -downloadComponent "
            f"MetalToolchain`, then retry.\n{compiler.stdout}"
        )


def _ensure_build_requires():
    subprocess.check_call([sys.executable, "-m", "pip", "install", *_BUILD_REQUIRES])


# Section 1: Prerequisites
_ensure_toolchain()
_ensure_build_requires()
os.chdir(root)


# Section 2: Build and install
from mlx.extension import BuildExtension, MetalExtension  # noqa: E402
from setuptools import find_packages, setup  # noqa: E402


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


metal_shader_sources = [
    "csrc/metal/radix_attention.metal",
    "csrc/metal/rope_pool_fused.metal",
]
cxx_sources = [
    "csrc/metal/bindings.cpp",
    "csrc/metal/metal_common.cpp",
    "csrc/metal/radix_attention.cpp",
    "csrc/metal/rope_pool_fused.cpp",
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"sgl_kernel": ["*.metallib", "*.pyi"]},
    include_package_data=True,
    ext_modules=[
        MetalExtension(
            name="sgl_kernel._metal",
            sources=[*cxx_sources, *metal_shader_sources],
            include_dirs=["csrc", "csrc/metal"],
            extra_compile_args={
                "cxx": ["-O3"],
                "metal": [
                    "-O3",
                    "-ffast-math",
                    f"-std={os.environ.get('SGL_METAL_STD', 'metal3.1')}",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
