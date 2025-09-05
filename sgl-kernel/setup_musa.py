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
import platform
import sys
from pathlib import Path

import torch
import torch_musa
from setuptools import find_packages, setup
from torch_musa.utils.simple_porting import SimplePorting
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
]

SimplePorting(
    cuda_dir_path="csrc",
    mapping_rule={
        "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
        "#include <ATen/cuda/Exceptions.h>": "#include \"torch_musa/csrc/aten/musa/Exceptions.h\"",
        "#include <THC/THCAtomics.cuh>": "#include <THC/THCAtomics.muh>",
        "#include <c10/cuda/CUDAException.h>": "#include \"torch_musa/csrc/core/MUSAException.h\"",
        "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
        "#include <c10/cuda/CUDAStream.h>": "#include \"torch_musa/csrc/core/MUSAStream.h\"",
        "#include \"custom_all_reduce.cuh\"": "#include \"custom_all_reduce.muh\"",
        "at::cuda": "at::musa",
        "c10::cuda": "c10::musa",
    }
).run()

sources = [
    "csrc_musa/allreduce/custom_all_reduce.mu",
    "csrc_musa/common_extension_musa.cc",
    # "csrc_musa/elementwise/activation.mu",
    "csrc_musa/grammar/apply_token_bitmask_inplace_cuda.mu",
    "csrc_musa/moe/moe_align_kernel.mu",
    "csrc_musa/moe/moe_topk_softmax_kernels.mu",
    # "csrc_musa/speculative/eagle_utils.mu",
    # "csrc_musa/kvcacheio/transfer.mu",
]

cxx_flags = ["force_mcc"]
libraries = ["c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "mp_22"
mtgpu_target = os.environ.get("MTGPU_TARGET", default_target)

if torch.musa.is_available():
    try:
        prop = torch.musa.get_device_properties(0)
        mtgpu_target = f"mp_{prop.major}{prop.minor}"
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.musa not available. Using default target: {mtgpu_target}")

if mtgpu_target not in ["mp_22", "mp_31"]:
    print(
        f"Warning: Unsupported GPU architecture detected '{mtgpu_target}'. Expected 'mp_22' or 'mp_31'."
    )
    sys.exit(1)

mcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-fPIC",
    "-std=c++17",
    f"--cuda-gpu-arch={mtgpu_target}",
    "-DUSE_MUSA"
]

ext_modules = [
    MUSAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "mcc": mcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
