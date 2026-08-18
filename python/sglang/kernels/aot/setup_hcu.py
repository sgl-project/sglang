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
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
host_arch = platform.machine().lower()

SUPPORTED_HCU_TARGETS = ("gfx928", "gfx936", "gfx938")
DEFAULT_HCU_TARGET = "gfx938"


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

# Keep the source list aligned with setup_rocm.py. USE_HCU selects the
# HCU-specific branch in moe_align_kernel.cu.
sources = [
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/allreduce/quick_all_reduce.cu",
    "csrc/common_extension_rocm.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/deepseek_v4_topk.cu",
    "csrc/elementwise/dsv4_norm_rope.cu",
    "csrc/elementwise/topk.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/elementwise/pos_enc.cu",
]

cxx_flags = ["-O3"]
libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
extra_link_args = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{host_arch}-linux-gnu",
]


def _get_hcu_target():
    hcu_target = os.environ.get("HCU_TARGET")
    if hcu_target:
        return hcu_target

    if not torch.cuda.is_available():
        print(
            "Warning: torch.cuda not available. "
            f"Using default HCU target: {DEFAULT_HCU_TARGET}"
        )
        return DEFAULT_HCU_TARGET

    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as exc:
        print(
            f"Warning: Failed to detect HCU properties: {exc}. "
            f"Using default HCU target: {DEFAULT_HCU_TARGET}"
        )
        return DEFAULT_HCU_TARGET


hcu_target = _get_hcu_target()

if hcu_target not in SUPPORTED_HCU_TARGETS:
    print(
        f"Warning: Unsupported HCU architecture detected '{hcu_target}'. "
        f"Expected one of: {', '.join(SUPPORTED_HCU_TARGETS)}."
    )
    sys.exit(1)

fp8_macro = "-DHIP_FP8_TYPE_FNUZ"

topk_dynamic_smem_bytes = 48 * 1024

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={hcu_target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-DUSE_HCU",
    fp8_macro,
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
]

if hcu_target == "gfx938":
    hipcc_flags.append("--gpu-max-threads-per-block=1024")

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
