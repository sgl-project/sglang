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

# The custom/deterministic/quick all-reduce collectives are CDNA-only: they use
# multi-GPU peer IPC and CDNA-specific buffer/scope semantics.
# They are appended below for CDNA targets only; on RDNA they are omitted and
# their registration is #ifdef'd out via -DSGL_IS_RDNA, so multi-GPU all-reduce
# falls back to RCCL and single-GPU never calls all-reduce.
sources = [
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
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

# Wave width is resolved at runtime (host) / per-arch constexpr (device) in
# include/utils.h, so no warp-size compile flag is passed here.
CDNA_TARGETS = ["gfx942", "gfx950"]
RDNA_TARGETS = ["gfx1100", "gfx1151", "gfx1201"]
SUPPORTED_TARGETS = CDNA_TARGETS + RDNA_TARGETS

if amdgpu_target not in SUPPORTED_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Supported: {SUPPORTED_TARGETS}."
    )
    sys.exit(1)

is_rdna = amdgpu_target in RDNA_TARGETS

# CDNA-only multi-GPU all-reduce collectives (see note above the sources list).
if not is_rdna:
    sources += [
        "csrc/allreduce/custom_all_reduce.hip",
        "csrc/allreduce/deterministic_all_reduce.hip",
        "csrc/allreduce/quick_all_reduce.cu",
    ]

fp8_macro = (
    "-DHIP_FP8_TYPE_FNUZ" if amdgpu_target == "gfx942" else "-DHIP_FP8_TYPE_E4M3"
)

# Dynamic shared-memory budget for the TopK kernels.
# - gfx942 (MI300/MI325): LDS is typically 64KB per workgroup -> keep dynamic smem <= ~48KB
#   (leaves room for static shared allocations in the kernel).
# - RDNA (gfx11xx/gfx12xx): 64KB LDS per workgroup -> same 48KB cap as gfx942.
# - gfx95x (MI350): LDS is larger (e.g. 160KB per CU) -> allow the original 128KB dynamic smem.
topk_dynamic_smem_bytes = (
    48 * 1024 if (amdgpu_target == "gfx942" or is_rdna) else 32 * 1024 * 4
)

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={amdgpu_target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    fp8_macro,
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
]

# On RDNA the CDNA-only all-reduce collectives are not built; guard their
# registration (common_extension_rocm.cc) and declarations (sgl_kernel_ops.h).
# The flag must reach BOTH compilers: hipcc for the .hip/.cu sources (headers)
# and the host C++ compiler for common_extension_rocm.cc (a .cc file), otherwise
# the registration is compiled in and links against the excluded symbols.
if is_rdna:
    hipcc_flags.append("-DSGL_IS_RDNA")
    cxx_flags.append("-DSGL_IS_RDNA")

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
