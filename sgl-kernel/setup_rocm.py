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

# The custom multi-GPU all-reduce collectives (custom/deterministic/quick) are
# CDNA-only (cross-GPU peer-IPC + wave64). They are appended below for CDNA
# targets only; on RDNA they are omitted from the build and #ifdef'd out of
# registration via -DSGL_IS_RDNA, so multi-GPU all-reduce falls back to RCCL.
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

# Supported architectures:
#   CDNA3  (gfx942, MI300X/MI325X) and CDNA3+ (gfx950, MI350X)
#   RDNA3.5 (gfx1151, Strix Halo / Ryzen AI Max) -- single-GPU only
CDNA_TARGETS = ["gfx942", "gfx950"]
RDNA_TARGETS = ["gfx1151"]
SUPPORTED_TARGETS = CDNA_TARGETS + RDNA_TARGETS

if amdgpu_target not in SUPPORTED_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Supported: {SUPPORTED_TARGETS}."
    )
    sys.exit(1)

is_cdna = amdgpu_target in CDNA_TARGETS
is_rdna = amdgpu_target in RDNA_TARGETS

fp8_macro = (
    "-DHIP_FP8_TYPE_FNUZ" if amdgpu_target == "gfx942" else "-DHIP_FP8_TYPE_E4M3"
)

# Dynamic shared-memory budget for the TopK kernels.
# - gfx950 (MI350): large LDS (~160KB/CU) -> allow the 128KB dynamic smem path.
# - gfx942 (MI300/MI325) and RDNA (gfx1151): 64KB LDS/workgroup -> cap at 48KB.
topk_dynamic_smem_bytes = 32 * 1024 * 4 if amdgpu_target == "gfx950" else 48 * 1024

# CDNA-only custom multi-GPU all-reduce kernels (see the sources note above).
if is_cdna:
    sources += [
        "csrc/allreduce/custom_all_reduce.hip",
        "csrc/allreduce/deterministic_all_reduce.hip",
        "csrc/allreduce/quick_all_reduce.cu",
    ]

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

# Host and device passes must agree on the logical warp/wave width (CDNA=64,
# RDNA=32); WARP_SIZE in include/utils.h reads SGL_ROCM_WARP_SIZE. On RDNA,
# SGL_IS_RDNA additionally #ifdef's the CDNA-only all-reduce ops out of the
# cxx-compiled registration TU (common_extension_rocm.cc) so it links against
# exactly the objects that were built.
_rocm_arch_flags = [f"-DSGL_ROCM_WARP_SIZE={64 if is_cdna else 32}"]
if is_rdna:
    _rocm_arch_flags.append("-DSGL_IS_RDNA")
hipcc_flags += _rocm_arch_flags
cxx_flags += _rocm_arch_flags

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
