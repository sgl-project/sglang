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
#   CDNA3:  gfx942  (MI300X/MI325X)
#   CDNA3+: gfx950  (MI350X)
#   RDNA3:  gfx1100, gfx1101, gfx1102  (RX 7900 series)
#   RDNA4:  gfx1200, gfx1201           (RX 9070 series)
CDNA_TARGETS = ["gfx942", "gfx950"]
RDNA3_TARGETS = ["gfx1100", "gfx1101", "gfx1102"]
RDNA4_TARGETS = ["gfx1200", "gfx1201"]
SUPPORTED_TARGETS = CDNA_TARGETS + RDNA3_TARGETS + RDNA4_TARGETS

if amdgpu_target not in SUPPORTED_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Supported: {SUPPORTED_TARGETS}"
    )
    sys.exit(1)

is_cdna = amdgpu_target in CDNA_TARGETS
is_rdna3 = amdgpu_target in RDNA3_TARGETS
is_rdna4 = amdgpu_target in RDNA4_TARGETS
is_rdna = is_rdna3 or is_rdna4

# common_extension_rocm.cc is built with cxx-only flags; HIP sources get hipcc_flags.
# Mirror RDNA macros on cxx so #ifndef SGL_IS_RDNA matches the linked objects (CDNA-only
# quick_all_reduce / qr_* are omitted on RDNA — otherwise we link references to qr_destroy
# without defining it).
if is_rdna:
    cxx_flags.append("-DSGL_IS_RDNA")
if is_rdna3:
    cxx_flags.append("-DSGL_IS_RDNA3")
if is_rdna4:
    cxx_flags.append("-DSGL_IS_RDNA4")

# FP8 format selection:
#   gfx942 (CDNA3):  FNUZ (non-IEEE, max=224)
#   gfx950 (CDNA3+): standard E4M3 (IEEE, same as NVIDIA)
#   RDNA3:           no hardware FP8 support
#   RDNA4:           standard E4M3 (IEEE, same as NVIDIA)
if amdgpu_target == "gfx942":
    fp8_macro = "-DHIP_FP8_TYPE_FNUZ"
elif amdgpu_target in ["gfx950"] + RDNA4_TARGETS:
    fp8_macro = "-DHIP_FP8_TYPE_E4M3"
else:
    fp8_macro = None  # RDNA3: no FP8

# Dynamic shared-memory budget for the TopK kernels.
# - gfx942 (MI300/MI325): LDS 64KB per workgroup -> cap at 48KB
# - gfx95x (MI350):       LDS 160KB per CU -> 128KB dynamic smem
# - RDNA3/4:              LDS 64KB per workgroup -> cap at 48KB
if amdgpu_target == "gfx950":
    topk_dynamic_smem_bytes = 32 * 1024 * 4
else:
    topk_dynamic_smem_bytes = 48 * 1024

sources = [
    "csrc/common_extension_rocm.cc",
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/elementwise/activation.cu",
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

# quick_all_reduce uses CDNA-specific MUBUF instructions; not available on RDNA.
if is_cdna:
    sources.append("csrc/allreduce/quick_all_reduce.cu")

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--amdgpu-target={amdgpu_target}",
    "-DENABLE_BF16",
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
]

if is_rdna:
    hipcc_flags.append("-DSGL_IS_RDNA")
if is_rdna3:
    hipcc_flags.append("-DSGL_IS_RDNA3")
if is_rdna4:
    hipcc_flags.append("-DSGL_IS_RDNA4")

if fp8_macro is not None:
    hipcc_flags.extend(["-DENABLE_FP8", fp8_macro])

# MoE topk: host and device must agree on logical warp width (64 CDNA, 32 RDNA).
_rocm_warp = 64 if is_cdna else 32
hipcc_flags.append(f"-DSGL_ROCM_WARP_SIZE={_rocm_warp}")
cxx_flags.append(f"-DSGL_ROCM_WARP_SIZE={_rocm_warp}")

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
