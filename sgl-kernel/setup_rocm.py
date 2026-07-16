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
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

rocm_arch_config = {
    # MI300/MI325 use the FNUZ FP8 encoding and expose 64 KiB of LDS.
    "gfx942": {
        "fp8_macro": "-DHIP_FP8_TYPE_FNUZ",
        "topk_dynamic_smem_bytes": 48 * 1024,
    },
    # MI350 uses the OCP E4M3 encoding and has enough LDS for the full budget.
    "gfx950": {
        "fp8_macro": "-DHIP_FP8_TYPE_E4M3",
        "topk_dynamic_smem_bytes": 128 * 1024,
    },
    # RDNA4 supports 32/64-lane wavefronts and OCP E4M3. Keep the TopK budget
    # conservative so the kernel leaves room within its 128 KiB LDS limit.
    "gfx1201": {
        "fp8_macro": "-DHIP_FP8_TYPE_E4M3",
        "topk_dynamic_smem_bytes": 48 * 1024,
    },
}

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

if amdgpu_target not in rocm_arch_config:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Expected one of: {', '.join(rocm_arch_config)}."
    )
    sys.exit(1)

arch_config = rocm_arch_config[amdgpu_target]
fp8_macro = arch_config["fp8_macro"]
topk_dynamic_smem_bytes = arch_config["topk_dynamic_smem_bytes"]

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    f"--offload-arch={amdgpu_target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    fp8_macro,
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
]

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
