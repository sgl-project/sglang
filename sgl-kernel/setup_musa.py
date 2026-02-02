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
import subprocess
import sys
from pathlib import Path

# isort: off
import torch
import torchada  # noqa: F401

# isort: on
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
third_party = Path(os.environ.get("SGLANG_MUSA_THIRD_PARTY_DIR", "build/_deps"))
arch = platform.machine().lower()


class _RepoInfo:
    """Configuration for a third-party git repository."""

    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_FLASHINFER_REPO = _RepoInfo(
    name="flashinfer",
    git_repository="https://github.com/flashinfer-ai/flashinfer.git",
    git_tag="bc29697ba20b7e6bdb728ded98f04788e16ee021",
    git_shallow=False,
)

_MUTLASS_REPO = _RepoInfo(
    name="mutlass",
    git_repository="https://github.com/MooreThreads/mutlass.git",
    git_tag="3abd6a728aacd190df0d922514aca8a8bc3c46b7",
    git_shallow=False,
)


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
    root / _FLASHINFER_REPO.source_dir / "include",
    root / _FLASHINFER_REPO.source_dir / "csrc",
    root / _MUTLASS_REPO.source_dir / "include",
]

sources = [
    "csrc/allreduce/custom_all_reduce.cu",
    "csrc/attention/merge_attn_states.cu",
    "csrc/common_extension_musa.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/concat_mla.cu",
    "csrc/elementwise/fused_add_rms_norm_kernel.mu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_fused_gate_musa.cu",
    "csrc/moe/kimi_k2_moe_fused_gate.cu",
    "csrc/moe/moe_sum.cu",
    "csrc/moe/moe_sum_reduce.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/quantization/gguf/gguf_kernel.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/speculative/ngram_utils.cu",
    "csrc/speculative/packbit.cu",
    "csrc/speculative/speculative_sampling.cu",
    "csrc/memory/store.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/gemm/awq_kernel.cu",
    "csrc/gemm/bmm_fp8.cu",
    "csrc/gemm/dsv3_fused_a_gemm.cu",
    "csrc/gemm/dsv3_router_gemm_bf16_out.cu",
    "csrc/gemm/dsv3_router_gemm_entry.cu",
    "csrc/gemm/dsv3_router_gemm_float_out.cu",
    "csrc/gemm/per_token_quant_fp8.cu",
    "csrc/gemm/per_tensor_quant_fp8.cu",
    "csrc/gemm/per_token_group_quant_8bit.cu",
    "csrc/gemm/per_token_group_quant_8bit_v2.cu",
    str(_FLASHINFER_REPO.source_dir / "csrc/norm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc/renorm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc/sampling.cu"),
]

cxx_flags = ["force_mcc"]
libraries = ["c10", "torch", "torch_python"]
extra_link_args = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{arch}-linux-gnu",
    "-lmublasLt",
]

default_target = "mp_31"
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
    "-x",
    "musa",
    "-mtgpu",
    "-Od3",
    "-ffast-math",
    "-fmusa-flush-denormals-to-zero",
    "-fno-strict-aliasing",
    "-DUSE_MUSA",
    "-DENABLE_BF16",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
]

if mtgpu_target == "mp_31":
    mcc_flags.extend(
        [
            "-DENABLE_FP8",
            "-DFLASHINFER_ENABLE_FP8",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
            "-DFLASHINFER_ENABLE_FP8_E5M2",
        ]
    )

ext_modules = [
    CUDAExtension(
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


class _CustomBuildExt(BuildExtension):
    """Custom build extension that clones third-party repositories before building."""

    @staticmethod
    def _clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
        """Clone a git repository and checkout a specific tag/commit."""
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        if not repo_path.exists():
            clone_cmd = ["git", "clone"]
            if git_shallow:
                clone_cmd += ["--depth", "1"]
            clone_cmd += [repo_url, str(repo_path)]
            subprocess.check_call(clone_cmd)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
        else:
            subprocess.check_call(["git", "fetch", "--all"], cwd=repo_path)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)

    def run(self):
        if os.environ.get("SGLANG_MUSA_SKIP_THIRD_PARTY", "0") == "1":
            print(
                "Skipping third-party repositories cloning (SGLANG_MUSA_SKIP_THIRD_PARTY=1)"
            )
        else:
            print("Cloning third-party repositories...")
            self._clone_and_checkout(
                _MUTLASS_REPO.source_dir,
                _MUTLASS_REPO.git_repository,
                _MUTLASS_REPO.git_tag,
                _MUTLASS_REPO.git_shallow,
            )
            self._clone_and_checkout(
                _FLASHINFER_REPO.source_dir,
                _FLASHINFER_REPO.git_repository,
                _FLASHINFER_REPO.git_tag,
                _FLASHINFER_REPO.git_shallow,
            )
            print("Third-party repositories ready.")

        super().run()


setup(
    name="sgl-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
