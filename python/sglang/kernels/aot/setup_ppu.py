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
from pathlib import Path

import torch
from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
third_party = Path(
    os.environ.get("SGL_KERNEL_THIRD_PARTY_DIR", root / "build" / "_deps")
)
arch = platform.machine().lower()

# ======================= CUDA Version Detection ======================= #
cuda_version = torch.version.cuda
if cuda_version is None:
    raise RuntimeError("CUDA is not available in the current PyTorch installation.")

cuda_version_tuple = tuple(int(x) for x in cuda_version.split(".")[:2])
print(f"Detected CUDA version: {cuda_version} ({cuda_version_tuple})")


# ======================= Third-Party Repository Info ======================= #
class _RepoInfo:
    """Configuration for a third-party git repository."""

    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_CUTLASS_REPO = _RepoInfo(
    name="cutlass",
    git_repository="https://github.com/NVIDIA/cutlass",
    git_tag="57e3cfb47a2d9e0d46eb6335c3dc411498efa198",
    git_shallow=False,
)

_CUTLASS_DIR = Path(os.environ.get("SGL_KERNEL_CUTLASS_DIR", _CUTLASS_REPO.source_dir))

_FLASHINFER_REPO = _RepoInfo(
    name="flashinfer",
    git_repository="https://github.com/flashinfer-ai/flashinfer.git",
    git_tag="bc29697ba20b7e6bdb728ded98f04788e16ee021",
    git_shallow=False,
)

_TRITON_REPO = _RepoInfo(
    name="triton",
    git_repository="https://github.com/triton-lang/triton",
    git_tag="v3.7.1",
    git_shallow=False,
)

ALL_REPOS = [
    _CUTLASS_REPO,
    _FLASHINFER_REPO,
    _TRITON_REPO,
]


# ======================= Version ======================= #
def _get_version():
    with open(root / "pyproject_ppu.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')
    return "0.0.0"


# ======================= Build Configuration ======================= #
operator_namespace = "sgl-kernel"

# Determine CXX ABI compatibility with PyTorch
torch_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}"

enable_bf16 = os.environ.get("SGL_KERNEL_ENABLE_BF16", "1").lower() in (
    "1",
    "true",
    "on",
    "yes",
)
enable_fp8 = os.environ.get("SGL_KERNEL_ENABLE_FP8", "1").lower() in (
    "1",
    "true",
    "on",
    "yes",
)
compile_threads = int(os.environ.get("SGL_KERNEL_COMPILE_THREADS", "32"))
if compile_threads < 1:
    compile_threads = 1

# ======================= NVCC Flags ======================= #
sgl_kernel_cuda_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-std=c++17",
    "-DFLASHINFER_ENABLE_F16",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    f"--threads={compile_threads}",
    # Suppress warnings
    "-Xcompiler=-Wno-clang-format-violations",
    "-Xcompiler=-Wno-conversion",
    "-Xcompiler=-Wno-deprecated-declarations",
    "-Xcompiler=-Wno-terminate",
    "-Xcompiler=-Wfatal-errors",
    "-Xcompiler=-ftemplate-backtrace-limit=1",
    "-Xcudafe=--diag_suppress=177",
    "-Xcudafe=--diag_suppress=2361",
    f"-Xcompiler={abi_flag}",
    abi_flag,
]

if arch == "aarch64":
    sgl_kernel_cuda_flags.append("-gencode=arch=compute_87,code=sm_87")

if enable_bf16:
    sgl_kernel_cuda_flags.append("-DFLASHINFER_ENABLE_BF16")

if enable_fp8:
    sgl_kernel_cuda_flags.extend(
        [
            "-DFLASHINFER_ENABLE_FP8",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
            "-DFLASHINFER_ENABLE_FP8_E5M2",
        ]
    )

# ======================= Include Directories ======================= #
include_dirs = [
    str(root / "include"),
    str(root / "csrc"),
    str(_CUTLASS_DIR / "include"),
    str(_CUTLASS_DIR / "tools" / "util" / "include"),
    str(_FLASHINFER_REPO.source_dir / "include"),
    str(_FLASHINFER_REPO.source_dir / "csrc"),
    str(_CUTLASS_DIR / "examples" / "77_blackwell_fmha"),
    str(_CUTLASS_DIR / "examples" / "common"),
]

# ======================= Source Files ======================= #
common_sources = [
    "csrc/allreduce/custom_all_reduce.cu",
    "csrc/attention/merge_attn_states.cu",
    "csrc/common_extension_ppu.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/concat_mla.cu",
    "csrc/elementwise/copy.cu",
    "csrc/elementwise/fused_add_rms_norm_kernel.cu",
    "csrc/elementwise/pos_enc.cu",
    "csrc/elementwise/topk.cu",
    "csrc/gemm/per_token_group_quant_8bit.cu",
    "csrc/gemm/per_token_group_quant_8bit_v2.cu",
    "csrc/gemm/per_token_quant_fp8.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/mamba/causal_conv1d.cu",
    "csrc/memory/weak_ref_tensor.cpp",
    "csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/fused_qknorm_rope_kernel.cu",
    "csrc/moe/moe_sum.cu",
    "csrc/moe/moe_sum_reduce.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/moe/prepare_moe_input.cu",
    "csrc/quantization/gguf/gguf_kernel.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/speculative/ngram_utils.cu",
    "csrc/speculative/packbit.cu",
    "csrc/speculative/speculative_sampling.cu",
]

# Third-party source files
flashinfer_sources = [
    str(_FLASHINFER_REPO.source_dir / "csrc" / "norm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc" / "renorm.cu"),
]

all_common_sources = common_sources + flashinfer_sources

# Libraries to link
libraries = ["c10", "cuda", "cublas", "cublasLt"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib"]


# ======================= Extension Modules ======================= #
def _make_common_ops():
    """common_ops for SM80/SM89."""
    nvcc_flags = sgl_kernel_cuda_flags + ["-use_fast_math"]
    return CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=all_common_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": ["-O3", "-std=c++17", abi_flag],
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )


def _make_spatial_ops():
    """Spatial ops extension for green contexts."""
    spatial_sources = [
        "csrc/spatial/greenctx_stream.cu",
        "csrc/spatial_extension.cc",
    ]

    return CUDAExtension(
        name="sgl_kernel.spatial_ops",
        sources=spatial_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": list(sgl_kernel_cuda_flags),
            "cxx": ["-O3", "-std=c++17", abi_flag],
        },
        libraries=["c10", "cuda"],
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )


# ======================= Clone Third-Party Repositories ======================= #
def _clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
    """Clone a git repository and checkout a specific tag/commit."""
    repo_path = Path(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if not repo_path.exists():
        clone_cmd = ["git", "clone"]
        if git_shallow:
            clone_cmd += ["--depth", "1"]
        clone_cmd += [repo_url, str(repo_path)]
        print(f"Cloning {repo_url} -> {repo_path}")
        subprocess.check_call(clone_cmd)
        subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)


if os.environ.get("SGL_KERNEL_SKIP_THIRD_PARTY", "0") != "1":
    for _repo in ALL_REPOS:
        _clone_and_checkout(
            _repo.source_dir,
            _repo.git_repository,
            _repo.git_tag,
            _repo.git_shallow,
        )


# ======================= Build Extension Modules List ======================= #
ext_modules = [
    _make_common_ops(),
    _make_spatial_ops(),
]


# ======================= Custom Build Extension ======================= #
class _CustomBuildExt(BuildExtension):
    """Custom build extension with ninja support."""

    pass


# ======================= Package Discovery ======================= #
_triton_kernels_root = _TRITON_REPO.source_dir / "python" / "triton_kernels"

_packages = find_namespace_packages(where="python")
_package_dir = {"": "python"}

# Discover triton_kernels packages (may not exist before first clone)
if _triton_kernels_root.exists():
    _triton_pkgs = find_namespace_packages(
        where=str(_triton_kernels_root),
        exclude=["tests", "tests.*"],
    )
else:
    _triton_pkgs = ["triton_kernels"]

for pkg in _triton_pkgs:
    _package_dir[pkg] = os.path.relpath(
        _triton_kernels_root / pkg.replace(".", os.sep), root
    )
_packages += _triton_pkgs


# ======================= Setup ======================= #
setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=_packages,
    package_dir=_package_dir,
    ext_modules=ext_modules,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
