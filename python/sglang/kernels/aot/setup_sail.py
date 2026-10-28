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

import atexit
import os
import platform
import shutil
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


_ACTLIZE_REPO = _RepoInfo(
    name="actlize",
    git_repository="https://github.com/t-head/actlize",
    git_tag="129651181ed29ec3d3e61df786f610926b739a9b",
    git_shallow=False,
)

_ACTLIZE_DIR = Path(os.environ.get("SGL_KERNEL_ACTLIZE_DIR", _ACTLIZE_REPO.source_dir))

_FLASHINFER_REPO = _RepoInfo(
    name="flashinfer",
    git_repository="https://github.com/flashinfer-ai/flashinfer.git",
    git_tag="bc29697ba20b7e6bdb728ded98f04788e16ee021",
    git_shallow=False,
)

_TRITON_REPO = _RepoInfo(
    name="triton",
    git_repository="https://github.com/triton-lang/triton",
    git_tag="v3.7.0",
    git_shallow=False,
)

ALL_REPOS = [
    _ACTLIZE_REPO,
    _FLASHINFER_REPO,
    _TRITON_REPO,
]


# ======================= Version ======================= #
def _get_version():
    with open(root / "pyproject_sail.toml") as f:
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

# ======================= hgcc (device compiler) Flags ======================= #
# No `-gencode`: hgcc takes `--gpu-architecture=ppu_XX` instead, and SAIL torch
# emits that itself from the PYTORCH_SAIL_ARCH environment variable (see
# torch.utils.cpp_extension._get_cuda_arch_flags). Passing a CUDA `-gencode` here
# makes hgcc fail outright, so the arch selection is left entirely to
# PYTORCH_SAIL_ARCH (`ppu_10`, `ppu_15`, or a semicolon-separated list).
sgl_kernel_cuda_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
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
    "-U__HGGC_NO_HALF_OPERATORS__",
    "-U__HGGC_NO_HALF_CONVERSIONS__",
    "-U__HGGC_NO_HALF2_OPERATORS__",
    "-U__HGGC_NO_BFLOAT16_CONVERSIONS__",
    f"--threads={compile_threads}",
    # Suppress warnings
    "-Xcompiler=-Wno-clang-format-violations",
    "-Xcompiler=-Wno-conversion",
    "-Xcompiler=-Wno-deprecated-declarations",
    "-Xcompiler=-Wno-terminate",
    "-Xcompiler=-Wfatal-errors",
    "-Xcompiler=-ftemplate-backtrace-limit=1",
    # `-Xcudafe` is an nvcc-only hand-off to cudafe++, which hgcc has no
    # counterpart for and rejects. The diagnostics it used to suppress (177
    # unused-variable, 2361) are nvcc frontend numbers with no hgcc analogue.
    f"-Xcompiler={abi_flag}",
    abi_flag,
]

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
# Upstream (non-sailified) include directories.  When sailify staging is
# active (see the "Sailify Source Staging" section below), the staged mirror
# directories are prepended so that converted headers shadow their upstream
# originals; the entries below then only act as a fallback for headers that
# were not part of the conversion closure.
base_include_dirs = [
    str(root / "include"),
    str(root / "csrc"),
    str(_ACTLIZE_DIR / "include"),
    str(_ACTLIZE_DIR / "tools" / "util" / "include"),
    str(_FLASHINFER_REPO.source_dir / "include"),
    str(_FLASHINFER_REPO.source_dir / "csrc"),
    str(_ACTLIZE_DIR / "examples" / "77_blackwell_fmha"),
    str(_ACTLIZE_DIR / "examples" / "common"),
]

# ======================= Source Files ======================= #
common_sources = [
    "csrc/allreduce/custom_all_reduce.cu",
    "csrc/attention/merge_attn_states.cu",
    "csrc/common_extension_sail.cc",
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
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/fused_qknorm_rope_kernel.cu",
    "csrc/moe/moe_sum.cu",
    "csrc/moe/moe_sum_reduce.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/moe/prepare_moe_input.cu",
    "csrc/quantization/gguf/gguf_kernel.cu",
]

# Third-party source files
flashinfer_sources = [
    str(_FLASHINFER_REPO.source_dir / "csrc" / "norm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc" / "renorm.cu"),
]

all_common_sources = common_sources + flashinfer_sources

# Libraries to link.
#
# The cuda_free PPU SDK ships no libcuda/libcublas/libcublasLt; the sailified
# sources call the native spellings instead, so link their PPU counterparts:
#   cudart  -> hggcrt1   (libhggcrt1.so -> libhggcrt.13.0.so; there is no
#                         unversioned libhggcrt.so to satisfy a plain -lhggcrt)
#   cublas  -> acblas    (PPU counterpart of cublas)
#   cublasLt-> acblasLt
# The CUDA driver API (libcuda) is dropped: no csrc translation unit calls it.
# `library_paths()` from SAIL torch already contributes $PPU_SDK/lib, so no
# explicit library_dirs is needed.
libraries = ["c10", "hggcrt1", "hggc", "acblas", "acblasLt"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib"]


# ======================= Extension Modules ======================= #
def _make_common_ops():
    """common_ops for SM80/SM89."""
    nvcc_flags = sgl_kernel_cuda_flags + ["-use_fast_math"]
    return CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=staged_common_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": ["-O3", "-std=c++17", abi_flag],
        },
        libraries=libraries,
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


# ======================= Sailify Source Staging ======================= #
# The upstream sources (sglang aot + flashinfer) target CUDA; the cuda_free
# PPU SDK needs their hggc equivalents.  Rather than modifying the upstream
# trees, every compiled source file -- plus every header they recursively
# #include -- is converted by sailify into a mirror tree under
# build/sailified/, and the extensions compile exclusively from that tree:
#
#   build/sailified/.ppu_compat/             compat layer (generated)
#   build/sailified/csrc/...                 aot sources + headers
#   build/sailified/include/...              aot headers
#   build/sailified/flashinfer/csrc/...      flashinfer sources + headers
#   build/sailified/flashinfer/include/...   flashinfer headers
#
# The staging tree is wiped and rebuilt on every invocation so no stale
# output survives an upstream or sailify change.

sailify_staging_dir = root / "build" / "sailified"

# Sailify toggle.
#
# SGL_KERNEL_USE_SAILIFY (default "1") enables the automatic cuda->hggc
# staging conversion described below; "0" skips staging and compiles the
# upstream sources directly, which requires a pre-sailified tree
# (root/.ppu_compat, see the skip branch).  The legacy
# SGL_KERNEL_SKIP_SAILIFY variable is kept for backwards compatibility and
# takes precedence when explicitly set.
if "SGL_KERNEL_SKIP_SAILIFY" in os.environ:
    skip_sailify = os.environ["SGL_KERNEL_SKIP_SAILIFY"] == "1"
else:
    skip_sailify = os.environ.get("SGL_KERNEL_USE_SAILIFY", "1") != "1"

# Targeted fixes applied to the *staged* copies after the generic sailify
# pass, for CUDA constructs the identifier-level mapping cannot express.
# The -U__HGGC_NO_HALF* flags in sgl_kernel_cuda_flags disable
# constructor-style half/bfloat16 conversions, so numeric casts must use the
# functional __float2half / __float2bfloat16 API instead.
_SAILIFY_PATCHES = {
    "csrc/moe/moe_sum_reduce.cu": [
        ("__half2float(__half(", "__half2float(__float2half("),
        ("__bfloat162float(__ppu_bfloat16(", "__bfloat162float(__float2bfloat16("),
    ],
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu": [
        ("-CUDART_INF_FP16", "__float2half(-INFINITY)"),
        ("-CUDART_INF_BF16", "__float2bfloat16(-INFINITY)"),
        ("__ppu_bfloat16(-INFINITY)", "__float2bfloat16(-INFINITY)"),
    ],
}


def _sailify_stage(tree_roots, source_files):
    """Convert *source_files* and their #include closure into the staging tree.

    Args:
        tree_roots: {upstream_root: staged_subdir} -- each file under an
            upstream root is written to
            sailified_dir/staged_subdir/<relpath from the root>.
        source_files: absolute paths of the .cu/.cc/.cpp files to convert.

    Returns:
        List of staged paths corresponding 1:1 to *source_files*.
    """
    try:
        from sailify.sailify_python import (
            _discover_includes,
            _ensure_ppu_compat,
            preprocessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "sailify is required for SAIL builds (pip install -e /path/to/sailify)"
        ) from exc

    staging = str(sailify_staging_dir)
    if sailify_staging_dir.exists():
        shutil.rmtree(sailify_staging_dir)
    sailify_staging_dir.mkdir(parents=True)

    # Longest root first so nested trees match the most specific entry.
    roots = sorted(
        ((os.path.realpath(r), sub) for r, sub in tree_roots.items()),
        key=lambda kv: -len(kv[0]),
    )

    def _staged_path(src_path):
        """Map an upstream file to (staged_path, project_directory anchor).

        The anchor is the directory that preprocessor must receive as
        project_directory so that output_directory + relpath(file, anchor)
        equals staged_path.
        """
        src_path = os.path.realpath(src_path)
        for base, sub in roots:
            if src_path.startswith(base + os.sep):
                rel = os.path.relpath(src_path, base)
                staged_rel = os.path.join(sub, rel) if sub else rel
                return os.path.join(staging, staged_rel), (
                    os.path.dirname(base) if sub else base
                )
        return None

    # Directories used to resolve #include targets while walking the
    # dependency closure (mirrors the upstream -I layout).
    header_include_dirs = []
    for base, _sub in roots:
        for name in ("include", "csrc"):
            d = os.path.join(base, name)
            if os.path.isdir(d) and d not in header_include_dirs:
                header_include_dirs.append(d)

    # BFS over the compiled sources and their #include closure.  Headers
    # outside the managed trees (system / torch / actlize SDK headers) are
    # resolved by the discovery walk but intentionally left unconverted.
    visited = {os.path.realpath(f) for f in source_files}
    pending = list(visited)
    stats = {}
    failures = []
    while pending:
        src_path = pending.pop(0)
        _discover_includes(src_path, header_include_dirs, visited, pending)
        mapped = _staged_path(src_path)
        if mapped is None:
            continue  # header outside the managed trees: used as-is
        staged_path, anchor = mapped
        result = preprocessor(
            output_directory=staging,
            filepath=src_path,
            all_files=[],
            stats=stats,
            clean_ctx=None,
            show_progress=False,
            project_directory=anchor,
        )
        if result.status != "ok":
            failures.append(f"{src_path}: {result.status}")
        elif not os.path.isfile(staged_path):
            failures.append(f"{src_path}: staged output {staged_path} missing")

    if failures:
        raise RuntimeError("sailify staging failed:\n  " + "\n  ".join(failures))

    # Regenerate the compat layer inside the staging tree.
    _ensure_ppu_compat(staging)

    # Apply targeted post-sailify patches to the staged copies only; the
    # upstream trees are never touched.
    for rel_path, replacements in _SAILIFY_PATCHES.items():
        staged = sailify_staging_dir / rel_path
        if not staged.is_file():
            print(f"sailify: patch target missing (skipped): {rel_path}")
            continue
        text = staged.read_text()
        for old, new in replacements:
            text = text.replace(old, new)
        staged.write_text(text)

    n_headers = max(len(stats.get("ppuified", [])) - len(source_files), 0)
    print(
        f"sailify: staged {len(source_files)} sources (+{n_headers} discovered "
        f"headers) into {sailify_staging_dir}"
    )

    return [_staged_path(f)[0] for f in source_files]


if skip_sailify:
    # Compile the upstream sources as-is.  The target toolchain is not
    # necessarily the cuda_free PPU SDK (this may be a plain CUDA build), so
    # no fail-fast on .ppu_compat: if it exists (pre-generated for the PPU
    # cuda_free toolchain) it is prepended to shadow the CUDA headers, and
    # otherwise the upstream include layout is used unchanged.
    print("WARNING: sailify disabled, compiling upstream sources directly")
    staged_common_sources = [str(root / s) for s in common_sources] + flashinfer_sources
    ppu_compat_dir = root / ".ppu_compat"
    if ppu_compat_dir.is_dir():
        print(f"note: using existing compat layer {ppu_compat_dir}")
        include_dirs = [str(ppu_compat_dir)] + base_include_dirs
    else:
        include_dirs = list(base_include_dirs)
else:
    _staged = _sailify_stage(
        {
            root: "",  # aot tree mirrors directly onto the staging root
            _FLASHINFER_REPO.source_dir: "flashinfer",
        },
        [str(root / s) for s in common_sources] + flashinfer_sources,
    )
    staged_common_sources = _staged
    # Staged directories shadow their upstream originals.  .ppu_compat MUST
    # stay first: SAIL torch injects its own (possibly older) `.ppu_compat`
    # through torch.utils.cpp_extension.include_paths(), which lands after
    # the user include_dirs.  Both copies contain compatible_wrapper.h, and
    # SAIL torch also passes `-include compatible_wrapper.h` on the hgcc
    # command line -- the compiler resolves that through -I order, so a
    # stale compat layer shadowing ours breaks the build with e.g.
    # "no template named 'maximum' in namespace 'hggc'".
    include_dirs = [
        str(sailify_staging_dir / ".ppu_compat"),
        str(sailify_staging_dir / "include"),
        str(sailify_staging_dir / "csrc"),
        str(sailify_staging_dir / "flashinfer" / "include"),
        str(sailify_staging_dir / "flashinfer" / "csrc"),
    ] + base_include_dirs

    # ======================= Post-Build Cleanup ======================= #
    # The build itself consumes the compat layer from
    # build/sailified/.ppu_compat (first -I entry above), but tooling on the
    # SAIL torch side may additionally drop a .ppu_compat directory into the
    # aot tree (it shows up on the hgcc -I line).  It is not part of the
    # upstream sources, so remove it after every compilation / packaging step
    # has finished.  In skip-sailify mode root/.ppu_compat is an optional
    # user-provided compat layer and must NOT be removed either (hence no
    # registration there).
    def _cleanup_upstream_ppu_compat():
        upstream_compat = root / ".ppu_compat"
        if upstream_compat.is_dir():
            shutil.rmtree(upstream_compat, ignore_errors=True)
            print(f"setup_sail: removed {upstream_compat} (post-build cleanup)")

    atexit.register(_cleanup_upstream_ppu_compat)


# ======================= Build Extension Modules List ======================= #
ext_modules = [
    _make_common_ops(),
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
