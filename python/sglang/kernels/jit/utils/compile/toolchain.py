"""The toolchain a JIT build runs on: compilers, tvm-ffi's headers, base flags.

sglang generates its own ``build.ninja`` rather than going through
``tvm_ffi.cpp.load_inline``, so the flags tvm-ffi used to supply implicitly have
to be stated here. That is the point: every flag that reaches the compiler is
now visible in one place and therefore hashable into the build key, instead of
living inside a dependency whose defaults we could only approximate by version
number.

Only tvm-ffi's *locations* are still consumed — its headers, its shared library,
and ``tvm_ffi.load_module`` for loading the result.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
from typing import List, Tuple

import torch

from sglang.kernels.jit.utils.arch import get_jit_cuda_arch
from sglang.kernels.jit.utils.common import cache_once, is_hip_runtime

logger = logging.getLogger(__name__)


@cache_once
def cuda_home() -> str:
    """CUDA install root, resolved the way tvm-ffi resolves it.

    `arch._jit_cuda_version` resolves nvcc the same way for its own purposes;
    the two must stay in agreement, since one picks the target and the other
    compiles for it.
    """
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured is not None:
        return configured
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))
    return "/usr/local/cuda"


@cache_once
def rocm_home() -> str:
    """ROCm install root, resolved the way tvm-ffi resolves it."""
    return os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm"


@cache_once
def device_compiler_path() -> str:
    """The nvcc/hipcc that JIT builds actually invoke.

    Resolved the same way tvm-ffi resolves it, so the binary the cache
    fingerprints is the binary that does the compiling.
    """
    if is_hip_runtime():
        return os.path.join(rocm_home(), "bin", "hipcc")
    return os.path.join(cuda_home(), "bin", "nvcc")


@cache_once
def host_compiler_path() -> str:
    """The C++ compiler host code is handed to.

    nvcc dispatches all host code to it, so its version decides both which
    system headers are pulled in and how that half is codegen'd.
    """
    return os.environ.get("CXX", "c++")


@cache_once
def gpu_arch_name() -> str:
    """The compile target as the vendor names it.

    On ROCm this is ``gcnArchName`` (``gfx942:sramecc+:xnack-``) rather than the
    CUDA-shaped ``(major, minor)`` capability: the latter maps gfx940/gfx941/
    gfx942 onto a single ``9.4``, which are three different compile targets.
    """
    if not is_hip_runtime():
        return get_jit_cuda_arch().target_name
    try:
        device = torch.cuda.current_device()
        return str(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        logger.warning("Cannot detect ROCm gcnArchName; the JIT cache target degrades.")
        return "unknown"


@cache_once
def toolkit_home() -> pathlib.Path:
    """The CUDA/ROCm root, derived from the compiler already resolved."""
    return pathlib.Path(device_compiler_path()).parent.parent


@cache_once
def tvm_ffi_paths() -> Tuple[Tuple[str, ...], str, str]:
    """``(include dirs, library dir, library name)`` for linking against tvm-ffi."""
    from tvm_ffi.libinfo import (
        find_dlpack_include_path,
        find_include_path,
        find_libtvm_ffi,
    )

    lib = pathlib.Path(find_libtvm_ffi())
    includes = tuple(dict.fromkeys([find_include_path(), find_dlpack_include_path()]))
    return includes, str(lib.parent), lib.stem.removeprefix("lib")


def target_flags() -> List[str]:
    """The device flags that pin the build to this GPU.

    Emitted from the architecture sglang already detected, rather than left to
    the compiler driver to probe: the value is part of the cache key, so it has
    to be decided here and not rediscovered at build time.
    """
    if is_hip_runtime():
        return [f"--offload-arch={gpu_arch_name()}"]
    arch = get_jit_cuda_arch()
    target = f"{arch.major}{arch.minor}{arch.suffix}"
    return [f"-gencode=arch=compute_{target},code=sm_{target}"]


def base_cxx_flags() -> List[str]:
    """Only what the platform requires; `-std`/`-O` arrive with the spec.

    Kept disjoint from ``arch.get_default_target_flags`` on purpose — supplying
    `-std=c++20` from both is what used to make nvcc warn about an incompatible
    redefinition on every single build.
    """
    return ["-fPIC"]


def base_cuda_flags() -> List[str]:
    if is_hip_runtime():
        return ["-fPIC", "-D__HIP_PLATFORM_AMD__=1", "-fno-gpu-rdc"]
    return ["-Xcompiler", "-fPIC"]


def base_include_paths() -> List[str]:
    includes, _, _ = tvm_ffi_paths()
    if is_hip_runtime():
        return [*includes, f"{rocm_home()}/include"]
    return list(includes)


def base_link_flags(*, with_device: bool) -> List[str]:
    """Link flags for a module, with the GPU runtime only when it has device code.

    A module built purely from ``.cpp`` sources must not drag in libcudart: CPU
    runners have no CUDA toolkit to link it from, and the module never calls it.
    tvm-ffi keyed this off the presence of ``.cu`` sources for the same reason.
    """
    _, lib_dir, lib_name = tvm_ffi_paths()
    flags = ["-shared", f"-L{lib_dir}", f"-l{lib_name}"]
    if not with_device:
        return flags
    if is_hip_runtime():
        return flags + [f"-L{rocm_home()}/lib", "-lamdhip64"]
    return flags + [f"-L{cuda_home()}/lib64", "-lcudart"]


def compilers() -> Tuple[str, str]:
    """``(host compiler, device compiler)`` as they will appear in build.ninja."""
    return host_compiler_path(), device_compiler_path()
