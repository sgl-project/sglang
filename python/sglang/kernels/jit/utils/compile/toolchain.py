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

import importlib.util
import logging
import os
import pathlib
import re
import shutil
import subprocess
from typing import List, Tuple

import torch

from sglang.kernels.jit.utils.arch import get_jit_cuda_arch
from sglang.kernels.jit.utils.common import cache_once, is_hip_runtime

logger = logging.getLogger(__name__)


def _declares_c23_rsqrt() -> bool:
    """True when the host libc declares C23 ``rsqrt``/``rsqrtf``.

    glibc >= 2.41 declares these as ``noexcept``. CUDA toolkits older than
    13.2 declare them with no exception specifier, and nvcc then rejects every
    translation unit that reaches ``<cmath>``:

        bits/mathcalls.h: error: exception specification is incompatible with
        that of previous function "rsqrt" (declared in crt/math_functions.h)

    which takes out every JIT kernel build on the host. Probed from the header
    rather than a glibc version so the check tracks the actual declaration.
    """
    for header in (
        "/usr/include/x86_64-linux-gnu/bits/mathcalls.h",
        "/usr/include/bits/mathcalls.h",
    ):
        try:
            with open(header, "r") as fh:
                return "(rsqrt," in fh.read()
        except OSError:
            continue
    return False


def _rsqrt_safe(home: str) -> bool:
    """True when ``home``'s headers can be compiled against this host's libc.

    CUDA 13.2 guards the clashing declarations behind ``_NV_RSQRT_SPECIFIER``;
    toolkits without that macro cannot build here once the libc declares C23
    ``rsqrt``. A toolkit whose header is unreadable is assumed fine, so an
    unfamiliar layout degrades to today's behaviour instead of being skipped.
    """
    if not _declares_c23_rsqrt():
        return True
    header = os.path.join(
        home, "targets", "x86_64-linux", "include", "crt", "math_functions.h"
    )
    if not os.path.exists(header):
        header = os.path.join(home, "include", "crt", "math_functions.h")
    try:
        with open(header, "r") as fh:
            return "_NV_RSQRT_SPECIFIER" in fh.read()
    except OSError:
        return True


def _pip_cuda_home() -> str | None:
    """Root of a pip-installed CUDA toolkit, if one shipped with the wheels."""
    for mod in ("nvidia.cu13", "nvidia.cu12"):
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        root = spec.submodule_search_locations[0]
        if os.path.exists(os.path.join(root, "bin", "nvcc")):
            return root
    return None


@cache_once
def cuda_home() -> str:
    """CUDA install root, resolved the way tvm-ffi resolves it.

    `arch._jit_cuda_version` resolves nvcc the same way for its own purposes;
    the two must stay in agreement, since one picks the target and the other
    compiles for it.

    One departure from tvm-ffi: if the toolkit we would otherwise pick cannot
    compile against this host's libc (see ``_rsqrt_safe``), fall back to a
    pip-installed toolkit that can. Without this, hosts with glibc >= 2.41 and
    a system CUDA < 13.2 -- Ubuntu 26.04 being the common case -- fail every
    JIT build with an error that names neither the toolkit nor the fix.
    """
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured is not None:
        return configured

    nvcc_path = shutil.which("nvcc")
    primary = (
        os.path.dirname(os.path.dirname(nvcc_path))
        if nvcc_path is not None
        else "/usr/local/cuda"
    )
    if _rsqrt_safe(primary):
        return primary

    fallback = _pip_cuda_home()
    if fallback is not None and _rsqrt_safe(fallback):
        logger.warning(
            "CUDA toolkit at %s cannot compile against this host's libc: it "
            "predates the _NV_RSQRT_SPECIFIER guard (CUDA 13.2) while glibc "
            "declares C23 rsqrt/rsqrtf as noexcept. Using the pip-installed "
            "toolkit at %s instead. Set CUDA_HOME to override.",
            primary,
            fallback,
        )
        return fallback

    logger.warning(
        "CUDA toolkit at %s predates the _NV_RSQRT_SPECIFIER guard (CUDA 13.2) "
        "while this host's glibc declares C23 rsqrt/rsqrtf as noexcept, so JIT "
        "builds are expected to fail with 'exception specification is "
        "incompatible with that of previous function rsqrt'. Install CUDA "
        ">= 13.2 and point CUDA_HOME at it.",
        primary,
    )
    return primary


@cache_once
def cuda_stubs_dir() -> str:
    """Directory holding the driver stub libraries (`libcuda.so`).

    A module that calls the driver API links `-lcuda` against this stub; the
    real `libcuda.so.1` is supplied by the installed driver at load time. Only
    the stub ships with the toolkit, so this path -- not a plain `-lcuda` -- is
    what makes the link resolve.
    """
    return os.path.join(cuda_home(), "lib64", "stubs")


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
    nvcc = os.path.join(cuda_home(), "bin", "nvcc")
    _warn_on_toolkit_version_skew(nvcc)
    return nvcc


def _header_cudart_version(home: str) -> Tuple[int, int] | None:
    """``(major, minor)`` from ``CUDART_VERSION`` in the toolkit's headers."""
    for rel in (
        ("targets", "x86_64-linux", "include", "cuda_runtime_api.h"),
        ("include", "cuda_runtime_api.h"),
    ):
        path = os.path.join(home, *rel)
        try:
            with open(path, "r") as fh:
                for line in fh:
                    if line.startswith("#define CUDART_VERSION"):
                        value = int(line.split()[2])
                        return value // 1000, (value % 1000) // 10
        except (OSError, IndexError, ValueError):
            continue
    return None


@cache_once
def _warn_on_toolkit_version_skew(nvcc: str) -> None:
    """Warn when nvcc and the CUDA headers beside it are different releases.

    The pip CUDA distribution is a set of independently versioned wheels that
    unpack into one tree, so ``nvidia-cuda-nvcc`` and ``nvidia-cuda-runtime``
    can disagree. CCCL asserts the two match and aborts the build with

        "CUDA compiler and CUDA toolkit headers are incompatible, please check
        your include paths"

    which points at include paths rather than at the package versions that
    actually differ. Only a warning: mixed minor versions are usually fine for
    builds that do not pull in CCCL.
    """
    header_version = _header_cudart_version(cuda_home())
    if header_version is None:
        return
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    match = re.search(r"release (\d+)\.(\d+)", out)
    if match is None:
        return
    nvcc_version = (int(match.group(1)), int(match.group(2)))
    if nvcc_version == header_version:
        return
    logger.warning(
        "CUDA toolkit at %s is inconsistent: nvcc is %d.%d but the CUDA runtime "
        "headers are %d.%d. CCCL-based builds (flashinfer) reject this with "
        "'CUDA compiler and CUDA toolkit headers are incompatible'. If these "
        "came from pip, align them, e.g. "
        "`pip install nvidia-cuda-runtime==%d.%d.*`.",
        cuda_home(),
        *nvcc_version,
        *header_version,
        *nvcc_version,
    )


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
    return flags + [f"-L{cuda_lib_dir()}", _cudart_link_flag()]


@cache_once
def cuda_lib_dir() -> str:
    """Directory holding the CUDA runtime libraries under ``cuda_home()``.

    A system toolkit keeps them in ``lib64``; the pip wheels use ``lib``.
    Hardcoding ``lib64`` makes the link step search a directory that does not
    exist on a pip-only install, so resolve it instead.
    """
    home = cuda_home()
    for name in ("lib64", "lib"):
        candidate = os.path.join(home, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(home, "lib64")


@cache_once
def _cudart_link_flag() -> str:
    """``-lcudart``, or a direct soname reference when the dev symlink is absent.

    The pip CUDA wheels are runtime-only: they ship ``libcudart.so.13`` but no
    unversioned ``libcudart.so``, so ``-lcudart`` fails with "cannot find
    -lcudart" even though the library is right there. ld's ``-l:`` form takes
    an exact filename, which lets a pip-only toolkit link without us having to
    write symlinks into someone's site-packages.
    """
    lib_dir = cuda_lib_dir()
    if os.path.exists(os.path.join(lib_dir, "libcudart.so")):
        return "-lcudart"
    try:
        versioned = sorted(
            name for name in os.listdir(lib_dir) if name.startswith("libcudart.so.")
        )
    except OSError:
        return "-lcudart"
    if versioned:
        return f"-l:{versioned[-1]}"
    return "-lcudart"


def compilers() -> Tuple[str, str]:
    """``(host compiler, device compiler)`` as they will appear in build.ninja."""
    return host_compiler_path(), device_compiler_path()
