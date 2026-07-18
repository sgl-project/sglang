"""JIT compilation: load_jit, the build cache, and C++ template arguments."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import pathlib
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple, TypeAlias, Union

import torch

from sglang.kernels._jit.arch import get_default_target_flags, get_jit_cuda_arch
from sglang.kernels._jit.common import cache_once, is_hip_runtime
from sglang.kernels._jit.deps import REGISTERED_DEPENDENCIES

if TYPE_CHECKING:
    from tvm_ffi import Module

logger = logging.getLogger(__name__)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_ANGLE_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*<(sgl_kernel/[^>]+)>", re.MULTILINE)


def _local_jit_source_hash(source_files: List[str]) -> str:
    """Hash JIT source contents so TVM-FFI cache keys track included headers."""
    digest = hashlib.sha256()
    seen: set[pathlib.Path] = set()
    stack = [pathlib.Path(path).resolve() for path in source_files]
    include_dir = KERNEL_PATH / "include"

    while stack:
        path = stack.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)

        data = path.read_bytes()
        # Relative to kernel root, not absolute: the key must track source
        # content, not install location (differs across runners / job dirs).
        try:
            ident = str(path.relative_to(KERNEL_PATH))
        except ValueError:
            ident = path.name
        digest.update(ident.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")

        text = data.decode("utf-8", errors="ignore")
        for include in _QUOTED_INCLUDE_RE.findall(text):
            include_path = (path.parent / include).resolve()
            if include_path.is_file():
                stack.append(include_path)
        for include in _ANGLE_INCLUDE_RE.findall(text):
            include_path = (include_dir / include).resolve()
            if include_path.is_file():
                stack.append(include_path)

    return digest.hexdigest()[:16]


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    # Resolve via the package spec so the lookup is location-independent.
    spec = importlib.util.find_spec("sglang.jit_kernel")
    assert spec is not None and spec.origin is not None
    cur_dir = pathlib.Path(spec.origin).parent.resolve()

    # first, try this directory structure
    def _environment_install():
        candidate = cur_dir.resolve()
        if (candidate / "include").exists() and (candidate / "csrc").exists():
            return candidate
        return None

    def _package_install():
        # TODO: support find path by package
        return None

    path = _environment_install() or _package_install()
    if path is None:
        raise RuntimeError("Cannot find sglang.jit_kernel path")
    return path


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float64: "double",
    torch.float32: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
    # The fnuz variants are the ROCm-side torch dtypes; fp8_*_t resolves to
    # the matching HIP type there (see HIP_FP8_TYPE_* in utils.cuh).
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.float8_e4m3fnuz: "fp8_e4m3_t",
    torch.float8_e5m2: "fp8_e5m2_t",
    torch.float8_e5m2fnuz: "fp8_e5m2_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
    torch.uint16: "uint16_t",
    torch.uint32: "uint32_t",
    torch.uint64: "uint64_t",
    torch.bool: "bool",
}


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


@cache_once
def _tvm_ffi_version() -> str:
    try:
        import tvm_ffi

        version = getattr(tvm_ffi, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib.metadata import version as dist_version

        return dist_version("apache-tvm-ffi")
    except Exception:
        return "unknown"


def _jit_build_dir_name(module_name: str) -> str:
    # Key on arch + tvm-ffi ABI too (module_name only hashes sources), so a
    # shared cache volume never reuses a cross-arch/ABI .so.
    arch = get_jit_cuda_arch().target_name
    return f"{module_name}__arch_{arch}__tvmffi_{_tvm_ffi_version()}"


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    build_directory: str | None = None,
    header_only: bool = True,
) -> Module:
    """
    Loading a JIT module from C++/CUDA source files.
    We define a wrapper as a tuple of (export_name, kernel_name),
    where `export_name` is the name used to called from Python,
    and `kernel_name` is the name of the kernel class in C++/CUDA source.

    :param args: Unique marker of the JIT module. Must be distinct for different kernels.
    :type args: str
    :param cpp_files: A list of C++ source files.
    :type cpp_files: List[str] | None
    :param cuda_files: A list of CUDA source files.
    :type cuda_files: List[str] | None
    :param cpp_wrappers: A list of C++ wrappers, defining the export name and kernel name.
    :type cpp_wrappers: List[Tuple[str, str]] | None
    :param cuda_wrappers: A list of CUDA wrappers, defining the export name and kernel name.
    :type cuda_wrappers: List[Tuple[str, str]] | None
    :param extra_cflags: Extra C++ compiler flags.
    :type extra_cflags: List[str] | None
    :param extra_cuda_cflags: Extra CUDA compiler flags.
    :type extra_cuda_cflags: List[str] | None
    :param extra_ldflags: Extra linker flags.
    :type extra_ldflags: List[str] | None
    :param extra_include_paths: Extra include paths.
    :type extra_include_paths: List[str] | None
    :param extra_dependencies: Extra dependencies for the JIT module, e.g., cutlass.
    :type extra_dependencies: List[str] | None
    :param build_directory: The build directory for JIT compilation.
    :type build_directory: str | None
    :param header_only: Whether the module is header-only.
                        If true, apply the wrappers to export given class/functions.
                        Otherwise, we must export from C++/CUDA side.
    :return: A just-in-time(JIT) compiled module.
    :rtype: Module
    """

    from tvm_ffi.cpp import load, load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cuda_files]

    for dep in set(extra_dependencies or []):
        if dep not in REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        extra_include_paths += REGISTERED_DEPENDENCIES[dep]()

    module_name = "sgl_kernel_jit_" + "_".join(str(arg) for arg in args)
    if cpp_files or cuda_files:
        module_name += "_" + _local_jit_source_hash(cpp_files + cuda_files)

    # A built .so under a deterministic dir is content-addressed: load it
    # directly to skip ninja, whose mtime check rebuilds every CI run (pip
    # install bumps dep header mtimes).
    if build_directory is None:
        cache_dir = os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
        build_directory = str(
            pathlib.Path(cache_dir).expanduser() / _jit_build_dir_name(module_name)
        )
    prebuilt = pathlib.Path(build_directory) / f"{module_name}.so"
    if prebuilt.is_file():
        from tvm_ffi import load_module

        try:
            module = load_module(str(prebuilt))
            logger.debug("Reused cached JIT module %s", module_name)
            return module
        except Exception:
            logger.warning(
                "Cached JIT module %s failed to load; rebuilding.", module_name
            )

    if header_only:
        cpp_wrappers = cpp_wrappers or []
        cuda_wrappers = cuda_wrappers or []
        cpp_sources = [f'#include "{path}"' for path in cpp_files]
        cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

        # include cuda files
        cuda_sources = [f'#include "{path}"' for path in cuda_files]
        cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
        with _jit_compile_context():
            return load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=get_default_target_flags() + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
                build_directory=build_directory,
            )
    else:
        assert cpp_wrappers is None and cuda_wrappers is None
        with _jit_compile_context():
            return load(
                module_name,
                cpp_files=cpp_files,
                cuda_files=cuda_files,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=get_default_target_flags() + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
                build_directory=build_directory,
            )


@contextmanager
def _jit_compile_context():
    if is_hip_runtime():
        yield  # TODO: support ROCm `TVM_FFI_ROCM_ARCH_LIST` if needed
        return
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old_value = os.environ.get(env_key, None)
    os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_value
