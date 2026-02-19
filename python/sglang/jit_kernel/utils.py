from __future__ import annotations

import functools
import os
import pathlib
import stat
import weakref
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, TypeAlias, TypeVar, Union

import torch
from torch.utils.dlpack import to_dlpack

if TYPE_CHECKING:
    from tvm_ffi import Module


F = TypeVar("F", bound=Callable[..., Any])


def cache_once(fn: F) -> F:
    """
    NOTE: `functools.lru_cache` is not compatible with `torch.compile`
    So we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items(), key=lambda x: x[0])))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).parent.resolve()

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
        raise RuntimeError("Cannot find sgl-kernel/jit path")
    return path


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool, torch.dtype]


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
}

# AMD/ROCm note:
# tvm_ffi's generic Python torch fallback path can break HIP graph capture and
# add avoidable per-call overhead. We keep a cached torch->tvm_ffi bridge and a
# HIP stream cache here so all HIP-enabled JIT kernels can share the same fix.
_tvm_tensor_cache: dict[int, tuple[weakref.ReferenceType[torch.Tensor], Any]] = {}
_hip_tvm_stream_cache: dict[tuple[int, int], int] = {}


@cache_once
def is_hip_runtime() -> bool:
    return bool(torch.version.hip)


@cache_once
def _prepare_rocm_cuda_shim() -> str:
    """Create a CUDA-home-compatible shim that forwards nvcc to hipcc."""
    shim_root = pathlib.Path.home() / ".cache" / "sglang_jit_rocm_cuda_shim"
    bin_dir = shim_root / "bin"
    lib64_dir = shim_root / "lib64"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib64_dir.mkdir(parents=True, exist_ok=True)

    libcudart = lib64_dir / "libcudart.so"
    if libcudart.exists() or libcudart.is_symlink():
        libcudart.unlink()
    libcudart.symlink_to("/opt/rocm/lib/libamdhip64.so")

    nvcc_path = bin_dir / "nvcc"
    wrapper = """#!/usr/bin/env bash
set -euo pipefail

depfile=""
out=""
src=""
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate-dependencies-with-compile)
      shift
      ;;
    --dependency-output)
      depfile="$2"
      shift 2
      ;;
    -gencode=*)
      shift
      ;;
    --expt-relaxed-constexpr)
      shift
      ;;
    -Xcompiler)
      # Keep the wrapped compiler flag, drop only the nvcc forwarding token.
      if [[ $# -ge 2 ]]; then
        args+=("$2")
      fi
      shift 2
      ;;
    -o)
      out="$2"
      args+=("$1" "$2")
      shift 2
      ;;
    -c)
      args+=("$1")
      if [[ $# -ge 2 ]]; then
        src="$2"
      fi
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${src}" ]]; then
  hip_src="${src}.hip.cu"
  /opt/rocm/bin/hipify-perl "${src}" > "${hip_src}"
  for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "${src}" ]]; then
      args[$i]="${hip_src}"
      break
    fi
  done
fi

/opt/rocm/bin/hipcc -DUSE_ROCM "${args[@]}"
rc=$?
if [[ $rc -ne 0 ]]; then
  exit $rc
fi

if [[ -n "${depfile}" ]]; then
  if [[ -n "${out}" && -n "${src}" ]]; then
    echo "${out}: ${src}" > "${depfile}"
  else
    : > "${depfile}"
  fi
fi
"""
    nvcc_path.write_text(wrapper)
    nvcc_path.chmod(
        nvcc_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    return str(shim_root)


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


def to_tvm_tensor_cached(tensor: torch.Tensor):
    """AMD/ROCm helper: cached torch->tvm_ffi conversion for HIP capture safety."""
    key = id(tensor)
    entry = _tvm_tensor_cache.get(key)
    if entry is not None:
        ref, cached = entry
        if ref() is tensor:
            return cached
        _tvm_tensor_cache.pop(key, None)
    from tvm_ffi import from_dlpack

    tvm_tensor = from_dlpack(to_dlpack(tensor))
    _tvm_tensor_cache[key] = (weakref.ref(tensor), tvm_tensor)
    return tvm_tensor


def hip_ensure_tvm_ffi_stream(tensor: torch.Tensor) -> None:
    """AMD/ROCm helper: keep tvm_ffi stream aligned with the current HIP stream."""
    from tvm_ffi.core import _env_set_current_stream

    dl_device_type, dl_device_id = tensor.__dlpack_device__()
    device_type = int(dl_device_type)
    device_id = int(dl_device_id)
    # Faster than constructing torch.cuda.current_stream(...) Python object.
    current_stream = int(torch._C._cuda_getCurrentRawStream(device_id))
    key = (device_type, device_id)
    if _hip_tvm_stream_cache.get(key) == current_stream:
        return
    _env_set_current_stream(device_type, device_id, current_stream)
    _hip_tvm_stream_cache[key] = current_stream


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
    build_directory: str | None = None,
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
    :param build_directory: The build directory for JIT compilation.
    :type build_directory: str | None
    :return: A just-in-time(JIT) compiled module.
    :rtype: Module
    """

    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "csrc" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "csrc" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    # Override TVM_FFI_CUDA_ARCH_LIST if it does not exist.
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    env_existed = env_key in os.environ
    cuda_home_existed = "CUDA_HOME" in os.environ
    old_cuda_home = os.environ.get("CUDA_HOME")

    # tvm_ffi currently assumes a CUDA-style toolchain. On ROCm, provide a shim.
    if is_hip_runtime():
        os.environ["CUDA_HOME"] = _prepare_rocm_cuda_shim()
        extra_cuda_cflags = ["-DUSE_ROCM"] + extra_cuda_cflags
    if not env_existed:
        os.environ[env_key] = _get_cuda_arch_list()
    try:
        return load_inline(
            "sgl_kernel_jit_" + "_".join(str(arg) for arg in args),
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            build_directory=build_directory,
        )
    finally:
        # Reset TVM_FFI_CUDA_ARCH_LIST to original state (not exist)
        if not env_existed:
            del os.environ[env_key]
        if is_hip_runtime():
            if cuda_home_existed:
                assert old_cuda_home is not None
                os.environ["CUDA_HOME"] = old_cuda_home
            else:
                os.environ.pop("CUDA_HOME", None)


@cache_once
def is_arch_support_pdl() -> bool:
    import torch

    device = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device)[0] >= 9


@cache_once
def _get_cuda_arch_list() -> str:
    """Get the correct CUDA architecture string for TVM_FFI_CUDA_ARCH_LIST."""
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}"
