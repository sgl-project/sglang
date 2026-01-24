from __future__ import annotations

import functools
import pathlib
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, TypeAlias, TypeVar, Union

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@lru_cache()
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


@cache_once
def is_arch_support_pdl() -> bool:
    import torch

    device = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device)[0] >= 9
