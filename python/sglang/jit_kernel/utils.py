from __future__ import annotations

import functools
import importlib.util
import logging
import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import torch

from sglang.utils import is_in_ci

if TYPE_CHECKING:
    from tvm_ffi import Module

F = TypeVar("F", bound=Callable[..., Any])
_FULL_TEST_ENV_VAR = "SGLANG_JIT_KERNEL_RUN_FULL_TESTS"

logger = logging.getLogger(__name__)


def should_run_full_tests() -> bool:
    return os.getenv(_FULL_TEST_ENV_VAR, "false").lower() == "true"


def get_ci_test_range(full_range: List[Any], ci_range: List[Any]) -> List[Any]:
    if should_run_full_tests():
        return full_range
    return ci_range if is_in_ci() else full_range


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
        raise RuntimeError("Cannot find sglang.jit_kernel path")
    return path


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool, torch.dtype]


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.bfloat16: "bf16_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}


# AMD/ROCm note:
@cache_once
def is_hip_runtime() -> bool:
    return bool(torch.version.hip)


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
    extra_dependencies: List[str] | None = None,
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
    :param extra_dependencies: Extra dependencies for the JIT module, e.g., cutlass.
    :type extra_dependencies: List[str] | None
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

    for dep in set(extra_dependencies or []):
        if dep not in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        extra_include_paths += _REGISTERED_DEPENDENCIES[dep]()

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "csrc" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "csrc" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    with _jit_compile_context():
        return load_inline(
            "sgl_kernel_jit_" + "_".join(str(arg) for arg in args),
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=_get_default_target_flags() + extra_cuda_cflags,
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            build_directory=build_directory,
        )


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


@cache_once
def _init_jit_cuda_arch_once():
    global _CUDA_ARCH
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        logger.warning("Cannot detect CUDA architecture.")
        major, minor = 0, 0  # invalid value to trigger compile error if used
    _CUDA_ARCH = ArchInfo(major, minor, "")


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


# NOTE: this might also be used in __main__.py for compile flags export
def _get_default_target_flags() -> List[str]:
    if is_hip_runtime():
        return ["-DUSE_ROCM", "-std=c++20", "-O3"]
    else:
        return [
            get_jit_cuda_arch().jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str = ""):
    """A context manager to temporarily override CUDA architecture."""
    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = ArchInfo(major, minor, suffix)
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


def get_jit_cuda_arch() -> ArchInfo:
    """Get the current CUDA architecture info."""
    _init_jit_cuda_arch_once()
    return _CUDA_ARCH


def is_arch_support_pdl() -> bool:
    if is_hip_runtime():
        return False
    return get_jit_cuda_arch().major >= 9


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


# NOTE: this might also be used in __main__.py for compile flags export
_REGISTERED_DEPENDENCIES: Dict[str, Callable[[], List[str]]] = {}


def register_dependency(name: str):
    def decorator(f: Callable[[], List[str]]) -> Callable[[], List[str]]:
        if name in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {name} already registered")
        _REGISTERED_DEPENDENCIES[name] = f
        return f

    return decorator


@register_dependency("flashinfer")
def get_flashinfer_include_paths() -> List[str]:
    include_paths: List[str] = []
    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is None:
        raise RuntimeError(
            "Cannot find flashinfer package. Please install flashinfer to get"
            "the required headers for JIT compilation."
        )

    flashinfer_data = flashinfer_root / "data"
    candidates = [
        flashinfer_data / "include",
        flashinfer_data / "csrc",
        flashinfer_data / "cutlass" / "include",
        flashinfer_data / "cutlass" / "tools" / "util" / "include",
        flashinfer_data / "spdlog" / "include",
    ]

    for path in candidates:
        if not path.exists():
            raise RuntimeError(
                f"Required header path {path} for flashinfer dependency not found."
                " Please check your flashinfer installation."
            )
        include_paths.append(str(path))
    return include_paths


@register_dependency("cutlass")
def get_cutlass_include_paths() -> List[str]:
    include_paths: List[str] = []

    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is not None:
        candidates = [
            flashinfer_root / "data" / "cutlass" / "include",
            flashinfer_root / "data" / "cutlass" / "tools" / "util" / "include",
        ]
        for path in candidates:
            if path.exists():
                include_paths.append(str(path))

    deep_gemm_root = _find_package_root("deep_gemm")
    if deep_gemm_root is not None:
        candidate = deep_gemm_root / "include"
        if candidate.exists():
            include_paths.append(str(candidate))

    # De-duplicate while preserving order.
    unique_paths = []
    seen = set()
    for path in include_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    if not unique_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for JIT compilation. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )
    return unique_paths


__all__ = [
    "should_run_full_tests",
    "get_ci_test_range",
    "cache_once",
    "is_hip_runtime",
    "make_cpp_args",
    "load_jit",
    "override_jit_cuda_arch",
    "get_jit_cuda_arch",
    "is_arch_support_pdl",
    "register_dependency",
]
