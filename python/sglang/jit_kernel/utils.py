from __future__ import annotations

import functools
import hashlib
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
        key = (args, tuple(sorted(kwargs.items())))
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
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]
JIT_SOURCE_EXTENSIONS = frozenset((".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"))


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
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


@dataclass(frozen=True)
class JitArtifactCacheInfo:
    """Information for external SGLang JIT artifact cache hooks."""

    module_name: str
    cache_key: str
    build_directory: pathlib.Path
    artifact_path: pathlib.Path


JitArtifactRestoreHook: TypeAlias = Callable[[JitArtifactCacheInfo], "Module | None"]
JitArtifactStoreHook: TypeAlias = Callable[[JitArtifactCacheInfo], None]
JitArtifactCacheHook: TypeAlias = Tuple[JitArtifactRestoreHook, JitArtifactStoreHook]
_JIT_ARTIFACT_CACHE_HOOKS: List[JitArtifactCacheHook] = []


def register_jit_artifact_cache(
    restore: JitArtifactRestoreHook,
    store: JitArtifactStoreHook,
) -> Callable[[], None]:
    """Register external restore/store hooks for compiled SGLang JIT artifacts.

    The restore hook should return a loaded module on a cache hit, or ``None`` on a
    miss. The store hook is called after a successful local build when
    ``info.artifact_path`` exists. Exceptions from hooks are logged and ignored so
    JIT compilation can still proceed.

    Returns a function that unregisters the hook pair.
    """

    hook = (restore, store)
    _JIT_ARTIFACT_CACHE_HOOKS.append(hook)

    def unregister() -> None:
        try:
            _JIT_ARTIFACT_CACHE_HOOKS.remove(hook)
        except ValueError:
            pass

    return unregister


def _file_digest(path: str) -> Tuple[str, str]:
    path_obj = pathlib.Path(path).resolve()
    digest = hashlib.sha256()
    with path_obj.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    try:
        source_id = str(path_obj.relative_to(KERNEL_PATH))
    except ValueError:
        source_id = str(path_obj)
    return source_id, digest.hexdigest()


@cache_once
def _jit_kernel_include_digest() -> str:
    digest = hashlib.sha256()
    root = KERNEL_PATH / "include"
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix not in JIT_SOURCE_EXTENSIONS:
            continue
        digest.update(repr(_file_digest(str(path))).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _tvm_ffi_cache_dir() -> pathlib.Path:
    from sglang.srt.environ import envs

    return pathlib.Path(envs.SGLANG_CACHE_DIR.get()).expanduser() / "tvm-ffi"


def _jit_module_extension() -> str:
    import tvm_ffi.cpp.extension as tvm_ext

    return ".dll" if tvm_ext.IS_WINDOWS else ".so"


def _jit_build_directory(
    module_name: str,
    *,
    cpp_files: List[str],
    cuda_files: List[str],
    cpp_wrappers: List[Tuple[str, str]],
    cuda_wrappers: List[Tuple[str, str]],
    extra_cflags: List[str],
    extra_cuda_cflags: List[str],
    target_arch: str | None,
    extra_ldflags: List[str],
    extra_include_paths: List[str],
    extra_dependencies: List[str] | None,
    header_only: bool,
) -> Tuple[pathlib.Path, str]:
    import tvm_ffi

    digest = hashlib.sha256()

    def add(value: object) -> None:
        digest.update(repr(value).encode())
        digest.update(b"\0")

    add(module_name)
    add(header_only)
    add(
        [
            str(pathlib.Path(path).relative_to(KERNEL_PATH / "csrc"))
            for path in cpp_files
        ]
    )
    add(
        [
            str(pathlib.Path(path).relative_to(KERNEL_PATH / "csrc"))
            for path in cuda_files
        ]
    )
    add(cpp_wrappers)
    add(cuda_wrappers)
    add(extra_cflags)
    add(extra_cuda_cflags)
    add(target_arch)
    add(extra_ldflags)
    add(extra_include_paths)
    add(sorted(extra_dependencies or []))
    add(getattr(tvm_ffi, "__version__", None))
    add(torch.version.cuda)
    add(torch.version.hip)
    add(_jit_kernel_include_digest())
    add([_file_digest(path) for path in sorted(cpp_files)])
    add([_file_digest(path) for path in sorted(cuda_files)])

    cache_key = digest.hexdigest()[:16]
    return _tvm_ffi_cache_dir() / f"{module_name}_{cache_key}", cache_key


def _jit_artifact_cache_info(
    module_name: str, cache_key: str, build_directory: pathlib.Path
) -> JitArtifactCacheInfo:
    return JitArtifactCacheInfo(
        module_name=module_name,
        cache_key=cache_key,
        build_directory=build_directory,
        artifact_path=build_directory / f"{module_name}{_jit_module_extension()}",
    )


def _restore_jit_artifact(info: JitArtifactCacheInfo) -> Module | None:
    for restore, _ in list(_JIT_ARTIFACT_CACHE_HOOKS):
        try:
            module = restore(info)
        except Exception:
            logger.warning(
                "SGLang JIT artifact restore hook failed for %s",
                info.module_name,
                exc_info=True,
            )
            continue
        if module is not None:
            return module
    return None


def _store_jit_artifact(info: JitArtifactCacheInfo) -> None:
    if not info.artifact_path.exists():
        return
    for _, store in list(_JIT_ARTIFACT_CACHE_HOOKS):
        try:
            store(info)
        except Exception:
            logger.warning(
                "SGLang JIT artifact store hook failed for %s",
                info.module_name,
                exc_info=True,
            )


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
        if dep not in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        extra_include_paths += _REGISTERED_DEPENDENCIES[dep]()

    module_name = "sgl_kernel_jit_" + "_".join(str(arg) for arg in args)
    artifact_info = None
    with _jit_compile_context():
        default_cuda_cflags = _get_default_target_flags()
        target_arch = (
            os.environ.get("TVM_FFI_ROCM_ARCH_LIST")
            if is_hip_runtime()
            else os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
        )

    if build_directory is None:
        build_dir, cache_key = _jit_build_directory(
            module_name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            cpp_wrappers=cpp_wrappers or [],
            cuda_wrappers=cuda_wrappers or [],
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=default_cuda_cflags + extra_cuda_cflags,
            target_arch=target_arch,
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            extra_dependencies=extra_dependencies,
            header_only=header_only,
        )
        build_directory = str(build_dir)
        artifact_info = _jit_artifact_cache_info(module_name, cache_key, build_dir)

        restored_module = _restore_jit_artifact(artifact_info)
        if restored_module is not None:
            return restored_module

    if header_only:
        cpp_wrappers = cpp_wrappers or []
        cuda_wrappers = cuda_wrappers or []
        cpp_sources = [f'#include "{path}"' for path in cpp_files]
        cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

        # include cuda files
        cuda_sources = [f'#include "{path}"' for path in cuda_files]
        cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
        with _jit_compile_context():
            module = load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=default_cuda_cflags + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
                build_directory=build_directory,
            )
        if artifact_info is not None:
            _store_jit_artifact(artifact_info)
        return module
    else:
        assert cpp_wrappers is None and cuda_wrappers is None
        with _jit_compile_context():
            module = load(
                module_name,
                cpp_files=cpp_files,
                cuda_files=cuda_files,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=default_cuda_cflags + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
                build_directory=build_directory,
            )
        if artifact_info is not None:
            _store_jit_artifact(artifact_info)
        return module


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


@cache_once
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
    "register_jit_artifact_cache",
    "JitArtifactCacheInfo",
]
