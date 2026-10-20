"""Header-only dependency registration (flashinfer, cutlass, mathdx, ...)."""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Callable, Dict, List, Optional


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


REGISTERED_DEPENDENCIES: Dict[str, Callable[[], List[str]]] = {}


def register_dependency(name: str):
    def decorator(f: Callable[[], List[str]]) -> Callable[[], List[str]]:
        if name in REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {name} already registered")
        REGISTERED_DEPENDENCIES[name] = f
        return f

    return decorator


@register_dependency("aiter")
def get_aiter_include_paths() -> List[str]:
    """aiter's C++ headers, for ROCm kernels that must compute exactly what aiter
    computes (hip_reduce.h, opus/opus.hpp)."""
    aiter_root = _find_package_root("aiter")
    if aiter_root is None:
        raise RuntimeError(
            "Cannot find the aiter package. Install aiter to get the required"
            " headers for JIT compilation."
        )

    include = aiter_root.parent / "csrc" / "include"
    if not (include / "hip_reduce.h").is_file():
        raise RuntimeError(
            f"aiter C++ headers not found under {include}; a kernel that includes"
            " them cannot be built against this aiter installation."
        )
    return [str(include)]


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


@register_dependency("flashinfer_nv_internal")
def get_flashinfer_nv_internal_include_paths() -> List[str]:
    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is None:
        raise RuntimeError(
            "Cannot find flashinfer package. Please install flashinfer to get "
            "the required NVFP4 headers for JIT compilation."
        )

    internal_root = flashinfer_root / "data" / "csrc" / "nv_internal"
    candidates = [internal_root, internal_root / "include"]
    for path in candidates:
        if not path.exists():
            raise RuntimeError(
                f"Required FlashInfer NVFP4 header path {path} was not found. "
                "Please install a FlashInfer build with nv_internal headers."
            )
    return [str(path) for path in candidates]


def get_mathdx_root() -> Optional[pathlib.Path]:
    """Locate the NVIDIA Math-DX install (cuBLASDx headers).

    Searches in order:
      1. ``$MATHDX_HOME`` env var (extracted Math-DX archive root).
      2. The ``nvidia-mathdx`` PyPI package, if installed.
    """
    env_home = os.environ.get("MATHDX_HOME")
    if env_home:
        candidate = pathlib.Path(env_home).expanduser().resolve()
        if (candidate / "include").exists():
            return candidate

    # The ``nvidia-mathdx`` wheel installs as the namespace package
    # ``nvidia.mathdx`` (no __init__, so spec.origin is None); resolve it via
    # submodule_search_locations rather than _find_package_root, which only
    # handles regular packages.
    spec = importlib.util.find_spec("nvidia.mathdx")
    if spec is not None:
        roots = list(spec.submodule_search_locations or [])
        if spec.origin is not None:
            roots.append(str(pathlib.Path(spec.origin).parent))
        for root in roots:
            candidate = pathlib.Path(root).resolve()
            if (candidate / "include").exists():
                return candidate

    return None


@register_dependency("mathdx")
def get_mathdx_include_paths() -> List[str]:
    root = get_mathdx_root()
    if root is None:
        raise RuntimeError(
            "Cannot find NVIDIA Math-DX (cuBLASDx) headers. "
            "Install the `nvidia-mathdx` package "
            "(`pip install nvidia-mathdx`) or set MATHDX_HOME to an "
            "extracted Math-DX archive root."
        )
    candidates = [root / "include"]
    cutlass = root / "external" / "cutlass" / "include"
    if cutlass.exists():
        candidates.append(cutlass)
    return [str(p) for p in candidates]


@register_dependency("cutlass")
def get_cutlass_include_paths() -> List[str]:
    """CUTLASS headers for JIT compilation.

    A probe only counts when it contains a real CUTLASS tree, verified by
    sentinel headers. An include dir that merely exists (e.g. deep_gemm's own
    headers) must not mask a missing tree: nvcc would run with no cutlass -I
    and die on the first cutlass/util include.

    Probes in priority order:
      1. ``$CUTLASS_HOME`` — explicit override (mirrors ``MATHDX_HOME``)
      2. flashinfer's vendored tree (pip wheel; source builds ship none)
      3. tilelang's vendored tree (``3rdparty/cutlass``)
      4. deep_gemm's include (only when it really contains ``cutlass/``)
    """

    def _validated(root: pathlib.Path) -> List[str]:
        paths: List[str] = []
        include = root / "include"
        if (include / "cutlass" / "cutlass.h").is_file():
            paths.append(str(include))
        util_include = root / "tools" / "util" / "include"
        if (util_include / "cutlass" / "util" / "packed_stride.hpp").is_file():
            paths.append(str(util_include))
        return paths

    include_paths: List[str] = []
    probed: List[str] = []

    env_home = os.environ.get("CUTLASS_HOME")
    if env_home:
        probed.append(f"CUTLASS_HOME={env_home}")
        include_paths.extend(_validated(pathlib.Path(env_home).expanduser().resolve()))

    for pkg, subdir in (
        ("flashinfer", "data/cutlass"),
        ("tilelang", "3rdparty/cutlass"),
    ):
        pkg_root = _find_package_root(pkg)
        if pkg_root is not None:
            root = pkg_root / subdir
            probed.append(str(root))
            include_paths.extend(_validated(root))

    deep_gemm_root = _find_package_root("deep_gemm")
    if deep_gemm_root is not None:
        probed.append(str(deep_gemm_root / "include"))
        include_paths.extend(_validated(deep_gemm_root))

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
            f"Probed (none contains a cutlass tree): {', '.join(probed) or 'nothing'}. "
            "Install a package that vendors CUTLASS (the flashinfer pip wheel or "
            "tilelang), or point CUTLASS_HOME at a CUTLASS source root."
        )
    return unique_paths
