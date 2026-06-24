"""Compatibility checks for the Triton APIs used by Artemis."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

QUALIFIED_TRITON_VERSION = "3.7.0+git5f3f125e"
QUALIFIED_TRITON_COMMIT = "5f3f125e8f63c24613f1f73b937442864f263f94"
MINIMUM_TRITON_VERSION = (3, 7, 0)


def _release_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return None
    major, minor, patch = match.groups(default="0")
    return int(major), int(minor), int(patch)


@lru_cache(maxsize=1)
def install_triton_compat_shims() -> bool:
    """Install narrow compatibility shims, or return false without Triton."""
    try:
        import triton
        import triton.language as tl
        from triton.compiler.compiler import CompiledKernel
    except ImportError:
        return False

    if not hasattr(tl, "constexpr_function"):
        constexpr_function = getattr(triton, "constexpr_function", None)
        if constexpr_function is None:
            return False
        tl.constexpr_function = constexpr_function
    if not hasattr(CompiledKernel, "num_ctas"):
        CompiledKernel.num_ctas = property(lambda kernel: kernel.metadata.num_ctas)
    if not hasattr(CompiledKernel, "cluster_dims"):
        CompiledKernel.cluster_dims = property(lambda kernel: (kernel.metadata.num_ctas, 1, 1))
    return True


@lru_cache(maxsize=1)
def ensure_triton_compat() -> str:
    """Validate Triton and ensure the required compatibility shims exist."""
    if not install_triton_compat_shims():
        raise RuntimeError(
            "ARTEMIS_KERNELS requires Triton with constexpr_function support; "
            f"qualified build={QUALIFIED_TRITON_VERSION!r}"
        )

    import triton

    module_version = str(getattr(triton, "__version__", "unknown"))
    try:
        version = distribution_version("triton")
    except PackageNotFoundError:
        version = module_version
    release = _release_tuple(version)
    if release is None or release < MINIMUM_TRITON_VERSION:
        raise RuntimeError(
            "ARTEMIS_KERNELS requires Triton >= 3.7.0; "
            f"found distribution={version!r} module={module_version!r}, "
            f"qualified build={QUALIFIED_TRITON_VERSION!r}"
        )

    return version


__all__ = [
    "MINIMUM_TRITON_VERSION",
    "QUALIFIED_TRITON_COMMIT",
    "QUALIFIED_TRITON_VERSION",
    "ensure_triton_compat",
    "install_triton_compat_shims",
]
