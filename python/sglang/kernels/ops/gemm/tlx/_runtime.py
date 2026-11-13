"""Lazy loading of the vendored TLX kernels."""

from __future__ import annotations

import functools
from importlib import import_module
from typing import Any


def require_tlx() -> None:
    """Make ``triton.language.extra.tlx`` importable, or raise if no TLX is installed."""
    try:
        import triton.language.extra.tlx  # noqa: F401  fbtriton ships it
    except ImportError:
        try:
            import utlx_plugin  # noqa: F401  registers it on stock Triton
        except ImportError as e:
            raise ImportError(
                "TLX kernels need triton.language.extra.tlx: install the uTLX plugin "
                "built for your triton-rocm (see sglang/kernels/ops/gemm/tlx/README.md), "
                "or use fbtriton"
            ) from e


@functools.cache
def load_kernel(module: str, name: str) -> Any:
    """Import a vendored kernel module after TLX is registered and return ``name`` from it."""
    require_tlx()
    from ._compat import add_missing_triton_knobs

    add_missing_triton_knobs()
    return getattr(import_module(module), name)
