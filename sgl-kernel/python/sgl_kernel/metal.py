"""Python entry points for the sgl_kernel Metal extension."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

_METALLIB_NAME = "sgl_metal_kernels.metallib"

try:
    from . import _metal

    _metallib_path = Path(_metal.__file__).resolve().parent / _METALLIB_NAME
    if not _metallib_path.is_file():
        raise ImportError(
            f"{_METALLIB_NAME} not found next to sgl_kernel._metal at {_metallib_path}"
        )
    _metal.register_library(str(_metallib_path))
except ImportError as _exc:  # pragma: no cover - import guarded at call time
    _metal = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None

# Python wrappers for the compiled `_metal.*` entry points go below. Each
# wrapper validates input shapes/dtypes and calls `mx.eval` on its operands
# before invoking the AOT C++ entry point.
