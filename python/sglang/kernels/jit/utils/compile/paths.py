"""Where the in-tree JIT sources live, and the compile defaults applied to them.

Kept in its own module so that both the build cache and the loader can depend
on it without depending on each other.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import List

from sglang.kernels.jit.utils.common import cache_once


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    spec = importlib.util.find_spec("sglang.kernels.jit")
    assert spec is not None and spec.origin is not None
    cur_dir = pathlib.Path(spec.origin).parent.resolve()

    candidate = cur_dir.resolve()
    if (candidate / "include").exists() and (candidate / "csrc").exists():
        return candidate

    raise RuntimeError("Cannot find sglang.kernels.jit path")


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_INCLUDE: List[str] = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS: List[str] = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS: List[str] = []
