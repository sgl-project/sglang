"""Backwards-compatible shim.

The Numba/nvmath-python fused IPM that used to live here has been replaced
by a CUDA C++ kernel JIT-compiled via sglang's ``load_jit`` infrastructure.
The new implementation lives in ``cuda_solver``. This module re-exports the
public API so any external import keeps working.
"""

from sglang.kernels.ops.lplb.cuda_solver import (  # noqa: F401
    solve_ipm,
    warmup,
)

__all__ = ["solve_ipm", "warmup"]
