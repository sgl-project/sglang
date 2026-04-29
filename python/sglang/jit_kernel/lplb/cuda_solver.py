"""JIT-compiled CUDA Interior Point Method LP solver.

Replaces the Numba/nvmath-python implementation in ``cublasdx_solver.py``.
The kernel is a single-block fused IPM defined in
``csrc/lplb/ipm.cuh`` and compiled per ``(NC, NV, BLOCK_DIM, SM_VER,
NUM_ITERS)`` tuple via sglang's ``tvm-ffi`` ``load_jit``.

Per-call CPU overhead is dominated by the pybind11 dispatch + four
``data_ptr()`` calls (~5–10 µs total), versus ~500–700 µs for the prior
Numba path (numba dispatcher chain + ``as_cuda_array`` per tensor).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_DIM = 256
DEFAULT_NUM_ITERS = 5

# Shapes that failed to compile or launch — never tried again.
_FUSED_REJECTED_SHAPES: set[Tuple[int, int]] = set()


def _sm_ver() -> int:
    arch = get_jit_cuda_arch()
    return arch.major * 100 + arch.minor * 10


@cache_once
def _ipm_module(nc: int, nv: int, block_dim: int, num_iters: int, sm_ver: int) -> "Module":
    """JIT-compile the IPM kernel for one shape. Cached for the process lifetime."""
    args = make_cpp_args(nc, nv, block_dim, sm_ver, num_iters)
    # The kernel uses cuBLASDx (header-only) for the GEMMs and a hand-written
    # block-level Cholesky for the POSV. No -rdc=true / static-lib linkage
    # required, so sglang's tvm-ffi load_jit handles the build with the
    # default flags.
    return load_jit(
        "lplb_ipm",
        *args,
        cuda_files=["lplb/ipm.cuh"],
        cuda_wrappers=[("ipm_solve", f"ipm_solve<{args}>")],
        extra_dependencies=["mathdx"],
    )


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "cuda",
) -> None:
    """JIT-compile the kernel for ``(nc, nv)`` so the first real solve isn't
    paying the compile cost. Safe to call when the kernel can't compile —
    becomes a no-op and adds the shape to ``_FUSED_REJECTED_SHAPES``.
    """
    if (nc, nv) in _FUSED_REJECTED_SHAPES:
        return
    try:
        module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
        # Trigger any first-call lazy initialization.
        A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
        b = torch.zeros(nc, dtype=torch.float32, device=device)
        c = torch.zeros(nv, dtype=torch.float32, device=device)
        result = torch.empty(nv, dtype=torch.float32, device=device)
        module.ipm_solve(A, b, c, result)
        logger.info(f"LPLB CUDA IPM solver: warmed up for (NC={nc}, NV={nv})")
    except Exception as e:  # pragma: no cover
        _FUSED_REJECTED_SHAPES.add((nc, nv))
        logger.warning(
            f"LPLB CUDA IPM solver: warmup failed for (NC={nc}, NV={nv}): {e}. "
            "Falling back to torch IPM for this shape."
        )


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
) -> torch.Tensor:
    """Drop-in replacement for ``torch_solver._solve_ipm_torch``.

    Runs the fused single-SM IPM kernel (cuBLASDx GEMM + cuSolverDx Cholesky)
    with the dispatch path described in the module docstring.

    Args:
        A: Constraint matrix, shape ``(NC, NV)``, float32, on CUDA.
        b: RHS vector, shape ``(NC,)``, float32, on CUDA.
        c: Objective coefficients, shape ``(NV,)``, float32, on CUDA.
        num_iters: Number of barrier iterations (default 5).

    Returns:
        x: Solution vector, shape ``(NV,)``, float32. The kernel writes 0.5
        for every entry on non-convergence (matches the prior Numba behavior).
    """
    assert A.is_cuda and b.is_cuda and c.is_cuda
    assert A.dtype == torch.float32
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    result = torch.empty(nv, dtype=torch.float32, device=A.device)
    module.ipm_solve(A, b, c, result)
    return result
