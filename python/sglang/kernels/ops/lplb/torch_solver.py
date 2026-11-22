"""IPM LP Solver entry point — dispatches to the fused JIT CUDA kernel.

Solves: min c^T x  subject to  Ax = b, x >= 0
using a barrier (interior point) method with 5 iterations.

The fused kernel lives in ``cuda_solver`` (CUDA C++ via ``load_jit``,
backed by header-only cuBLASDx + a hand-written block Cholesky). This
module is the public-facing import surface for callers (``LPLBSolver``)
and resolves/caches the backend on first use.

LPLB requires Hopper-class hardware and Math-DX cuBLASDx headers. If
either is missing, ``warmup`` raises; there is no silent fallback.
"""

import logging

import torch

logger = logging.getLogger(__name__)


# Backend dispatch state (resolved on first call, cached afterwards)
_BACKEND_CHECKED = False
_FUSED_AVAILABLE = False
_FUSED_WARMUP = None  # type: ignore[assignment]
_FUSED_ASSERT_FITS = None  # type: ignore[assignment]


def _init_fused_backend() -> None:
    """Resolve the fused backend once. Records WHY it's disabled when it is."""
    global _BACKEND_CHECKED, _FUSED_AVAILABLE
    global _FUSED_WARMUP, _FUSED_ASSERT_FITS

    if _BACKEND_CHECKED:
        return
    _BACKEND_CHECKED = True

    if not torch.cuda.is_available():
        logger.info("LPLB fused solver disabled: CUDA not available")
        return

    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        logger.info(
            f"LPLB fused solver disabled: GPU SM {cap[0]}.{cap[1]} < 9.0 "
            "(requires Hopper or newer)"
        )
        return

    try:
        from sglang.kernels.ops.lplb.cuda_solver import warmup as fused_warmup
        from sglang.kernels.ops.lplb.shmem_budget import assert_fits
    except ImportError as e:
        logger.info(
            f"LPLB fused solver disabled: {e}. "
            "Install Math-DX cuBLASDx via `pip install nvidia-mathdx` "
            "or set MATHDX_HOME to an extracted archive."
        )
        return

    _FUSED_WARMUP = fused_warmup
    _FUSED_ASSERT_FITS = assert_fits
    _FUSED_AVAILABLE = True
    logger.info("LPLB fused solver enabled (CUDA C++ via load_jit, cuBLASDx)")


def _unavailable_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        return f"GPU SM {cap[0]}.{cap[1]} < 9.0 (requires Hopper or newer)"
    return (
        "Math-DX cuBLASDx headers not found — install via "
        "`pip install nvidia-mathdx` or set MATHDX_HOME"
    )


def warmup(nc: int, nv: int, num_iters: int = 5, device: str = "cuda") -> None:
    """Pre-JIT-compile the fused kernel for a given (NC, NV) shape.

    Call once per unique shape at solver construction time to hide the
    20-40s JIT compilation cost. Raises if the fused backend is
    unavailable, the shape exceeds the shmem budget, or the kernel
    fails to compile/launch.
    """
    _init_fused_backend()
    if not _FUSED_AVAILABLE:
        raise RuntimeError(f"LPLB fused solver unavailable: {_unavailable_reason()}")
    _FUSED_ASSERT_FITS(nc, nv)
    _FUSED_WARMUP(nc, nv, num_iters=num_iters, device=device)
