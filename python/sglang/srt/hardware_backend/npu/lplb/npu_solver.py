"""Ascend NPU backend for the LPLB fused operators.

The Triton-Ascend kernels live in sgl-kernel-npu, in
``sgl_kernel_npu.operators.moe.lplb`` (``lp_prep``, ``lp_post``, ``solve_ipm``
and ``dispatch_probability``), together with their ``torch_native`` references
and tests.  This module is the thin adapter that exposes them with the same
function surface as :mod:`sglang.kernels.ops.lplb.cuda_solver` so that
``LPLBSolver`` and the probability dispatcher can select a backend per platform
without knowing any launch details.

Only the functions used on the LPLB hot path are exposed.  They intentionally
keep the CUDA signatures so the call sites stay backend-agnostic.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_NUM_ITERS = 5

# The NPU kernels are always the Triton-Ascend implementation.  Pin it instead
# of relying on the operator's ``auto`` resolution so that a future default
# change cannot silently switch this backend.
_IMPL = "auto"

# ``device`` is part of the key only to make repeated calls for a different
# device cheap; in practice every rank has one NPU.
_WARMED_UP: set[tuple[int, int, int, str]] = set()


class _Ops(NamedTuple):
    lp_prep: object
    lp_post: object
    solve_ipm: object
    dispatch_probability: object


@lru_cache(maxsize=None)
def _ops() -> _Ops:
    """Import the registered LPLB operators once, on first use."""
    from sgl_kernel_npu.operators.moe import (
        dispatch_probability,
        lp_post,
        lp_prep,
        solve_ipm,
    )

    return _Ops(
        lp_prep=lp_prep,
        lp_post=lp_post,
        solve_ipm=solve_ipm,
        dispatch_probability=dispatch_probability,
    )


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "npu",
) -> None:
    """Compile the IPM kernel for ``(nc, nv)`` before the first request.

    Mirrors ``cuda_solver.warmup``: the first launch pays the Triton-Ascend
    compilation cost, so do it once during solver construction.  Repeated
    calls for the same shape are skipped.
    """
    key = (int(nc), int(nv), int(num_iters), str(device))
    if key in _WARMED_UP:
        return

    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    _ops().solve_ipm(A, b, c, num_iters=num_iters, impl=_IMPL)

    _WARMED_UP.add(key)
    logger.info("LPLB Ascend NPU IPM solver: warmed up for (NC=%d, NV=%d)", nc, nv)


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Triton-Ascend interior-point solver.

    Args:
        A: Constraint matrix, shape ``(NC, NV)``, float32, on NPU.
        b: RHS vector, shape ``(NC,)``, float32, on NPU.
        c: Objective coefficients, shape ``(NV,)``, float32, on NPU.
        num_iters: Number of barrier iterations (default 5).
        result: Optional ``(NV,)`` float32 buffer the kernel writes into; a
            fresh tensor is allocated when omitted, matching the CUDA backend.

    Returns:
        x: Solution vector, shape ``(NV,)``, float32.  The kernel writes 0.5
        for every entry on non-convergence.
    """
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"
    assert A.dtype == torch.float32
    return _ops().solve_ipm(A, b, c, num_iters=num_iters, result=result, impl=_IMPL)


def prep_lp_inputs(
    A_full: torch.Tensor,
    b: torch.Tensor,
    t1: torch.Tensor,
    global_counts: torch.Tensor,
    log_single: torch.Tensor,
    log_replicated: torch.Tensor,
    B1: torch.Tensor,
    A_base_row_sum: torch.Tensor,
) -> None:
    """Build the IPM inputs in place, matching ``cuda_solver.prep_lp_inputs``."""
    _ops().lp_prep(
        A_full,
        b,
        t1,
        global_counts,
        log_single,
        log_replicated,
        B1,
        A_base_row_sum,
        impl=_IMPL,
    )


def extract_log2phy_prob(
    log2phy_prob: torch.Tensor,
    x: torch.Tensor,
    t1: torch.Tensor,
    phy_single: torch.Tensor,
    phy_replicated: torch.Tensor,
    log2phy: torch.Tensor,
) -> None:
    """Turn the IPM output into ``log2phy_prob`` (in place).

    ``lp_post`` takes the replicated-ratio prefix of the IPM solution rather
    than the full ``(NV,)`` vector, so slice it here.  Only the first
    ``NUM_RED_PHY`` entries of ``x`` are the replica ratios; the rest are the
    per-GPU slack variables, ``M`` and the Big-M column.
    """
    num_red_phy = phy_replicated.shape[0]
    _ops().lp_post(
        log2phy_prob,
        x[:num_red_phy],
        t1,
        phy_single,
        phy_replicated,
        log2phy,
        impl=_IMPL,
    )


def dispatch_probability(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample a physical expert per ``(token, slot)`` on Ascend NPU.

    Same contract as ``cuda_solver.dispatch_probability``: inverse-CDF sample
    over the per-row LP probabilities, falling back to a uniform choice over
    valid replicas when a row sums to zero.  The operator materialises
    non-contiguous ``topk_ids`` (e.g. routed columns of a fused shared-expert
    layout) and preserves the input shape and dtype.
    """
    n = topk_ids.numel()
    if random_vals is not None:
        assert random_vals.shape == (n,), (
            f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"
        )
    return _ops().dispatch_probability(
        topk_ids, log2phy_prob, log2phy_map, random_vals, impl=_IMPL
    )
