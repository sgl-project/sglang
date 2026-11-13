"""gfx950 BF16 routed MoE vendored from ``tlx.ops.moe`` in facebookexperimental/triton.

Built on the ``gemm.tlx`` GEMM core; provenance, re-sync and the uTLX runtime are
in ``sglang/kernels/ops/gemm/tlx/README.md``. Import is lazy, so callers can probe
``moe_supported()`` and fall back without pulling the kernel in.

Coverage: one call aligns tokens to local experts, runs the gate/up GEMM with
fused SiLU, the down GEMM, the optional per-expert post-norm and the routing-
weighted combine, with no host reads of device data (HIP-graph capturable).
BF16 only; K, I and H multiples of 128; 1..512 local experts;
``next_pow2(topk) * next_pow2(H) <= 2**16``; gfx950 only, not checked here.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.gemm.tlx._runtime import load_kernel

# Kernel limits, mirrored from the ``tlx.ops.moe`` wrapper and its gfx950 catalog entry.
_MOE_DIM_MULTIPLE = 128
_MOE_MAX_LOCAL_EXPERTS = 512
_MOE_MAX_FINALIZE_TILE = 1 << 16


def moe_supported(
    x: torch.Tensor, topk_ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Host-only check that ``moe`` handles these shapes; never reads device data."""
    if x.dtype != torch.bfloat16:
        return False
    if w13.dtype != x.dtype or w2.dtype != x.dtype:
        return False
    num_experts, two_i, k = w13.shape
    inter = two_i // 2
    hidden = w2.shape[1]
    if two_i % 2 or k != x.shape[1] or w2.shape != (num_experts, hidden, inter):
        return False
    if any(d <= 0 or d % _MOE_DIM_MULTIPLE for d in (k, inter, hidden)):
        return False
    if not 1 <= num_experts <= _MOE_MAX_LOCAL_EXPERTS:
        return False
    topk = topk_ids.shape[1]
    tile = (1 << (topk - 1).bit_length()) * (1 << (hidden - 1).bit_length())
    return 1 <= tile <= _MOE_MAX_FINALIZE_TILE


def moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_start: int,
    post_norm_weight: torch.Tensor | None,
    post_norm_eps: float,
    post_norm_gain: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed SiLU-gated MoE; a None ``post_norm_weight`` means no post-norm."""
    kernel = load_kernel("sglang.kernels.ops.moe.tlx.moe_gfx950", "moe")
    return kernel(
        x,
        topk_weights,
        topk_ids,
        w13,
        w2,
        expert_start,
        post_norm_weight,
        post_norm_eps,
        post_norm_gain,
        out,
    )


__all__ = ["moe", "moe_supported"]
