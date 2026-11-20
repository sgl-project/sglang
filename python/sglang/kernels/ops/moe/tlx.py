"""gfx950 BF16 routed MoE from fbtriton ``tlx.ops.moe``, vendored as ``sglang.kernels.tlx_ops.kernels.moe.gfx950``.

Coverage: ``moe`` there aligns tokens to local experts, runs the gate/up GEMM with
fused SiLU, the down GEMM, the optional per-expert post-norm (a None weight means no
norm) and the routing-weighted combine, with no host reads of device data
(HIP-graph capturable). BF16 only; K, I and H multiples of 128; 1..512 local
experts; ``next_pow2(topk) * next_pow2(H) <= 2**16``; gfx950 only, not checked here.
Runtime requirements are in ``sglang/kernels/jit/utils/tlx_runtime.py``.
"""

import torch

# Kernel limits, mirrored from the ``tlx.ops.moe`` wrapper and its gfx950 catalog entry.
_MOE_DIM_MULTIPLE = 128
_MOE_MAX_LOCAL_EXPERTS = 512
_MOE_MAX_FINALIZE_TILE = 1 << 16


def moe_supported(
    x: torch.Tensor, topk_ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Host-only check that the vendored ``moe`` handles these shapes; never reads device data."""
    if x.dtype != torch.bfloat16 or w13.dtype != x.dtype or w2.dtype != x.dtype:
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
