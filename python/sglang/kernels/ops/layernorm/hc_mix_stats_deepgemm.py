"""Compensated FP32 mHC projections for SM100 batches with at least 128 rows.

The small-row and batch-invariant paths remain in mhc.py. Native TF32 discards
too much of the FP32 projection weights, so evaluate their high and residual
components separately and bound accumulation length with a fixed split count.
"""

import torch

_NUM_SPLITS = 16


def split_tf32_hc_weight(weight: torch.Tensor):
    assert weight.dtype == torch.float32 and weight.is_contiguous()
    high = (weight.view(torch.int32) & -8192).view(torch.float32)
    return high, weight - high


def hc_mix_stats_sinkhorn_deepgemm(
    x_flat: torch.Tensor,
    weight_parts,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
):
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    assert x_flat.dtype == torch.bfloat16 and x_flat.is_contiguous()
    m, k = x_flat.shape
    high, low = weight_parts
    assert k == 20480 and high.shape == low.shape == (24, k)
    dev = x_flat.device
    pre = torch.empty((m, 4), dtype=torch.float32, device=dev)
    post = torch.empty_like(pre)
    comb = torch.empty((m, 4, 4), dtype=torch.float32, device=dev)
    if m == 0:
        return pre, post, comb

    mix_hi = torch.empty((_NUM_SPLITS, m, 24), dtype=torch.float32, device=dev)
    mix_lo = torch.empty_like(mix_hi)
    sq = torch.empty((_NUM_SPLITS, m), dtype=torch.float32, device=dev)
    unused_sq = torch.empty_like(sq)
    tf32_hc_prenorm_gemm(x_flat, high, mix_hi, sq, _NUM_SPLITS)
    tf32_hc_prenorm_gemm(x_flat, low, mix_lo, unused_sq, _NUM_SPLITS)
    _hc_mix_reduce_sinkhorn_kernel[(m,)](
        mix_hi,
        sq,
        hc_scale,
        hc_base,
        pre,
        post,
        comb,
        m,
        1.0 / k,
        rms_eps,
        MIX=24,
        HC=4,
        NUM_SLICES=_NUM_SPLITS,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        part_mix_residual_ptr=mix_lo,
        num_warps=1,
    )
    return pre, post, comb
