"""DeepSeek-V4.1 mHC (each hyper-connection collapses with the previous one's ``pre``) as one TOP program.

Serving and the trainer run the same three kernels at every row count; each was measured row-invariant on H200
(rows at M equal the same rows inside a larger batch, bit for bit):

    mix stats  hc_mix_stats_sinkhorn: split-K partial GEMM at a K-fixed slice count, then reduce + Sinkhorn
    combine    hc_combine to BF16, then the sublayer's RMSNorm (the fused hc_combine_norm is not row-invariant)
    post       mhc_post_split_h

Serving's row-count-selected alternatives (the DeepGEMM and BF16x3 prenorm GEMMs, the TileLang post, the fused
combine+norm, the Blackwell boundary fusions, the stats side stream's overlap) are not part of the program.
"""

import torch


def mix_stats(x, fn, scale, base, *, mult, sinkhorn_iters, rms_eps, eps):
    """(pre [T, mult], post [T, mult], comb [T, mult, mult]) FP32 from the streams ``x`` [T, mult, H] BF16."""
    if x.dtype != torch.bfloat16 or x.ndim != 3 or fn.dtype != torch.float32:
        raise NotImplementedError(
            "[top] V4.1 mHC stats need BF16 streams [T, mult, H] and an FP32 hc_fn"
        )
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn

    return hc_mix_stats_sinkhorn(
        x.flatten(1).contiguous(),
        fn,
        scale,
        base,
        mult,
        sinkhorn_iters,
        rms_eps,
        eps,
    )


def collapse(x, pre):
    """``sum_k pre[k] * x[k]`` in BF16: the head input before the final norm."""
    if x.dtype != torch.bfloat16 or pre.dtype != torch.float32:
        raise NotImplementedError(
            "[top] V4.1 mHC combine needs BF16 streams and FP32 pre"
        )
    from sglang.kernels.ops.layernorm.mhc import mhc_combine

    return mhc_combine(
        x.flatten(1).contiguous(),
        pre.contiguous(),
        x.shape[1],
        torch.bfloat16,
    )


def combine(x, pre, norm):
    """``norm(collapse(x, pre))``: a sublayer's input."""
    return norm(collapse(x, pre))


def post(y, residual, post_mix, comb):
    """``R'[i] = post[i] * y + sum_j comb[j][i] * R[j]``."""
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

    if y.shape[0] == 0:
        return residual.new_empty(residual.shape)
    return mhc_post_split_h(
        y.contiguous(), residual.contiguous(), post_mix.contiguous(), comb.contiguous()
    )
