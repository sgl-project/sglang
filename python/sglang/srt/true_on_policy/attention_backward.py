"""First-order FA4 adjoint for the shared dense TOP forward.

The saved output and natural-log LSE come from SGLang's batch-invariant forward.
Upstream FA4 supplies the packed BF16 Q/K/V adjoint, not a replacement forward.
LSE is auxiliary metadata, not a differentiable output of this adapter.
"""

import torch
from torch.autograd.function import once_differentiable


class _DenseVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, scale):
        # Optional CUDA dependencies are resolved only on the training path.
        from flash_attn.cute.interface import _flash_attn_bwd

        from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_varlen_func

        out, lse = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=scale,
            causal=True,
            num_splits=1,
            pack_gqa=True,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
        ctx.max_q, ctx.max_k, ctx.scale = max_q, max_k, scale
        ctx.attention_backward = _flash_attn_bwd
        ctx.mark_non_differentiable(lse)
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, _dlse):
        if dout is None:
            return (None,) * 8
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        # Forward packing leaves out/LSE in canonical layout; retain the adjoint policy.
        dq, dk, dv = ctx.attention_backward(
            q=q,
            k=k,
            v=v,
            out=out,
            dout=dout,
            lse=lse,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=ctx.max_q,
            max_seqlen_k=ctx.max_k,
            softmax_scale=ctx.scale,
            causal=True,
            pack_gqa=False,
            deterministic=True,
        )
        return dq, dk, dv, None, None, None, None, None


def dense_varlen_forward(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=True,
    num_splits=1,
    pack_gqa=True,
    **kwargs,
):
    """Retain backward only for the packed causal dense TOP training operation."""
    if kwargs or not causal or num_splits != 1 or not pack_gqa:
        raise NotImplementedError(
            "TOP FA4 backward requires packed-GQA causal attention with split1"
        )
    if any(t.ndim != 3 for t in (q, k, v)):
        raise ValueError(
            "TOP FA4 backward requires packed [tokens, heads, head_dim] Q/K/V"
        )
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("TOP FA4 backward requires packed sequence boundaries")
    return _DenseVarlenAttention.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
    )
