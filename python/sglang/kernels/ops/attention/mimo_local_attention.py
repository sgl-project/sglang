"""Attention for MiMo's independent groups of at most four audio tokens."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mimo_local_attention_kernel(
    Q,
    K,
    V,
    O,
    num_batch_heads,
    NUM_HEADS: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vt: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    batch, head = bh // NUM_HEADS, bh % NUM_HEADS
    token, dim = tl.arange(0, 4), tl.arange(0, 16)
    valid = (bh[:, None, None] < num_batch_heads) & (token[None, :, None] < NUM_TOKENS)
    q = tl.load(
        Q
        + batch[:, None, None] * stride_qb
        + head[:, None, None] * stride_qh
        + token[None, :, None] * stride_qt
        + dim[None, None, :],
        valid,
        other=0,
    ).to(tl.float32)
    k = tl.load(
        K
        + batch[:, None, None] * stride_kb
        + head[:, None, None] * stride_kh
        + token[None, :, None] * stride_kt
        + dim[None, None, :],
        valid,
        other=0,
    ).to(tl.float32)
    v = tl.load(
        V
        + batch[:, None, None] * stride_vb
        + head[:, None, None] * stride_vh
        + token[None, :, None] * stride_vt
        + dim[None, None, :],
        valid,
        other=0,
    ).to(tl.float32)
    scores = tl.sum(q[:, :, None, :] * k[:, None, :, :], axis=3) * SCALE
    allowed = token[None, None, :] < NUM_TOKENS
    if CAUSAL:
        allowed = allowed & (token[None, None, :] <= token[None, :, None])
    scores = tl.where(allowed, scores, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, axis=2)[:, :, None])
    denominator = tl.sum(weights, axis=2)
    # Match the tested cuDNN path: round the unnormalized exponential weights,
    # then normalize the weighted sum. Rounding normalized weights differs.
    weights = weights.to(Q.dtype.element_ty).to(tl.float32)
    result = tl.sum(weights[:, :, :, None] * v[:, None, :, :], axis=2)
    result /= denominator[:, :, None]
    offsets = (
        (batch[:, None, None] * NUM_TOKENS + token[None, :, None]) * NUM_HEADS
        + head[:, None, None]
    ) * 16 + dim[None, None, :]
    tl.store(O + offsets, result, valid)


def mimo_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 0.25,
    is_causal: bool = False,
) -> torch.Tensor:
    """Attend within audio groups of 1–4 tokens with head dimension 16.

    Q/K/V have shape [batch, head, token, dim]; output is contiguous
    [batch, token, head, dim]. Only the optional causal mask is supported.
    """
    assert q.is_cuda and q.device == k.device == v.device
    assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.bfloat16, torch.float16)
    assert q.shape == k.shape == v.shape and q.ndim == 4
    batch, heads, tokens, dim = q.shape
    assert 0 < tokens <= 4 and dim == 16
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    out = torch.empty((batch, tokens, heads, dim), device=q.device, dtype=q.dtype)
    if batch * heads:
        _mimo_local_attention_kernel[(triton.cdiv(batch * heads, 4),)](
            q,
            k,
            v,
            out,
            batch * heads,
            heads,
            tokens,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            scale,
            is_causal,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out
