"""Attention for MiMo's independent groups of at most four audio tokens."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mimo_local_attention(
    Q,
    K,
    V,
    O,
    COUNT,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QT: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KT: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    batch, head = bh // HEADS, bh % HEADS
    token, dim = tl.arange(0, 4), tl.arange(0, 16)
    valid = (bh[:, None, None] < COUNT) & (token[None, :, None] < TOKENS)
    q = tl.load(
        Q
        + batch[:, None, None] * QB
        + head[:, None, None] * QH
        + token[None, :, None] * QT
        + dim[None, None, :],
        valid,
        other=0,
    ).to(tl.float32)
    k = tl.load(
        K
        + batch[:, None, None] * KB
        + head[:, None, None] * KH
        + token[None, :, None] * KT
        + dim[None, None, :],
        valid,
        other=0,
    ).to(tl.float32)
    v = tl.load(
        V
        + batch[:, None, None] * VB
        + head[:, None, None] * VH
        + token[None, :, None] * VT
        + dim[None, None, :],
        valid,
        other=0,
    ).to(tl.float32)
    scores = tl.sum(q[:, :, None, :] * k[:, None, :, :], axis=3) * SCALE
    allowed = token[None, None, :] < TOKENS
    if CAUSAL:
        allowed = allowed & (token[None, None, :] <= token[None, :, None])
    scores = tl.where(allowed, scores, -float("inf"))
    probabilities = tl.exp(scores - tl.max(scores, axis=2)[:, :, None])
    denominator = tl.sum(probabilities, axis=2)
    probabilities = probabilities.to(Q.dtype.element_ty).to(tl.float32)
    result = tl.sum(probabilities[:, :, :, None] * v[:, None, :, :], axis=2)
    result /= denominator[:, :, None]
    offsets = (
        (batch[:, None, None] * TOKENS + token[None, :, None]) * HEADS
        + head[:, None, None]
    ) * 16 + dim[None, None, :]
    tl.store(O + offsets, result, valid)


def mimo_local_attention(q, k, v, scale=0.25, is_causal=False):
    """Return contiguous [batch, token, head, dim] attention without a mask."""
    assert q.is_cuda and q.device == k.device == v.device
    assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.bfloat16, torch.float16)
    assert q.shape == k.shape == v.shape and q.ndim == 4
    batch, heads, tokens, dim = q.shape
    assert 0 < tokens <= 4 and dim == 16
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    out = torch.empty((batch, tokens, heads, dim), device=q.device, dtype=q.dtype)
    if batch * heads:
        _mimo_local_attention[(triton.cdiv(batch * heads, 4),)](
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
