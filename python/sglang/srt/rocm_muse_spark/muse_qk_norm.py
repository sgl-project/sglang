"""Fused per-head RMSNorm for q and k (+ query scale) in one Triton launch.

Replaces, on ROCm decode/prefill:
    q = F.rms_norm(q.view(-1, D), (D,), None, eps).view_as(q).to(bf16) * scale_query_by
    k = F.rms_norm(k.view(-1, D), (D,), None, eps).view_as(k).to(bf16)
(5 kernels: two norms, two casts/copies of the split views, one scalar multiply) with a single
kernel. Rounding matches the eager path bit for bit: normed values are rounded to bf16 first and the
scale is applied to the rounded value (RNE), exactly like ``bf16_tensor * python_float``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_pairs(x, cos, sin, H: tl.constexpr, D: tl.constexpr):
    bf = tl.bfloat16
    a, b = tl.split(tl.reshape(x.to(bf).to(tl.float32), (H, D // 2, 2)))
    ra = (a * cos).to(bf).to(tl.float32) - (b * sin).to(bf).to(tl.float32)
    rb = (b * cos).to(bf).to(tl.float32) + (a * sin).to(bf).to(tl.float32)
    return tl.reshape(tl.join(ra, rb), (H, D))


@triton.jit
def _qk_norm_kernel(
    Q,
    K,
    QO,
    KO,
    q_stride,
    k_stride,
    eps,
    q_scale,
    POS,
    COS_SIN,
    cos_sin_stride,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    t = tl.program_id(0)
    offs_q = tl.arange(0, HQ * D)
    q = tl.load(Q + t * q_stride + offs_q).to(tl.float32)
    q2 = tl.reshape(q, (HQ, D))
    rstd_q = tl.rsqrt(tl.sum(q2 * q2, axis=1) / D + eps)
    qn = (q2 * rstd_q[:, None]).to(tl.bfloat16).to(tl.float32) * q_scale
    offs_k = tl.arange(0, HK * D)
    k = tl.load(K + t * k_stride + offs_k).to(tl.float32)
    k2 = tl.reshape(k, (HK, D))
    rstd_k = tl.rsqrt(tl.sum(k2 * k2, axis=1) / D + eps)
    kn = k2 * rstd_k[:, None]
    if HAS_ROPE:
        # GPT-J (interleaved) rotary as the separate rotary kernel computes it: on the stored BF16 values, with the
        # fp32 cos | sin cache row held in BF16 and each product rounded to BF16 before the add.
        cs = COS_SIN + tl.load(POS + t) * cos_sin_stride + tl.arange(0, D // 2)
        cos = tl.load(cs).to(tl.bfloat16).to(tl.float32)[None, :]
        sin = tl.load(cs + D // 2).to(tl.bfloat16).to(tl.float32)[None, :]
        qn = _rope_pairs(qn, cos, sin, HQ, D)
        kn = _rope_pairs(kn, cos, sin, HK, D)
    tl.store(QO + t * (HQ * D) + offs_q, tl.reshape(qn, (HQ * D,)).to(tl.bfloat16))
    tl.store(KO + t * (HK * D) + offs_k, tl.reshape(kn, (HK * D,)).to(tl.bfloat16))


def qk_norm_supported(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> bool:
    return (
        q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and q.dim() == 2
        and k.dim() == 2
        and q.stride(1) == 1
        and k.stride(1) == 1
        and q.shape[1] % head_dim == 0
        and k.shape[1] % head_dim == 0
        and (head_dim & (head_dim - 1)) == 0
        and q.shape[0] == k.shape[0]
    )


def fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    eps: float,
    q_scale: float,
    rope: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """q: [T, HQ*D] (any row stride, e.g. a split view of the qkv buffer), k: [T, HK*D]. Returns
    contiguous bf16 (normed_q * q_scale, normed_k). ``rope`` = (positions [T], fp32 cos | sin cache
    [positions, head_dim]) also applies the GPT-J rotary to both, as the separate rotary kernel would."""
    T = q.shape[0]
    qo = torch.empty(T, q.shape[1], dtype=torch.bfloat16, device=q.device)
    ko = torch.empty(T, k.shape[1], dtype=torch.bfloat16, device=k.device)
    if T == 0:
        return qo, ko
    _qk_norm_kernel[(T,)](
        q,
        k,
        qo,
        ko,
        q.stride(0),
        k.stride(0),
        float(eps),
        float(q_scale),
        rope[0] if rope is not None else None,
        rope[1] if rope is not None else None,
        rope[1].stride(0) if rope is not None else 0,
        HQ=q.shape[1] // head_dim,
        HK=k.shape[1] // head_dim,
        D=head_dim,
        HAS_ROPE=rope is not None,
        # gfx1250: one wave per token is faster than four at 8K-16K rows and ties at 128-512.
        num_warps=1,
    )
    return qo, ko
