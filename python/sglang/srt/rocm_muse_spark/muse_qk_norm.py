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
def _qk_norm_kernel(
    Q,
    K,
    QO,
    KO,
    q_stride,
    k_stride,
    eps,
    q_scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
):
    t = tl.program_id(0)
    offs_q = tl.arange(0, HQ * D)
    q = tl.load(Q + t * q_stride + offs_q).to(tl.float32)
    q2 = tl.reshape(q, (HQ, D))
    rstd_q = tl.rsqrt(tl.sum(q2 * q2, axis=1) / D + eps)
    qn = (q2 * rstd_q[:, None]).to(tl.bfloat16).to(tl.float32) * q_scale
    tl.store(QO + t * (HQ * D) + offs_q, tl.reshape(qn, (HQ * D,)).to(tl.bfloat16))
    offs_k = tl.arange(0, HK * D)
    k = tl.load(K + t * k_stride + offs_k).to(tl.float32)
    k2 = tl.reshape(k, (HK, D))
    rstd_k = tl.rsqrt(tl.sum(k2 * k2, axis=1) / D + eps)
    tl.store(
        KO + t * (HK * D) + offs_k,
        tl.reshape(k2 * rstd_k[:, None], (HK * D,)).to(tl.bfloat16),
    )


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
    q: torch.Tensor, k: torch.Tensor, head_dim: int, eps: float, q_scale: float
):
    """q: [T, HQ*D] (any row stride, e.g. a split view of the qkv buffer), k: [T, HK*D]. Returns
    contiguous bf16 (normed_q * q_scale, normed_k)."""
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
        HQ=q.shape[1] // head_dim,
        HK=k.shape[1] // head_dim,
        D=head_dim,
        num_warps=4,
    )
    return qo, ko
