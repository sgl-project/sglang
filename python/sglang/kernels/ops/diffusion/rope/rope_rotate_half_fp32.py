# SPDX-License-Identifier: Apache-2.0
"""Paired split-half RoPE with FP32 products and sum, then one output cast.

Unlike the BF16-rounded RoPE kernel, this matches Cosmos/Anima eager math.
Cos/sin have shape (S, D) and are shared by every batch and attention head.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_rotate_half_fp32_kernel(
    Q,
    K,
    Cos,
    Sin,
    OutQ,
    OutK,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    col = offsets % D
    rotated = offsets + tl.where(col < D // 2, D // 2, -D // 2)
    freq = ((offsets // (H * D)) % S) * D + col
    c = tl.load(Cos + freq, mask, other=0)
    s = tl.load(Sin + freq, mask, other=0)
    if tl.program_id(1) == 0:
        x = tl.load(Q + offsets, mask, other=0).to(tl.float32)
        r = tl.load(Q + rotated, mask, other=0).to(tl.float32)
    else:
        x = tl.load(K + offsets, mask, other=0).to(tl.float32)
        r = tl.load(K + rotated, mask, other=0).to(tl.float32)
    r = tl.where(col < D // 2, -r, r)
    # Preserve the two FP32 multiplication boundaries; do not contract to FMA.
    out = x * c + r * s
    if tl.program_id(1) == 0:
        tl.store(OutQ + offsets, out, mask)
    else:
        tl.store(OutK + offsets, out, mask)


def can_use_fused_rope_rotate_half_fp32(q, k, cos, sin):
    """Conservative inference-only contract; unsupported inputs stay eager."""
    return (
        q.is_cuda
        and torch.version.hip is None
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.ndim == 4
        and all(n > 0 for n in q.shape)
        and q.shape[-1] == 128
        and k.shape == q.shape
        and k.dtype == q.dtype
        and cos.shape == sin.shape == (q.shape[1], q.shape[-1])
        and cos.dtype == sin.dtype == torch.float32
        and all(t.device == q.device and t.is_contiguous() for t in (q, k, cos, sin))
        and not (
            torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, cos, sin))
        )
    )


def fused_rope_rotate_half_fp32(q, k, cos, sin):
    """Return fresh Q/K tensors; never mutate inputs or allocate FP32 activations."""
    if not can_use_fused_rope_rotate_half_fp32(q, k, cos, sin):
        raise ValueError(
            "Expected contiguous CUDA FP16/BF16 Q/K [B,S,H,128] and FP32 cos/sin [S,128]"
        )
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    _, seq, heads, dim = q.shape
    with torch.cuda.device(q.device):
        _rope_rotate_half_fp32_kernel[(triton.cdiv(q.numel(), 1024), 2)](
            q,
            k,
            cos,
            sin,
            out_q,
            out_k,
            q.numel(),
            seq,
            heads,
            dim,
            BLOCK=1024,
            enable_fp_fusion=False,
        )
    return out_q, out_k
