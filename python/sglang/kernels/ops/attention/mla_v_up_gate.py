"""Fused absorbed-MLA value up-projection + output gate for decode batches.

One launch for the per-head bmm and Kimi-K3's output gate:
    y[t, h] = bf16(attn[t, h] @ w_vc[h])                      # [T, H, V]
    out     = bf16(y * bf16(sigmoid(gate)))                   # [T, H * V]
with the rounding points of the unfused bmm + mla_output_gate pair. The bmm is
tiny ([T, 512] x [512, 128] per head) and pure launch latency as a batched GEMM.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

MAX_TOKENS: int = 64


@triton.jit
def _mla_v_up_gate_kernel(
    attn_ptr,  # [T, H, L]
    w_ptr,  # [H, L, V]
    gate_ptr,  # [T, H * V]
    out_ptr,  # [T, H * V]
    T,
    stride_at,
    stride_ah,
    stride_wh,
    stride_wl,
    stride_wv,
    stride_gt,
    stride_ot,
    L: tl.constexpr,
    V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_L: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    h = tl.program_id(0)
    v0 = tl.program_id(1) * BLOCK_V
    t0 = tl.program_id(2) * BLOCK_T
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_v = v0 + tl.arange(0, BLOCK_V)
    mask_t = offs_t < T

    acc = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
    for l0 in tl.range(0, L, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        a = tl.load(
            attn_ptr + offs_t[:, None] * stride_at + h * stride_ah + offs_l[None, :],
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        w = tl.load(
            w_ptr
            + h * stride_wh
            + offs_l[:, None] * stride_wl
            + offs_v[None, :] * stride_wv
        )
        acc += tl.dot(a, w)

    y = acc.to(tl.bfloat16)
    cols = h * V + offs_v
    if HAS_GATE:
        g = tl.load(
            gate_ptr + offs_t[:, None] * stride_gt + cols[None, :],
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float32)
        sig = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16)
        y = (y.to(tl.float32) * sig.to(tl.float32)).to(tl.bfloat16)
    tl.store(
        out_ptr + offs_t[:, None] * stride_ot + cols[None, :], y, mask=mask_t[:, None]
    )


def covered(
    attn: torch.Tensor, w_vc: torch.Tensor, gate: Optional[torch.Tensor]
) -> bool:
    T, H, L = attn.shape
    return (
        0 < T <= MAX_TOKENS
        and w_vc.dtype == torch.bfloat16
        and w_vc.dim() == 3
        and w_vc.shape[0] == H
        and w_vc.shape[1] == L
        and attn.stride(-1) == 1
        and triton.next_power_of_2(L) == L
        and w_vc.shape[2] % 32 == 0
        and (
            gate is None
            or (gate.shape == (T, H * w_vc.shape[2]) and gate.stride(-1) == 1)
        )
    )


def mla_v_up_gate(
    attn: torch.Tensor,
    w_vc: torch.Tensor,
    gate: Optional[torch.Tensor],
) -> torch.Tensor:
    """attn [T, H, L] x w_vc [H, L, V] (any strides) -> [T, H * V] bf16, gated
    by sigmoid(gate) [T, H * V] when given."""
    T, H, L = attn.shape
    V = w_vc.shape[2]
    out = torch.empty((T, H * V), dtype=torch.bfloat16, device=attn.device)
    block_t = 16 if T <= 16 else 32
    block_v = 32
    gate_arg = gate if gate is not None else out
    _mla_v_up_gate_kernel[(H, V // block_v, triton.cdiv(T, block_t))](
        attn,
        w_vc,
        gate_arg,
        out,
        T,
        attn.stride(0),
        attn.stride(1),
        w_vc.stride(0),
        w_vc.stride(1),
        w_vc.stride(2),
        gate_arg.stride(0),
        out.stride(0),
        L=L,
        V=V,
        BLOCK_T=block_t,
        BLOCK_V=block_v,
        BLOCK_L=128,
        HAS_GATE=gate is not None,
        num_warps=4,
    )
    return out
