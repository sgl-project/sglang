# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernels for the VDN-H3 (Video DeltaNet MiniMax-H3) linear branch.

The branch is bandwidth-bound: every stage walks a ~1.4 GiB [T, H, d] bf16
tensor at the paper workload (105k rows x 56 heads x 128). Eager, each stage
is several full passes (the 5-tap temporal conv alone reads its input five
times); these kernels read each operand once and round once at the store.

    vdn_temporal_conv_act   5-tap depthwise temporal conv + SiLU [+ L2 norm]
                            (port of OpenVDN's _tconv_act_kernel)
    vdn_silu_l2norm         SiLU [+ L2 norm] over head_dim, strided input ok
    vdn_frame_stats_prep    the four GEMM operands of the frame statistics
                            (kf16, kf32, kf32 * beta, v * beta) in [F, H, S, d]
                            off one read of k and one of v -- bitwise equal
                            to the eager chain (widening casts, same products)
    vdn_linear_epilogue     RMSNorm(d) * gate with the [F, H, S, d] -> [F*S, H*d]
                            transpose folded into the store

Inference-only: they are not bitwise against the eager bf16 chains (one
rounding instead of one per op; about one bf16 ulp, closer to fp32), which is
the same contract OpenVDN documents for its inference kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_T = 16
_BLOCK_ROWS = 32


def _check_head_dim(head_dim: int) -> None:
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(
            f"head_dim must be a power of two >= 16 (tl.arange), got {head_dim}"
        )


# --------------------------------------------------------------------------
# temporal conv + SiLU + L2 norm
# --------------------------------------------------------------------------


@triton.jit
def _tconv_act_kernel(
    X,
    W,
    OUT,
    T,
    S_,
    C_,
    BLOCK_T: tl.constexpr,
    D_: tl.constexpr,
    L2: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    chan = pid_h * D_ + tl.arange(0, D_)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = rows < T

    acc = tl.zeros((BLOCK_T, D_), dtype=tl.float32)
    for dt in tl.static_range(5):
        r = rows + dt - 2
        ok = valid & (r >= 0) & (r < T)  # zero padding, both ends
        v = tl.load(
            X + (r[:, None] * S_ + pid_s) * C_ + chan[None, :],
            mask=ok[:, None],
            other=0.0,
        ).to(tl.float32)
        wd = tl.load(W + chan * 5 + dt).to(tl.float32)
        acc += v * wd[None, :]

    y = acc * tl.sigmoid(acc)  # SiLU
    if L2:
        inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
        y = y * inv[:, None]
    tl.store(
        OUT + (rows[:, None] * S_ + pid_s) * C_ + chan[None, :],
        y.to(OUT.dtype.element_ty),
        mask=valid[:, None],
    )


def vdn_temporal_conv_act(
    x: torch.Tensor,
    w: torch.Tensor,
    heads: int,
    head_dim: int,
    l2norm: bool,
) -> torch.Tensor:
    """x [T, S, C] bf16 contiguous, w [C, 5] -> [T * S, heads, head_dim]."""
    if not x.is_cuda:
        raise ValueError("vdn_temporal_conv_act is a Triton kernel; x must be on CUDA")
    _check_head_dim(head_dim)
    T, S_, C_ = x.shape
    if C_ != heads * head_dim:
        raise ValueError(f"C={C_} != heads*head_dim={heads * head_dim}")
    if w.shape != (C_, 5):
        raise ValueError(f"w must be [C, 5], got {tuple(w.shape)}")
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    _tconv_act_kernel[(triton.cdiv(T, _BLOCK_T), S_, heads)](
        x, w, out, T, S_, C_, BLOCK_T=_BLOCK_T, D_=head_dim, L2=l2norm, num_warps=4, num_stages=2
    )
    return out.view(T * S_, heads, head_dim)


# --------------------------------------------------------------------------
# SiLU + L2 norm on a (possibly strided) [N, H, d] tensor
# --------------------------------------------------------------------------


@triton.jit
def _silu_l2norm_kernel(
    X,
    OUT,
    N,
    stride_n,
    stride_h,
    H,
    BLOCK_N: tl.constexpr,
    D_: tl.constexpr,
    L2: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = rows < N
    offs = tl.arange(0, D_)
    x = tl.load(
        X + rows[:, None] * stride_n + pid_h * stride_h + offs[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    y = x * tl.sigmoid(x)
    if L2:
        inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
        y = y * inv[:, None]
    tl.store(
        OUT + (rows[:, None] * H + pid_h) * D_ + offs[None, :],
        y.to(OUT.dtype.element_ty),
        mask=valid[:, None],
    )


def vdn_silu_l2norm(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    """tokens [N, H, d] (last dim contiguous) -> contiguous [N, H, d]."""
    if not tokens.is_cuda:
        raise ValueError("vdn_silu_l2norm is a Triton kernel; tokens must be on CUDA")
    N, H, D = tokens.shape
    _check_head_dim(D)
    if tokens.stride(-1) != 1:
        tokens = tokens.contiguous()
    out = torch.empty((N, H, D), dtype=tokens.dtype, device=tokens.device)
    if N == 0:
        return out
    _silu_l2norm_kernel[(triton.cdiv(N, _BLOCK_ROWS), H)](
        tokens, out, N, tokens.stride(0), tokens.stride(1), H,
        BLOCK_N=_BLOCK_ROWS, D_=D, L2=l2norm, num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# frame statistics prologue
# --------------------------------------------------------------------------


@triton.jit
def _frame_stats_prep_kernel(
    K,
    V,
    BETA,
    K16,
    K32,
    KB32,
    VB,
    S_,
    H,
    BLOCK_S: tl.constexpr,
    D_: tl.constexpr,
):
    pid_s = tl.program_id(0)
    f = tl.program_id(1)
    h = tl.program_id(2)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = s < S_
    offs = tl.arange(0, D_)
    rows = f * S_ + s  # token rows
    src = (rows[:, None] * H + h) * D_ + offs[None, :]  # [F*S, H, d]
    dst = ((f * H + h) * S_ + s)[:, None] * D_ + offs[None, :]  # [F, H, S, d]
    k = tl.load(K + src, mask=valid[:, None], other=0.0)
    v = tl.load(V + src, mask=valid[:, None], other=0.0)
    beta = tl.load(BETA + rows * H + h, mask=valid, other=0.0)
    k32 = k.to(tl.float32)
    beta32 = beta.to(tl.float32)
    tl.store(K16 + dst, k, mask=valid[:, None])
    tl.store(K32 + dst, k32, mask=valid[:, None])
    tl.store(KB32 + dst, k32 * beta32[:, None], mask=valid[:, None])
    vb = (v.to(tl.float32) * beta32[:, None]).to(VB.dtype.element_ty)
    tl.store(VB + dst, vb, mask=valid[:, None])


def vdn_frame_stats_prep(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """key/value [F*S, H, d] bf16 contiguous, beta [F*S, H] bf16 ->
    (k16 [F,H,S,d] bf16, k32 fp32, k32*beta fp32, v*beta bf16), all contiguous."""
    if not key.is_cuda:
        raise ValueError("vdn_frame_stats_prep is a Triton kernel; inputs must be on CUDA")
    rows, H, D = key.shape
    _check_head_dim(D)
    if rows != num_frames * tokens_per_frame:
        raise ValueError(f"{rows} rows != {num_frames} x {tokens_per_frame}")
    key = key.contiguous()
    value = value.contiguous()
    beta = beta.to(key.dtype).contiguous()
    shape = (num_frames, H, tokens_per_frame, D)
    k16 = torch.empty(shape, dtype=key.dtype, device=key.device)
    k32 = torch.empty(shape, dtype=torch.float32, device=key.device)
    kb32 = torch.empty(shape, dtype=torch.float32, device=key.device)
    vb = torch.empty(shape, dtype=value.dtype, device=key.device)
    _frame_stats_prep_kernel[(triton.cdiv(tokens_per_frame, _BLOCK_ROWS), num_frames, H)](
        key, value, beta, k16, k32, kb32, vb, tokens_per_frame, H,
        BLOCK_S=_BLOCK_ROWS, D_=D, num_warps=4,
    )
    return k16, k32, kb32, vb


# --------------------------------------------------------------------------
# readout epilogue: RMSNorm(d) * gate, [F, H, S, d] -> [F*S, H*d]
# --------------------------------------------------------------------------


@triton.jit
def _linear_epilogue_kernel(
    R,
    W,
    G,
    OUT,
    S_,
    H,
    eps,
    BLOCK_S: tl.constexpr,
    D_: tl.constexpr,
):
    pid_s = tl.program_id(0)
    f = tl.program_id(1)
    h = tl.program_id(2)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = s < S_
    offs = tl.arange(0, D_)
    src = ((f * H + h) * S_ + s)[:, None] * D_ + offs[None, :]
    rows = f * S_ + s
    dst = (rows[:, None] * H + h) * D_ + offs[None, :]
    r = tl.load(R + src, mask=valid[:, None], other=0.0).to(tl.float32)
    ms = tl.sum(r * r, axis=1) / D_
    w = tl.load(W + offs).to(tl.float32)
    g = tl.load(G + dst, mask=valid[:, None], other=0.0).to(tl.float32)
    y = r * (1.0 / tl.sqrt(ms + eps))[:, None] * w[None, :] * g
    tl.store(OUT + dst, y.to(OUT.dtype.element_ty), mask=valid[:, None])


def vdn_linear_epilogue(
    readout: torch.Tensor,
    norm_weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """readout [F, H, S, d] bf16 contiguous, norm_weight [d], gate [F*S, H, d]
    -> [F*S, H*d] bf16."""
    if not readout.is_cuda:
        raise ValueError("vdn_linear_epilogue is a Triton kernel; readout must be on CUDA")
    F, H, S_, D = readout.shape
    _check_head_dim(D)
    readout = readout.contiguous()
    gate = gate.reshape(F * S_, H, D).to(readout.dtype).contiguous()
    out = torch.empty((F * S_, H * D), dtype=readout.dtype, device=readout.device)
    _linear_epilogue_kernel[(triton.cdiv(S_, _BLOCK_ROWS), F, H)](
        readout, norm_weight.contiguous(), gate, out, S_, H, float(eps),
        BLOCK_S=_BLOCK_ROWS, D_=D, num_warps=4,
    )
    return out


__all__ = [
    "vdn_frame_stats_prep",
    "vdn_linear_epilogue",
    "vdn_silu_l2norm",
    "vdn_temporal_conv_act",
]
