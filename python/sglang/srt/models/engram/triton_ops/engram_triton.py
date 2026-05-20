"""Triton kernels for Engram forward fusion."""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _pick_block_h(hidden_size: int) -> int:
    if hidden_size <= 64:
        return 64
    if hidden_size <= 128:
        return 128
    if hidden_size <= 256:
        return 256
    return 512


def _pick_block_c(hidden_size: int) -> int:
    if hidden_size <= 64:
        return 64
    if hidden_size <= 128:
        return 128
    if hidden_size <= 256:
        return 256
    return 512


@triton.jit
def _gate_value_kernel(
    normed_key_ptr,
    normed_query_ptr,
    value_ptr,
    output_ptr,
    stride_nk_b,
    stride_nk_l,
    stride_nk_g,
    stride_nk_h,
    stride_nq_b,
    stride_nq_l,
    stride_nq_g,
    stride_nq_h,
    stride_val_b,
    stride_val_l,
    stride_val_h,
    stride_out_b,
    stride_out_l,
    stride_out_g,
    stride_out_h,
    B,
    L,
    G,
    scale,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * L * G
    if pid >= total:
        return

    g = pid % G
    pid = pid // G
    l = pid % L
    b = pid // L

    acc = tl.zeros([1], dtype=tl.float32)
    for h in tl.static_range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        nk = tl.load(
            normed_key_ptr
            + b * stride_nk_b
            + l * stride_nk_l
            + g * stride_nk_g
            + offs * stride_nk_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        nq = tl.load(
            normed_query_ptr
            + b * stride_nq_b
            + l * stride_nq_l
            + g * stride_nq_g
            + offs * stride_nq_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(nk * nq, axis=0)

    gate = acc / scale
    abs_gate = tl.maximum(tl.abs(gate), eps)
    sign_gate = tl.where(gate > 0, 1.0, tl.where(gate < 0, -1.0, 0.0))
    gate = tl.sqrt(abs_gate) * sign_gate
    gate = tl.sigmoid(gate)

    for h in tl.static_range(0, H, BLOCK_H):
        offs = h + tl.arange(0, BLOCK_H)
        mask = offs < H
        val = tl.load(
            value_ptr + b * stride_val_b + l * stride_val_l + offs * stride_val_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = val * gate
        tl.store(
            output_ptr
            + b * stride_out_b
            + l * stride_out_l
            + g * stride_out_g
            + offs * stride_out_h,
            out,
            mask=mask,
        )


def gate_value_fused(
    normed_key: torch.Tensor,
    normed_query: torch.Tensor,
    value_proj: torch.Tensor,
    hidden_size: int,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if normed_key.device.type != "cuda":
        raise RuntimeError("Triton kernel requires CUDA tensors")

    if out is None:
        out = torch.empty(
            (*normed_key.shape[:-1], hidden_size),
            device=normed_key.device,
            dtype=value_proj.dtype,
        )

    B, L, G, H = normed_key.shape
    if H != hidden_size:
        raise ValueError("hidden_size mismatch for gate_value_fused")

    grid = (B * L * G,)
    block_h = _pick_block_h(H)
    num_warps = 4 if block_h <= 128 else 8

    _gate_value_kernel[grid](
        normed_key,
        normed_query,
        value_proj,
        out,
        normed_key.stride(0),
        normed_key.stride(1),
        normed_key.stride(2),
        normed_key.stride(3),
        normed_query.stride(0),
        normed_query.stride(1),
        normed_query.stride(2),
        normed_query.stride(3),
        value_proj.stride(0),
        value_proj.stride(1),
        value_proj.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        L,
        G,
        float(math.sqrt(hidden_size)),
        H=H,
        eps=eps,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )

    return out


@triton.jit
def _rms_norm_and_transpose_kernel(
    X_ptr,  # (B, T, G, C)
    W_ptr,  # (G, C)
    Y_ptr,  # (B, G*C, T)
    eps,
    B,
    T,
    G,
    C,
    stride_xb,
    stride_xt,
    stride_xg,
    stride_xc,
    stride_yb,
    stride_ygc,
    stride_yt,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    b = row_idx // (T * G)
    rem = row_idx % (T * G)
    t = rem // G
    g = rem % G

    X_ptr += b * stride_xb + t * stride_xt + g * stride_xg
    W_ptr += g * C

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C
    x = tl.load(X_ptr + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / C
    rsqrt = tl.math.rsqrt(var + eps)
    out = xf * rsqrt * w

    y_base = Y_ptr + b * stride_yb + g * C * stride_ygc + t * stride_yt
    tl.store(y_base + cols * stride_ygc, out, mask=mask)


def shortconv_preprocess(x, weights, eps):
    B, T, G, C = x.shape
    y = torch.empty((B, G * C, T), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = triton.next_power_of_2(C)

    grid = (B * T * G,)

    _rms_norm_and_transpose_kernel[grid](
        x,
        weights,
        y,
        eps,
        B,
        T,
        G,
        C,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@triton.jit
def _rms_norm_fused_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_row,
    N_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_row_ptr = X_ptr + row_idx * stride_row
    y_row_ptr = Y_ptr + row_idx * stride_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_cols

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N_cols
    inv_std = tl.extra.cuda.libdevice.rsqrt(var + eps)

    w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
    y = x * inv_std * w

    tl.store(y_row_ptr + cols, y, mask=mask)


def rms_norm_group_fused(x, weights, eps=1e-5):
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N)
    M, _ = x_2d.shape

    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)

    _rms_norm_fused_kernel[(M,)](
        x_2d,
        weights,
        y,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if N <= 1024 else 8,
    )
    return y.reshape(*orig_shape)
