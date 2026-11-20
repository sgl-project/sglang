# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D", "NB"],
# )
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_x = (o_t[:, None] < T) & (o_d[None, :] < D)
    p_x = x + o_t[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


@triton.jit(do_not_specialize=["num_tokens"])
def gdn_prefill_qkv_prepare_kernel(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    num_tokens,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    eps,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    V_HEADS: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    # Flatten token/head rows like l2norm_fwd, preserving its reduction tree.
    q_rows = num_tokens * Q_HEADS
    k_rows = num_tokens * K_HEADS
    v_rows = num_tokens * V_HEADS
    q_blocks = tl.cdiv(q_rows, BLOCK_ROWS)
    k_blocks = tl.cdiv(k_rows, BLOCK_ROWS)

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_DIM)[None, :]

    if pid < q_blocks:
        row = pid * BLOCK_ROWS + rows
        token = row // Q_HEADS
        head = row % Q_HEADS
        mask = (row < q_rows) & (cols < Q_DIM)
        offsets = token * stride_q_t + head * stride_q_h + cols * stride_q_d
        q_values = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
        q_values = (
            q_values / tl.sqrt(tl.sum(q_values * q_values, axis=1) + eps)[:, None]
        )
        tl.store(q_out + row * Q_DIM + cols, q_values, mask=mask)
    elif pid < q_blocks + k_blocks:
        row = (pid - q_blocks) * BLOCK_ROWS + rows
        token = row // K_HEADS
        head = row % K_HEADS
        mask = (row < k_rows) & (cols < K_DIM)
        offsets = token * stride_k_t + head * stride_k_h + cols * stride_k_d
        k_values = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
        k_values = (
            k_values / tl.sqrt(tl.sum(k_values * k_values, axis=1) + eps)[:, None]
        )
        tl.store(k_out + row * K_DIM + cols, k_values, mask=mask)
    else:
        row = (pid - q_blocks - k_blocks) * BLOCK_ROWS + rows
        token = row // V_HEADS
        head = row % V_HEADS
        mask = (row < v_rows) & (cols < V_DIM)
        offsets = token * stride_v_t + head * stride_v_h + cols * stride_v_d
        v_values = tl.load(v + offsets, mask=mask)
        tl.store(v_out + row * V_DIM + cols, v_values, mask=mask)


def gdn_prefill_qkv_prepare_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize Q/K and materialize strided Q/K/V in a single launch.

    Match l2norm_fwd rounding and retain contiguous V storage.
    """
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError(
            "GDN fused prepare requires equal Q/K [T, Hqk, D] shapes and "
            "V [T, Hv, D], got "
            f"{q.shape=}, {k.shape=}, {v.shape=}"
        )
    if v.shape[0] != q.shape[0] or v.shape[2] != q.shape[2]:
        raise ValueError(
            "GDN fused prepare requires common token and head-dim axes, got "
            f"{q.shape=}, {v.shape=}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("GDN fused prepare requires Q/K/V on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("GDN fused prepare requires equal Q/K/V dtypes")

    T, H_QK, D = q.shape
    H_V = v.shape[1]
    if D > 512:
        raise ValueError(f"GDN fused prepare supports head dim <= 512, got {D}")
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    copy_v = not v.is_contiguous()
    v_out = torch.empty(v.shape, dtype=v.dtype, device=v.device) if copy_v else v
    if T == 0:
        return q_out, k_out, v_out
    block_rows = 16
    grid = (
        2 * triton.cdiv(T * H_QK, block_rows)
        + (triton.cdiv(T * H_V, block_rows) if copy_v else 0),
    )
    gdn_prefill_qkv_prepare_kernel[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        T,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        eps,
        Q_HEADS=H_QK,
        K_HEADS=H_QK,
        V_HEADS=H_V,
        Q_DIM=D,
        K_DIM=D,
        V_DIM=D,
        BLOCK_DIM=triton.next_power_of_2(D),
        BLOCK_ROWS=block_rows,
        num_warps=8,
        num_stages=3,
    )
    return q_out, k_out, v_out


class L2NormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        return l2norm_fwd(x, eps, output_dtype)


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
