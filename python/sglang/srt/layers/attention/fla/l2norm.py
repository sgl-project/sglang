# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


@triton.jit
def fused_l2norm_qk_kernel(
    q, k, q_out, k_out, eps,
    T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):
    """Fused L2 norm for q and k (D <= 512). Each block processes both q and k."""
    i_t = tl.program_id(0)
    
    rows = i_t * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    row_mask = rows < T
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None] * D + cols[None, :]
    
    # Process q
    b_q = tl.load(q + offs, mask=mask, other=0.0).to(tl.float32)
    b_q_var = tl.sum(b_q * b_q, axis=1)
    b_q_out = b_q / tl.sqrt(b_q_var + eps)[:, None]
    tl.store(q_out + offs, b_q_out.to(q_out.dtype.element_ty), mask=mask)
    
    # Process k
    b_k = tl.load(k + offs, mask=mask, other=0.0).to(tl.float32)
    b_k_var = tl.sum(b_k * b_k, axis=1)
    b_k_out = b_k / tl.sqrt(b_k_var + eps)[:, None]
    tl.store(k_out + offs, b_k_out.to(k_out.dtype.element_ty), mask=mask)


@triton.jit
def fused_l2norm_qk_kernel1(
    q, k, q_out, k_out, D, BD: tl.constexpr, eps,
):
    """Fused L2 norm for q and k (D > 512). Each block processes both q and k."""
    i_t = tl.program_id(0)
    
    cols = tl.arange(0, BD)
    mask = cols < D
    offs = i_t * D + cols
    
    # Process q
    b_q = tl.load(q + offs, mask=mask, other=0.0).to(tl.float32)
    b_q_var = tl.sum(b_q * b_q, axis=0)
    b_q_out = b_q / tl.sqrt(b_q_var + eps)
    tl.store(q_out + offs, b_q_out, mask=mask)
    
    # Process k
    b_k = tl.load(k + offs, mask=mask, other=0.0).to(tl.float32)
    b_k_var = tl.sum(b_k * b_k, axis=0)
    b_k_out = b_k / tl.sqrt(b_k_var + eps)
    tl.store(k_out + offs, b_k_out, mask=mask)


def fused_l2norm_qk(
    q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused L2 norm for q and k in a single kernel launch."""
    q_shape_og, k_shape_og = q.shape, k.shape
    q_flat = q.view(-1, q.shape[-1])
    k_flat = k.view(-1, k.shape[-1])
    T, D = q_flat.shape
    
    if output_dtype is None:
        q_out, k_out = torch.empty_like(q_flat), torch.empty_like(k_flat)
    else:
        q_out = torch.empty_like(q_flat, dtype=output_dtype)
        k_out = torch.empty_like(k_flat, dtype=output_dtype)
    
    MAX_FUSED_SIZE = 65536 // q.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    
    if D <= 512:
        BT = 16
        fused_l2norm_qk_kernel[(triton.cdiv(T, BT),)](
            q_flat, k_flat, q_out, k_out, eps,
            T=T, D=D, BT=BT, BD=BD, num_warps=8, num_stages=3,
        )
    else:
        fused_l2norm_qk_kernel1[(T,)](
            q_flat, k_flat, q_out, k_out, eps=eps, D=D, BD=BD,
            num_warps=8, num_stages=3,
        )
    
    return q_out.view(q_shape_og), k_out.view(k_shape_og)


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
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


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
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
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
