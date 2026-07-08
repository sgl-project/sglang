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
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output (or write straight into a caller-provided dense buffer,
    # e.g. the tree-verify stash, skipping a separate copy)
    if out is not None:
        y = out.view(-1, x.shape[-1])
        assert y.shape == x.shape and y.is_contiguous()
    elif output_dtype is None:
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


@triton.jit(do_not_specialize=["R"])
def l2norm_fwd_kernel_strided(
    x,
    y,
    eps,
    R,
    stride_t,
    stride_h,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """Row-wise L2 norm where rows live at two-level strides (token, head):
    normalizes strided q/k projection slices without a contiguous() copy.
    Output is dense [R, D]."""
    i_r = tl.program_id(0)
    i_t = i_r // H
    i_h = i_r % H
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(
        x + i_t * stride_t + i_h * stride_h + cols, mask=mask, other=0.0
    ).to(tl.float32)
    b_y = b_x / tl.sqrt(tl.sum(b_x * b_x, 0) + eps)
    tl.store(y + i_r * D + cols, b_y.to(y.dtype.element_ty), mask=mask)


def l2norm_fwd_strided(
    x: torch.Tensor, eps: float = 1e-6, out: Optional[torch.Tensor] = None
):
    """L2-normalize the last dim of a [..., H, D] tensor whose last dim is
    dense but whose leading dims may be strided (e.g. q/k slices of the fused
    qkv projection output). Returns a contiguous tensor of x's shape (or
    writes into the dense buffer `out`)."""
    assert x.stride(-1) == 1
    if x.dim() == 4:
        assert x.shape[0] == 1
        _, T, H, D = x.shape
        stride_t, stride_h = x.stride(1), x.stride(2)
    else:
        assert x.dim() == 3
        T, H, D = x.shape
        stride_t, stride_h = x.stride(0), x.stride(1)
    if out is not None:
        y = out.view(-1, D)
        assert y.shape == (T * H, D) and y.is_contiguous()
    else:
        y = torch.empty(T * H, D, dtype=x.dtype, device=x.device)
    l2norm_fwd_kernel_strided[(T * H,)](
        x=x,
        y=y,
        eps=eps,
        R=T * H,
        stride_t=stride_t,
        stride_h=stride_h,
        H=H,
        D=D,
        BD=triton.next_power_of_2(D),
        num_warps=4 if D <= 512 else 8,
    )
    return y.view(x.shape) if out is None else out


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
