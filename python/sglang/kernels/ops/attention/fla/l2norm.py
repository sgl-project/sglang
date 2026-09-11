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


@triton.jit(do_not_specialize=["T"])
def gdn_prefill_qkv_prepare_kernel(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    T,
    H_QK: tl.constexpr,
    H_V: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Materialize strided Q/K/V into token-major tensors in one launch."""
    token_block = tl.program_id(0)
    head_idx = tl.program_id(1)

    if head_idx < H_QK:
        # Match l2norm_fwd_kernel's block layout so the BF16 reduction tree is
        # unchanged for strided inputs.
        q_block = tl.make_block_ptr(
            q + head_idx * q_stride_h,
            (T, D),
            (q_stride_t, q_stride_d),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        k_block = tl.make_block_ptr(
            k + head_idx * k_stride_h,
            (T, D),
            (k_stride_t, k_stride_d),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        q_values = tl.load(q_block, boundary_check=(0, 1)).to(tl.float32)
        k_values = tl.load(k_block, boundary_check=(0, 1)).to(tl.float32)
        q_output_block = tl.make_block_ptr(
            q_out + head_idx * D,
            (T, D),
            (H_QK * D, 1),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        k_output_block = tl.make_block_ptr(
            k_out + head_idx * D,
            (T, D),
            (H_QK * D, 1),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        tl.store(
            q_output_block,
            q_values.to(q_output_block.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            k_output_block,
            k_values.to(k_output_block.dtype.element_ty),
            boundary_check=(0, 1),
        )
    else:
        value_head = head_idx - H_QK
        v_block = tl.make_block_ptr(
            v + value_head * v_stride_h,
            (T, D),
            (v_stride_t, v_stride_d),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        v_values = tl.load(v_block, boundary_check=(0, 1))
        v_output_block = tl.make_block_ptr(
            v_out + value_head * D,
            (T, D),
            (H_V * D, 1),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        tl.store(v_output_block, v_values, boundary_check=(0, 1))


def gdn_prefill_qkv_prepare_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q/K/V for FlashInfer, materializing only strided inputs."""
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
    if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
        return l2norm_fwd(q, eps), l2norm_fwd(k, eps), v

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    v_out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    BT = 16
    BD = triton.next_power_of_2(D)
    grid = (triton.cdiv(T, BT), H_QK + H_V)
    gdn_prefill_qkv_prepare_kernel[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        T=T,
        H_QK=H_QK,
        H_V=H_V,
        D=D,
        BT=BT,
        BD=BD,
        num_warps=4,
        num_stages=2,
    )
    return l2norm_fwd(q_out, eps), l2norm_fwd(k_out, eps), v_out


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
