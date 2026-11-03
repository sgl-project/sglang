# SPDX-License-Identifier: Apache-2.0
"""Fused norm kernels for the MiniMax-H3 video VAE decoder.

The eager block casts the residual to FP32, runs RMSNorm, then adds
``update * scale`` as a separate kernel. Under FP16 autocast that add
promotes the residual to FP32, so later layers carry an FP32 residual and an
FP16 attention/FFN update. ``h3_vae_scale_add_rmsnorm`` does the add and the
next RMSNorm in one launch: the sum stays in registers for the variance, and
the stored residual matches the eager FP32 add.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.numerics import mul_rn_f32

_MAX_BLOCK = 4096


@triton.jit
def _rmsnorm_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    rows,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_KIND: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * weight
    if OUT_KIND == 1:
        y = y.to(tl.float16)
    elif OUT_KIND == 2:
        y = y.to(tl.bfloat16)
    tl.store(out_ptr + row * D + cols, y, mask=mask)


@triton.jit
def _scale_add_rmsnorm_kernel(
    res_in_ptr,
    update_ptr,
    res_out_ptr,
    out_ptr,
    scale_ptr,
    weight_ptr,
    rows,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    residual = tl.load(res_in_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Same rounding as the eager FP32 residual add: a rounded multiply, not an FMA.
    acc = residual + mul_rn_f32(update, scale)
    tl.store(res_out_ptr + row * D + cols, acc, mask=mask)
    rstd = tl.rsqrt(tl.sum(acc * acc, axis=0) / D + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * D + cols, acc * rstd * weight, mask=mask)


def _out_kind(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise ValueError(f"unsupported H3 VAE norm dtype {dtype}")


def _launch_config(dim: int) -> tuple[int, int]:
    block = triton.next_power_of_2(dim)
    if block > _MAX_BLOCK:
        raise ValueError(f"H3 VAE fused RMSNorm dim {dim} exceeds {_MAX_BLOCK}")
    num_warps = min(8, max(1, block // 128))
    return block, num_warps


def _rows_of(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if tensor.shape[-1] > _MAX_BLOCK or tensor.numel() == 0:
        raise ValueError("H3 VAE fused RMSNorm input is empty or wider than one tile")
    flat = tensor.reshape(-1, tensor.shape[-1])
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return flat, flat.shape[0], flat.shape[1]


def h3_vae_rmsnorm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """FP32 RMSNorm. The stored dtype matches ``hidden`` so it lines up with ``.to(dtype)``."""
    flat, rows, dim = _rows_of(hidden)
    if weight.shape != (dim,) or weight.device != hidden.device:
        raise ValueError("H3 VAE RMSNorm weight must match the last dimension")
    out = torch.empty(flat.shape, device=hidden.device, dtype=hidden.dtype)
    block, num_warps = _launch_config(dim)
    _rmsnorm_kernel[(rows,)](
        out,
        flat,
        weight.contiguous(),
        rows,
        float(eps),
        D=dim,
        BLOCK=block,
        OUT_KIND=_out_kind(hidden.dtype),
        num_warps=num_warps,
    )
    return out.view(hidden.shape)


def h3_vae_scale_add_rmsnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual + update * scale`` in FP32, then RMSNorm of that sum.

    The residual is updated in place when it is already a contiguous FP32
    tensor. Otherwise a new FP32 residual is returned, matching the dtype the
    eager autocast add promotes to.
    """
    if residual.shape != update.shape:
        raise ValueError(
            "H3 VAE scale-add expects residual and update to share a shape"
        )
    flat, rows, dim = _rows_of(residual)
    update_flat, _, update_dim = _rows_of(update)
    if update_dim != dim or scale.shape != (dim,) or weight.shape != (dim,):
        raise ValueError("H3 VAE scale-add vectors must match the residual width")
    if (
        scale.device != residual.device
        or weight.device != residual.device
        or update.device != residual.device
    ):
        raise ValueError("H3 VAE scale-add tensors must share a device")
    inplace = (
        residual.dtype == torch.float32
        and flat.data_ptr() == residual.data_ptr()
        and flat.is_contiguous()
    )
    res_out = (
        flat
        if inplace
        else torch.empty(flat.shape, device=residual.device, dtype=torch.float32)
    )
    normed = torch.empty(flat.shape, device=residual.device, dtype=torch.float32)
    block, num_warps = _launch_config(dim)
    _scale_add_rmsnorm_kernel[(rows,)](
        flat,
        update_flat,
        res_out,
        normed,
        scale.contiguous(),
        weight.contiguous(),
        rows,
        float(eps),
        D=dim,
        BLOCK=block,
        num_warps=num_warps,
    )
    out_shape = residual.shape
    residual_out = res_out.view(out_shape)
    if inplace:
        return residual, normed.view(out_shape)
    return residual_out, normed.view(out_shape)
