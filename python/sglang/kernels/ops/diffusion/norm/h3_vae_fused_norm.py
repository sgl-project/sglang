# SPDX-License-Identifier: Apache-2.0
"""Fused norm kernels for the MiniMax-H3 video VAE decoder.

The eager block casts the residual to FP32, runs RMSNorm, then adds
``update * scale`` as a separate kernel. Under FP16 autocast that add
promotes the residual to FP32, so later layers carry an FP32 residual and an
FP16 attention/FFN update. ``h3_vae_scale_add_rmsnorm`` does the add and the
next RMSNorm in one launch: the sum stays in registers for the variance, and
the stored residual matches the eager FP32 add.

``h3_vae_qk_rmsnorm_rope`` is the affine-free head-64 path: RMSNorm, round to
the activation dtype, then partial NeoX RoPE on the first 48 channels.
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


@triton.jit
def _qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    cos_ptr,
    sin_ptr,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    seq_len,
    n_heads,
    eps,
):
    row = tl.program_id(0).to(tl.int64)
    heads = n_heads
    token = row // heads
    head = row % heads
    seq = token % seq_len
    batch = token // seq_len
    cols = tl.arange(0, 64)
    rope_mask = cols < 48
    q_base = q_ptr + batch * stride_qb + seq * stride_qs + head * stride_qh
    k_base = k_ptr + batch * stride_kb + seq * stride_ks + head * stride_kh
    q = tl.load(q_base + cols).to(tl.float32)
    k = tl.load(k_base + cols).to(tl.float32)
    q = (q * tl.rsqrt(tl.sum(q * q, axis=0) / 64 + eps)).to(tl.float16)
    k = (k * tl.rsqrt(tl.sum(k * k, axis=0) / 64 + eps)).to(tl.float16)
    cos = tl.load(cos_ptr + seq * 48 + cols, mask=rope_mask, other=0.0).to(tl.float16)
    sin = tl.load(sin_ptr + seq * 48 + cols, mask=rope_mask, other=0.0).to(tl.float16)
    # NeoX rotate-half: [-x2, x1] on the first 48 channels.
    src = tl.where(cols < 24, cols + 24, cols - 24)
    sign = tl.where(cols < 24, -1.0, 1.0)
    q_rot = (tl.gather(q.to(tl.float32), src, 0) * sign).to(tl.float16)
    k_rot = (tl.gather(k.to(tl.float32), src, 0) * sign).to(tl.float16)
    yq = (
        (q.to(tl.float32) * cos.to(tl.float32)).to(tl.float16).to(tl.float32)
        + (q_rot.to(tl.float32) * sin.to(tl.float32)).to(tl.float16).to(tl.float32)
    ).to(tl.float16)
    yk = (
        (k.to(tl.float32) * cos.to(tl.float32)).to(tl.float16).to(tl.float32)
        + (k_rot.to(tl.float32) * sin.to(tl.float32)).to(tl.float16).to(tl.float32)
    ).to(tl.float16)
    yq = tl.where(rope_mask, yq, q)
    yk = tl.where(rope_mask, yk, k)
    tl.store(qo_ptr + row * 64 + cols, yq)
    tl.store(ko_ptr + row * 64 + cols, yk)


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


def h3_vae_qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Affine-free RMSNorm plus partial NeoX RoPE for head 64 / rotary 48.

    Returns None when the tensors are outside that contract so the caller can
    keep the unfused norm and RoPE.
    """
    if (
        not query.is_cuda
        or query.shape != key.shape
        or query.dtype != key.dtype
        or query.dtype != torch.float16
        or query.dim() != 4
        or query.shape[-1] != 64
        or query.stride(-1) != 1
        or key.stride(-1) != 1
        or cos.shape != sin.shape
        or cos.dim() != 2
        or cos.shape[-1] != 48
        or cos.shape[0] != query.shape[1]
        or cos.device != query.device
        or sin.device != query.device
        or key.device != query.device
    ):
        return None
    batch, seq_len, n_heads, _ = query.shape
    rows = batch * seq_len * n_heads
    if rows == 0:
        return None
    query_out = torch.empty(rows, 64, device=query.device, dtype=query.dtype)
    key_out = torch.empty(rows, 64, device=key.device, dtype=key.dtype)
    _qk_rmsnorm_rope_kernel[(rows,)](
        query,
        key,
        query_out,
        key_out,
        cos.contiguous(),
        sin.contiguous(),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        seq_len,
        n_heads,
        float(eps),
        num_warps=1,
    )
    view = (batch, seq_len, n_heads, 64)
    return query_out.view(view), key_out.view(view)
