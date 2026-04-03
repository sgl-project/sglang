"""
INT8 KV cache quantization/dequantization helpers.

This module provides Triton kernels with torch fallbacks for:
1) per-token per-head asymmetric INT8 quantization into KV pool,
2) full-pool dequantization,
3) indexed gather + dequantization.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _get_launch_config(head_dim: int) -> Tuple[int, int]:
    if head_dim <= 0:
        raise ValueError(f"Invalid head_dim={head_dim}")
    block_d = min(triton.next_power_of_2(head_dim), 1024)
    if block_d <= 64:
        num_warps = 2
    elif block_d <= 128:
        num_warps = 4
    elif block_d <= 256:
        num_warps = 8
    else:
        num_warps = 16
    return block_d, num_warps


def _torch_quantize_to_int8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_fp32 = x.to(torch.float32)
    x_min = x_fp32.amin(dim=-1, keepdim=True)
    x_max = x_fp32.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min) / 255.0
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    zp = (x_min + x_max) * 0.5
    q = torch.clamp(torch.round((x_fp32 - zp) / scale), -128, 127).to(torch.int8)
    return q, scale, zp


@triton.jit
def _quantize_to_int8_kernel(
    src_ptr,
    loc_ptr,
    q_ptr,
    scale_ptr,
    zp_ptr,
    src_s0,
    src_s1,
    src_s2,
    q_s0,
    q_s1,
    q_s2,
    scale_s0,
    scale_s1,
    num_tokens,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    loc = tl.load(loc_ptr + pid_tok).to(tl.int32)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim

    src_ptrs = src_ptr + pid_tok * src_s0 + pid_head * src_s1 + offs * src_s2
    x = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_min = tl.min(x, axis=0)
    x_max = tl.max(x, axis=0)
    scale = (x_max - x_min) / 255.0
    scale = tl.where(scale > 1e-6, scale, 1.0)
    zp = (x_min + x_max) * 0.5

    q = (x - zp) / scale
    q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
    q = tl.maximum(q, -128.0)
    q = tl.minimum(q, 127.0)

    q_ptrs = q_ptr + loc * q_s0 + pid_head * q_s1 + offs * q_s2
    tl.store(q_ptrs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + loc * scale_s0 + pid_head * scale_s1, scale)
    tl.store(zp_ptr + loc * scale_s0 + pid_head * scale_s1, zp)


@triton.jit
def _dequant_from_int8_kernel(
    q_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    q_s0,
    q_s1,
    q_s2,
    scale_s0,
    scale_s1,
    out_s0,
    out_s1,
    out_s2,
    num_tokens,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_tok >= num_tokens:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim

    q_ptrs = q_ptr + pid_tok * q_s0 + pid_head * q_s1 + offs * q_s2
    q = tl.load(q_ptrs, mask=mask, other=0).to(tl.float32)

    scale = tl.load(scale_ptr + pid_tok * scale_s0 + pid_head * scale_s1).to(
        tl.float32
    )
    zp = tl.load(zp_ptr + pid_tok * scale_s0 + pid_head * scale_s1).to(tl.float32)
    out = q * scale + zp

    out_ptrs = out_ptr + pid_tok * out_s0 + pid_head * out_s1 + offs * out_s2
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _gather_dequant_from_int8_kernel(
    loc_ptr,
    q_ptr,
    scale_ptr,
    zp_ptr,
    out_ptr,
    q_s0,
    q_s1,
    q_s2,
    scale_s0,
    scale_s1,
    out_s0,
    out_s1,
    out_s2,
    num_tokens,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_tok >= num_tokens:
        return

    loc = tl.load(loc_ptr + pid_tok).to(tl.int32)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim

    q_ptrs = q_ptr + loc * q_s0 + pid_head * q_s1 + offs * q_s2
    q = tl.load(q_ptrs, mask=mask, other=0).to(tl.float32)

    scale = tl.load(scale_ptr + loc * scale_s0 + pid_head * scale_s1).to(tl.float32)
    zp = tl.load(zp_ptr + loc * scale_s0 + pid_head * scale_s1).to(tl.float32)
    out = q * scale + zp

    out_ptrs = out_ptr + pid_tok * out_s0 + pid_head * out_s1 + offs * out_s2
    tl.store(out_ptrs, out, mask=mask)


def _scatter_quant_tensor_to_pool(
    src: torch.Tensor,
    loc: torch.Tensor,
    q_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    zp_pool: torch.Tensor,
) -> None:
    if src.ndim != 3:
        raise ValueError(f"src must be [tokens, heads, dim], got {tuple(src.shape)}")
    if q_pool.ndim != 3:
        raise ValueError(f"q_pool must be [pool, heads, dim], got {tuple(q_pool.shape)}")
    if scale_pool.ndim != 3 or scale_pool.shape[-1] != 1:
        raise ValueError(
            f"scale_pool must be [pool, heads, 1], got {tuple(scale_pool.shape)}"
        )
    if zp_pool.ndim != 3 or zp_pool.shape[-1] != 1:
        raise ValueError(f"zp_pool must be [pool, heads, 1], got {tuple(zp_pool.shape)}")
    if src.shape[0] != loc.numel():
        raise ValueError(f"src tokens {src.shape[0]} != len(loc) {loc.numel()}")
    if src.shape[1] != q_pool.shape[1] or src.shape[2] != q_pool.shape[2]:
        raise ValueError(
            f"src shape {tuple(src.shape)} incompatible with pool shape {tuple(q_pool.shape)}"
        )

    if not src.is_cuda:
        q, s, z = _torch_quantize_to_int8(src)
        q_pool[loc] = q
        scale_pool[loc, :, 0] = s.squeeze(-1).to(scale_pool.dtype)
        zp_pool[loc, :, 0] = z.squeeze(-1).to(zp_pool.dtype)
        return

    if loc.dtype != torch.int32:
        loc = loc.to(torch.int32)
    loc = loc.contiguous()
    src = src.contiguous()

    num_tokens, num_heads, head_dim = src.shape
    block_d, num_warps = _get_launch_config(head_dim)
    grid = (num_tokens, num_heads)

    _quantize_to_int8_kernel[grid](
        src,
        loc,
        q_pool,
        scale_pool,
        zp_pool,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        q_pool.stride(0),
        q_pool.stride(1),
        q_pool.stride(2),
        scale_pool.stride(0),
        scale_pool.stride(1),
        num_tokens,
        head_dim=head_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )


def scatter_quant_kv_to_pool(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_int8_pool: torch.Tensor,
    v_int8_pool: torch.Tensor,
    k_scale_pool: torch.Tensor,
    k_zp_pool: torch.Tensor,
    v_scale_pool: torch.Tensor,
    v_zp_pool: torch.Tensor,
) -> None:
    _scatter_quant_tensor_to_pool(cache_k, loc, k_int8_pool, k_scale_pool, k_zp_pool)
    _scatter_quant_tensor_to_pool(cache_v, loc, v_int8_pool, v_scale_pool, v_zp_pool)


def dequant_int8_kv(
    q_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    zp_pool: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if q_pool.ndim != 3 or scale_pool.ndim != 3 or zp_pool.ndim != 3:
        raise ValueError("Expected q/scale/zp tensors in [tokens, heads, ...] format")
    if scale_pool.shape != zp_pool.shape:
        raise ValueError(
            f"scale and zp shapes must match: {scale_pool.shape} vs {zp_pool.shape}"
        )
    if scale_pool.shape[:2] != q_pool.shape[:2] or scale_pool.shape[2] != 1:
        raise ValueError(
            f"Invalid scale/zp shape for q_pool {tuple(q_pool.shape)}: {tuple(scale_pool.shape)}"
        )

    if out is None:
        out = torch.empty(q_pool.shape, dtype=out_dtype, device=q_pool.device)
    elif out.shape != q_pool.shape:
        raise ValueError(f"out shape {tuple(out.shape)} != q_pool shape {tuple(q_pool.shape)}")

    if not q_pool.is_cuda:
        out.copy_(
            q_pool.to(torch.float32)
            * scale_pool[:, :, 0].to(torch.float32).unsqueeze(-1)
            + zp_pool[:, :, 0].to(torch.float32).unsqueeze(-1)
        )
        return out

    num_tokens, num_heads, head_dim = q_pool.shape
    block_d, num_warps = _get_launch_config(head_dim)
    grid = (num_tokens, num_heads)

    _dequant_from_int8_kernel[grid](
        q_pool,
        scale_pool,
        zp_pool,
        out,
        q_pool.stride(0),
        q_pool.stride(1),
        q_pool.stride(2),
        scale_pool.stride(0),
        scale_pool.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_tokens,
        head_dim=head_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


def gather_dequant_kv_from_pool(
    loc: torch.Tensor,
    q_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    zp_pool: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if loc.ndim != 1:
        raise ValueError(f"loc must be 1-D, got {tuple(loc.shape)}")
    if out is None:
        out = torch.empty(
            (loc.numel(), q_pool.shape[1], q_pool.shape[2]),
            dtype=out_dtype,
            device=q_pool.device,
        )

    if not q_pool.is_cuda:
        gather = q_pool[loc].to(torch.float32)
        scale = scale_pool[loc, :, 0].to(torch.float32).unsqueeze(-1)
        zp = zp_pool[loc, :, 0].to(torch.float32).unsqueeze(-1)
        out.copy_(gather * scale + zp)
        return out

    if loc.dtype != torch.int32:
        loc = loc.to(torch.int32)
    loc = loc.contiguous()

    num_tokens = loc.numel()
    num_heads = q_pool.shape[1]
    head_dim = q_pool.shape[2]
    block_d, num_warps = _get_launch_config(head_dim)
    grid = (num_tokens, num_heads)

    _gather_dequant_from_int8_kernel[grid](
        loc,
        q_pool,
        scale_pool,
        zp_pool,
        out,
        q_pool.stride(0),
        q_pool.stride(1),
        q_pool.stride(2),
        scale_pool.stride(0),
        scale_pool.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_tokens,
        head_dim=head_dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out
