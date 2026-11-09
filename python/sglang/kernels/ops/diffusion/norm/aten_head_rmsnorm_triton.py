# SPDX-License-Identifier: Apache-2.0
"""Bit-exact replica of aten's fused per-head RMSNorm on ROCm.

For bf16 rows of 128, torch's ROCm ``vectorized_layer_norm_kernel<bf16, float,
rms_norm=true>`` runs one 256-thread block per row, of which 32 lanes each load
4 contiguous elements and accumulate their squares serially in fp32 (bf16
squares are exact in fp32, so FMA contraction cannot change the sum). A
shfl-down tree folds the 32 lane partials at offsets 16/8/4/2/1, the sum is
divided by 128, ``rsqrtf(mean + eps)`` scales each element and the weight
multiplies last: ``w * (rstd * x)``. One block per 256-byte row leaves that
kernel at ~0.6 TB/s; this kernel keeps the arithmetic and reads strided
fused-qkv views directly, absorbing the ``.contiguous()`` the eager path needs.
It can also apply the bf16 NeoX ``rotary_embedding`` that follows the norm.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_HEAD_DIM = 128
_ROWS_PER_PROG = 16


@triton.jit
def _aten_head_rmsnorm_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    cache_ptr,
    positions_ptr,
    token_stride,
    nheads,
    n_rows,
    eps,
    cache_stride,
    head_dim: tl.constexpr,
    rows_per_prog: tl.constexpr,
    rope_half: tl.constexpr,
):
    rows = tl.program_id(0) * rows_per_prog + tl.arange(0, rows_per_prog)
    row_mask = rows < n_rows
    tokens = (rows // nheads).to(tl.int64)
    heads = rows % nheads
    base = x_ptr + tokens * token_stride + heads * head_dim
    offs = tl.arange(0, head_dim)
    x = tl.load(base[:, None] + offs[None, :], mask=row_mask[:, None], other=0.0)
    x = x.to(tl.float32)

    # lane t owns elements 4t..4t+3 and adds their squares in order
    lanes = tl.reshape(x, (rows_per_prog, head_dim // 4, 2, 2), can_reorder=False)
    even, odd = tl.split(lanes)
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)
    acc = x0 * x0
    acc = acc + x1 * x1
    acc = acc + x2 * x2
    acc = acc + x3 * x3
    # shfl-down fold over the 32 lane partials: acc[i] += acc[i + half]
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 16), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 8), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 4), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 2), can_reorder=False), axis=1)
    mean = tl.sum(acc, axis=1) / head_dim
    rstd = tl.rsqrt(mean + eps)

    weight = tl.load(weight_ptr + offs).to(tl.float32)
    y = (weight[None, :] * (rstd[:, None] * x)).to(tl.bfloat16)
    if rope_half > 0:
        # NeoX RoPE as the bf16 rotary_embedding kernel computes it: every
        # product and the sum round to bf16; dims past 2 * rope_half pass through
        rotated = offs < 2 * rope_half
        first = offs < rope_half
        partner = tl.where(first, offs + rope_half, offs - rope_half)
        rope_mask = row_mask[:, None] & rotated[None, :]
        xp = tl.load(base[:, None] + partner[None, :], mask=rope_mask, other=0.0)
        wp = tl.load(weight_ptr + partner, mask=rotated, other=0.0).to(tl.float32)
        yp = (wp[None, :] * (rstd[:, None] * xp.to(tl.float32))).to(tl.bfloat16)
        cache_row = tl.load(positions_ptr + tokens, mask=row_mask) * cache_stride
        cos_offs = cache_row[:, None] + (offs % rope_half)[None, :]
        cos = tl.load(cache_ptr + cos_offs, mask=rope_mask, other=0.0)
        sin = tl.load(cache_ptr + cos_offs + rope_half, mask=rope_mask, other=0.0)
        a = (y.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        b = (yp.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        rope = tl.where(first[None, :], a - b, a + b).to(tl.bfloat16)
        y = tl.where(rotated[None, :], rope, y)
    out_rows = rows.to(tl.int64)[:, None] * head_dim
    tl.store(y_ptr + out_rows + offs[None, :], y, mask=row_mask[:, None])


def _token_stride(x: torch.Tensor) -> int | None:
    """Uniform token stride of a ``[..., heads, 128]`` view with packed heads."""
    if x.dim() < 3 or x.stride(-1) != 1 or x.stride(-2) != _HEAD_DIM:
        return None
    token_stride = x.stride(-3)
    expected = token_stride
    for dim in range(x.dim() - 4, -1, -1):
        expected *= x.shape[dim + 1]
        if x.shape[dim] > 1 and x.stride(dim) != expected:
            return None
    return token_stride


def can_use_aten_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    return (
        torch.version.hip is not None
        and x.is_cuda
        and weight is not None
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.numel() > 0
        and x.shape[-1] == _HEAD_DIM
        and weight.shape == (_HEAD_DIM,)
        and weight.is_contiguous()
        and _token_stride(x) is not None
    )


def can_use_aten_head_rmsnorm_rope(
    x: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> bool:
    """NeoX RoPE over a ``[positions, rot_dim]`` cos|sin cache, rot_dim <= 128."""
    rot_dim = cos_sin_cache.shape[-1]
    return (
        cos_sin_cache.dim() == 2
        and cos_sin_cache.dtype == x.dtype
        and cos_sin_cache.device == x.device
        and cos_sin_cache.stride(-1) == 1
        and 0 < rot_dim <= _HEAD_DIM
        and rot_dim % 2 == 0
        and positions.dtype == torch.int64
        and positions.device == x.device
        and positions.is_contiguous()
        and positions.numel() == x.numel() // (x.shape[-2] * _HEAD_DIM)
    )


def aten_head_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """``F.rms_norm(x, (128,), weight, eps)`` bit-for-bit, as a contiguous tensor.

    With ``cos_sin_cache``/``positions`` the NeoX ``rotary_embedding`` that
    follows the norm is applied in the same pass, also bit-for-bit.
    """
    nheads = x.shape[-2]
    n_rows = x.numel() // _HEAD_DIM
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n_rows, _ROWS_PER_PROG),)
    rope = cos_sin_cache is not None
    with torch.get_device_module().device(x.device):
        _aten_head_rmsnorm_kernel[grid](
            y,
            x,
            weight,
            cos_sin_cache if rope else x,
            positions if rope else x,
            _token_stride(x),
            nheads,
            n_rows,
            eps,
            cos_sin_cache.stride(0) if rope else 0,
            head_dim=_HEAD_DIM,
            rows_per_prog=_ROWS_PER_PROG,
            rope_half=cos_sin_cache.shape[-1] // 2 if rope else 0,
            num_warps=4,
        )
    return y


__all__ = [
    "aten_head_rmsnorm",
    "can_use_aten_head_rmsnorm",
    "can_use_aten_head_rmsnorm_rope",
]
