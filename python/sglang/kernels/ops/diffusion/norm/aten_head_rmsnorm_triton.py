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
    token_stride,
    nheads,
    n_rows,
    eps,
    head_dim: tl.constexpr,
    rows_per_prog: tl.constexpr,
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
    y = weight[None, :] * (rstd[:, None] * x)
    out_rows = rows.to(tl.int64)[:, None] * head_dim
    tl.store(
        y_ptr + out_rows + offs[None, :], y.to(tl.bfloat16), mask=row_mask[:, None]
    )


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


def aten_head_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``F.rms_norm(x, (128,), weight, eps)`` bit-for-bit, as a contiguous tensor."""
    nheads = x.shape[-2]
    n_rows = x.numel() // _HEAD_DIM
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n_rows, _ROWS_PER_PROG),)
    with torch.get_device_module().device(x.device):
        _aten_head_rmsnorm_kernel[grid](
            y,
            x,
            weight,
            _token_stride(x),
            nheads,
            n_rows,
            eps,
            head_dim=_HEAD_DIM,
            rows_per_prog=_ROWS_PER_PROG,
            num_warps=4,
        )
    return y


__all__ = ["aten_head_rmsnorm", "can_use_aten_head_rmsnorm"]
