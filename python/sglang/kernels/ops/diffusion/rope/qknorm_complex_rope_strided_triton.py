"""Q/K RMSNorm + complex RoPE that reads heads in place from a packed projection.

``x`` is ``[1, S, H, 128]`` with unit head-dim stride and heads 128 apart, at any
token stride (e.g. a slice of a packed ``[S, 3 * H * 128]`` Q/K/V output). The
output is contiguous. The rounding points follow the eager chain; the variance
reduction order differs, so the result matches eager to rounding error.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_HEAD_DIM = 128
_ROWS_PER_PROGRAM = 4


@triton.jit
def _qknorm_complex_rope_strided_kernel(
    x_ptr,
    weight_ptr,
    rope_ptr,
    out_ptr,
    num_rows,
    heads,
    token_stride,
    eps,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, 128)
    mask = row[:, None] < num_rows
    token = row // heads
    head = row % heads
    dtype = out_ptr.dtype.element_ty
    src = x_ptr + token[:, None] * token_stride + head[:, None] * 128 + cols[None, :]
    x = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) * (1.0 / 128) + eps)
    x = (x * inv[:, None]).to(dtype).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    x = (x * weight[None, :]).to(dtype).to(tl.float32)
    real, imag = tl.split(tl.reshape(x, (BLOCK_ROWS, 64, 2)))
    rotation = tl.load(
        rope_ptr + token[:, None] * 128 + cols[None, :], mask=mask, other=0.0
    )
    cos, sin = tl.split(tl.reshape(rotation, (BLOCK_ROWS, 64, 2)))
    out = tl.reshape(
        tl.join(real * cos - imag * sin, real * sin + imag * cos), (BLOCK_ROWS, 128)
    )
    tl.store(out_ptr + row[:, None] * 128 + cols[None, :], out.to(dtype), mask=mask)


def can_use_qknorm_complex_rope_strided(
    x: torch.Tensor, weight: torch.Tensor, rope: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim == 4
        and x.shape[0] == 1
        and x.shape[-1] == _HEAD_DIM
        and x.numel() > 0
        and x.stride(-1) == 1
        and x.stride(-2) == _HEAD_DIM
        and weight.shape == (_HEAD_DIM,)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
        and rope.dtype == torch.complex64
        and rope.device == x.device
        and rope.is_contiguous()
        and rope.shape == (x.shape[1], _HEAD_DIM // 2)
    )


def qknorm_complex_rope_strided(
    x: torch.Tensor, weight: torch.Tensor, rope: torch.Tensor, eps: float
) -> torch.Tensor:
    if not can_use_qknorm_complex_rope_strided(x, weight, rope):
        raise RuntimeError("invalid input for qknorm_complex_rope_strided")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    num_rows = x.shape[1] * x.shape[2]
    with torch.cuda.device(x.device):
        _qknorm_complex_rope_strided_kernel[
            (triton.cdiv(num_rows, _ROWS_PER_PROGRAM),)
        ](
            x,
            weight,
            torch.view_as_real(rope),
            out,
            num_rows,
            x.shape[2],
            x.stride(1),
            eps,
            BLOCK_ROWS=_ROWS_PER_PROGRAM,
            num_warps=4,
        )
    return out
