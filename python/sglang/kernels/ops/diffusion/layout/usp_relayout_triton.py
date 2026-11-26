# SPDX-License-Identifier: Apache-2.0
"""Triton Ulysses output-head merge, ``[W, S, B, H, D] -> [B, S, W, H, D]``.

Each output row gathers one contiguous ``H * D`` chunk per rank, so a program
copies whole chunks instead of decomposing every element index. The output may
be a row-strided view, e.g. the attention columns of a wider GEMM input.
"""

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit(do_not_specialize=["seq_len", "batch", "out_row_stride"])
def _usp_merge_heads_rows_kernel(
    out_ptr,
    x_ptr,
    seq_len,
    batch,
    out_row_stride,
    CHUNK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch_index = row // seq_len
    sequence = row % seq_len
    cols = tl.arange(0, BLOCK)
    mask = cols < CHUNK
    for rank in tl.static_range(WORLD):
        source = ((rank * seq_len + sequence) * batch + batch_index) * CHUNK
        values = tl.load(x_ptr + source + cols, mask=mask)
        tl.store(
            out_ptr + row * out_row_stride + rank * CHUNK + cols, values, mask=mask
        )


def _row_stride(out: torch.Tensor, x: torch.Tensor) -> int | None:
    """Row stride of ``out`` viewed as ``[B * S, W * H * D]`` rows, or None."""
    world, seq, batch, heads, head_dim = x.shape
    expected = (batch, seq, world * heads, head_dim)
    if out.shape != expected or out.dtype != x.dtype or out.device != x.device:
        return None
    if out.stride(-1) != 1 or out.stride(-2) != head_dim:
        return None
    if batch > 1 and out.stride(0) != seq * out.stride(1):
        return None
    return out.stride(1)


def can_use_usp_merge_heads_triton(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> bool:
    return (
        torch.version.hip is not None
        and x.is_cuda
        and x.dim() == 5
        and x.dtype in _SUPPORTED_DTYPES
        and x.numel() > 0
        and x.is_contiguous()
        and (out is None or _row_stride(out, x) is not None)
    )


def usp_merge_heads_triton(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x.permute(2, 1, 0, 3, 4)`` materialized, as ``[B, S, W * H, D]``.

    ``out`` may be any ``[B, S, W * H, D]`` view whose heads are contiguous
    within a row; it is filled and returned. ROCm only.
    """
    if not can_use_usp_merge_heads_triton(x, out):
        raise ValueError("invalid input for usp_merge_heads_triton")
    world, seq, batch, heads, head_dim = x.shape
    if out is None:
        out = x.new_empty((batch, seq, world * heads, head_dim))
    chunk = heads * head_dim
    with torch.cuda.device(x.device):
        _usp_merge_heads_rows_kernel[(batch * seq,)](
            out,
            x,
            seq,
            batch,
            _row_stride(out, x),
            CHUNK=chunk,
            WORLD=world,
            BLOCK=triton.next_power_of_2(chunk),
            num_warps=4,
        )
    return out
