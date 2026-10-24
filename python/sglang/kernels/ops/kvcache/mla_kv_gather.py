"""Compact FP8 MLA row gathers and capture-safe physical-row shadow refreshes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _refresh_mla_kv_shadow_kernel(
    src_ptr,
    dst_ptr,
    idx_ptr,
    n_live_ptr,
    max_rows,
    src_stride,
    dst_stride,
    row_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_live = tl.minimum(tl.load(n_live_ptr).to(tl.int64), max_rows)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < row_dim
    start = pid.to(tl.int64) * ROWS_PER_PROG
    step = nprog.to(tl.int64) * ROWS_PER_PROG
    while start < n_live:
        for r in tl.static_range(ROWS_PER_PROG):
            i = start + r
            if i < n_live:
                row = tl.load(idx_ptr + i).to(tl.int64)
                vals = tl.load(src_ptr + row * src_stride + offs, mask=mask)
                if (
                    vals.dtype == tl.float8e5
                    and dst_ptr.dtype.element_ty == tl.bfloat16
                ):
                    # E5M2 is the high byte of FP16. Decode that exact layout to
                    # preserve Inf/NaN; direct E5M2 -> BF16 lowering can rebias
                    # these encodings into finite values.
                    bits = vals.to(tl.uint8, bitcast=True).to(tl.uint16) << 8
                    vals = bits.to(tl.uint16).to(tl.float16, bitcast=True)
                tl.store(
                    dst_ptr + row * dst_stride + offs,
                    vals.to(dst_ptr.dtype.element_ty),
                    mask=mask,
                )
        start += step


def _row_view(buf: torch.Tensor) -> torch.Tensor:
    return buf.view(buf.shape[0], -1)


def gather_mla_kv_rows(
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    dst_dtype: torch.dtype,
) -> torch.Tensor:
    """Gather physical rows as bytes, then convert to the query dtype."""
    src_dtype = kv_buffer.dtype
    rows = torch.index_select(kv_buffer.view(torch.uint8), 0, kv_indices)
    return rows.view(src_dtype).to(dst_dtype)


def refresh_mla_kv_shadow(
    kv_buffer: torch.Tensor,
    shadow: torch.Tensor,
    kv_indices: torch.Tensor,
    n_live: torch.Tensor,
    num_programs: int = 1024,
    rows_per_program: int = 4,
) -> None:
    """Refresh physical rows; the live count is read on device during graph replay."""
    max_rows = kv_indices.numel()
    if max_rows == 0:
        return
    dim = kv_buffer.shape[-1]
    src = _row_view(kv_buffer)
    dst = _row_view(shadow)
    _refresh_mla_kv_shadow_kernel[(num_programs,)](
        src,
        dst,
        kv_indices,
        n_live,
        max_rows,
        src.stride(0),
        dst.stride(0),
        row_dim=dim,
        BLOCK_D=triton.next_power_of_2(dim),
        ROWS_PER_PROG=rows_per_program,
    )
