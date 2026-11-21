"""Materialize the complete visible KV history for KPool single-token decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _kpool_short_decode_indices_kernel(
    Table,
    Lengths,
    Out,
    table_stride,
    table_width,
    rows,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(Lengths + row, mask=row < rows, other=0)
    ids = tl.load(
        Table + row.to(tl.int64) * table_stride + col,
        mask=(col < length) & (col < table_width),
        other=-1,
    )
    tl.store(Out + row.to(tl.int64) * WIDTH + col, ids, mask=col < WIDTH)


def kpool_short_decode_indices(
    table: torch.Tensor, lengths: torch.Tensor, out_rows: int, width: int
) -> torch.Tensor:
    """Copy real slots in sequence order; mask both the token and graph padding."""
    assert table.stride(1) == 1 and lengths.is_contiguous()
    assert lengths.numel() <= out_rows and table.shape[0] >= lengths.numel()
    out = lengths.new_empty((out_rows, width), dtype=torch.int32)
    if out_rows:
        _kpool_short_decode_indices_kernel[(out_rows, triton.cdiv(width, 256))](
            table,
            lengths,
            out,
            table.stride(0),
            table.shape[1],
            lengths.numel(),
            WIDTH=width,
            TILE=256,
        )
    return out
