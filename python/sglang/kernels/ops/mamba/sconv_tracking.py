"""Prepare short-convolution checkpoint windows in a single Triton launch.

Live rows select the window ending at the last complete cache-chunk boundary;
padded rows are zeroed so graph replay never reads stale token indices.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["live_rows"])
def _track_conv_indices_kernel(
    query_start_loc,
    track_seqlens,
    prefix_lens,
    output,
    live_rows,
    ROWS: tl.constexpr,
    QUERY_END: tl.constexpr,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    ZERO_PREFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // WIDTH
    window = offsets % WIDTH
    live = rows < live_rows
    tracked = tl.load(track_seqlens + rows, mask=live, other=0).to(tl.int64)
    if ZERO_PREFIX:
        prefix = tl.full((BLOCK,), 0, tl.int64)
    else:
        prefix = tl.load(prefix_lens + rows, mask=live, other=0).to(tl.int64)
    delta = tracked - prefix
    # Triton's signed integer division truncates toward zero; Torch's //
    # floors, including masked requests whose tracked length precedes prefix.
    aligned = tl.where(
        delta < 0,
        -((-delta + CHUNK - 1) // CHUNK) * CHUNK,
        (delta // CHUNK) * CHUNK,
    )
    query_start = tl.load(query_start_loc + rows, mask=live, other=0).to(tl.int64)
    total = tl.load(query_start_loc + QUERY_END).to(tl.int64)
    index = query_start + aligned - WIDTH + window
    index = tl.minimum(tl.maximum(index, 0), total - 1)
    index = tl.where(live, index, 0)
    tl.store(output + offsets, index, mask=rows < ROWS)


def fill_track_conv_indices(
    *,
    query_start_loc: torch.Tensor,
    track_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor | None,
    output: torch.Tensor,
    live: int,
    chunk_size: int,
) -> None:
    """Fill a stable track-index buffer with one launch and no temporaries."""
    assert output.ndim == 2 and output.dtype == torch.int64 and output.is_contiguous()
    rows, width = output.shape
    assert 0 <= live <= rows and width > 0 and chunk_size > 0
    assert query_start_loc.ndim == track_seqlens.ndim == 1
    assert query_start_loc.dtype in (torch.int32, torch.int64)
    assert track_seqlens.dtype in (torch.int32, torch.int64)
    assert query_start_loc.numel() == rows + 1 and track_seqlens.numel() >= live
    assert query_start_loc.is_contiguous() and track_seqlens.is_contiguous()
    assert (
        query_start_loc.is_cuda
        and track_seqlens.device == output.device == query_start_loc.device
    )
    if prefix_lens is not None:
        assert prefix_lens.ndim == 1 and prefix_lens.numel() >= live
        assert prefix_lens.dtype in (torch.int32, torch.int64)
        assert prefix_lens.is_contiguous() and prefix_lens.device == output.device
    if rows == 0:
        return
    _track_conv_indices_kernel[(triton.cdiv(rows * width, 128),)](
        query_start_loc,
        track_seqlens,
        prefix_lens if prefix_lens is not None else track_seqlens,
        output,
        live,
        ROWS=rows,
        QUERY_END=query_start_loc.numel() - 1,
        WIDTH=width,
        CHUNK=chunk_size,
        ZERO_PREFIX=prefix_lens is None,
        BLOCK=128,
    )
