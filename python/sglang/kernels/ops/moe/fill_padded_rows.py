"""Fused padded-row fill for MoE top-k outputs.

Migrated from ``sglang.srt.layers.moe.topk`` (RFC #29630, Phase 2.5), where two
near-identical copies had accumulated; this keeps the later, runtime-winning
copy (explicit raises instead of asserts).
"""

import torch
import triton
import triton.language as tl

# One program per row is launch bound for large buffers of narrow rows: nearly
# every program finds its row live and exits, so the cost scales with the buffer
# rather than with the padding tail. For >= 2 * 1024 rows of <= 2 KB, 1024
# striding programs fill the tail instead; every other shape keeps the per-row
# launch, which wider rows already run at full HBM speed.
_MAX_PROGRAMS = 1024
_MAX_STRIDED_ROW_BYTES = 2048


@triton.jit
def _fill_padded_rows_kernel(
    out_ptr,
    num_token_non_padded_ptr,
    n_rows,
    n_cols,
    fill_value,
    stride_row,
    BLOCK_COLS: tl.constexpr,
    STRIDED: tl.constexpr,
):
    if STRIDED:
        # Program p fills its rows p, p + grid, ... that are padding. Padding is a
        # suffix, so start at the last one and walk down to the first live row.
        pid = tl.program_id(0)
        grid = tl.num_programs(0)
        row = pid + ((n_rows - 1 - pid) // grid) * grid
        n_valid = tl.load(num_token_non_padded_ptr)
        if row >= n_valid:
            cols = tl.arange(0, BLOCK_COLS)
            mask = cols < n_cols
            fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
            while row >= n_valid:
                ptrs = out_ptr + row.to(tl.int64) * stride_row + cols
                tl.store(ptrs, fill, mask=mask)
                row -= grid
    else:
        # One program per row. Kept as a separate compile-time branch: the
        # strided body run one row per program is slower.
        row = tl.program_id(0)
        n_valid = tl.load(num_token_non_padded_ptr)
        if row >= n_valid:
            cols = tl.arange(0, BLOCK_COLS)
            mask = cols < n_cols
            ptrs = out_ptr + row.to(tl.int64) * stride_row + cols
            fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
            tl.store(ptrs, fill, mask=mask)


def _use_strided(n_rows: int, row_bytes: int) -> bool:
    return n_rows >= 2 * _MAX_PROGRAMS and row_bytes <= _MAX_STRIDED_ROW_BYTES


def _can_fuse_padded_region(x: torch.Tensor) -> bool:
    # The fused kernel needs a row-major 2D tensor; fall back to eager otherwise.
    return x.dim() == 2 and x.stride(1) == 1


def _fill_padded_rows(
    x: torch.Tensor,
    num_token_non_padded: torch.Tensor,
    fill_value,
) -> None:
    """Set ``x[row, :] = fill_value`` for every padded row (row index
    ``>= num_token_non_padded``) using a single Triton launch.

    Replaces the eager ``arange + (>=) + boolean index_put_`` sequence, which
    issues several launch-latency-bound kernels per call. The grid depends only
    on the shape and the pad count is read from device memory inside the kernel,
    so this is safe to capture inside a CUDA/HIP graph.
    """
    # Metadata-only checks (no device sync): the kernel reads a single scalar
    # routing count from device memory, so it must be a 1-element integer tensor
    # on the same device as ``x``. Use explicit raises (not asserts) so the
    # checks survive ``python -O`` and invalid inputs fail loudly instead of
    # turning into opaque Triton/memory errors.
    if not isinstance(num_token_non_padded, torch.Tensor):
        raise TypeError("num_token_non_padded must be a torch.Tensor")
    if num_token_non_padded.numel() != 1:
        raise ValueError(
            "num_token_non_padded must be a single-element tensor, got shape "
            f"{tuple(num_token_non_padded.shape)}"
        )
    if num_token_non_padded.dtype.is_floating_point:
        raise TypeError(
            "num_token_non_padded must be an integer tensor, got "
            f"{num_token_non_padded.dtype}"
        )
    if num_token_non_padded.device != x.device:
        raise ValueError("num_token_non_padded and x must be on the same device")
    n_rows, n_cols = x.shape
    strided = _use_strided(n_rows, n_cols * x.element_size())
    _fill_padded_rows_kernel[(_MAX_PROGRAMS if strided else n_rows,)](
        x,
        num_token_non_padded,
        n_rows,
        n_cols,
        fill_value,
        x.stride(0),
        BLOCK_COLS=triton.next_power_of_2(n_cols),
        STRIDED=strided,
    )


@triton.jit
def _fill_dp_pad_rows_kernel(
    out_ptr,
    dp_real_rows_ptr,
    rows_per_rank,
    n_cols,
    fill_value,
    stride_row,
    BLOCK_COLS: tl.constexpr,
):
    # One program per row. DP rank r owns rows [r * rows_per_rank,
    # (r + 1) * rows_per_rank); the first dp_real_rows[r] of them are real.
    row = tl.program_id(0)
    rank = row // rows_per_rank
    if row - rank * rows_per_rank >= tl.load(dp_real_rows_ptr + rank):
        cols = tl.arange(0, BLOCK_COLS)
        ptrs = out_ptr + row.to(tl.int64) * stride_row + cols
        fill = tl.full((BLOCK_COLS,), fill_value, dtype=out_ptr.dtype.element_ty)
        tl.store(ptrs, fill, mask=cols < n_cols)


def _fill_dp_pad_rows(
    x: torch.Tensor,
    dp_real_rows: torch.Tensor,
    fill_value,
) -> None:
    """Set ``x[row, :] = fill_value`` for the padding rows of a MAX_LEN DP gather.

    ``x`` holds one equal slot per DP rank, in rank order, and slot ``r`` starts
    with ``dp_real_rows[r]`` real rows. The counts are read from device memory,
    so this is safe to capture inside a CUDA graph.
    """
    if not isinstance(dp_real_rows, torch.Tensor):
        raise TypeError("dp_real_rows must be a torch.Tensor")
    if dp_real_rows.dim() != 1 or dp_real_rows.dtype.is_floating_point:
        raise TypeError(
            "dp_real_rows must be a 1-D integer tensor, got "
            f"{dp_real_rows.dtype} {tuple(dp_real_rows.shape)}"
        )
    if dp_real_rows.device != x.device:
        raise ValueError("dp_real_rows and x must be on the same device")
    n_rows, n_cols = x.shape
    num_ranks = dp_real_rows.numel()
    if num_ranks == 0 or n_rows % num_ranks != 0:
        raise ValueError(f"{n_rows} rows do not split into {num_ranks} equal DP slots")
    if n_rows == 0:
        return
    _fill_dp_pad_rows_kernel[(n_rows,)](
        x,
        dp_real_rows,
        n_rows // num_ranks,
        n_cols,
        fill_value,
        x.stride(0),
        BLOCK_COLS=triton.next_power_of_2(n_cols),
    )
