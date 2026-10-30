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
