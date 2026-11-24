"""Split-row argmax for wide rows (HIP).

torch.argmax reduces a vocabulary-wide row with one block per row; at verify
widths that reads the logits at a fraction of HBM bandwidth. Two launches here:
per-split (max, first index) partials, then a per-row pick of the first winning
split. Matches torch.argmax exactly: the lowest index wins a tie and the first
NaN wins over any number.
"""

import torch
import triton
import triton.language as tl

_MIN_VOCAB = 8192


@triton.jit
def _argmax_partial_kernel(
    logits,
    row_stride,
    partial_vals,
    partial_idx,
    vocab_size,
    num_splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    cols = split * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_size
    x = tl.load(logits + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    is_nan = x != x
    nan_col = tl.min(tl.where(is_nan, cols, 2147483647), axis=0)
    best = tl.max(tl.where(is_nan, -float("inf"), x), axis=0)
    col = tl.min(tl.where((x == best) & mask, cols, 2147483647), axis=0)
    # An all -inf split still reports its first column.
    col = tl.where(col == 2147483647, split * BLOCK, col)
    has_nan = nan_col != 2147483647
    out = row * num_splits + split
    tl.store(partial_vals + out, tl.where(has_nan, float("nan"), best))
    tl.store(partial_idx + out, tl.where(has_nan, nan_col, col))


@triton.jit
def _argmax_final_kernel(
    partial_vals,
    partial_idx,
    out,
    num_splits,
    SPLITS_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, SPLITS_BLOCK)
    mask = s < num_splits
    v = tl.load(partial_vals + row * num_splits + s, mask=mask, other=-float("inf"))
    idx = tl.load(partial_idx + row * num_splits + s, mask=mask, other=2147483647)
    is_nan = (v != v) & mask
    nan_split = tl.min(tl.where(is_nan, s, SPLITS_BLOCK), axis=0)
    best = tl.max(tl.where(is_nan, -float("inf"), v), axis=0)
    win = tl.min(tl.where((v == best) & mask, s, SPLITS_BLOCK), axis=0)
    win = tl.where(nan_split < SPLITS_BLOCK, nan_split, win)
    win = tl.where(win == SPLITS_BLOCK, 0, win)
    col = tl.sum(tl.where(s == win, idx, 0), axis=0)
    tl.store(out + row, col.to(tl.int64))


def can_row_argmax(logits: torch.Tensor) -> bool:
    return (
        logits.is_cuda
        and logits.ndim == 2
        and logits.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and logits.stride(1) == 1
        and logits.shape[0] > 0
        and _MIN_VOCAB <= logits.shape[1] < 2**31
    )


def row_argmax(logits: torch.Tensor) -> torch.Tensor:
    """int64 ``[rows]`` argmax of a ``[rows, vocab]`` tensor (see module docstring)."""
    rows, vocab_size = logits.shape
    block = 4096 if rows >= 16 else 2048
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    partial_idx = torch.empty(
        (rows, num_splits), dtype=torch.int32, device=logits.device
    )
    out = torch.empty((rows,), dtype=torch.int64, device=logits.device)
    _argmax_partial_kernel[(rows, num_splits)](
        logits,
        logits.stride(0),
        partial_vals,
        partial_idx,
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _argmax_final_kernel[(rows,)](
        partial_vals,
        partial_idx,
        out,
        num_splits,
        SPLITS_BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return out
