"""Split-row temperature softmax for the speculative sampling shapes."""

import torch
import triton
import triton.language as tl

# Split low-row-count reductions across CTAs and fuse temperature scaling.
_MIN_BLOCK = 2048
_TARGET_CTAS = 512
ROW_LIMIT = 32


@triton.jit
def _partial_max_sumexp_kernel(
    logits,
    temperatures,
    partial_max,
    partial_sum,
    logits_row_stride,
    vocab_size,
    num_splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    t = tl.load(temperatures + row)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    x = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    x = x / t
    block_max = tl.max(x, axis=0)
    # Fully masked splits contribute zero without evaluating -inf - (-inf).
    safe_block_max = tl.where(block_max == -float("inf"), 0.0, block_max)
    block_sum = tl.sum(tl.where(mask, tl.exp(x - safe_block_max), 0.0), axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_max + out_offset, block_max)
    tl.store(partial_sum + out_offset, block_sum)


@triton.jit
def _combine_kernel(
    partial_max,
    partial_sum,
    row_max,
    row_sum,
    num_splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    m = tl.load(
        partial_max + row * num_splits + offsets, mask=mask, other=-float("inf")
    )
    s = tl.load(partial_sum + row * num_splits + offsets, mask=mask, other=0.0)
    m_all = tl.max(m, axis=0)
    s_all = tl.sum(tl.where(mask, s * tl.exp(m - m_all), 0.0), axis=0)
    tl.store(row_max + row, m_all)
    tl.store(row_sum + row, s_all)


@triton.jit
def _normalize_kernel(
    logits,
    temperatures,
    row_max,
    row_sum,
    probs,
    logits_row_stride,
    probs_row_stride,
    vocab_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    t = tl.load(temperatures + row)
    m = tl.load(row_max + row)
    s = tl.load(row_sum + row)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    x = tl.load(logits + row * logits_row_stride + offsets, mask=mask, other=0.0)
    p = tl.exp(x / t - m) / s
    tl.store(probs + row * probs_row_stride + offsets, p, mask=mask)


def _split_geometry(rows: int, vocab_size: int):
    """Choose a block width and split count near the target grid size."""
    splits = min(
        triton.cdiv(_TARGET_CTAS, rows), max(1, triton.cdiv(vocab_size, _MIN_BLOCK))
    )
    block = max(_MIN_BLOCK, triton.next_power_of_2(triton.cdiv(vocab_size, splits)))
    return block, triton.cdiv(vocab_size, block)


def temperature_softmax(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    """Apply temperature softmax, splitting eligible rows across CTAs."""
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or logits.shape[0] == 0
        or logits.shape[1] == 0
        or logits.shape[0] > ROW_LIMIT
        or logits.stride(1) != 1
        or not logits.is_cuda
    ):
        return torch.softmax(logits / temperatures, dim=-1)

    rows, vocab_size = logits.shape
    t = temperatures.reshape(-1)
    if (
        t.shape[0] != rows
        or t.stride(0) != 1
        or t.dtype != logits.dtype
        or t.device != logits.device
    ):
        return torch.softmax(logits / temperatures, dim=-1)

    block, num_splits = _split_geometry(rows, vocab_size)
    if num_splits == 1:
        return torch.softmax(logits / temperatures, dim=-1)

    probs = torch.empty_like(logits)
    partial_max = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    row_max = torch.empty((rows,), dtype=torch.float32, device=logits.device)
    row_sum = torch.empty((rows,), dtype=torch.float32, device=logits.device)

    _partial_max_sumexp_kernel[(rows, num_splits)](
        logits,
        t,
        partial_max,
        partial_sum,
        logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _combine_kernel[(rows,)](
        partial_max,
        partial_sum,
        row_max,
        row_sum,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    _normalize_kernel[(rows, num_splits)](
        logits,
        t,
        row_max,
        row_sum,
        probs,
        logits.stride(0),
        probs.stride(0),
        vocab_size,
        BLOCK=block,
        num_warps=8,
    )
    return probs
