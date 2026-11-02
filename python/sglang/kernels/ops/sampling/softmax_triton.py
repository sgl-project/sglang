"""Split-row temperature softmax for sampling workloads."""

import torch
import triton
import triton.language as tl

# Split row reductions across CTAs and fuse temperature scaling. _MAX_BLOCK keeps
# many-row batches from getting one huge block per row, which spills registers.
_MIN_BLOCK = 2048
_MAX_BLOCK = 4096
_TARGET_CTAS = 512


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
def _normalize_kernel(
    logits,
    temperatures,
    partial_max,
    partial_sum,
    probs,
    logits_row_stride,
    probs_row_stride,
    vocab_size,
    num_splits,
    BLOCK: tl.constexpr,
    SPLITS_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    s_offsets = tl.arange(0, SPLITS_BLOCK)
    s_mask = s_offsets < num_splits
    pm = tl.load(
        partial_max + row * num_splits + s_offsets, mask=s_mask, other=-float("inf")
    )
    ps = tl.load(partial_sum + row * num_splits + s_offsets, mask=s_mask, other=0.0)
    m = tl.max(pm, axis=0)
    s = tl.sum(tl.where(s_mask, ps * tl.exp(pm - m), 0.0), axis=0)
    t = tl.load(temperatures + row)
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
    block = min(block, _MAX_BLOCK)
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
    _normalize_kernel[(rows, num_splits)](
        logits,
        t,
        partial_max,
        partial_sum,
        probs,
        logits.stride(0),
        probs.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        SPLITS_BLOCK=triton.next_power_of_2(num_splits),
        num_warps=8,
    )
    return probs
