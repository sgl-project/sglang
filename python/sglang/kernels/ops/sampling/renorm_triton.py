"""ROCm-compatible top-k / top-p probability renormalization fallbacks."""

from __future__ import annotations

from typing import Union

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024


@triton.jit
def _next_float_up(x):
    """Smallest float strictly greater than a non-negative ``x``.

    For non-negative finite floats the IEEE-754 bit pattern read as a signed
    integer is monotonic in the value, so incrementing it steps to the adjacent
    representable float. Probabilities are non-negative, so this is all the top-p
    search needs from a ``nextafter`` toward ``+inf``.
    """
    return (x.to(tl.int32, bitcast=True) + 1).to(tl.float32, bitcast=True)


@triton.jit
def _mask_and_partial_sum_kernel(
    probs_ptr,
    pivots_ptr,
    out_ptr,
    partial_sums_ptr,
    vocab_size: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size
    row_offsets = row.to(tl.int64) * vocab_size + offsets

    probs = tl.load(probs_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    pivot = tl.load(pivots_ptr + row)
    kept = tl.where(mask & (probs >= pivot), probs, 0.0)

    tl.store(out_ptr + row_offsets, kept, mask=mask)
    tl.store(partial_sums_ptr + row * num_chunks + chunk, tl.sum(kept, axis=0))


@triton.jit
def _normalize_kernel(
    out_ptr,
    row_sums_ptr,
    numel,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    row = offsets // vocab_size
    values = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    denominator = tl.load(row_sums_ptr + row, mask=mask, other=1.0)
    tl.store(out_ptr + offsets, values / denominator, mask=mask)


@triton.jit
def _top_p_renorm_kernel(
    probs_ptr,
    top_ps_ptr,
    out_ptr,
    vocab_size: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Find each row's top-p pivot by value, then threshold and renormalize it.

    One program owns one row. The pivot is not found by ranking the row but by
    searching the *value* axis: ``f(x) = sum(probs[probs > x])`` is non-increasing,
    so the pivot is the largest ``x`` with ``f(x) >= p``. Each round evaluates two
    interior points of the bracket, keeps the third that still contains the answer,
    and snaps the new bounds onto values that actually occur in the row, which is
    what makes the round count depend on the data rather than on the float exponent
    range. Every round is a streaming pass with block reductions, so nothing is
    sorted, nothing is materialized, and the host is never asked anything.
    """
    row = tl.program_id(0)
    row_start = row.to(tl.int64) * vocab_size
    p = tl.load(top_ps_ptr + row)

    row_sum = 0.0
    max_val = 0.0
    for chunk in range(num_chunks):
        offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        values = tl.load(probs_ptr + row_start + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        row_sum += tl.sum(values, axis=0)
        max_val = tl.maximum(max_val, tl.max(values, axis=0))

    # Budgeting against the row's own total rather than against 1.0 keeps top_p=1 a
    # no-op on a peaked row, whose leading terms round up to one on their own.
    target = p * row_sum

    # Loop invariant: f(low) >= target and f(high) < target, where
    # f(x) = sum(probs[probs > x]). min_gt_low and max_le_high are the smallest and
    # largest values of the row still inside the bracket, so the search stops once
    # they are the same value or two adjacent floats -- at that point no
    # representable pivot is left to test.
    low = 0.0
    high = max_val
    kept_sum = row_sum
    # p >= 1 keeps the whole row, so the bracket never has to move: low stays at 0
    # and the row is simply rescaled by its own sum.
    searching = tl.where(p < 1.0, 1, 0)

    while searching == 1:
        pivot_low = (high + 2.0 * low) / 3.0
        pivot_high = (2.0 * high + low) / 3.0

        sum_gt_pivot_low = 0.0
        sum_gt_pivot_high = 0.0
        min_gt_low = high
        max_le_high = low
        for chunk in range(num_chunks):
            offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            values = tl.load(probs_ptr + row_start + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            sum_gt_pivot_low += tl.sum(tl.where(values > pivot_low, values, 0.0), axis=0)
            sum_gt_pivot_high += tl.sum(
                tl.where(values > pivot_high, values, 0.0), axis=0
            )
            min_gt_low = tl.minimum(
                min_gt_low, tl.min(tl.where(mask & (values > low), values, high), axis=0)
            )
            max_le_high = tl.maximum(
                max_le_high,
                tl.max(tl.where(mask & (values <= high), values, low), axis=0),
            )

        # Branch-free bracket update: raise the floor to whichever interior point
        # still retains enough mass, otherwise pull the ceiling down.
        take_high = sum_gt_pivot_high >= target
        take_low = (sum_gt_pivot_high < target) & (sum_gt_pivot_low >= target)
        next_low = tl.where(take_high, pivot_high, tl.where(take_low, pivot_low, low))
        next_high = tl.where(
            take_high,
            high,
            tl.where(
                take_low,
                tl.minimum(pivot_high, max_le_high),
                tl.minimum(pivot_low, max_le_high),
            ),
        )
        kept_sum = tl.where(
            take_high,
            sum_gt_pivot_high,
            tl.where(take_low, sum_gt_pivot_low, kept_sum),
        )
        low = next_low
        high = next_high

        # Done once no representable float separates the two surviving values. The
        # adjacency test is what keeps the loop finite: once the interior points
        # round onto the bounds themselves the bracket can no longer shrink, and
        # testing only ``min_gt_low < max_le_high`` would spin forever.
        searching = tl.where(
            (min_gt_low < max_le_high) & (_next_float_up(min_gt_low) < max_le_high),
            1,
            0,
        )

    denominator = tl.maximum(kept_sum, 1e-8)
    for chunk in range(num_chunks):
        offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        values = tl.load(probs_ptr + row_start + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        kept = tl.where(values > low, values / denominator, 0.0)
        tl.store(out_ptr + row_start + offsets, kept, mask=mask)


def _prepare_probs(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={tuple(probs.shape)}")
    if not probs.is_cuda:
        raise ValueError("renorm kernels require a CUDA/HIP tensor")
    return probs.float().contiguous()


def _renorm_from_pivots(probs_fp32: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    batch_size, vocab_size = probs_fp32.shape
    num_chunks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    out = torch.empty_like(probs_fp32)
    partial_sums = torch.empty(
        (batch_size, num_chunks), device=probs_fp32.device, dtype=torch.float32
    )
    _mask_and_partial_sum_kernel[(batch_size, num_chunks)](
        probs_fp32,
        pivots,
        out,
        partial_sums,
        vocab_size=vocab_size,
        num_chunks=num_chunks,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )

    row_sums = partial_sums.sum(dim=1)
    _normalize_kernel[(triton.cdiv(out.numel(), _BLOCK_SIZE),)](
        out,
        row_sums,
        out.numel(),
        vocab_size=vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return out


def top_p_renorm_probs_triton(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Apply exact top-p thresholding and renormalize each probability row.

    The threshold, the mask and the renormalization all happen inside one kernel, so
    the cost is a small number of streaming passes over the batch and no vocabulary-
    sized sort, no auxiliary buffer, and no device-to-host transfer. See
    :func:`_top_p_renorm_kernel` for how the pivot is located.
    """
    probs_fp32 = _prepare_probs(probs)
    batch_size, vocab_size = probs_fp32.shape
    if batch_size == 0 or vocab_size == 0:
        return probs_fp32

    if isinstance(top_p, torch.Tensor):
        top_ps = top_p.to(device=probs.device, dtype=torch.float32).reshape(-1)
        if top_ps.numel() == 1:
            top_ps = top_ps.expand(batch_size)
        elif top_ps.numel() != batch_size:
            raise ValueError(
                f"top_p must be scalar or have one value per row, got "
                f"{top_ps.numel()} values for {batch_size} rows"
            )
    else:
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p values must be in (0, 1]")
        top_ps = torch.full(
            (batch_size,), float(top_p), device=probs.device, dtype=torch.float32
        )

    out = torch.empty_like(probs_fp32)
    _top_p_renorm_kernel[(batch_size,)](
        probs_fp32,
        top_ps.contiguous(),
        out,
        vocab_size=vocab_size,
        num_chunks=triton.cdiv(vocab_size, _BLOCK_SIZE),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return out


def top_k_renorm_probs_triton(
    probs: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    """Apply exact top-k thresholding and renormalize each probability row.

    Sorting uses PyTorch's device kernels because a vocabulary-sized in-register
    Triton sort does not scale to 100K+ vocabularies. Triton performs the
    bandwidth-heavy masking, partial reduction, and normalization.
    """
    probs_fp32 = _prepare_probs(probs)
    batch_size, vocab_size = probs_fp32.shape
    if batch_size == 0 or vocab_size == 0:
        return probs_fp32

    if isinstance(top_k, torch.Tensor):
        top_ks = top_k.to(device=probs.device, dtype=torch.int64).reshape(-1)
        if top_ks.numel() == 1:
            top_ks = top_ks.expand(batch_size)
        elif top_ks.numel() != batch_size:
            raise ValueError(
                f"top_k must be scalar or have one value per row, got "
                f"{top_ks.numel()} values for {batch_size} rows"
            )
    else:
        top_ks = torch.full(
            (batch_size,), int(top_k), device=probs.device, dtype=torch.int64
        )

    # Match FlashInfer's threshold semantics: sort descending, keep the k highest
    # probabilities, and retain all ties at the pivot.
    sorted_probs = torch.sort(probs_fp32, dim=-1, descending=True).values
    cutoff = (top_ks - 1).clamp_(min=0, max=vocab_size - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1).contiguous()

    return _renorm_from_pivots(probs_fp32, pivots)


__all__ = ["top_k_renorm_probs_triton", "top_p_renorm_probs_triton"]
