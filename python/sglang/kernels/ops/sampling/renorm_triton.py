"""Triton top-k / top-p probability renormalization for devices without the AOT kernel.

Pivot selection is shared with the portable path in :mod:`.renorm`; only the
bandwidth-bound part is in Triton. Applying a pivot in eager torch costs five
traversals of a (batch, vocab) tensor -- mask, materialize, reduce, divide, select --
which at a 100K+ vocabulary dominates the pivot search itself. Summing without
materializing the masked copy, then folding the reciprocal into the write, brings
that down to one read plus one read-modify-write.
"""

from __future__ import annotations

from typing import Union

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.sampling.renorm import (
    per_row_threshold,
    top_k_pivots,
    top_p_pivots,
)

_BLOCK_SIZE = 1024


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
    chunk = tl.program_id(1).to(tl.int64)
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
    scale = tl.where(denominator > 0.0, 1.0 / denominator, 0.0)
    tl.store(out_ptr + offsets, values * scale, mask=mask)


def apply_pivot_triton(probs: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    """Keep ``probs >= pivots`` row-wise and renormalize what survives."""
    batch_size, vocab_size = probs.shape
    num_chunks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    grid = (batch_size, num_chunks)

    out = torch.empty_like(probs)
    partial_sums = torch.empty(
        (batch_size, num_chunks), device=probs.device, dtype=torch.float32
    )
    _mask_and_partial_sum_kernel[grid](
        probs,
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


def top_k_renorm_probs_triton(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    *,
    max_top_k: int | None = None,
) -> torch.Tensor:
    """Keep every entry at least as likely as the k-th largest, then renormalize."""
    assert probs.ndim == 2
    probs = probs.float().contiguous()
    if probs.shape[0] == 0:
        return probs.clone()
    assert probs.shape[1] > 0

    top_ks = per_row_threshold(top_k, probs=probs, dtype=torch.int64).clamp(
        1, probs.shape[1]
    )
    return apply_pivot_triton(probs, top_k_pivots(probs, top_ks, max_top_k=max_top_k))


def top_p_renorm_probs_triton(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep the nucleus -- every entry at least as likely as its pivot -- and
    renormalize."""
    assert probs.ndim == 2
    probs = probs.float().contiguous()
    if probs.shape[0] == 0:
        return probs.clone()
    assert probs.shape[1] > 0

    top_ps = per_row_threshold(top_p, probs=probs, dtype=torch.float32).clamp(0.0, 1.0)
    return apply_pivot_triton(probs, top_p_pivots(probs, top_ps))


__all__ = [
    "apply_pivot_triton",
    "top_k_renorm_probs_triton",
    "top_p_renorm_probs_triton",
]
