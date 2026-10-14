"""ROCm-compatible top-p probability renormalization fallback."""

from __future__ import annotations

from typing import Union

import torch
import triton
import triton.language as tl

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
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size
    row_offsets = row * vocab_size + offsets

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
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    row = offsets // vocab_size
    values = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    denominator = tl.load(row_sums_ptr + row, mask=mask, other=1.0)
    tl.store(out_ptr + offsets, values / denominator, mask=mask)


def top_p_renorm_probs_triton(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Apply exact top-p thresholding and renormalize each probability row.

    Sorting and prefix sums use PyTorch's device kernels because a vocabulary-sized
    in-register Triton sort does not scale to 100K+ vocabularies. Triton performs
    the bandwidth-heavy masking, partial reduction, and normalization.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={tuple(probs.shape)}")
    if not probs.is_cuda:
        raise ValueError("top_p_renorm_probs_triton requires a CUDA/HIP tensor")

    probs_fp32 = probs.float().contiguous()
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

    # Match FlashInfer's threshold semantics: sort ascending, discard the prefix
    # whose cumulative mass is below 1 - p, and retain all ties at the pivot.
    sorted_probs = torch.sort(probs_fp32, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(
        1
    )
    cutoff.clamp_(max=vocab_size - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1).contiguous()

    num_chunks = triton.cdiv(vocab_size, _BLOCK_SIZE)
    out = torch.empty_like(probs_fp32)
    partial_sums = torch.empty(
        (batch_size, num_chunks), device=probs.device, dtype=torch.float32
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


__all__ = ["top_p_renorm_probs_triton"]
