"""ROCm-compatible top-k / top-p probability renormalization fallbacks."""

from __future__ import annotations

from typing import Union

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_BLOCK_SIZE = 1024

# Nucleus size beyond which the top-p prefix search cannot answer and has to fall
# back to a sort. Real decode distributions need a handful of entries; flat ones
# (high temperature, early generation) can need far more, so the fallback must stay
# correct, not fast.
_TOP_P_PREFIX = 4096


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


def _top_p_pivots_sorted(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Exact top-p pivot for every row, from a full ascending sort.

    Matches FlashInfer's threshold semantics: discard the smallest entries whose
    cumulative mass stays below ``1 - p`` and retain all ties at the pivot. Exact for
    any input and free of host synchronization, but it pays an ``O(V log V)`` sort over
    a 100K+ vocabulary, and that cost grows with the number of rows.
    """
    vocab_size = probs.shape[1]
    sorted_probs = torch.sort(probs, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(
        1
    )
    cutoff.clamp_(max=vocab_size - 1)
    return sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1)


def _top_p_pivots_prefix(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Top-p pivot from a bounded prefix, falling back to a sort for rows it misses.

    Walking a descending prefix is the mirror image of walking the ascending CDF: an
    entry is kept exactly while the mass above it still leaves at least ``1 - p``
    behind. Budgeting against the row's own total rather than against ``1.0`` keeps
    ``top_p=1`` a no-op instead of truncating the tail of a peaked row, whose leading
    terms round up to one on their own.

    A row whose nucleus runs past the prefix has its pivot outside the prefix, and only
    a sort can find it. Those rows are flagged by the last prefix entry still being
    kept, and re-resolved exactly. Reading that flag costs one device-to-host transfer,
    which is what :func:`top_p_pivots` weighs against the sort it avoids.
    """
    vocab_size = probs.shape[1]
    prefix = min(_TOP_P_PREFIX, vocab_size)

    budget = probs.sum(dim=-1) - (1.0 - top_ps)
    values = torch.topk(probs, prefix, dim=-1).values
    within = values.cumsum(dim=-1) <= budget.unsqueeze(1)
    position = within.sum(dim=-1).clamp(max=prefix - 1)
    pivots = values.gather(1, position.unsqueeze(1)).squeeze(1)

    overflow = within[:, -1]
    if prefix < vocab_size and bool(overflow.any()):
        rows = overflow.nonzero(as_tuple=True)[0]
        pivots[rows] = _top_p_pivots_sorted(probs[rows], top_ps[rows])
    return pivots


def top_p_pivots(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Per-row top-p pivot for ``probs``, one value per row.

    Both paths implement the same threshold; they differ only in cost. The prefix path
    is worth its host synchronization once the sort it replaces is large enough to
    outweigh a fixed queue drain, so the choice turns on the row count alone -- which
    the host already knows, making the dispatch itself free. The crossover was measured
    on MI355X with a ~151K vocabulary under speculative decoding, where the verify batch
    is ``requests x draft tokens``: below it the sort wins on both throughput and TPOT,
    above it the sort starts costing TPOT.

    Descending and ascending accumulation round differently, so on a row flat enough
    that thousands of entries sit within a few ULPs of each other the two paths can land
    on adjacent entries. Ascending is the better-conditioned order, which is why it
    stays the default for the batch sizes where it is affordable.
    """
    if probs.shape[0] >= envs.SGLANG_OPT_TOP_P_PREFIX_MIN_ROWS.get():
        return _top_p_pivots_prefix(probs, top_ps)
    return _top_p_pivots_sorted(probs, top_ps)


def top_p_renorm_probs_triton(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Apply exact top-p thresholding and renormalize each probability row.

    Pivot selection is delegated to :func:`top_p_pivots`, which picks between a sort
    and a bounded prefix search by row count. Triton performs the bandwidth-heavy
    masking, partial reduction, and normalization.
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

    pivots = top_p_pivots(probs_fp32, top_ps).contiguous()

    return _renorm_from_pivots(probs_fp32, pivots)


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
