"""ROCm-compatible top-k / top-p probability renormalization fallbacks."""

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


_TOP_P_BLOCK = 1024
_TOP_P_CHUNK = 4096
_TOP_P_FANOUT = 8
# Probabilities are in [0, 1], so their fp32 keys lie in [0, _ONE_KEY], fewer than
# 2**30 values; each pass divides the search interval by _TOP_P_FANOUT (2**3).
_ONE_KEY = 0x3F800000
_TOP_P_PASSES = 10


@triton.jit
def _top_p_pass_kernel(
    probs_ptr,
    top_ps_ptr,
    out_ptr,
    part_ptr,
    tail_ptr,
    bounds_ptr,
    num_rows,
    vocab_size,
    num_tiles,
    PASS: tl.constexpr,
    PASSES: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    FANOUT: tl.constexpr,
    ONE_KEY: tl.constexpr,
):
    # The pivot is the smallest probability v with mass(probs <= v) >= 1 - top_p,
    # i.e. the ascending sort / cumsum / searchsorted pivot, found without a sort:
    # non-negative fp32 values order like their int32 bit patterns, so bisect on
    # the keys with FANOUT candidates per pass. Each launch is one pass over
    # (row, tile) programs. Passes 0..PASSES-1 store per-tile candidate masses in
    # part[pass, row, tile, :]; from pass 1 on, every program of a row first folds
    # the previous pass's masses into the same next [lo, hi] (bounds). Pass
    # PASSES writes the renormalized row.
    # The denominator is summed from the kept probabilities (tail) rather than
    # taken as total - mass(key < pivot): for small top_p both terms are near 1
    # and the difference loses most of its precision.
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    j = tl.arange(0, FANOUT)
    t = tl.arange(0, NUM_CHUNKS)
    tile_ptr = probs_ptr + row * vocab_size + tile * CHUNK
    offs = tl.arange(0, BLOCK)
    stride_pass = num_rows * NUM_CHUNKS * FANOUT
    stride_row = NUM_CHUNKS * FANOUT

    if PASS == 0:
        lo = 0
        hi = ONE_KEY
    else:
        if PASS == 1:
            lo = 0
            hi = ONE_KEY
        else:
            lo = tl.load(bounds_ptr + ((PASS - 1) * num_rows + row) * 2)
            hi = tl.load(bounds_ptr + ((PASS - 1) * num_rows + row) * 2 + 1)
        step = (hi - lo + FANOUT) // FANOUT
        cand = tl.minimum(lo + (j + 1) * step - 1, hi)
        prev = part_ptr + (PASS - 1) * stride_pass + row * stride_row
        mass = tl.load(
            prev + t[:, None] * FANOUT + j[None, :],
            mask=(t < num_tiles)[:, None],
            other=0.0,
        )
        mass = tl.sum(mass, axis=0)
        target = 1.0 - tl.load(top_ps_ptr + row)
        # The last candidate is hi; if rounding leaves even it short, keep hi.
        first = tl.minimum(
            tl.min(tl.where(mass >= target, j, FANOUT), axis=0), FANOUT - 1
        )
        prev_cand = tl.sum(tl.where(j == first - 1, cand, 0), axis=0)
        hi = tl.sum(tl.where(j == first, cand, 0), axis=0)
        lo = tl.where(first == 0, lo, prev_cand + 1)
        if tile == 0:
            tl.store(bounds_ptr + (PASS * num_rows + row) * 2, lo)
            tl.store(bounds_ptr + (PASS * num_rows + row) * 2 + 1, hi)

    if PASS < PASSES:
        step = (hi - lo + FANOUT) // FANOUT
        cand = tl.minimum(lo + (j + 1) * step - 1, hi)
        # Candidate j's interval starts at cand[j - 1] + 1 (lo for j = 0); the
        # next fold makes that start the new lo.
        cand_lo = tl.minimum(lo + j * step, hi + 1)
        mass_acc = tl.zeros([BLOCK, FANOUT], dtype=tl.float32)
        if PASS == PASSES - 1:
            tail_acc = tl.zeros([BLOCK, FANOUT], dtype=tl.float32)
        for start in range(0, CHUNK, BLOCK):
            idx = tile * CHUNK + start + offs
            p = tl.load(tile_ptr + start + offs, mask=idx < vocab_size, other=0.0)
            key = p.to(tl.int32, bitcast=True)
            mass_acc += tl.where(key[:, None] <= cand[None, :], p[:, None], 0.0)
            if PASS == PASSES - 1:
                tail_acc += tl.where(key[:, None] >= cand_lo[None, :], p[:, None], 0.0)
        out_part = part_ptr + PASS * stride_pass + row * stride_row + tile * FANOUT
        tl.store(out_part + j, tl.sum(mass_acc, axis=0))
        if PASS == PASSES - 1:
            out_tail = tail_ptr + row * stride_row + tile * FANOUT
            tl.store(out_tail + j, tl.sum(tail_acc, axis=0))
    else:
        tail = tl.load(
            tail_ptr + row * stride_row + t * FANOUT + first,
            mask=t < num_tiles,
            other=0.0,
        )
        kept_sum = tl.sum(tail, axis=0)
        out_tile_ptr = out_ptr + row * vocab_size + tile * CHUNK
        for start in range(0, CHUNK, BLOCK):
            idx = tile * CHUNK + start + offs
            m = idx < vocab_size
            p = tl.load(tile_ptr + start + offs, mask=m, other=0.0)
            key = p.to(tl.int32, bitcast=True)
            tl.store(
                out_tile_ptr + start + offs,
                tl.where(key >= lo, p / kept_sum, 0.0),
                mask=m,
            )


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

    Same pivot as FlashInfer's: sort ascending, discard the prefix whose
    cumulative mass is below 1 - p, and retain all ties at the pivot. The
    kernel finds it by bisection instead of a vocabulary-wide sort and scan.
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
    num_chunks = triton.next_power_of_2(triton.cdiv(vocab_size, _TOP_P_CHUNK))
    device = probs.device
    part = torch.empty(
        (_TOP_P_PASSES, batch_size, num_chunks, _TOP_P_FANOUT),
        dtype=torch.float32,
        device=device,
    )
    tail = torch.empty(
        (batch_size, num_chunks, _TOP_P_FANOUT), dtype=torch.float32, device=device
    )
    bounds = torch.empty(
        (_TOP_P_PASSES + 1, batch_size, 2), dtype=torch.int32, device=device
    )
    top_ps = top_ps.contiguous()
    num_tiles = triton.cdiv(vocab_size, _TOP_P_CHUNK)
    grid = (batch_size, num_tiles)
    for pass_id in range(_TOP_P_PASSES + 1):
        _top_p_pass_kernel[grid](
            probs_fp32,
            top_ps,
            out,
            part,
            tail,
            bounds,
            batch_size,
            vocab_size,
            num_tiles,
            PASS=pass_id,
            PASSES=_TOP_P_PASSES,
            NUM_CHUNKS=num_chunks,
            CHUNK=_TOP_P_CHUNK,
            BLOCK=_TOP_P_BLOCK,
            FANOUT=_TOP_P_FANOUT,
            ONE_KEY=_ONE_KEY,
            num_warps=4,
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
