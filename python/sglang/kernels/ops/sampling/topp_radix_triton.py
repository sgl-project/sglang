"""Sort-free top-p pivot for ROCm: mass-weighted radix select over fp32 bit patterns.

Reference semantics (sglang top_p_renorm_probs_triton / FlashInfer): sort ascending,
cdf = cumsum, cutoff = first index with cdf >= 1 - p, pivot = sorted[cutoff], keep x >= pivot.
Equivalently pivot = min{v in row : G(v) >= 1 - p}, G(v) = sum of x with x <= v. Probs are
non-negative, so their fp32 bit patterns order like the values; the pivot's bit pattern is
found BITS bits at a time. When 1 - p <= 0 the search returns 0.0, which keeps every
entry (the reference returns the row minimum; the kept set is the same).

Pass p (one launch over (rows, blocks)) first finishes the digit selection of pass p - 1:
every program sums the previous pass's per-block partial bin masses (a few hundred floats)
and derives (prefix, mass below prefix); block 0 stores that state for pass p + 1. Then it
writes its own partial bin masses for elements matching the prefix. Partials are stored per
block instead of accumulated with fp32 atomics, which contend badly on ROCm. A final
single-block-per-row launch selects the last digit: PASSES + 1 launches in total.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_BITS = 4
_BINS = 1 << _BITS
_PASSES = 32 // _BITS


@triton.jit
def _select(
    part_ptr,
    state_ptr,
    target,
    row,
    rows,
    NBLK,
    P: tl.constexpr,
    NBLK_P2: tl.constexpr,
    BINS: tl.constexpr,
    BITS: tl.constexpr,
):
    """Digit selection for pass P from its partials and the state left by pass P - 1."""
    bins = tl.arange(0, BINS)
    blks = tl.arange(0, NBLK_P2)
    ptrs = part_ptr + ((P * rows + row) * NBLK + blks[:, None]) * BINS + bins[None, :]
    h = tl.sum(tl.load(ptrs, mask=blks[:, None] < NBLK, other=0.0), axis=0)
    if P > 0:
        prefix = tl.load(state_ptr + (P - 1) * rows * 2 + row * 2).to(
            tl.int32, bitcast=True
        )
        below = tl.load(state_ptr + (P - 1) * rows * 2 + row * 2 + 1)
    else:
        prefix = tl.zeros((), dtype=tl.int32)
        below = tl.zeros((), dtype=tl.float32)
    cum = below + tl.cumsum(h, axis=0)
    first_ok = tl.min(tl.where(cum >= target, bins, BINS), axis=0)
    last_nonempty = tl.max(tl.where(h > 0, bins, -1), axis=0)
    d = tl.where(first_ok < BINS, first_ok, tl.maximum(last_nonempty, 0))
    below = below + tl.sum(tl.where(bins < d, h, 0.0), axis=0)
    return (prefix << BITS) | d, below


@triton.jit
def _pass_kernel(
    bits_ptr,
    probs_ptr,
    part_ptr,
    state_ptr,
    target_ptr,
    V,
    NBLK,
    PASS: tl.constexpr,
    NBLK_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
    BITS: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    rows = tl.num_programs(0)
    shift_hi: tl.constexpr = 32 - BITS * PASS
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    base = row.to(tl.int64) * V
    b = tl.load(bits_ptr + base + offs, mask=m, other=0)
    x = tl.load(probs_ptr + base + offs, mask=m, other=0.0)
    if PASS > 0:
        target = tl.load(target_ptr + row)
        prefix, below = _select(
            part_ptr, state_ptr, target, row, rows, NBLK, PASS - 1, NBLK_P2, BINS, BITS
        )
        if blk == 0:
            tl.store(
                state_ptr + (PASS - 1) * rows * 2 + row * 2 + 0,
                prefix.to(tl.float32, bitcast=True),
            )
            tl.store(state_ptr + (PASS - 1) * rows * 2 + row * 2 + 1, below)
        m = m & ((b >> shift_hi) == prefix)
    digit = (b >> (shift_hi - BITS)) & (BINS - 1)
    x = tl.where(m, x, 0.0)
    bins = tl.arange(0, BINS)
    part = tl.sum(tl.where(bins[:, None] == digit[None, :], x[None, :], 0.0), axis=1)
    tl.store(part_ptr + ((PASS * rows + row) * NBLK + blk) * BINS + bins, part)


@triton.jit
def _final_kernel(
    part_ptr,
    state_ptr,
    target_ptr,
    pivot_ptr,
    NBLK,
    PASSES: tl.constexpr,
    NBLK_P2: tl.constexpr,
    BINS: tl.constexpr,
    BITS: tl.constexpr,
):
    row = tl.program_id(0)
    rows = tl.num_programs(0)
    target = tl.load(target_ptr + row)
    prefix, below = _select(
        part_ptr, state_ptr, target, row, rows, NBLK, PASSES - 1, NBLK_P2, BINS, BITS
    )
    tl.store(pivot_ptr + row, prefix)


def top_p_pivots_radix(probs_fp32: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    rows, V = probs_fp32.shape
    dev = probs_fp32.device
    bits = probs_fp32.view(torch.int32)
    target = (1.0 - top_ps.to(torch.float32)).contiguous()
    nblk = triton.cdiv(V, _BLOCK)
    nblk_p2 = triton.next_power_of_2(nblk)
    part = torch.empty((_PASSES, rows, nblk, _BINS), device=dev, dtype=torch.float32)
    state = torch.empty((_PASSES, rows, 2), device=dev, dtype=torch.float32)
    for p in range(_PASSES):
        _pass_kernel[(rows, nblk)](
            bits,
            probs_fp32,
            part,
            state,
            target,
            V,
            nblk,
            PASS=p,
            NBLK_P2=nblk_p2,
            BLOCK=_BLOCK,
            BINS=_BINS,
            BITS=_BITS,
            num_warps=8,
        )
    pivot = torch.empty(rows, device=dev, dtype=torch.int32)
    _final_kernel[(rows,)](
        part,
        state,
        target,
        pivot,
        nblk,
        PASSES=_PASSES,
        NBLK_P2=nblk_p2,
        BINS=_BINS,
        BITS=_BITS,
    )
    return pivot.view(torch.float32)
