"""Sequence-aware deterministic top-K selection for Double Sparsity decode.

Replaces the two full-width ``torch.topk`` passes of the selection pipeline
with a multi-pass radix select whose work is proportional to each request's
LIVE window (per-row ``seq_lens`` bounds every scan; token blocks entirely
past a row's length exit immediately), while keeping the exact DS selection
contract:

  * (score DESCENDING, then logical position ASCENDING) tie-break — ties are
    admitted lowest-position-first by construction (per-block prefix counts +
    in-block cumsum over ascending offsets; the only atomics are histogram
    COUNT increments, which are order-independent), so the result is
    bit-deterministic run-to-run and across TP ranks.
  * output: ``selected_indices`` int32 ``[bs, max_top_k]`` in ascending
    position order with ``-1`` padding, plus ``valid_lengths`` int32 ``[bs]``.
  * ``-inf`` (masked/unwritten slots) and NaN are never selected;
    ``valid_lengths = min(num_selectable, max_top_k)`` per row. ``+inf`` is
    not producible by the scorer but, if present, ranks as the maximal score
    — matching the torch reference selector and the AOT op exactly.
  * graph-safe: fixed grids, no host syncs, allocation-free given the scratch
    bundle (all intermediates are caller-owned buffers).

Algorithm (keys are the standard order-preserving uint32 mapping of fp32 —
``-0.0`` is canonicalized to ``+0.0`` so equal scores tie by position):

  1. Four rounds of (per-row 256-bin histogram over the live window on the
     next 8 key bits, restricted to positions matching the key prefix so far;
     one-program-per-row threshold scan) narrow to the exact threshold key T
     and the tie quota r (rows with fewer finite scores than K resolve
     naturally: the scan targets min(K, num_finite)).
  2. A per-block count pass (strictly-above / tied-at-T per block), a
     per-row exclusive block-prefix pass, and an emission pass that scatters
     each selected position directly into its ascending output slot.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    _TRITON_AVAILABLE = False


RADIX_BITS = 8
NUM_BINS = 1 << RADIX_BITS
NUM_ROUNDS = 4  # 32-bit keys


if _TRITON_AVAILABLE:

    @triton.jit
    def _key_of(s):
        # Order-preserving uint32 key of fp32 held in int64 lanes.
        # -0.0 + 0.0 == +0.0 canonicalizes the zero pair so equal scores tie.
        s = s + 0.0
        bits = s.to(tl.int32, bitcast=True).to(tl.int64)
        upos = (bits & 0xFFFFFFFF) | 0x80000000
        uneg = (~bits) & 0xFFFFFFFF
        return tl.where(bits >= 0, upos, uneg)

    @triton.jit
    def _radix_hist_kernel(
        scores_ptr,  # fp32/bf16 [bs, width] (loads upcast to fp32)
        seq_lens_ptr,  # int32 [bs]
        prefix_ptr,  # int64 [bs] accumulated high key bits (<< shift aligned)
        hist_ptr,  # int32 [bs, NUM_BINS] (pre-zeroed)
        width: tl.constexpr,
        scores_stride_b: tl.constexpr,
        SHIFT: tl.constexpr,  # bits below the digit being counted
        PREFIX_SHIFT: tl.constexpr,  # SHIFT + radix bits (start of the fixed prefix)
        PREFIX_BITS: tl.constexpr,  # number of key bits already fixed (0 on the first radix pass)
        NBINS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        start = blk * BLOCK
        seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
        n = tl.minimum(seq_len, width)
        if start >= n:
            return
        offs = start + tl.arange(0, BLOCK)
        in_win = offs < n
        s = tl.load(
            scores_ptr + row * scores_stride_b + offs, mask=in_win, other=float("-inf")
        ).to(
            tl.float32
        )  # exact upcast: bf16 input compares as its fp32 value
        finite = in_win & (s == s) & (s != float("-inf"))
        key = _key_of(s)
        if PREFIX_BITS > 0:
            want = tl.load(prefix_ptr + row)
            count_mask = finite & ((key >> PREFIX_SHIFT) == (want >> PREFIX_SHIFT))
        else:
            count_mask = finite
        digit = (key >> SHIFT) & (NBINS - 1)
        tl.atomic_add(
            hist_ptr + row * NBINS + digit.to(tl.int32),
            1,
            mask=count_mask,
        )

    @triton.jit
    def _radix_scan_kernel(
        hist_ptr,  # int32 [bs, NUM_BINS]
        prefix_ptr,  # int64 [bs] in/out
        quota_ptr,  # int32 [bs] in/out (k_target in, tie quota state out)
        lengths_out_ptr,  # int32 [bs] valid_lengths (written in the first round)
        max_top_k: tl.constexpr,
        SHIFT: tl.constexpr,
        FIRST_ROUND: tl.constexpr,
        NBINS: tl.constexpr,
    ):
        row = tl.program_id(0)
        bins = tl.arange(0, NBINS)
        h = tl.load(hist_ptr + row * NBINS + bins).to(tl.int64)
        total = tl.sum(h, axis=0)
        if FIRST_ROUND:
            k64 = tl.minimum(total, max_top_k)
            tl.store(lengths_out_ptr + row, k64.to(tl.int32))
        else:
            k64 = tl.load(quota_ptr + row).to(tl.int64)
        # above_excl[b] = count of keys with digit > b among candidates.
        incl = tl.cumsum(h, axis=0)  # inclusive ascending
        above_excl = total - incl
        # The threshold digit b*: above_excl[b] < k <= above_excl[b] + h[b].
        is_thr = (above_excl < k64) & ((above_excl + h) >= k64) & (h > 0)
        bstar = tl.sum(tl.where(is_thr, bins.to(tl.int64), 0), axis=0)
        above_at = tl.sum(tl.where(is_thr, above_excl, 0), axis=0)
        new_quota = k64 - above_at  # candidates needed from the b* bin
        prev = tl.load(prefix_ptr + row)
        tl.store(prefix_ptr + row, prev | (bstar << SHIFT))
        tl.store(quota_ptr + row, new_quota.to(tl.int32))

    @triton.jit
    def _block_count_kernel(
        scores_ptr,
        seq_lens_ptr,
        thr_ptr,  # int64 [bs] full threshold key T
        above_ptr,  # int32 [bs, nblocks]
        tie_ptr,  # int32 [bs, nblocks]
        width: tl.constexpr,
        scores_stride_b: tl.constexpr,
        nblocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        start = blk * BLOCK
        seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
        n = tl.minimum(seq_len, width)
        if start >= n:
            tl.store(above_ptr + row * nblocks + blk, 0)
            tl.store(tie_ptr + row * nblocks + blk, 0)
            return
        offs = start + tl.arange(0, BLOCK)
        in_win = offs < n
        s = tl.load(
            scores_ptr + row * scores_stride_b + offs, mask=in_win, other=float("-inf")
        ).to(
            tl.float32
        )  # exact upcast: bf16 input compares as its fp32 value
        finite = in_win & (s == s) & (s != float("-inf"))
        key = _key_of(s)
        t = tl.load(thr_ptr + row)
        above = finite & (key > t)
        tie = finite & (key == t)
        tl.store(above_ptr + row * nblocks + blk, tl.sum(above.to(tl.int32), axis=0))
        tl.store(tie_ptr + row * nblocks + blk, tl.sum(tie.to(tl.int32), axis=0))

    @triton.jit
    def _block_prefix_kernel(
        above_ptr,  # int32 [bs, nblocks] in: counts, out untouched
        tie_ptr,
        above_pref_ptr,  # int32 [bs, nblocks] out: exclusive prefixes
        tie_pref_ptr,
        nblocks: tl.constexpr,
        NBLOCK_POW2: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, NBLOCK_POW2)
        m = offs < nblocks
        a = tl.load(above_ptr + row * nblocks + offs, mask=m, other=0).to(tl.int64)
        t = tl.load(tie_ptr + row * nblocks + offs, mask=m, other=0).to(tl.int64)
        a_excl = tl.cumsum(a, axis=0) - a
        t_excl = tl.cumsum(t, axis=0) - t
        tl.store(above_pref_ptr + row * nblocks + offs, a_excl.to(tl.int32), mask=m)
        tl.store(tie_pref_ptr + row * nblocks + offs, t_excl.to(tl.int32), mask=m)

    @triton.jit
    def _emit_kernel(
        scores_ptr,
        seq_lens_ptr,
        thr_ptr,  # int64 [bs]
        quota_ptr,  # int32 [bs] tie quota r
        above_pref_ptr,  # int32 [bs, nblocks] exclusive prefix of strictly-above
        tie_pref_ptr,  # int32 [bs, nblocks] exclusive prefix of ties
        out_ptr,  # int32 [bs, out_width] (pre-filled with -1)
        width: tl.constexpr,
        scores_stride_b: tl.constexpr,
        out_stride_b: tl.constexpr,
        nblocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        start = blk * BLOCK
        seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
        n = tl.minimum(seq_len, width)
        if start >= n:
            return
        offs = start + tl.arange(0, BLOCK)
        in_win = offs < n
        s = tl.load(
            scores_ptr + row * scores_stride_b + offs, mask=in_win, other=float("-inf")
        ).to(
            tl.float32
        )  # exact upcast: bf16 input compares as its fp32 value
        finite = in_win & (s == s) & (s != float("-inf"))
        key = _key_of(s)
        t = tl.load(thr_ptr + row)
        r = tl.load(quota_ptr + row).to(tl.int64)
        above = finite & (key > t)
        tie = finite & (key == t)
        tie_pref = tl.load(tie_pref_ptr + row * nblocks + blk).to(tl.int64)
        above_pref = tl.load(above_pref_ptr + row * nblocks + blk).to(tl.int64)
        # Global ascending tie rank; admit the lowest-position r ties.
        tie_i = tie.to(tl.int64)
        tie_rank = tie_pref + tl.cumsum(tie_i, axis=0) - tie_i
        admit_tie = tie & (tie_rank < r)
        selected = above | admit_tie
        # Output slot = number of selected positions before this one (global).
        admitted_tie_before_block = tl.minimum(tie_pref, r)
        sel_i = selected.to(tl.int64)
        local_sel = tl.cumsum(sel_i, axis=0) - sel_i
        slot = above_pref + admitted_tie_before_block + local_sel
        tl.store(
            out_ptr + row * out_stride_b + slot.to(tl.int32),
            offs.to(tl.int32),
            mask=selected,
        )


def select_topk_sequence_order_triton(
    token_scores: torch.Tensor,  # fp32 or bf16 [bs, width] (kernels upcast in-register)
    seq_lens: torch.Tensor,  # int32 [bs]
    max_top_k: int,
    *,
    out_indices: torch.Tensor,  # int32 [bs_buf >= bs, >= max_top_k]
    out_lengths: torch.Tensor,  # int32 [bs_buf >= bs]
    scratch_hist: torch.Tensor,  # int32 [bs_buf, NUM_BINS]
    scratch_key_prefix: torch.Tensor,  # int64 [bs_buf]
    scratch_quota: torch.Tensor,  # int32 [bs_buf]
    scratch_block_above: torch.Tensor,  # int32 [bs_buf, nblocks]
    scratch_block_tie: torch.Tensor,  # int32 [bs_buf, nblocks]
    scratch_above_pref: torch.Tensor,  # int32 [bs_buf, nblocks]
    scratch_tie_pref: torch.Tensor,  # int32 [bs_buf, nblocks]
    block: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the sequence-aware deterministic top-K over caller-owned buffers.

    All scratch must be sized for ``nblocks = ceil(width / block)``. The call
    performs no allocations; every launch has a fixed grid, so the whole
    sequence is CUDA-graph capturable.
    """
    assert _TRITON_AVAILABLE, "Triton is required for the fast top-k path"
    bs, width = token_scores.shape
    nblocks = (width + block - 1) // block
    assert scratch_block_above.shape[1] >= nblocks, "scratch nblocks too small"

    hist = scratch_hist[:bs]
    key_prefix = scratch_key_prefix[:bs]
    quota = scratch_quota[:bs]

    key_prefix.zero_()
    out_indices[:bs, :max_top_k].fill_(-1)

    grid_wide = (bs, nblocks)
    for rnd in range(NUM_ROUNDS):
        shift = 32 - RADIX_BITS * (rnd + 1)
        hist.zero_()
        _radix_hist_kernel[grid_wide](
            token_scores,
            seq_lens,
            key_prefix,
            hist,
            width=width,
            scores_stride_b=token_scores.stride(0),
            SHIFT=shift,
            PREFIX_SHIFT=shift + RADIX_BITS,
            PREFIX_BITS=RADIX_BITS * rnd,
            NBINS=NUM_BINS,
            BLOCK=block,
        )
        _radix_scan_kernel[(bs,)](
            hist,
            key_prefix,
            quota,
            out_lengths,
            max_top_k=max_top_k,
            SHIFT=shift,
            FIRST_ROUND=(rnd == 0),
            NBINS=NUM_BINS,
        )

    _block_count_kernel[grid_wide](
        token_scores,
        seq_lens,
        key_prefix,
        scratch_block_above,
        scratch_block_tie,
        width=width,
        scores_stride_b=token_scores.stride(0),
        nblocks=scratch_block_above.shape[1],
        BLOCK=block,
    )
    nb_pow2 = 1
    while nb_pow2 < nblocks:
        nb_pow2 *= 2
    _block_prefix_kernel[(bs,)](
        scratch_block_above,
        scratch_block_tie,
        scratch_above_pref,
        scratch_tie_pref,
        nblocks=scratch_block_above.shape[1],
        NBLOCK_POW2=nb_pow2,
    )
    _emit_kernel[grid_wide](
        token_scores,
        seq_lens,
        key_prefix,
        quota,
        scratch_above_pref,
        scratch_tie_pref,
        out_indices,
        width=width,
        scores_stride_b=token_scores.stride(0),
        out_stride_b=out_indices.stride(0),
        nblocks=scratch_block_above.shape[1],
        BLOCK=block,
    )
    return out_indices, out_lengths
