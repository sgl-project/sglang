# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 68506823: atom/models/deepseek_v41/mono/kernels/topk.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A CTA's exact top-k of up to ``LDS_KEYS`` fp32 values, the answer of aiter's
``top_k_per_row_decode(stable=True)``: ranked by the fp32 bits' u32 order key
(-0 below +0, NaN above +inf), ties to the lowest index, emitted ascending by
index, -1 padded. P0.7 pinned the key and the tie rule against aiter;
``lds_topk_timing.py CHECK=1`` (skill mono-indexer) pins this routine against
that rule on ties, +-0, NaN and +-inf at every edge n.

Thread i holds ``per_thread`` consecutive indices from i x per_thread, as
keys in LDS, so thread order is index order and one block scan places the
winners. Four 8-bit radix passes find the k-th key; integer counts, so the
answer does not depend on arrival order.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32

from sglang.srt.layers.mono.common.device.ops import (
    block_excl_scan,
    traced,
    wave_incl_scan,
    wave_max,
)
from sglang.srt.layers.mono.common.plan.execution import THREADS, WAVES

PER_THREAD = 64  # the largest a CTA's LDS takes alone: 32768 keys
LDS_KEYS = THREADS * PER_THREAD
INT_MIN = -(2**31)


def order_key(bits):
    """u32 order key of fp32 bits, held in an Int32."""
    return (bits < 0).select(~bits, bits | INT_MIN)


def key_bits(key):
    """The fp32 bits ``order_key`` came from."""
    return (key < 0).select(key ^ INT_MIN, ~key)


def ult(a, b):
    """a < b as u32."""
    return (a ^ INT_MIN) < (b ^ INT_MIN)


# a histogram: 256 bins, then a dummy bin a lane: a key outside the pass's
# prefix counts there, so the adds need no branch and no two lanes share one
HIST_WORDS = 256 + 64


def lds_words(per_thread: int = PER_THREAD) -> dict[str, int]:
    """The LDS the routine needs, in 32-bit words."""
    return {"keys": THREADS * per_thread, "hist": 3 * HIST_WORDS, "buf": WAVES}


def _no_extra():
    return None


@traced
def lds_topk(n, k, load, emit, lds, tid, per_thread=PER_THREAD, after_keys=None):
    """Top ``k`` of ``n`` <= THREADS x per_thread values, given a quad at a
    time: ``load(q, i)`` -> the 4 values (fp32 bits as Int32) at indices i ..
    i + 3, for i = per_thread tid + 4 q clamped to the quad holding index n - 1
    (values past n are read but never counted) -> ``emit(pos, d, i, extra)`` by
    the thread holding each winner (index i = per_thread tid + d), pos its
    place ascending, and ``emit(pos, None, -1, extra)`` for pos in [min(n, k),
    k). The keys stay in ``lds["keys"]`` (``key_bits`` of one is its value).
    Barriers included; every thread of the CTA calls it. ``extra`` is what
    ``after_keys()`` returns (None without it), called once the keys are in,
    before the passes: loads it issues overlap the passes without holding up
    the keys' own (the load counter is in order)."""
    lane, wave = tid % 64, tid // 64
    keys, hist, buf = lds["keys"], lds["hist"], lds["buf"]
    base = tid * per_thread
    # quads past the last value's re-read it: in the row, never counted
    last = fx.max(n - 1, 0) & ~3
    kv = []
    for q in range_constexpr(per_thread // 4):
        quad = load(q, fx.min(base + 4 * q, last))
        for e in range_constexpr(4):
            kv.append(order_key(quad[e]))
            fx.ptr_store(kv[-1], keys + (base + 4 * q + e))
    for r in range_constexpr((2 * HIST_WORDS + THREADS - 1) // THREADS):
        if tid + THREADS * r < 2 * HIST_WORDS:  # histograms 0 and 1
            fx.ptr_store(Int32(0), hist + (tid + THREADS * r))
    gpu.barrier()
    extra = (after_keys or _no_extra)()
    prefix = Int32(0)
    mask = Int32(0)
    need = Int32(k)
    n = fx.max(n, 0)
    # n <= k takes every key: no k-th to find
    if n > k:
        # three histograms in rotation: pass p counts into p % 3 and clears
        # (p + 2) % 3, which every wave finished reading before this pass's barrier
        for p in range_constexpr(4):
            shift = 24 - 8 * p
            h = hist + (p % 3) * HIST_WORDS
            dummy = 256 + lane
            if const_expr(p == 0):
                # a thread's keys are consecutive and logits share their top byte:
                # a run of one bin goes in as one add at its end (the others add 0
                # to the lane's own dummy bin), not an add a key on one hot bin
                run_bin, run = Int32(dummy), Int32(0)
                for d in range_constexpr(per_thread):
                    bin_ = (base + d < n).select((kv[d] >> shift) & 255, dummy)
                    same = bin_ == run_bin
                    fx.llvm.atomic_add(
                        h + same.select(dummy, run_bin), same.select(0, run),
                        syncscope="workgroup",
                    )  # fmt: skip
                    run = same.select(run + 1, 1)
                    run_bin = bin_
                fx.llvm.atomic_add(h + run_bin, run, syncscope="workgroup")
            else:
                for d in range_constexpr(per_thread):
                    key = kv[d]
                    counted = (base + d < n) & ((key & mask) == prefix)
                    bin_ = counted.select((key >> shift) & 255, dummy)
                    fx.llvm.atomic_add(h + bin_, Int32(1), syncscope="workgroup")
            gpu.barrier()
            if (p + 2 < 4) & (tid < HIST_WORDS):
                fx.ptr_store(Int32(0), hist + ((p + 2) % 3) * HIST_WORDS + tid)
            # lane l: bins 255 - 4 l - q, highest first
            cs = [fx.ptr_load(h + (255 - 4 * lane - q)) for q in range(4)]
            s4 = cs[0] + cs[1] + cs[2] + cs[3]
            before = wave_incl_scan(s4, lane) - s4
            found = Int32(-1)
            for q in range_constexpr(4):
                hit = (before < need) & (need <= before + cs[q])
                found = hit.select(
                    ((255 - 4 * lane - q) << 16) | (need - before), found
                )
                before = before + cs[q]
            found = wave_max(found)
            prefix = prefix | (((found >> 16) & 255) << shift)
            mask = mask | Int32(
                (255 << shift) - (1 << 32) if shift == 24 else 255 << shift
            )
            need = found & 0xFFFF
    take_all = n <= k
    # winners: keys above the k-th, then its ties lowest index first; one scan of
    # (above, ties) packed in 16-bit halves gives both prefixes
    gts = [(base + d < n) & ult(prefix, kv[d]) for d in range(per_thread)]
    eqs = [(base + d < n) & (kv[d] == prefix) for d in range(per_thread)]
    above, ties = Int32(0), Int32(0)
    for d in range_constexpr(per_thread):
        above = above + gts[d].select(1, 0)
        ties = ties + eqs[d].select(1, 0)
    packed, _ = block_excl_scan(above | (ties << 16), lane, wave, buf)
    ab = packed & 0xFFFF
    tb = packed >> 16
    for d in range_constexpr(per_thread):
        i = base + d
        gt, tie = gts[d], eqs[d]
        win = (i < n) & (take_all | gt | (tie & (tb < need)))
        at = take_all.select(i, ab + fx.min(tb, need))
        if win:
            emit(at, d, i, extra)
        ab = ab + gt.select(1, 0)
        tb = tb + tie.select(1, 0)
    n_out = fx.min(n, k)
    for r in range_constexpr((k + THREADS - 1) // THREADS):
        pos = tid + THREADS * r
        if (pos >= n_out) & (pos < k):
            emit(pos, None, Int32(-1), extra)
    gpu.barrier()
