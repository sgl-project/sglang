# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Default-off one-kernel all-reduce sites for the DFlash MLA draft layers (SGLANG_OPT_DFLASH_DRAFT_AR_SITE).

Each draft layer ends its attention and its MLP with the same chain: all-reduce of the row-parallel projection's BF16
partials -> grouped conv side 1 (DFlashGroupedConv.finish) -> fused add-RMSNorm. One kernel per site replaces the
three. Each rank convolves its own partial row (side 1) BEFORE the reduction and multicasts it into slot ``rank`` of
every rank's Lamport mailbox; every rank then sums the TP slots in FP32 in rank order and rounds once (so every rank
holds the same bits), adds the residual in FP32 and RMS-normalizes that FP32 sum, as the fused add-RMSNorm does; the
residual is updated in place with the sum rounded to BF16.

The conv is linear in the reduced row and its coefficients come from the replicated layer input (the same on every
rank), so conv(sum_r p_r) == sum_r conv(p_r) in exact arithmetic; only the rounding points move: each rank's convolved
partial is rounded to BF16 before the sum, where the served chain rounds the sum, the conv and the add.

The attention site normalizes with the layer's post-attention norm; the MLP site with the NEXT layer's input norm, or
the model's final norm after the last layer, so with the sites engaged the model skips those norms
(DFlashDraftModel._forward_ar_sites). Consecutive sites alternate between the mailbox's two parities (attention 0,
MLP 1): a rank can run at most one site ahead of a peer, because each site waits for every peer's push of that site,
and a peer pushes only after it has read and re-armed its slots of the site before.

The site's form (one 8-CTA cluster per row, 896 columns per CTA, DSMEM cluster sum of squares) is based on earlier
NVIDIA work. The mailbox protocol
(multicast publish, Lamport poll with a fail-stop deadline, sentinel re-arm and its fence) is the megakernel plugin's
(sgl_nvidia_mega kimi_k3 cutedsl/common/mailbox.py).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import fake, stream, to_cute
from sgl_nvidia_mega.kimi_k3.cutedsl.common.mailbox import K23_ARM, poll, rearm, rearm_fence
from sgl_nvidia_mega.kimi_k3.cutedsl.common.twoshot import push as twoshot_push
from sgl_nvidia_mega.kimi_k3.cutedsl.common.twoshot import reduce as twoshot_reduce
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import (
    alloc,
    asm,
    at,
    bf16,
    bf16x2,
    bits_float,
    fence_proxy_async_global,
    fma,
    launch,
    load_words,
    multicast_words,
    pack_pair,
    push_float,
    store_words,
    wait,
    wait_deadline_check,
    wait_deadline_start,
    warp_sum,
)

logger = logging.getLogger(__name__)

HIDDEN = 7168
CAP = 64  # rows per mailbox slot: 8 draft blocks of 8
CLUSTER = 8
COLUMNS = HIDDEN // CLUSTER  # 896 per CTA = 112 threads x 8 columns
THREADS = 128  # per CTA (launch); SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM stores one sum of squares per thread
TAPS = 2
GROUP = 16
GROUPS = HIDDEN // GROUP  # 448
SENTINEL = -0x80000000  # int32 view of an unpublished BF16x2 word (+0.0, -0.0)
GATE_ROWS = 8  # SGLANG_OPT_DFLASH_DRAFT_AR_GATE: polled sites of at most one draft block
# SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_CAP: the two-shot box's rows (wide decode tier: 9-16 requests = 72-128 rows). The site's
# grid is one 8-CTA cluster of 128 threads per row; its cross-rank polls need every CTA resident, so the cap stays at 128
# until a residency check covers more (the poll's fail-stop deadline bounds a starved wait).
TWOSHOT_MAX_CAP = 128
# SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BOUND_ROWS: CTAs per SM the launch bound asks for (<= 85 registers at 128 threads; measured 80).
BOUND_BLOCKS = 6


# SGLANG_OPT_DFLASH_DRAFT_AR_VALID (SM107): validity-reporting bulk copies (PTX ISA 9.4; CCCL cuda::ptx).
# ``cp.async.bulk ... .mbarrier::report::validity::per_16bytes::80000000`` copies global memory into shared memory and the
# layout::v1 mbarrier it completes also reports whether the FIRST 32-bit word of any copied 16 B chunk was 0x80000000.
# The writer contract that makes that check cover a whole chunk: every 16 B chunk of a slot is written by ONE aligned
# 16 B store (the ring epilogue's and the site's v4 multimem.st), and no published word is 0x80000000 (pack_pair
# policy "latent": no -0.0 half). The slots are armed with 0x80000000 in every word (DraftArMailbox, rearm), so the
# LSU poll and the validity copy read the same mailbox.
@asm()
def init_valid(barrier):
    """layout::v1 mbarrier with one arrival (the caller fences the init)."""
    return "mbarrier.init.layout::v1.shared::cta.b64 [$0], 1;", "r,~{memory}", (barrier,)


@asm()
def expect_valid(barrier, nbytes):
    return "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}", (barrier, Int32(nbytes))


@asm()
def copy_valid(dst, src, nbytes, barrier):
    """Bulk copy of ``nbytes`` (a multiple of 16) from global ``src`` into this CTA's shared ``dst``, reporting armed
    16 B chunks on the layout::v1 ``barrier``."""
    code = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    code += ".mbarrier::report::validity::per_16bytes::80000000 [$0], [$1], $2, [$3];"
    return code, "r,l,r,r,~{memory}", (dst, src, Int32(nbytes), barrier)


@asm((Int32, Int32))
def try_wait_valid(barrier, phase):
    """(phase complete, some copied chunk was armed) of a layout::v1 barrier."""
    code = "{ .reg .pred p, q; mbarrier.try_wait.parity.phase_type::primary.shared::cta"
    code += ".b64 p|q, [$2], $3; selp.u32 $0, 1, 0, p; selp.u32 $1, 1, 0, q; }"
    return code, "=r,=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait_valid(barrier, phase):
    """Wait for ``phase`` of a layout::v1 barrier; nonzero if a copied chunk was armed."""
    done, armed = try_wait_valid(barrier, phase)
    while done == 0:
        done, armed = try_wait_valid(barrier, phase)
    return armed


class DraftArSite:
    """One site for ``rows`` rows (compile-time) on ``tp`` ranks; ``block_size`` is the draft block (conv position =
    row % block_size); ``deadline_ns`` the poll's fail-stop deadline (0 = none). ``early``: trigger the PDL dependent
    right after this kernel's griddepcontrol.wait instead of after the norm stores; only for a dependent that reads
    nothing but static operands before its own wait (the split-K conv projection), see DraftArSites.run. ``polled``
    (SGLANG_OPT_DFLASH_DRAFT_AR_RING): the projection before the site already multicast this rank's contribution
    (dflash_draft_ar_ring); the site reads no partial and publishes nothing, it only polls. ``gate``
    (SGLANG_OPT_DFLASH_DRAFT_AR_GATE): before the CTA polls, lane s of warp 0 waits for one fragment of source s (this
    CTA's last 8 columns), the GATED_AR form; every
    fragment is still read and re-armed by its owning thread. ``deferred``
    (SGLANG_OPT_DFLASH_DRAFT_AR_DEFERRED, polled sites only): the SITE_DEFERRED form.
    The site runs before its griddepcontrol.wait and waits at the end. Its ring producer releases it only after the
    producer's own wait (fenced trigger), so every kernel before the producer has completed when the site starts. Before
    its wait the site reads only the slots (Lamport-polled, written by the producers), the residual (written by the site
    before, older than the producer), the norm weight (static), and it writes the residual (no kernel since the site before
    reads it) and a static output buffer that only the readers of this site's previous call read (all before the producer).
    It completes only after its wait, so its dependents' waits also cover the producer. ``valid``
    (SGLANG_OPT_DFLASH_DRAFT_AR_VALID, polled sites only, Rubin): the VALID_AR form:
    thread 0 lands the CTA's columns of every TP slot in shared memory with validity-reporting bulk copies, re-issued
    until no copied chunk was armed (warp 0 waits; the poll's fail-stop deadline also bounds the rounds); every thread
    then reads its 16 B per slot from shared memory, and its owner still re-arms the slot fragment. Replaces ``gate``.
    ``smemsum`` (SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM): the norm's sum of squares goes
    through shared memory instead of the warp shuffles, see finish()."""

    def __init__(
        self, rows: int, tp: int, block_size: int, deadline_ns: int, early: bool = False, polled: bool = False,
        gate: bool = False, deferred: bool = False, valid: bool = False, smemsum: bool = False, cap: int = CAP,
    ):  # fmt: skip
        assert 0 < rows <= cap
        assert polled or not deferred, "only a polled site can run before its wait"
        assert polled or not valid, "only a polled site lands its slots with validity copies"
        self.rows, self.tp, self.block_size, self.deadline_ns = rows, tp, block_size, deadline_ns
        self.early, self.polled = early, polled
        self.valid = valid and tp > 1
        self.gate = gate and tp > 1 and not self.valid
        self.deferred = deferred
        self.fence_offload = False  # two-shot only (DraftArSiteTwoShot)
        self.smemsum = smemsum

    @cute.jit
    def __call__(self, partial, proj, base, residual, gamma, norm_out, box, mc, parity, rank, eps, stream):
        args = (partial, proj, base, residual, gamma, norm_out, box, mc, parity, rank, eps, stream)
        launch(self.kernel, args, (CLUSTER, self.rows, 1), THREADS, cluster=CLUSTER)

    @cute.kernel
    def kernel(self, args):
        partial, proj, base, residual, gamma, norm_out, box, mc, parity, rank, eps = args
        part, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, THREADS if self.smemsum else 4)
        peers = alloc(smem, Float32, CLUSTER)
        bar = smem.allocate(Uint64, 8)
        stage, vbar = None, None
        if cutlass.const_expr(self.valid):
            stage = alloc(smem, Uint32, self.tp * COLUMNS // 2, 128)  # the CTA's columns of every TP slot
            vbar = smem.allocate(Uint64, 8)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            if cutlass.const_expr(self.valid):
                init_valid(vbar)
            cute.arch.mbarrier_init_fence()
            # Peers complete this barrier with st.async: arm it at cluster scope (PTX ISA, st.async / mbarrier).
            cute.arch.mbarrier_arrive_and_expect_tx(bar, CLUSTER * 4, **K23_ARM)
        if cutlass.const_expr(self.valid):
            cute.arch.sync_warp()  # warp 0 waits on vbar
        cute.arch.cluster_arrive_relaxed()
        live = tid < COLUMNS // 8
        col = part * COLUMNS + cutlass.min(tid, COLUMNS // 8 - 1) * 8
        slot = HIDDEN * CAP  # elements per source slot
        local = box.iterator + (parity * self.tp * CAP + row) * HIDDEN + col
        # Every input (partial, conv projection, residual) is written by earlier kernels: read after the wait. A deferred
        # site starts only after its producer's wait (see the class doc) and waits at the end instead.
        if cutlass.const_expr(not self.deferred):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.early):
            # Consumers still read this kernel's outputs only after their own wait (grid completion); the trigger
            # carries no data, so the poll's deadline trap below still publishes nothing.
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.polled):
            if live:
                words = self.contribution(partial, proj, base, row, col)
                offset = ((parity * self.tp + rank) * CAP + row) * HIDDEN + col
                multicast_words(mc + Int64(offset) * 2, words, kind="bf16x2")
        weights = load_words(gamma.iterator + col, count=4)
        res = load_words(residual.iterator + row * HIDDEN + col, count=4, mutable=True)
        if cutlass.const_expr(self.gate):
            # Fewer polls in flight while the peers' stores land: lane s waits for source s's last fragment of this CTA.
            # It only reads (the owner, thread COLUMNS // 8 - 1, polls and re-arms that fragment after the barrier).
            if tid < self.tp:
                last = box.iterator + (parity * self.tp * CAP + row) * HIDDEN + part * COLUMNS + (COLUMNS - 8)
                poll(last + tid * slot, slot, 1, self.deadline_ns)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.valid):
            if tid < 32:
                first = box.iterator + (parity * self.tp * CAP + row) * HIDDEN + part * COLUMNS
                self.land(first, stage, vbar, tid)
            cute.arch.sync_threads()
        x = [Float32(0)] * 8
        total = Float32(0)
        if live:
            if cutlass.const_expr(self.valid):
                got = self.staged(stage, tid)
            else:
                got = poll(local, slot, self.tp, self.deadline_ns)
            for s in cutlass.range_constexpr(self.tp):
                rearm(local + s * slot)
            # Order the re-arms before this thread's later stores and its exit (and before the end trigger when the
            # trigger is not early).
            rearm_fence()
            acc = [Float32(0)] * 8
            for s in cutlass.range_constexpr(self.tp):
                for k in cutlass.range_constexpr(4):
                    for h in (False, True):
                        e = 2 * k + int(h)
                        value = bf16(got[s, k], h)
                        acc[e] = value if s == 0 else acc[e] + value
            out = []
            for k in cutlass.range_constexpr(4):
                pair = bf16x2(acc[2 * k], acc[2 * k + 1])  # the reduced row, rounded once
                # The norm reads the FP32 sum; only the stored residual is rounded.
                x[2 * k] = bf16(pair, False) + bf16(res[k], False)
                x[2 * k + 1] = bf16(pair, True) + bf16(res[k], True)
                out.append(bf16x2(x[2 * k], x[2 * k + 1]))
            store_words(residual.iterator + row * HIDDEN + col, tuple(out))
            for e in cutlass.range_constexpr(8):
                total = fma(x[e], x[e], total)
        self.finish(x, total, live, col, row, part, weights, eps, norm_out, sums, peers, bar)
        if cutlass.const_expr(self.deferred):
            cute.arch.griddepcontrol_wait()  # completes only after the producer: the dependents' waits cover it

    @cute.jit
    def land(self, first, stage, vbar, tid):
        """Warp 0 (``valid``): rounds of thread 0's validity copies of the CTA's COLUMNS of every TP slot (``first``:
        slot 0's) into ``stage`` until no copied 16 B chunk was armed; lane 0's report decides each round for the warp.
        Every copy of a round is re-issued (a chunk can land between the copies)."""
        nbytes: cutlass.Constexpr = COLUMNS * 2
        slot = HIDDEN * CAP
        if tid == 0:
            # The slots were re-armed by generic stores (two sites ago); this thread's bulk copies read them through
            # the async proxy (generic-to-async proxy fence, PTX memory model).
            fence_proxy_async_global()
        phase, armed, rounds = Int32(0), Int32(1), Int32(0)
        started, polls = Int64(0), Int32(0)
        while armed != 0:
            if rounds == 1:
                started = wait_deadline_start(self.deadline_ns)
            if rounds > 0:
                polls = wait_deadline_check(started, polls, self.deadline_ns)
            if tid == 0:
                expect_valid(vbar, self.tp * nbytes)
                for s in cutlass.range_constexpr(self.tp):
                    source = s
                    copy_valid(stage.iterator + s * (COLUMNS // 2), first + source * slot, nbytes, vbar)
            armed = cute.arch.shuffle_sync(wait_valid(vbar, phase), 0)
            phase = phase ^ 1
            rounds = rounds + 1

    @cute.jit
    def staged(self, stage, tid):
        """This thread's 16 B of every TP slot from the landed stage, as ``poll``'s [tp, 4] words."""
        got = cute.make_rmem_tensor((self.tp, 4), Uint32)
        for s in cutlass.range_constexpr(self.tp):
            words = load_words(stage.iterator + s * (COLUMNS // 2) + 4 * tid, count=4, shared=True)
            for k in cutlass.range_constexpr(4):
                got[s, k] = words[k]
        return got

    @cute.jit
    def contribution(self, partial, proj, base, row, col):
        """This rank's 16 bytes for the slots: the partial's 8 columns convolved (side 1), with no -0.0 half (the
        slots' sentinel)."""
        cur = load_words(partial.iterator + row * HIDDEN + col, count=4, mutable=True)
        y = [Float32(0)] * 8
        for k in cutlass.range_constexpr(4):
            for h in (False, True):
                y[2 * k + int(h)] = bf16(cur[k], h)
        # coefficient(tap) = base[tap, c] + delta[row, tap, c // 16]; delta = proj[row, (TAPS + tap) * GROUPS + g], the
        # side-1 half of the [rows, 2, TAPS, GROUPS] projection, as dflash_grouped_conv reads it.
        group = col // GROUP
        position = row % self.block_size
        coef = []
        for tap in cutlass.range_constexpr(TAPS):
            index = row * (2 * TAPS * GROUPS) + (TAPS + tap) * GROUPS + group
            word = load_words(proj.iterator + (index // 2) * 2, count=1, mutable=True)[0]
            delta = bits_float((word >> (Uint32(index % 2) * Uint32(16))) << Uint32(16))
            b = load_words(base.iterator + tap * HIDDEN + col, count=4)
            coef.append([bf16(b[k], h) + delta for k in range(4) for h in (False, True)])
        prev = [Float32(0)] * 8
        if position > 0:
            before = load_words(partial.iterator + (row - 1) * HIDDEN + col, count=4, mutable=True)
            for k in cutlass.range_constexpr(4):
                for h in (False, True):
                    prev[2 * k + int(h)] = bf16(before[k], h)
        for e in cutlass.range_constexpr(8):
            y[e] = fma(coef[1][e], prev[e], coef[0][e] * y[e])
        return tuple(pack_pair(y[2 * k], y[2 * k + 1], policy="latent") for k in range(4))

    @cute.jit
    def finish(self, x, total, live, col, row, part, weights, eps, norm_out, sums, peers, bar):
        """The row's sum of squares (warp, CTA, then DSMEM across the cluster in CTA order), then this thread's 8
        normalized columns (x * rstd * w, rounded once)."""
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        if cutlass.const_expr(self.smemsum):
            # SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM: every thread stores its sum of squares (the non-live threads their 0.0, which
            # warp_sum also adds); the pushers below rebuild the warp sums after the CTA barrier (smem_total).
            sums[tid] = total
        else:
            total = warp_sum(total)
            if lane == 0:
                sums[warp] = total
        cute.arch.sync_threads()
        if cutlass.const_expr(self.fence_offload):
            # SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_FENCE_OFFLOAD: the live threads re-armed their gather words before this barrier
            # without a fence; the first non-live thread issues the same rearm_fence here, after the barrier (CTA-scope
            # synchronization with every re-arming thread, then the fence's cumulativity), so the fence overlaps the sum of
            # squares and the norm instead of preceding the residual add. Every later use of the gather row still follows it:
            # the next write into this rank's gather row of the parity is an owner's multicast two sites later, after this
            # rank's push of the site in between, which starts only after this grid completed (this thread included).
            if tid == COLUMNS // 8:
                rearm_fence()
        cute.arch.cluster_wait()  # every CTA's barrier is initialized and armed
        if tid < CLUSTER:
            if cutlass.const_expr(self.smemsum):
                mine = self.smem_total(sums)
            else:
                mine = (sums[0] + sums[1]) + (sums[2] + sums[3])
            push_float(peers.iterator + part, mine, bar, tid)
        wait(bar)
        variance = Float32(0)
        for c in cutlass.range_constexpr(CLUSTER):
            variance += peers[c]
        scale = cute.math.rsqrt(fma(variance, Float32(1.0 / HIDDEN), eps), fastmath=False)
        if live:
            normed = []
            for k in cutlass.range_constexpr(4):
                pair = []
                for h in (False, True):
                    pair.append(x[2 * k + int(h)] * scale * bf16(weights[k], h))
                normed.append(bf16x2(pair[0], pair[1]))
            store_words(norm_out.iterator + row * HIDDEN + col, tuple(normed))
        if cutlass.const_expr(not self.early):
            cute.arch.griddepcontrol_launch_dependents()
        # No CTA exits before every DSMEM push into the cluster has landed.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.jit
    def smem_total(self, sums):
        """SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM: the CTA's sum of squares from the THREADS per-thread values in ``sums``, with the
        same adds in the same order as warp_sum + (s0 + s1) + (s2 + s3): warp_sum's xor butterfly leaves lane 0 with
        v[j] + v[j + s] for j < s at s = 16, 8, 4, 2, 1 (IEEE add is commutative, so own + partner order does not matter).
        Same bits as the shuffle form; no warp shuffle, no reconvergence after the per-lane gather poll."""
        warps = []
        for w in cutlass.range_constexpr(THREADS // 32):
            v = [sums[32 * w + j] for j in range(32)]
            for shift in (16, 8, 4, 2, 1):
                v = [v[j] + v[j + shift] for j in range(shift)]
            warps.append(v[0])
        return (warps[0] + warps[1]) + (warps[2] + warps[3])


class DraftArSiteTwoShot(DraftArSite):
    """SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS: the unpolled site over the megakernel plugin's two-shot Lamport exchange
    (sgl_nvidia_mega kimi_k3 cutedsl/common/twoshot.py, the target o-projection's), the ArNormTwoShot form, with
    this site's conv side 1 kept.

    Every thread computes its rank's convolved 8 columns exactly as the one-shot site does (``contribution``) and stores
    them into the owner rank's receive slot ``rank`` (owner = column // (HIDDEN / TP)). Each CTA then owns ``share``
    8-column chunks of this rank's HIDDEN / TP columns of its row: its first threads poll the TP slots of a chunk, sum
    them in FP32 in rank order, round once (no -0.0) and multicast the BF16 sum into every rank's gather row, and re-arm
    the slots. After a CTA barrier every thread polls its 8 columns of the gather row, re-arms them, adds the residual in
    FP32 and normalizes, as the one-shot site. The one-shot site sums the same TP BF16 words in the same order and rounds
    once at the same point, so both forms give the same bits; each GPU receives ~2 x rows x HIDDEN x 2 bytes instead of
    TP x rows x HIDDEN x 2.

    Re-arm and parity (as the one-shot site and the plugin's o-projection): consecutive sites alternate the box's two
    parities. An owner re-arms its slots of a site before it can run its next site; a peer can push into a parity again
    only two sites later, after it has polled the owner's gather chunk of the site in between, which the owner multicasts
    only after its own push of that site, i.e. after it re-armed. A rank's gather row of a parity is re-armed by that rank
    right after its poll; an owner multicasts into it again only for the site two later, after that rank's push of it.
    ``batched`` (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED): the owner polls its TP slots with the plugin reduce's batched
    form (the TWOSHOT_BATCHED form: every slot re-loaded each pass, all loads in flight) instead of
    re-loading only the pending slots one dependent round trip after another; the same words are summed in the same order,
    so the same bits. ``min_blocks`` (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BOUND_ROWS): a launch bound of that many CTAs per SM, so the
    compiler keeps registers low enough that more rows stay resident at one row per cluster; same arithmetic, same bits."""

    def __init__(
        self, rows: int, tp: int, block_size: int, deadline_ns: int, early: bool = False,
        batched: bool = False, fence_offload: bool = False, drr_defer: bool = False, drr_catchup: int = 0,
        smemsum: bool = False, cap: int = CAP, min_blocks: int = 0,
    ):  # fmt: skip
        assert CAP <= cap <= TWOSHOT_MAX_CAP
        super().__init__(rows, tp, block_size, deadline_ns, early, smemsum=smemsum, cap=cap)
        assert COLUMNS // 8 < 128, "fence offload needs a non-live thread"
        # SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR: ``drr_defer`` leaves this site's gather row published (no re-arm, no
        # fence; the next site of the forward re-arms it); ``drr_catchup`` (+1 / -1 = the other parity's gather row, in
        # parities, relative to this one; 0 = none): this site's non-live threads re-arm the previous site's gather row
        # (same rows and columns) and fence, while the live threads compute and push.
        assert drr_catchup in (-1, 0, 1)
        assert (COLUMNS // 8) % (128 - COLUMNS // 8) == 0, "DRR: the non-live threads split the row's words evenly"
        self.drr_defer = drr_defer
        self.drr_catchup = drr_catchup
        self.fence_offload = fence_offload and not drr_defer
        assert tp > 1 and HIDDEN % (8 * tp) == 0
        self.owned = HIDDEN // tp
        self.owned_chunks = self.owned // 8
        self.share = -(-self.owned_chunks // CLUSTER)  # owned chunks per CTA
        assert self.share <= COLUMNS // 8
        self.batched = batched
        self.min_blocks = min_blocks

    @cute.jit
    def __call__(self, partial, proj, base, residual, gamma, norm_out, recv, peers, gather, gather_mc, rank, eps, stream):
        args = (partial, proj, base, residual, gamma, norm_out, recv, peers, gather, gather_mc, rank, eps, stream)
        if cutlass.const_expr(self.min_blocks > 0):
            launch(self.kernel, args, (CLUSTER, self.rows, 1), THREADS, cluster=CLUSTER, min_blocks_per_mp=self.min_blocks)
        else:
            launch(self.kernel, args, (CLUSTER, self.rows, 1), THREADS, cluster=CLUSTER)

    @cute.kernel
    def kernel(self, args):
        partial, proj, base, residual, gamma, norm_out, recv, peers, gather, gather_mc, rank, eps = args
        part, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, THREADS if self.smemsum else 4)
        cluster_sums = alloc(smem, Float32, CLUSTER)
        bar = smem.allocate(Uint64, 8)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init_fence()
            # Peers complete this barrier with st.async: arm it at cluster scope (PTX ISA, st.async / mbarrier).
            cute.arch.mbarrier_arrive_and_expect_tx(bar, CLUSTER * 4, **K23_ARM)
        cute.arch.cluster_arrive_relaxed()
        live = tid < COLUMNS // 8
        col = part * COLUMNS + cutlass.min(tid, COLUMNS // 8 - 1) * 8
        # Every input (partial, conv projection, residual) is written by earlier kernels: read after the wait.
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.drr_catchup != 0):
            # SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR catch-up: the previous site (the other parity, same rows, same columns)
            # left this CTA's gather words published; its non-live threads re-arm them (7 words each) and fence, while the
            # live threads compute and push. Every later write into that row is an owner's multicast of the site after
            # this one, which needs this rank's push of that site, which starts only after this grid completed.
            if tid >= COLUMNS // 8:  # the non-live threads
                first = (tid - COLUMNS // 8) * ((COLUMNS // 8) // (128 - COLUMNS // 8))
                for j in cutlass.range_constexpr((COLUMNS // 8) // (128 - COLUMNS // 8)):
                    rearm(at(gather, 0, row, part * COLUMNS + (first + j) * 8) + self.drr_catchup * CAP * HIDDEN)
                rearm_fence()
        if live:
            words = self.contribution(partial, proj, base, row, col)
            twoshot_push(recv, peers, rank, row, col, words)
        if tid < self.share:
            chunk = part * self.share + tid
            if chunk < self.owned_chunks:
                twoshot_reduce(
                    recv, gather, gather_mc, rank, row, rank * self.owned + chunk * 8, self.deadline_ns, batched=self.batched
                )
        cute.arch.sync_threads()  # the gather row's pollers after this CTA's owners
        # The norm weights and the residual load while the gather row is in flight.
        weights = load_words(gamma.iterator + col, count=4)
        res = load_words(residual.iterator + row * HIDDEN + col, count=4, mutable=True)
        x = [Float32(0)] * 8
        total = Float32(0)
        if live:
            source = at(gather, 0, row, col)
            got = poll(source, 0, 1, self.deadline_ns)
            if cutlass.const_expr(not self.drr_defer):
                rearm(source)
            if cutlass.const_expr(not self.fence_offload and not self.drr_defer):
                rearm_fence()  # else one non-live thread fences in finish() after the CTA barrier (or DRR: the next site)
            out = []
            for k in cutlass.range_constexpr(4):
                # The reduced row (rounded once by its owner); the norm reads the FP32 sum with the residual.
                x[2 * k] = bf16(got[0, k], False) + bf16(res[k], False)
                x[2 * k + 1] = bf16(got[0, k], True) + bf16(res[k], True)
                out.append(bf16x2(x[2 * k], x[2 * k + 1]))
            store_words(residual.iterator + row * HIDDEN + col, tuple(out))
            for e in cutlass.range_constexpr(8):
                total = fma(x[e], x[e], total)
        self.finish(x, total, live, col, row, part, weights, eps, norm_out, sums, cluster_sums, bar)


class DraftArSiteTwoShotRows(DraftArSiteTwoShot):
    """SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_MAXCL: the two-shot site for more than ``max_clusters`` rows, with R = ceil(rows /
    max_clusters) rows per CTA so the grid stays at <= max_clusters 8-CTA clusters (clusters beyond what stays co-resident
    run as a second wave).

    Row k of CTA-row y is y + k * grid_rows. Per row everything is the parent's: the same contribution (conv side 1, rounded,
    no -0.0), the same push into the owner's receive slot, the same owner reduce (threads [k * share, (k + 1) * share) take
    row k's owned chunks, the parent's chunk mapping), the same gather poll + re-arm, residual add and norm (warp sums, CTA
    sums, DSMEM cluster sums in CTA order, rsqrt). So the bits equal the parent's and the chunked one-shot site's; only which
    CTA runs which row changes, and every row's slot words still have one pusher, one owner thread and one poller per rank.
    All of a CTA's rows push before any of its owners poll. The cluster-sum barrier expects CLUSTER x 4 B per row; an invalid
    tail row (beyond ``rows``; the same for the whole cluster, which spans columns) pushes a zero sum, so the count is static."""

    def __init__(
        self, rows: int, tp: int, block_size: int, deadline_ns: int, early: bool = False,
        cap: int = CAP, max_clusters: int = 64, batched: bool = False,
    ):  # fmt: skip
        super().__init__(rows, tp, block_size, deadline_ns, early, batched, cap=cap)
        assert 0 < max_clusters < rows
        self.rpc = -(-rows // max_clusters)  # rows per CTA
        self.grid_rows = -(-rows // self.rpc)
        assert self.rpc * self.share <= 128

    @cute.jit
    def __call__(self, partial, proj, base, residual, gamma, norm_out, recv, peers, gather, gather_mc, rank, eps, stream):
        args = (partial, proj, base, residual, gamma, norm_out, recv, peers, gather, gather_mc, rank, eps, stream)
        launch(self.kernel, args, (CLUSTER, self.grid_rows, 1), 128, cluster=CLUSTER)

    @cute.kernel
    def kernel(self, args):
        partial, proj, base, residual, gamma, norm_out, recv, peers, gather, gather_mc, rank, eps = args
        part, crow, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, 4 * self.rpc)
        cluster_sums = alloc(smem, Float32, CLUSTER * self.rpc)
        bar = smem.allocate(Uint64, 8)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init_fence()
            # Peers complete this barrier with st.async: arm at cluster scope (KNOWN K-23); CLUSTER x 4 B per row.
            cute.arch.mbarrier_arrive_and_expect_tx(bar, CLUSTER * 4 * self.rpc, **K23_ARM)
        cute.arch.cluster_arrive_relaxed()
        live = tid < COLUMNS // 8
        col = part * COLUMNS + cutlass.min(tid, COLUMNS // 8 - 1) * 8
        # Every input (partial, conv projection, residual) is written by earlier kernels: read after the wait.
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        # Every row's contribution first (all of this CTA's pushes in flight before any of its owners polls).
        for k in cutlass.range_constexpr(self.rpc):
            row = crow + k * self.grid_rows
            if row < self.rows:
                if live:
                    words = self.contribution(partial, proj, base, row, col)
                    twoshot_push(recv, peers, rank, row, col, words)
        # Owners: threads [k * share, (k + 1) * share) reduce row k's owned chunks (the parent's mapping per row).
        for k in cutlass.range_constexpr(self.rpc):
            row = crow + k * self.grid_rows
            if row < self.rows:
                if tid >= k * self.share:
                    if tid < (k + 1) * self.share:
                        chunk = part * self.share + (tid - k * self.share)
                        if chunk < self.owned_chunks:
                            twoshot_reduce(
                                recv, gather, gather_mc, rank, row, rank * self.owned + chunk * 8, self.deadline_ns,
                                batched=self.batched,
                            )  # fmt: skip
        cute.arch.sync_threads()  # the gather rows' pollers after this CTA's owners
        weights = load_words(gamma.iterator + col, count=4)
        xs = []
        totals = []
        for k in cutlass.range_constexpr(self.rpc):
            row = crow + k * self.grid_rows
            x = [Float32(0)] * 8
            total = Float32(0)
            if row < self.rows:
                if live:
                    res = load_words(residual.iterator + row * HIDDEN + col, count=4, mutable=True)
                    source = at(gather, 0, row, col)
                    got = poll(source, 0, 1, self.deadline_ns)
                    rearm(source)
                    rearm_fence()
                    out = []
                    for j in cutlass.range_constexpr(4):
                        # The reduced row (rounded once by its owner); the norm reads the FP32 sum with the residual.
                        x[2 * j] = bf16(got[0, j], False) + bf16(res[j], False)
                        x[2 * j + 1] = bf16(got[0, j], True) + bf16(res[j], True)
                        out.append(bf16x2(x[2 * j], x[2 * j + 1]))
                    store_words(residual.iterator + row * HIDDEN + col, tuple(out))
                    for e in cutlass.range_constexpr(8):
                        total = fma(x[e], x[e], total)
            xs.append(x)
            totals.append(total)
        # The parent's finish, per row: warp sums -> CTA sums -> DSMEM cluster sums in CTA order -> rsqrt.
        for k in cutlass.range_constexpr(self.rpc):
            t = warp_sum(totals[k])
            if lane == 0:
                sums[k * 4 + warp] = t
        cute.arch.sync_threads()
        cute.arch.cluster_wait()  # every CTA's barrier is initialized and armed
        if tid < CLUSTER:
            for k in cutlass.range_constexpr(self.rpc):
                mine = (sums[k * 4] + sums[k * 4 + 1]) + (sums[k * 4 + 2] + sums[k * 4 + 3])
                push_float(cluster_sums.iterator + k * CLUSTER + part, mine, bar, tid)
        wait(bar)
        for k in cutlass.range_constexpr(self.rpc):
            row = crow + k * self.grid_rows
            variance = Float32(0)
            for c in cutlass.range_constexpr(CLUSTER):
                variance += cluster_sums[k * CLUSTER + c]
            scale = cute.math.rsqrt(fma(variance, Float32(1.0 / HIDDEN), eps), fastmath=False)
            if row < self.rows:
                if live:
                    normed = []
                    for j in cutlass.range_constexpr(4):
                        pair = []
                        for h in (False, True):
                            pair.append(xs[k][2 * j + int(h)] * scale * bf16(weights[j], h))
                        normed.append(bf16x2(pair[0], pair[1]))
                    store_words(norm_out.iterator + row * HIDDEN + col, tuple(normed))
        if cutlass.const_expr(not self.early):
            cute.arch.griddepcontrol_launch_dependents()
        # No CTA exits before every DSMEM push into the cluster has landed.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()


class DraftArMailbox:
    """Sentinel-armed multicast mailbox [2 parities, TP, CAP, HIDDEN] BF16 on the TP group (collective)."""

    def __init__(self, group, device, tp: int, rank: int):
        self.tp, self.rank = tp, rank
        self.buffer = symm_mem.empty((2, tp, CAP, HIDDEN), dtype=torch.bfloat16, device=device)
        self.handle = symm_mem.rendezvous(self.buffer, getattr(group, "group_name", group))
        self.mc = int(self.handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("SGLANG_OPT_DFLASH_DRAFT_AR_SITE needs multicast memory on the draft TP group")
        self.buffer.view(torch.int32).fill_(SENTINEL)
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)


@cache
def compile_site(
    rows: int, tp: int, block_size: int, deadline_ns: int, early: bool = False, polled: bool = False,
    gate: bool = False, deferred: bool = False, valid: bool = False, smemsum: bool = False,
):  # fmt: skip
    site = DraftArSite(rows, tp, block_size, deadline_ns, early, polled, gate, deferred, valid, smemsum)
    args = (
        fake((rows, HIDDEN)),  # partial
        fake((rows, 2 * TAPS * GROUPS)),  # conv projection
        fake((TAPS, HIDDEN)),  # side-1 base kernel
        fake((rows, HIDDEN)),  # residual
        fake((HIDDEN,)),  # norm weight
        fake((rows, HIDDEN)),  # norm output
        fake((2, tp, CAP, HIDDEN)),  # mailbox
        Int64(0),
        Int32(0),
        Int32(0),
        Float32(1e-6),
        make_fake_stream(),
    )
    return cute.compile(site, *args)


@cache
def compile_site_twoshot(
    rows: int, tp: int, block_size: int, deadline_ns: int, early: bool = False, batched: bool = False,
    fence_offload: bool = False, drr_defer: bool = False, drr_catchup: int = 0, smemsum: bool = False,
    cap: int = CAP, min_blocks: int = 0,
):  # fmt: skip
    site = DraftArSiteTwoShot(
        rows, tp, block_size, deadline_ns, early, batched, fence_offload, drr_defer, drr_catchup, smemsum,
        cap=cap, min_blocks=min_blocks,
    )
    args = (
        fake((rows, HIDDEN)),  # partial
        fake((rows, 2 * TAPS * GROUPS)),  # conv projection
        fake((TAPS, HIDDEN)),  # side-1 base kernel
        fake((rows, HIDDEN)),  # residual
        fake((HIDDEN,)),  # norm weight
        fake((rows, HIDDEN)),  # norm output
        fake((tp, cap, HIDDEN // tp)),  # this rank's receive slots of the parity
        fake((tp,), None, torch.int64),  # the ranks' receive slot bases of the parity
        fake((1, cap, HIDDEN)),  # this rank's gather row of the parity
        Int64(0),
        Int32(0),
        Float32(1e-6),
        make_fake_stream(),
    )
    return cute.compile(site, *args)


@cache
def compile_site_twoshot_rows(
    rows: int, tp: int, block_size: int, deadline_ns: int, early: bool, cap: int, max_clusters: int,
    batched: bool = False,
):  # fmt: skip
    site = DraftArSiteTwoShotRows(rows, tp, block_size, deadline_ns, early, cap, max_clusters, batched)
    args = (
        fake((rows, HIDDEN)),  # partial
        fake((rows, 2 * TAPS * GROUPS)),  # conv projection
        fake((TAPS, HIDDEN)),  # side-1 base kernel
        fake((rows, HIDDEN)),  # residual
        fake((HIDDEN,)),  # norm weight
        fake((rows, HIDDEN)),  # norm output
        fake((tp, cap, HIDDEN // tp)),  # this rank's receive slots of the parity
        fake((tp,), None, torch.int64),  # the ranks' receive slot bases of the parity
        fake((1, cap, HIDDEN)),  # this rank's gather row of the parity
        Int64(0),
        Int32(0),
        Float32(1e-6),
        make_fake_stream(),
    )
    return cute.compile(site, *args)


class DraftArSites:
    """The draft model's all-reduce sites: one mailbox on the draft TP group and one compiled site per row count."""

    def __init__(
        self, group, device, deadline_ns: int, early_trigger: bool = False, gate: Optional[bool] = None,
        deferred: Optional[bool] = None, twoshot_rows: Optional[int] = None, twoshot_cap: Optional[int] = None,
        twoshot_maxcl: Optional[int] = None, twoshot_batched: Optional[bool] = None, twoshot_bound_rows: Optional[int] = None,
    ):  # fmt: skip
        from sglang.srt.environ import envs

        self.tp, self.rank = group.world_size, group.rank_in_group
        self.deadline_ns = deadline_ns
        self.early_trigger = early_trigger
        # Polled sites of at most GATE_ROWS rows gate their poll (SGLANG_OPT_DFLASH_DRAFT_AR_GATE).
        self.gate = envs.SGLANG_OPT_DFLASH_DRAFT_AR_GATE.get() if gate is None else gate
        # Polled sites of at most GATE_ROWS rows after a trigger-after-wait ring producer run deferred
        # (SGLANG_OPT_DFLASH_DRAFT_AR_DEFERRED); their outputs go to one static buffer per (norm, rows).
        self.deferred = envs.SGLANG_OPT_DFLASH_DRAFT_AR_DEFERRED.get() if deferred is None else deferred
        self._deferred_out = {}
        self.deferred_runs = 0  # tests: deferred launches so far
        # Polled sites of at most GATE_ROWS rows land their slots with validity copies (SGLANG_OPT_DFLASH_DRAFT_AR_VALID,
        # Rubin); they are not gated.
        self.valid = envs.SGLANG_OPT_DFLASH_DRAFT_AR_VALID.get()
        self.valid_runs = 0  # tests: validity-landed launches so far
        self.mailbox = DraftArMailbox(group.device_group, device, self.tp, self.rank)
        # Unpolled sites of at least this many rows exchange over the two-shot box (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS;
        # 0 = never). Same bits as the one-shot site.
        self.twoshot_rows = envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS.get() if twoshot_rows is None else twoshot_rows
        self.twoshot_runs = 0  # tests: two-shot launches so far
        # Rows of the two-shot box (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_CAP): sites of more than CAP rows (the one-shot
        # mailbox's) exist only on the two-shot path.
        self.twoshot_cap = envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_CAP.get() if twoshot_cap is None else twoshot_cap
        # Two-shot sites of more than this many rows run several rows per CTA (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_MAXCL; 0 = one
        # row per cluster at every row count). Same bits.
        self.twoshot_maxcl = envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_MAXCL.get() if twoshot_maxcl is None else twoshot_maxcl
        if self.twoshot_maxcl < 0:
            raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_MAXCL={self.twoshot_maxcl}: needs >= 0")
        self.twoshot_rows_runs = 0  # tests: several-rows-per-CTA two-shot launches so far
        # Two-shot owners poll their slots in the batched form (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED). Same bits.
        self.twoshot_batched = bool(
            envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED.get() if twoshot_batched is None else twoshot_batched
        )
        self.twoshot_batched_runs = 0  # tests: batched two-shot launches so far
        # Two-shot sites move their gather-side re-arm fence to one non-live thread after the norm's first CTA barrier
        # (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_FENCE_OFFLOAD). Same bits.
        self.twoshot_fence_offload = bool(envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_FENCE_OFFLOAD.get())
        self.twoshot_foff_runs = 0  # tests: fence-offload two-shot launches so far
        # Two-shot sites of one draft forward defer their gather re-arm to the next site (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR).
        # Sites are counted inside engage() only; the forward's last site re-arms its own row, and a row still pending
        # when engage() exits (a forward that stopped early) is re-armed there by a stream-ordered fill. Same bits.
        self.twoshot_drr = bool(envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR.get())
        self.twoshot_drr_runs = 0  # tests: two-shot launches with a deferred re-arm or a catch-up so far
        self.twoshot_drr_flushes = 0  # tests: pending rows re-armed by a fill
        # Every site sums the norm's squares through shared memory instead of warp shuffles (SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM).
        # Same bits.
        self.smemsum = bool(envs.SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM.get())
        self.smemsum_runs = 0  # tests: shared-memory-sum launches so far
        self._ts_index = None  # two-shot sites run so far in this forward (None outside engage())
        self._ts_sites = 0
        self._ts_pending = None  # (parity, rows) whose gather row a site left published
        # One-row-per-cluster two-shot sites of more than this many rows compile with a launch bound of BOUND_BLOCKS CTAs per SM
        # (SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BOUND_ROWS; 0 = never). Same bits.
        self.twoshot_bound_rows = (
            envs.SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BOUND_ROWS.get() if twoshot_bound_rows is None else twoshot_bound_rows
        )
        if self.twoshot_bound_rows < 0:
            raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BOUND_ROWS={self.twoshot_bound_rows}: needs >= 0")
        self.twoshot_bound_runs = 0  # tests: bounded two-shot launches so far
        self._twoshot_logged = set()
        # The deferred re-arm addresses the other parity's gather row at the one-shot box pitch (CAP rows) and runs only in the
        # one-row-per-cluster two-shot kernel: not together with a two-shot cap above CAP or several rows per CTA.
        if self.twoshot_drr and (self.twoshot_cap != CAP or self.twoshot_maxcl > 0):
            raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR needs SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_CAP=64 and _MAXCL=0")
        self.twoshot = None
        if self.twoshot_rows > 0 and self.tp > 1:
            if HIDDEN % (8 * self.tp):
                raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS needs {HIDDEN} columns in 8-column chunks per rank")
            if not (CAP <= self.twoshot_cap <= TWOSHOT_MAX_CAP and self.twoshot_cap % 8 == 0):
                raise ValueError(
                    f"SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_CAP={self.twoshot_cap}: needs a multiple of 8 in [{CAP}, {TWOSHOT_MAX_CAP}]"
                )
            if self.twoshot_cap > CAP and self.twoshot_rows > CAP:
                raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS must be <= 64 when the two-shot cap is above 64")
            from sgl_nvidia_mega.kimi_k3.cutedsl.common.twoshot import TwoShot

            self.twoshot = TwoShot(group.device_group, device, self.tp, self.rank, HIDDEN, self.twoshot_cap, 2)
        # SGLANG_OPT_DFLASH_DRAFT_AR_RING (install_draft_ar_ring): the sublayer's o_proj / down_proj publishes its
        # convolved partial inside ring_scope; the site after it then only polls.
        self.rings = False
        self.ring_pending = None  # (parity, conv projection, conv) inside ring_scope
        self._published = None  # (parity, rows) once the ring projection published
        self._published_after_wait = False  # that ring projection releases its dependents only after its wait

    def ring_address(self, parity: int) -> int:
        """The multicast address of this rank's slot of ``parity`` ([CAP, HIDDEN] BF16)."""
        return self.mailbox.mc + ((parity * self.tp + self.rank) * CAP) * HIDDEN * 2

    @contextmanager
    def ring_scope(self, parity: int, projection, conv):
        """Around the sublayer whose row-parallel projection feeds the site of ``parity``: a ring projection inside
        publishes the site's contribution (conv side 1 from ``projection`` / ``conv``) itself."""
        if not self.rings:
            yield
            return
        self.ring_pending, self._published = (parity, projection.contiguous(), conv), None
        try:
            yield
        finally:
            self.ring_pending = None

    def ring_published(self, parity: int, rows: int, after_wait: bool = False) -> None:
        self._published = (parity, rows)
        self._published_after_wait = after_wait

    def max_rows(self) -> int:
        """Rows a site can take: the one-shot mailbox's CAP, or the two-shot box's cap when two-shot sites are on (every
        row count above CAP is then >= twoshot_rows, see __init__)."""
        return self.twoshot_cap if self.twoshot is not None else CAP

    def covers(self, hidden_states: torch.Tensor) -> bool:
        from sglang.srt.runtime_context import get_forward

        # Under sequence parallelism the row-parallel linears reduce-scatter whatever reduce_results says.
        return (
            not get_forward().sp_active
            and hidden_states.is_cuda
            and hidden_states.dim() == 2
            and 0 < hidden_states.shape[0] <= self.max_rows()
            and hidden_states.shape[1] == HIDDEN
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
        )

    @contextmanager
    def engage(self, layers):
        """The draft layers' o_proj / down_proj return their partials (the sites reduce them) inside the block."""
        linears = [linear for layer in layers for linear in (layer.self_attn.o_proj, layer.mlp.down_proj)]
        saved = [linear.reduce_results for linear in linears]
        for linear in linears:
            linear.reduce_results = False
        if self.twoshot_drr and self.twoshot is not None:
            self._ts_index, self._ts_sites, self._ts_pending = 0, 2 * len(layers), None
        try:
            yield
        finally:
            for linear, value in zip(linears, saved):
                linear.reduce_results = value
            if self._ts_pending is not None:
                self._ts_flush()
            self._ts_index, self._ts_pending = None, None

    def _ts_flush(self) -> None:
        """Re-arm the gather row a deferring two-shot site left published (stream-ordered after it, as the served
        in-site re-arm plus its kernel boundary)."""
        parity, rows = self._ts_pending
        self._ts_pending = None
        if not self.twoshot_drr_flushes:
            logger.warning("SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR: a forward left a deferred gather row; re-armed by a fill")
        self.twoshot_drr_flushes += 1
        self.twoshot.views(parity)[2][0, :rows].view(torch.int32).fill_(SENTINEL)

    def run(self, parity: int, partial, projection, conv, residual, norm, successor=None) -> torch.Tensor:
        """Reduce ``partial`` [rows, HIDDEN] over the TP group with ``conv``'s side 1 (``projection`` = the whole
        [rows, 2 * TAPS * GROUPS] kernel projection of its prepare), add it into ``residual`` in place and return
        ``norm``(residual). ``successor``: the DFlashGroupedConv whose prepare_projection consumes the result next
        (None when something else does)."""
        rows = partial.shape[0]
        partial, projection = partial.contiguous(), projection.contiguous()
        # The ring projection already published this site's contribution: poll only (a second publish of the same
        # slot could land after a peer read and re-armed it).
        polled = self._published == (parity, rows)
        deferred = bool(self.deferred and polled and self._published_after_wait and rows <= GATE_ROWS)
        self._published, self._published_after_wait = None, False
        if self.twoshot is not None and not polled and rows >= self.twoshot_rows:
            normed = torch.empty_like(partial)
            early = self._early_for(successor, normed)
            drr_defer, drr_catchup = False, 0
            if self._ts_index is not None:
                if self._ts_pending is not None and self._ts_pending != (parity ^ 1, rows):
                    self._ts_flush()
                if self._ts_pending is not None:
                    drr_catchup = 1 - 2 * parity  # the other parity's gather row, in parities
                drr_defer = self._ts_index != self._ts_sites - 1
                self._ts_index += 1
                self._ts_pending = (parity, rows) if drr_defer else None
                if not self.twoshot_drr_runs:
                    logger.info("SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR ENGAGED: two-shot AR sites re-arm the previous "
                                "site's gather row from their non-live threads (first at %d rows)", rows)
                self.twoshot_drr_runs += 1
            self._note_smemsum(rows)
            several = 0 < self.twoshot_maxcl < rows
            if several:
                code = compile_site_twoshot_rows(
                    rows, self.tp, conv.block_size, self.deadline_ns, early, self.twoshot_cap,
                    self.twoshot_maxcl, self.twoshot_batched,
                )  # fmt: skip
                self.twoshot_rows_runs += 1
            else:
                bound = 0 < self.twoshot_bound_rows < rows
                code = compile_site_twoshot(
                    rows, self.tp, conv.block_size, self.deadline_ns, early, self.twoshot_batched,
                    self.twoshot_fence_offload, drr_defer, drr_catchup, self.smemsum,
                    cap=self.twoshot_cap, min_blocks=BOUND_BLOCKS if bound else 0,
                )  # fmt: skip
                if bound and not self.twoshot_bound_runs:
                    logger.info(
                        "DFLASH draft two-shot AR sites: launch bound %d CTAs / SM above %d rows (first at %d rows)", BOUND_BLOCKS,
                        self.twoshot_bound_rows, rows,
                    )  # fmt: skip
                self.twoshot_bound_runs += int(bound)
                if self.twoshot_fence_offload and not self.twoshot_foff_runs:
                    logger.info("SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_FENCE_OFFLOAD ENGAGED: two-shot AR sites fence the gather re-arm "
                                "from one non-live thread after the norm's CTA barrier (first at %d rows)", rows)
                self.twoshot_foff_runs += int(self.twoshot_fence_offload)
            if self.twoshot_batched and not self.twoshot_batched_runs:
                logger.info("DFLASH draft two-shot AR sites: batched owner polls (first at %d rows)", rows)
            self.twoshot_batched_runs += int(self.twoshot_batched)
            if rows > CAP and rows not in self._twoshot_logged:
                self._twoshot_logged.add(rows)
                logger.info(
                    "DFLASH draft two-shot AR site engaged at %d rows (cap %d, %d rows per CTA)", rows, self.twoshot_cap,
                    -(-rows // self.twoshot_maxcl) if several else 1,
                )  # fmt: skip
            recv, peers, gather, gather_mc = self.twoshot.views(parity)
            code(
                to_cute(partial), to_cute(projection), to_cute(conv.base_kernel.detach()[1]), to_cute(residual),
                to_cute(norm.weight.detach()), to_cute(normed), to_cute(recv), to_cute(peers), to_cute(gather),
                Int64(gather_mc), Int32(self.rank), Float32(norm.variance_epsilon), stream(partial.device),
            )  # fmt: skip
            self.twoshot_runs += 1
            return normed
        assert rows <= CAP, f"a {rows}-row draft AR site needs the two-shot path"
        if deferred:
            # A deferred site writes before its wait: never into memory the caching allocator could have handed out from
            # a tensor a still-running earlier kernel reads.
            key = (id(norm), rows)
            normed = self._deferred_out.get(key)
            if normed is None or normed.device != partial.device:
                normed = self._deferred_out[key] = torch.empty_like(partial)
            self.deferred_runs += 1
        else:
            normed = torch.empty_like(partial)
        early = self._early_for(successor, normed)
        if self._ts_pending is not None:
            self._ts_flush()
        valid = bool(self.valid and polled and rows <= GATE_ROWS)
        self.valid_runs += int(valid)
        gate = bool(self.gate and polled and rows <= GATE_ROWS and not valid)
        self._note_smemsum(rows)
        code = compile_site(
            rows, self.tp, conv.block_size, self.deadline_ns, early, polled, gate, deferred, valid, self.smemsum,
        )
        mailbox = self.mailbox
        code(
            to_cute(partial), to_cute(projection), to_cute(conv.base_kernel.detach()[1]), to_cute(residual),
            to_cute(norm.weight.detach()), to_cute(normed), to_cute(mailbox.buffer), Int64(mailbox.mc), Int32(parity),
            Int32(self.rank), Float32(norm.variance_epsilon), stream(partial.device),
        )  # fmt: skip
        return normed

    def _note_smemsum(self, rows: int) -> None:
        if not self.smemsum:
            return
        if not self.smemsum_runs:
            logger.info("SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM ENGAGED: draft AR sites sum the norm's squares in shared memory "
                        "(first at %d rows)", rows)
        self.smemsum_runs += 1

    def _early_for(self, successor, normed: torch.Tensor) -> bool:
        """The early trigger only when the next kernel is the plugin split-K projection of ``normed``: before its own
        wait it reads only its weight and TMA maps. Any other successor (the stock GEMM, the head after the last
        layer) is not shown to wait before its first load, so it keeps the trigger after the norm stores. The projection
        must also release its own dependents only after its wait: otherwise they too could start while this site runs."""
        if not self.early_trigger or successor is None:
            return False
        projection = successor.split_k_projection
        return (
            projection is not None
            and projection.covers(normed)
            and projection.trigger_follows_wait(normed.shape[0])
        )


def install_draft_ar_site(draft_model) -> Optional[DraftArSites]:
    """Give an MLA DFlash draft (RMSNorm, conv taps 2, group 16, hidden 7168) its all-reduce sites; collective over the
    draft TP group. Returns None at TP 1 (no all-reduce)."""
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashMLAAttention
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.dflash_draft_collectives import serving_deadline_ns

    group = get_parallel().tp_group
    if group.world_size == 1:
        return None
    if draft_model.draft_all_reduce is not None:
        raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_SITE replaces SGLANG_OPT_DFLASH_DRAFT_PK3_ALL_REDUCE: set one.")
    norms = [draft_model.norm]
    for layer in draft_model.layers:
        if type(layer) is not DFlashDecoderLayer or not isinstance(layer.self_attn, DFlashMLAAttention):
            raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_SITE needs an MLA DFlash draft")
        for conv in (layer.attention_conv, layer.mlp_conv):
            if conv is None or conv.taps != TAPS or conv.group_size != GROUP or conv.num_groups != GROUPS:
                raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_AR_SITE needs conv taps {TAPS}, group {GROUP}, hidden {HIDDEN}")
            if conv.base_kernel.dtype != torch.bfloat16:
                raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_SITE needs BF16 conv base kernels")
        for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
            if not linear.reduce_results or getattr(linear, "use_decode_attn_tp", False):
                raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_SITE needs o_proj / down_proj to all-reduce their outputs")
        norms += [layer.input_layernorm, layer.post_attention_layernorm]
    for norm in norms:
        if type(norm) is not RMSNorm or norm.weight.dtype != torch.bfloat16 or norm.weight.shape != (HIDDEN,):
            raise ValueError("SGLANG_OPT_DFLASH_DRAFT_AR_SITE needs BF16 RMSNorm draft norms of the hidden size")
    from sglang.srt.environ import envs

    device = next(draft_model.parameters()).device
    early_trigger = envs.SGLANG_OPT_DFLASH_DRAFT_AR_SITE_EARLY_TRIGGER.get()
    sites = draft_model.ar_sites = DraftArSites(group, device, serving_deadline_ns(), early_trigger=early_trigger)
    logger.info(
        "DFLASH draft all-reduce sites on %d layers (TP %d), early trigger %s, gated poll %s, deferred %s, valid %s, "
        "two-shot from %s rows up to %s, several rows per CTA above %s rows, smem sum %s",
        len(draft_model.layers), sites.tp, early_trigger, sites.gate, sites.deferred, sites.valid,
        sites.twoshot_rows if sites.twoshot is not None else "no", sites.max_rows(),
        sites.twoshot_maxcl if sites.twoshot is not None and sites.twoshot_maxcl > 0 else "no", sites.smemsum,
    )  # fmt: skip
    if envs.SGLANG_OPT_DFLASH_DRAFT_AR_RING.get():
        from sglang.srt.speculative.dflash_draft_ar_ring import install_draft_ar_ring

        install_draft_ar_ring(draft_model, sites)
    return sites
