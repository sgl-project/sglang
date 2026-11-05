# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""MLA decode under decode context parallelism (DCP = TP, rank d owns head block d)
without collectives, three kernels per layer. Each rank's KV cache holds 1 / DCP of
every context.

  MlaQuery     FP8 query rows [B, H, 576] (+ the KV insert where this rank owns the
               slot); each CTA multicasts its bytes into every rank's Lamport q
               mailbox [B, DCP*H, 576], then polls its share until all landed
  tokenspeed   all DCP*H heads over the local KV shard -> BF16 rows + LSE (log2)
  DcpValue     re-arms the q mailbox; rows pushed to their head owner's Lamport
               mailbox (peer stores); the owner LSE-merges its H heads over the DCP
               sources, then W_UV and the gate (padding from the global lengths)
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from sglang.kernels.jit.kimi_k3.primitives import (
    alloc,
    at,
    bf16x2,
    bits_float,
    float_bits,
    launch,
    load_volatile,
    load_words,
    mma_into,
    no_neg_zero,
    pack_pair,
    published,
    sigmoid,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    wait_deadline_check,
    wait_deadline_start,
    warp_sum,
    zeros,
)
from sglang.kernels.ops.communication.kimi_k3_specialized.mla_query import (
    _require_deadline_ns,
)

LSE_SENT = 0xFFFFFFFF  # a NaN pattern the merge never sends


EMPTY_LSE = -3.0e38


Q_SENT = 0xFFFFFFFF  # four FP8 0xFF bytes: never written (satfinite, canonical NaN)


PAIR_SENT = 0x80000000  # BF16 pair (+0, -0); sent pairs have no negative zero


class DcpValue:
    """The DCP tail of one MLA layer, one 512-thread CTA per (head, 16 requests).
    Every rank re-arms this parity's q mailbox (tokenspeed has read it) and pushes all
    its local tokenspeed rows (+ LSE) to their head owner's Lamport mailbox (empty
    local shards: zeros, LSE -inf). Warp r then LSE-weights request t0 + r of this
    head over the DCP sources (exp2(lse - max) / sum in FP32), the CTA merges the
    polled rows in FP32
    (one BF16 rounding, re-arming what it read) into shared memory, and W_UV (BF16
    MMA, FP32 accumulation) and the sigmoid gate follow as in MlaValue: warp w owns
    outputs 32 (w // 4) + 8 (w % 4) .. + 7. Rows are tokens, ``q`` per request."""

    def __init__(self, heads, size, batch, sms=148, q=1, *, deadline_ns):
        self.deadline_ns = _require_deadline_ns(deadline_ns)
        if size not in (8, 16) or size * heads != 96:
            raise ValueError("Kimi DCP value has unsupported rank/head geometry")
        if not 1 <= batch <= 64 or q <= 0 or batch % q or sms <= 0:
            raise ValueError("Kimi DCP value has invalid prepared rows/query/SM bounds")
        self.heads, self.size, self.batch = heads, size, batch
        self.q = q
        # Push-only CTAs (z > 0) spread the NVLink stores over more SMs: up to one
        # wave, at least 4 16 B chunks per thread.
        merge = heads * ((batch + 15) // 16)
        chunks = batch * size * heads * 64
        pushers = min(max(1, sms // merge), -(-chunks // (merge * 512 * 4)))
        self.grid = (heads, (batch + 15) // 16, pushers)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 512)

    @cute.kernel
    def kernel(self, args):
        rows, rows_lse, lens, box, lbox, peers, lpeers, mailbox = args[0:8]
        w_uv, gate, totals, output, parity, rank = args[8:14]
        H, D, B, cap = self.heads, self.size, self.batch, box.shape[1]
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (16, 544), 128)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, pusher = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        threads = H * self.grid[1] * self.grid[2] * 512
        me = ((pusher * self.grid[1] + tile) * H + head) * 512 + tid
        for c in range(me, B * D * H * 36, threads):  # 16 B q mailbox chunks
            store_words(mailbox.iterator + c * 16, (Uint32(Q_SENT),) * 4)
        for i in range(me, B * D * H * 64, threads):
            r, k = i // 64, i % 64  # r = b * D * H + global head
            b, gh = r // (D * H), r % (D * H)
            words = (Uint32(0),) * 4
            if lens[b // self.q] > 0:
                words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
            words = tuple(no_neg_zero(w) for w in words)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(peers[gh // H] + (Int64(slot) * 512 + 8 * k) * 2, words)
        for i in range(me, B * D * H, threads):
            b, gh = i // (D * H), i % (D * H)
            value, n = rows_lse[b, 0, gh], lens[b // self.q]
            bad = (value != value) | (value <= Float32(EMPTY_LSE))  # NaN, -inf
            if (n <= 0) | bad | (value >= Float32(-EMPTY_LSE)):  # or +inf
                value = Float32(EMPTY_LSE)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(lpeers[gh // H] + Int64(slot) * 4, (float_bits(value),))
        cute.arch.griddepcontrol_launch_dependents()
        if pusher == 0:
            self.merge(args, latent, scales, wb, tid)

    @cute.jit
    def merge(self, args, latent, scales, wb, tid):
        """The owner's merge of request rows t0.. of this head, then W_UV + gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, tile, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = tile * 16
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < B:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t0 + warp, head)
                word = load_volatile(source)[0]
                started, polls = Int64(0), Int32(0)
                if word == Uint32(LSE_SENT):
                    started = wait_deadline_start(self.deadline_ns)
                while word == Uint32(LSE_SENT):
                    polls = wait_deadline_check(started, polls, self.deadline_ns)
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            scale = Float32(0)
            if total > 0:
                scale = weight / total
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        chunk = tid % 64
        for half in cutlass.range_constexpr(2):  # rows r and r + 8, 16 B chunk
            r = tid // 64 + 8 * half
            merged = zeros(8, Float32)
            if t0 + r < B:
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, t0 + r, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        words = loads[i]
                        started, polls = Int64(0), Int32(0)
                        if not published(words):
                            started = wait_deadline_start(self.deadline_ns)
                        while not published(words):
                            polls = wait_deadline_check(
                                started, polls, self.deadline_ns
                            )
                            words = load_volatile(sources[i], count=4)
                        store_words(sources[i], (Uint32(PAIR_SENT),) * 4)
                        w = scales[r, s0 + i]
                        for e in cutlass.range_constexpr(4):
                            lo, hi = unpack_bf16(words[e])
                            merged[2 * e] += w * lo
                            merged[2 * e + 1] += w * hi
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            store_shared_words(latent.iterator + r * 544 + 8 * chunk, pairs)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < B:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if totals[t // self.q] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e])) * sigmoid(gates[r][e], False)
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))
