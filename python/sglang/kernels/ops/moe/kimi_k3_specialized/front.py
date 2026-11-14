# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""MoE front: the weight-stationary [router shard; down shard] GEMM publishes logits
and the latent shard into a multicast mailbox row; the routing kernel polls all 896
logits, ranks the top16 (sigmoid + bias, renormalized) and Lamport-copies the latent."""

import cutlass as c
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64

from sglang.kernels.jit.kimi_k3.primitives import (
    alloc,
    asm,
    bits_float,
    div_rn,
    float_bits,
    launch,
    redux,
    sigmoid,
    store_words,
    view,
)
from sglang.kernels.ops.communication.kimi_k3_specialized.experts import plan_chunk
from sglang.kernels.ops.communication.kimi_k3_specialized.mailbox import poll, rearm
from sglang.kernels.ops.communication.kimi_k3_specialized.shared import situ


@asm(Int32)
def shared_add(address, value):
    code = "atom.shared.add.u32 $0,[$1],$2;"
    return code, "=r,r,r,~{memory}", (address, Int32(value))


@cute.jit
def ordered(score):
    """Monotone uint32 key of a float; NaN maps to 0 (below every number)."""
    bits = Uint32(float_bits(score))
    key = ~bits if (bits & Uint32(0x80000000)) != 0 else bits | Uint32(0x80000000)
    return key if score == score else Uint32(0)


class RouteSitu:
    def __init__(
        self, batch, shared, renorm, early, roles, polled, plan=False, deadline_ns=0
    ):  # fmt: skip
        # B>8: few registers, so routing CTAs co-reside with the front and the fused
        # shared expert, and polled logits skip the front's completion wait.
        self.lean = batch > 8
        self.batch, self.shared, self.experts, self.renorm = batch, shared, 896, renorm
        # early: release the persistent experts at entry (safe once down drains first);
        # polled: logits arrive as FP32 words in mailbox columns [3584, 5376) (Lamport);
        # plan: the copy role also writes the latent as MXFP8 into the experts' plan.
        self.early, self.polled, self.plan = early, polled, plan
        self.deadline_ns = deadline_ns
        self.roles = tuple(r for r, on in zip(("route", "copy", "situ"), roles) if on)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, len(self.roles) * self.batch, 256)

    @cute.kernel
    def kernel(self, args):
        logits, bias, ids, weights, mailbox, latent, gate_up, hidden = args[0:8]
        scale, beta, linear_beta = args[8:11]
        plan = args[11] if self.plan else None
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        token, role = cta % self.batch, cta // self.batch
        if c.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        for r in c.range_constexpr(len(self.roles)):
            if role == r:
                if c.const_expr(self.roles[r] == "route"):
                    self.route(logits, bias, ids, weights, scale, token, tid, mailbox)
                elif c.const_expr(self.roles[r] == "copy"):
                    # Lamport: no dependency wait; the sentinel poll is the barrier.
                    base = mailbox.iterator + token * mailbox.shape[2] + tid * 8
                    if tid < 192:
                        self.copy(base, latent, token, tid, 2, plan)
                    else:
                        self.copy(base, latent, token, tid, 1, plan)
                else:
                    cute.arch.griddepcontrol_wait()
                    rounds = (self.shared + 255) // 256  # issue every load first
                    pair = cute.make_rmem_tensor((rounds, 2), c.Float32)
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            pair[i, 0] = gate_up[token, tid + 256 * i]
                            pair[i, 1] = gate_up[token, self.shared + tid + 256 * i]
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            gate = c.Float32(c.BFloat16(pair[i, 0]))
                            up = c.Float32(c.BFloat16(pair[i, 1]))
                            value = situ(gate, up, beta, linear_beta)
                            hidden[token, tid + 256 * i] = c.BFloat16(value)
        # Late release: the persistent experts must not take SMs from the down GEMM.
        if c.const_expr(not self.early):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def select(self, key, threshold, buffers):
        """B>8 candidates (keys at or above ``threshold``) append to ``dense`` at a
        per-warp offset from the shared count, then rank themselves: 8 threads per
        candidate (4 past 32 candidates) each count their share of the candidates above
        it (key, then lower expert). Ranks below 16 store the expert and its sigmoid."""
        dense, sig, chosen, picked, found = buffers
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        packed = cute.make_rmem_tensor(4, Uint64)
        ballots = cute.make_rmem_tensor(4, Uint32)
        total = Int32(0)
        for i in c.range_constexpr(4):  # packed 0: not a candidate
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballots[i] = cute.arch.vote_ballot_sync(hit)
            packed[i] = Uint64(0)
            if hit:
                packed[i] = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
            total += cute.arch.popc(ballots[i])
        offset = Int32(0)
        if (lane == 0) & (total > 0):
            offset = shared_add(found.iterator, total)
        offset = cute.arch.shuffle_sync(offset, 0)
        below = (Uint32(1) << lane) - Uint32(1)
        for i in c.range_constexpr(4):
            if packed[i] != Uint64(0):
                dense[offset + cute.arch.popc(ballots[i] & below)] = packed[i]
            offset += cute.arch.popc(ballots[i])
        cute.arch.sync_threads()
        n = found[0]
        if n <= 32:  # usually: 8 threads per candidate, 4 slots each
            part, index = tid % 8, tid // 8
            mine, rank = Uint64(0), Int32(0)
            if index < n:
                mine = Uint64(dense[index])
            for k in c.range_constexpr(4):  # slots past n do not count
                j = part + 8 * k
                rank += Int32((j < n) & (Uint64(dense[j]) > mine))
            for shift in c.range_constexpr(3):
                rank += cute.arch.shuffle_sync_bfly(rank, 1 << shift)
            if (part == 0) & (index < n) & (rank < 16):
                expert = Int32(~Uint32(mine))
                chosen[rank], picked[rank] = expert, sig[expert]
        else:
            part = tid % 4
            for first in c.range(0, n, 64, unroll=1):
                index = first + tid // 4
                mine, rank = Uint64(0), Int32(0)
                if index < n:
                    mine = Uint64(dense[index])
                if n <= 64:  # every load issues at once; slots past n do not count
                    for k in c.range_constexpr(16):
                        j = part + 4 * k
                        rank += Int32((j < n) & (Uint64(dense[j]) > mine))
                else:
                    for j in c.range(part, n, 4, unroll=2):
                        rank += Int32(Uint64(dense[j]) > mine)
                rank += cute.arch.shuffle_sync_bfly(rank, 1)
                rank += cute.arch.shuffle_sync_bfly(rank, 2)
                if (part == 0) & (index < n) & (rank < 16):
                    expert = Int32(~Uint32(mine))
                    chosen[rank], picked[rank] = expert, sig[expert]
        cute.arch.sync_threads()

    @cute.jit
    def publish(self, chosen, picked, ids, weights, scale, token):
        """Warp 0: the 16 ids by rank and their sigmoids renormalized (lane r holds
        rank r; the butterfly sums in the same order for every lane)."""
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        if tid < 32:
            value = c.Float32(0)
            if lane < 16:
                ids[token, lane] = chosen[lane]
                value = picked[lane]
            total = value
            for shift in c.range_constexpr(4):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane < 16:
                value = value * scale
                if c.const_expr(self.renorm):
                    value = div_rn(value, total + c.Float32(1e-20))
                weights[token, lane] = c.BFloat16(value)

    def expert(self, tid, i):
        return tid * 4 + i if self.polled else tid + i * 256

    @cute.jit
    def copy(self, base, latent, token, tid, count: c.Constexpr[int], plan=None):
        got = poll(base, 2048, count, self.deadline_ns)
        for i in c.range_constexpr(count):
            dst = latent.iterator + token * latent.stride[0] + (tid + 256 * i) * 8
            words = [got[i, k] for k in range(4)]
            store_words(dst, words)
            rearm(base + i * 2048)
            if c.const_expr(plan is not None):
                plan_chunk(plan, token, tid + 256 * i, words)

    @cute.jit
    def route(self, logits, bias, ids, weights, scale, token, tid, mailbox):
        """Exact top16 by threads [0, 256): candidates at or above the 16th largest
        lane-group maximum are compacted and ranked by an unrolled 64-way compare."""
        warp, lane = tid // 32, tid % 32
        smem = c.utils.SmemAllocator()
        km = alloc(smem, Uint32, (8, 32))
        cand, dense = (alloc(smem, Uint64, shape) for shape in ((8, 128), 64))
        sig = alloc(smem, c.Float32, 1024)
        counts = alloc(smem, Int32, 8)
        cutoff = alloc(smem, Uint32, 1)
        chosen = alloc(smem, Int32, 16)
        if c.const_expr(self.lean):  # candidate count; the chosen experts' sigmoids
            found, picked = alloc(smem, Int32, 1), alloc(smem, c.Float32, 16)
        # Thread t owns experts t + 256 i, or 4 t + i when polling 16-byte chunks.
        offsets = cute.make_rmem_tensor(4, c.Float32)
        for i in c.range_constexpr(4):  # bias is a weight: load before the wait
            offsets[i] = c.Float32(0)
            if self.expert(tid, i) < self.experts:
                offsets[i] = bias[self.expert(tid, i)]
        raw = cute.make_rmem_tensor(4, c.Float32)
        key, top = cute.make_rmem_tensor(4, Uint32), Uint32(0)
        for i in c.range_constexpr(4):
            key[i] = Uint32(0)
        if c.const_expr(self.polled):
            if tid < 224:
                cell = mailbox.iterator + token * mailbox.shape[2] + 3584 + tid * 8
                got = poll(cell, 0, 1, self.deadline_ns)
                rearm(cell)
                for i in c.range_constexpr(4):
                    raw[i] = bits_float(got[0, i])
                if c.const_expr(self.lean):  # all four experts: one block, no branches
                    scores = [sigmoid(raw[i]) for i in range(4)]
                    for i in c.range_constexpr(4):
                        sig[4 * tid + i] = scores[i]
                        key[i] = ordered(scores[i] + offsets[i])
            if c.const_expr(not self.lean):
                cute.arch.griddepcontrol_wait()
        else:
            cute.arch.griddepcontrol_wait()
            for i in c.range_constexpr(4):
                if self.expert(tid, i) < self.experts:
                    raw[i] = logits[token, self.expert(tid, i)]
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            if c.const_expr(not (self.lean and self.polled)):  # noqa: SIM102
                if expert < self.experts:
                    score = sigmoid(raw[i])
                    sig[expert] = score
                    key[i] = ordered(score + offsets[i])
            top = max(top, key[i])
        km[warp, lane] = top
        if c.const_expr(self.lean):  # noqa: SIM102
            if tid == 0:
                found[0] = Int32(0)
        cute.arch.sync_threads()
        if warp == 0:
            group = km[0, lane]
            for w in c.range_constexpr(1, 8):
                group = max(group, km[w, lane])
            km[0, lane] = group
            cute.arch.sync_warp()
            rank = Int32(0)
            for j in c.range_constexpr(32):
                rank += Int32(km[0, j] > group)
            low = redux(group if rank < 16 else Uint32(0xFFFFFFFF))
            if lane == 0:
                cutoff[0] = low
        cute.arch.sync_threads()
        if c.const_expr(self.lean):
            dense = view(cand.iterator, 0, Uint64, (1024,))
            self.select(key, cutoff[0], (dense, sig, chosen, picked, found))
            self.publish(chosen, picked, ids, weights, scale, token)
            return
        threshold, offset = cutoff[0], Int32(0)
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballot = cute.arch.vote_ballot_sync(hit)
            if hit:
                below = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
                packed = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
                cand[warp, offset + below] = packed
            offset += cute.arch.popc(ballot)
        if lane == 0:
            counts[warp] = offset
        cute.arch.sync_threads()
        if warp == 0:
            n = Int32(0)
            for v in c.range_constexpr(8):
                n += counts[v]
            for half in c.range_constexpr(2):  # compact the regions densely
                index = lane + 32 * half
                rest, region = Int32(index), Int32(0)
                for v in c.range_constexpr(8):
                    if (region == v) & (rest >= counts[v]):
                        rest -= counts[v]
                        region += 1
                value = Uint64(0)
                if index < n:
                    value = cand[region, rest]
                dense[index] = value
            cute.arch.sync_warp()
            if n <= 64:
                for half in c.range_constexpr(2):
                    mine, rank = dense[lane + 32 * half], Int32(0)
                    for j in c.range_constexpr(64):
                        rank += Int32(dense[j] > mine)
                    if (lane + 32 * half < n) & (rank < 16):
                        chosen[rank] = Int32(~Uint32(mine))
            else:  # rare: rank every candidate against all regions
                for q in range(lane, n, 32):
                    rest, region = Int32(q), Int32(0)
                    for v in c.range_constexpr(8):
                        if (region == v) & (rest >= counts[v]):
                            rest -= counts[v]
                            region += 1
                    mine, rank = cand[region, rest], Int32(0)
                    for v in c.range_constexpr(8):
                        for j in range(counts[v]):
                            rank += Int32(cand[v, j] > mine)
                    if rank < 16:
                        chosen[rank] = Int32(~Uint32(mine))
            cute.arch.sync_warp()
            expert, value = chosen[lane % 16], c.Float32(0)
            if lane < 16:
                ids[token, lane] = expert
                value = sig[expert]
            total = value
            for shift in c.range_constexpr(4):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane < 16:
                value = value * scale
                if c.const_expr(self.renorm):
                    value = div_rn(value, total + c.Float32(1e-20))
                weights[token, lane] = c.BFloat16(value)


FRONT = {
    560: "1-4:4.14.0.1.0,5-32:16.7.16.5.2,33-64:16.14.16.5.2",
    288: "1-4:4.14.0.1.0,5-64:16.7.16.3.2",
}


WIDTH = 3584 + 2 * 896  # front mailbox row: BF16 latent, then FP32 logit words


FUSED, SPLIT = "9-64:1", "9-12:6,13-24:4,25-64:2"


SPLITS = {8: ("25-32:3", "13-16:6,25-32:4,33-64:3"), 16: ("", "25-32:3")}


MERGED = {8: "41-64:1", 16: "33-64:1"}


SHARED_TACTICS = {
    768: (
        "1-8:128.8.4.6,9-16:64.16.4.10,17-32:64.32.4.8,33-64:64.64.4.5",
        "1-8:128.8.1.6,9-16:64.16.1.6,17-32:64.32.1.6,33-64:64.64.1.6",
    ),
    384: (
        "1-16:64.8.4.8,17-32:64.16.4.10,33-64:64.32.4.8",
        "1-8:128.8.1.3,9-16:64.16.1.3,17-32:64.32.1.3,33-64:64.64.1.3",
    ),
}
