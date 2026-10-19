# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Derived from NVIDIA/cutlass v4.8.0 examples/python/CuTeDSL/cute_ext/blackwell/attention/
# (gqa_decode_simple.py, gqa_decode_opt.py; BSD-3-Clause): warp-specialized split-KV decode structure,
# rewritten for paged FP8 KV.

# Modified by NVIDIA: paged FP8 verify with explicit local reduction.

# Modified by NVIDIA: separate softmax and value consumer groups.
# Modified by NVIDIA: separate key and value pipeline handoffs.

# Modified by NVIDIA: runtime attention scales, page-tail bounds and reusable buffers.
# Modified by NVIDIA: stock split-KV rounding (page splits, one FP8 P rounding, bf16 partials).
# Modified by NVIDIA: q1 decode with the trtllm-gen generation kernel's KV split and rounding (two streams per CTA).

"""Paged FP8 q1 decode (draft model) with trtllm-gen's rounding structure.

Same KV split as trtllm-gen's multi-CTA generation kernel: one CTA per SM, ceil(steps / CTAs) contiguous 256-token
steps per CTA. Each CTA keeps two softmax streams (the two 128-token halves of every step) with a running max of the
raw scores; P x 448 = ex2(fma(s, c, -fma(m, c, -log2 448))) is rounded once to e4m3, the CTA writes bf16(O / 448)
with fp32 maxima and denominators, and a small kernel merges the CTAs in fp32. Only fp32 association differs.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as ex
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.experimental import primitives as prim

LOG2_448 = math.log2(448.0)


@cute.experimental.jit
def load_keys(
    k: cute.Tensor,
    table: cute.Tensor,
    sk: cute.Tensor,
    barrier: cute.Pointer,
    first_page: cutlass.Int32,
    stage: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    length: cutlass.Int32,
):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tile * 256)
    for half in cutlass.range_constexpr(tile // 64):
        page_index = first_page + half
        page = cutlass.Int32(0)
        if page_index < cute.ceil_div(length, 64):
            page = table[0, page_index]
        source = cute.local_tile(k, (64, 256), (page, 0))
        dest = cute.local_tile(sk[None, None, stage], (64, 256), (half, 0))
        ex.tma_load(source, dest, barrier.value, update_expect_tx=False)


@cute.experimental.jit
def load_values(
    v: cute.Tensor,
    table: cute.Tensor,
    sv: cute.Tensor,
    barrier: cute.Pointer,
    first_page: cutlass.Int32,
    stage: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    length: cutlass.Int32,
):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tile * 256)
    for half in cutlass.range_constexpr(tile // 64):
        page_index = first_page + half
        page = cutlass.Int32(0)
        if page_index < cute.ceil_div(length, 64):
            page = table[0, page_index]
        source = cute.local_tile(v, (256, 64), (0, page))
        dest = cute.local_tile(sv[None, None, stage], (256, 64), (0, half))
        ex.tma_load(source, dest, barrier.value, update_expect_tx=False)


class Q1SplitDecode:
    def __init__(self, heads=4, nsplit=152, stages=2, slots=2):
        assert stages >= 1 and slots in (2, 3, 4) and 1 <= heads <= 12
        self.heads, self.nsplit, self.stages, self.slots = heads, nsplit, stages, slots
        self.rows = heads
        self.width = 32
        self.tile = 128
        assert self.width * slots * 3 <= 512

    @cute.experimental.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        table: cute.Tensor,
        lengths: cute.Tensor,
        partial: cute.Tensor,
        maxima: cute.Tensor,
        denom: cute.Tensor,
        qk_scale: cutlass.Float32,
        out_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        dtype = cutlass.Float8E4M3FN
        kq = cute.make_tiled_mma(
            tcgen05.MmaFP8Op(
                dtype,
                cutlass.Float32,
                (128, self.width),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        vp = cute.make_tiled_mma(
            tcgen05.MmaFP8Op(
                dtype,
                cutlass.Float32,
                (128, self.width),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.MN,
                OperandMajorMode.MN,
            )
        )
        self.kernel(
            q, k, v, table, lengths, partial, maxima, denom, qk_scale, out_scale, kq, vp
        ).launch(grid=(self.nsplit, 1, 1), block=(384, 1, 1), stream=stream)

    @cute.experimental.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        table: cute.Tensor,
        lengths: cute.Tensor,
        partial: cute.Tensor,
        maxima: cute.Tensor,
        denom: cute.Tensor,
        qk_scale: cutlass.Float32,
        out_scale: cutlass.Float32,
        kq: cute.TiledMma,
        vp: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        local_tid = tid % 128
        width, rows = self.width, self.rows
        dtype, f32 = cutlass.Float8E4M3FN, cutlass.Float32
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, dtype
        )
        v_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, dtype
        )
        p_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW32, dtype
        )
        sq = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(k_atom, (width, 256), order=(1, 0)),
            alignment=128,
        )
        sk = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(k_atom, (self.tile, 256, self.stages), order=(1, 0, 2)),
            alignment=128,
        )
        sv = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(v_atom, (256, self.tile, self.stages), order=(0, 1, 2)),
            alignment=128,
        )
        sp = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(p_atom, (width, self.tile, self.slots), order=(0, 1, 2)),
            alignment=128,
        )
        # Per-thread denominators of both streams, padded against bank conflicts.
        shared_l = ex.allocate(
            f32,
            cutlass.AddressSpace.smem,
            cute.make_layout((2, 128, rows), stride=(128 * (rows + 1), rows + 1, 1)),
            alignment=128,
        )
        shared = ex.allocate(
            f32,
            cutlass.AddressSpace.smem,
            cute.make_layout((4, width), stride=(width, 1)),
            alignment=128,
        )
        alphas = ex.allocate(
            f32,
            cutlass.AddressSpace.smem,
            cute.make_layout((width, self.slots), stride=(1, width)),
            alignment=128,
        )
        stream_off = ex.allocate(
            f32,
            cutlass.AddressSpace.smem,
            cute.make_layout((2, width), stride=(width, 1)),
            alignment=128,
        )
        barriers = ex.allocate(
            cutlass.Int64,
            cutlass.AddressSpace.smem,
            cute.make_layout(4 * self.stages + 5 * self.slots + 2),
            alignment=8,
        )
        bq = barriers.iterator
        bk = bq + 1
        bv = bk + self.stages
        bkfree = bv + self.stages
        bvfree = bkfree + self.stages
        bs = bvfree + self.stages
        bsfree = bs + self.slots
        bp = bsfree + self.slots
        bo = bp + self.slots
        bfree = bo + self.slots
        bdenom = bfree + self.slots
        if tid < 4 * self.stages + 5 * self.slots + 2:
            count = cutlass.Int32(1)
            if (
                tid >= 4 * self.stages + 1 + 2 * self.slots
                and tid < 4 * self.stages + 1 + 3 * self.slots
            ):
                count = cutlass.Int32(128)
            if tid == 4 * self.stages + 5 * self.slots + 1:
                count = cutlass.Int32(128)
            cute.arch.mbarrier_init(barriers.iterator + tid, count)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        ts = ex.allocate(
            f32,
            cutlass.AddressSpace.tmem,
            ex.make_tmem_layout_acc(kq, (128, width, 256), self.slots),
            alignment=32,
        )
        to = ex.allocate(
            f32,
            cutlass.AddressSpace.tmem,
            ex.make_tmem_layout_acc(vp, (128, width, 128), 2 * self.slots),
            alignment=32,
        )
        copy_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(width)), f32
        )
        s_plane = ts[(None, None), 0, 0, 0]
        copy_s = tcgen05.make_tmem_copy(copy_atom, s_plane).get_slice(local_tid)
        rlayout = copy_s.partition_D(s_plane).layout
        # trtllm-gen split: one CTA per SM, ceil(steps / CTAs) contiguous 256-token steps each.
        length = lengths[0]
        spc = cute.ceil_div(cute.ceil_div(length, 256), self.nsplit)
        begin_tok = split * spc * 256
        end_tok = begin_tok + spc * 256
        if end_tok > length:
            end_tok = length
        nblk = cutlass.Int32(0)
        if begin_tok < end_tok:
            nblk = cute.ceil_div(end_tok - begin_tok, self.tile)
        npair = cute.ceil_div(nblk, 2)
        begin_page = split * spc * 4
        cute.arch.sync_threads()
        if warp < 4:
            scores = cute.make_rmem_tensor((width, 1), f32)
            running_m = cute.make_rmem_tensor((2,), f32)
            running_off = cute.make_rmem_tensor((2,), f32)
            lane_off = cute.make_rmem_tensor((1,), f32)
            lane_alpha = cute.make_rmem_tensor((1,), f32)
            running_l = cute.make_rmem_tensor((2, rows), f32)
            packed = cute.make_rmem_tensor((width, 1), dtype)
            running_m.fill(-f32.inf)
            running_off.fill(-f32.inf)
            running_l.fill(0)
            for pair in cutlass.range(0, npair):
                for st in cutlass.range_constexpr(2):
                    it = pair * 2 + st
                    if it < nblk:
                        slot = it % self.slots
                        cute.arch.mbarrier_wait(bs + slot, phase=(it // self.slots) % 2)
                        current_s = ts[(None, None), 0, 0, slot]
                        current_copy = tcgen05.make_tmem_copy(
                            copy_atom, current_s
                        ).get_slice(local_tid)
                        sub_scores = cute.make_tensor(scores.iterator, rlayout)
                        ex.partition_and_copy(current_copy, current_s, sub_scores)
                        cute.arch.fence_view_async_tmem_load()
                        prim.barrier_cta_sync(1, thread_count=128)
                        if warp == 0:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(bsfree + slot)
                        token = begin_tok + it * self.tile + local_tid
                        for j in cutlass.range_constexpr(rows):
                            score = scores[j, 0]
                            if token >= end_tok:
                                score = -f32.inf
                            scores[j, 0] = score
                            maximum = cute.arch.warp_redux_sync(
                                score, kind="fmax", nan=True
                            )
                            if lane == j % 32:
                                shared[warp, j] = maximum
                        prim.barrier_cta_sync(1, thread_count=128)
                        # The value consumer releases both the probability and alpha slot.
                        cute.arch.mbarrier_wait(
                            bfree + slot, phase=(it // self.slots + 1) % 2
                        )
                        raw = running_m[st]
                        if lane < rows:
                            for w in cutlass.range_constexpr(4):
                                raw = cute.arch.fmax(raw, shared[w, lane])
                        else:
                            raw = f32(0)
                        # trtllm-gen exponent: off = fma(m, c, -log2 448), P448 = ex2(fma(s, c, -off)).
                        off = cute.math.fma(raw, qk_scale, f32(-LOG2_448))
                        safe = off
                        if raw == -f32.inf:
                            safe = f32(0)
                        lane_off[0] = safe
                        lane_alpha[0] = cute.math.exp2(
                            running_off[st] - safe, fastmath=True
                        )
                        running_m[st] = raw
                        running_off[st] = off
                        if warp == 0:
                            alphas[lane, slot] = lane_alpha[0]
                        for j in cutlass.range_constexpr(rows):
                            offset = cute.arch.shuffle_sync(lane_off[0], j)
                            alpha = cute.arch.shuffle_sync(lane_alpha[0], j)
                            total = running_l[st, j] * alpha
                            probability = cute.math.exp2(
                                cute.math.fma(scores[j, 0], qk_scale, -offset),
                                fastmath=True,
                            )
                            total += probability
                            scores[j, 0] = probability
                            running_l[st, j] = total
                        for j in cutlass.range_constexpr(rows, width):
                            scores[j, 0] = f32(0)
                        packed[None, 0].store(scores[None, 0].load().to(dtype))
                        cute.autovec_copy(packed[None, 0], sp[None, local_tid, slot])
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(bp + slot)
            for st in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(rows):
                    shared_l[st, local_tid, j] = running_l[st, j]
            if warp == 0:
                if lane < rows:
                    stream_off[0, lane] = running_off[0]
                    stream_off[1, lane] = running_off[1]
            cute.arch.mbarrier_arrive(bdenom)
        elif warp < 8:
            value = cute.make_rmem_tensor(rlayout, f32)
            accum = cute.make_rmem_tensor((2, rows, 2), f32)
            accum.fill(0)
            for pair in cutlass.range(0, npair):
                for st in cutlass.range_constexpr(2):
                    it = pair * 2 + st
                    if it < nblk:
                        slot = it % self.slots
                        cute.arch.mbarrier_wait(bo + slot, phase=(it // self.slots) % 2)
                        for dm in cutlass.range_constexpr(2):
                            o_plane = to[(None, None), 0, 0, 2 * slot + dm]
                            copy_o = tcgen05.make_tmem_copy(
                                copy_atom, o_plane
                            ).get_slice(local_tid)
                            ex.partition_and_copy(copy_o, o_plane, value)
                            cute.arch.fence_view_async_tmem_load()
                            for j in cutlass.range_constexpr(rows):
                                accum[st, j, dm] = (
                                    accum[st, j, dm] * alphas[j, slot] + value[j]
                                )
                        prim.barrier_cta_sync(2, thread_count=128)
                        if warp == 4:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(bfree + slot)
            cute.arch.mbarrier_wait(bdenom, phase=0)
            for j in cutlass.range_constexpr(rows):
                o0 = stream_off[0, j]
                o1 = stream_off[1, j]
                top = cute.arch.fmax(o0, o1)
                w0 = f32(0)
                w1 = f32(0)
                if o0 != -f32.inf:
                    w0 = cute.math.exp2(o0 - top, fastmath=True)
                if o1 != -f32.inf:
                    w1 = cute.math.exp2(o1 - top, fastmath=True)
                for dm in cutlass.range_constexpr(2):
                    merged = accum[0, j, dm] * w0 + accum[1, j, dm] * w1
                    partial[split, j, local_tid + dm * 128] = cutlass.BFloat16(
                        merged * out_scale
                    )
        else:
            if warp == 8:
                q_frag = kq.get_slice(0).partition_B(sq)
                atom_kq = cute.make_mma_atom(kq.op)
                cute.arch.mbarrier_wait(bq, phase=0)
                for it in cutlass.range(0, nblk):
                    slot, stage = it % self.slots, it % self.stages
                    cute.arch.mbarrier_wait(
                        bsfree + slot, phase=(it // self.slots + 1) % 2
                    )
                    cute.arch.mbarrier_wait(bk + stage, phase=(it // self.stages) % 2)
                    ktile = cute.local_tile(sk[None, None, stage], (128, 256), (0, 0))
                    k_frag = kq.get_slice(0).partition_A(ktile)
                    atom_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(cute.size(k_frag, mode=[2])):
                        ex.dot(
                            atom_kq,
                            k_frag[None, None, kk],
                            q_frag[None, None, kk],
                            ts[None, None, None, slot],
                        )
                        atom_kq.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(bs + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bkfree + stage, None, tcgen05.CtaGroup.ONE)
            elif warp == 9:
                atom_vp = cute.make_mma_atom(vp.op)
                for it in cutlass.range(0, nblk):
                    slot, stage = it % self.slots, it % self.stages
                    cute.arch.mbarrier_wait(bp + slot, phase=(it // self.slots) % 2)
                    cute.arch.mbarrier_wait(bv + stage, phase=(it // self.stages) % 2)
                    p_frag = vp.get_slice(0).partition_B(sp[None, None, slot])
                    for dm in cutlass.range_constexpr(2):
                        vtile = cute.local_tile(
                            sv[None, None, stage], (128, self.tile), (dm, 0)
                        )
                        v_frag = vp.get_slice(0).partition_A(vtile)
                        atom_vp.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range_constexpr(cute.size(v_frag, mode=[2])):
                            ex.dot(
                                atom_vp,
                                v_frag[None, None, kk],
                                p_frag[None, None, kk],
                                to[None, None, None, 2 * slot + dm],
                            )
                            atom_vp.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(bo + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bvfree + stage, None, tcgen05.CtaGroup.ONE)
            elif warp == 10:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bq, width * 256)
                ex.tma_load(
                    cute.local_tile(q, (width, 256), (0, 0)),
                    sq,
                    bq.value,
                    update_expect_tx=False,
                )
                for stage in cutlass.range_constexpr(self.stages):
                    if stage < nblk:
                        load_keys(
                            k,
                            table,
                            sk,
                            bk + stage,
                            begin_page + stage * (self.tile // 64),
                            stage,
                            self.tile,
                            length,
                        )
                for it in cutlass.range(0, nblk):
                    stage = it % self.stages
                    cute.arch.mbarrier_wait(
                        bvfree + stage, phase=(it // self.stages + 1) % 2
                    )
                    load_values(
                        v,
                        table,
                        sv,
                        bv + stage,
                        begin_page + it * (self.tile // 64),
                        stage,
                        self.tile,
                        length,
                    )
                    if it + self.stages < nblk:
                        cute.arch.mbarrier_wait(
                            bkfree + stage, phase=(it // self.stages) % 2
                        )
                        load_keys(
                            k,
                            table,
                            sk,
                            bk + stage,
                            begin_page + (it + self.stages) * (self.tile // 64),
                            stage,
                            self.tile,
                            length,
                        )
        cute.arch.mbarrier_wait(bdenom, phase=0)
        for chunk in cutlass.range_constexpr(cute.ceil_div(rows, 12)):
            column = warp + 12 * chunk
            if column < rows:
                o0 = stream_off[0, column]
                o1 = stream_off[1, column]
                top = cute.arch.fmax(o0, o1)
                total = f32(0)
                for st in cutlass.range_constexpr(2):
                    part = (
                        shared_l[st, lane, column] + shared_l[st, lane + 32, column]
                    ) + (
                        shared_l[st, lane + 64, column]
                        + shared_l[st, lane + 96, column]
                    )
                    for offset in cutlass.range_constexpr(5):
                        part += cute.arch.shuffle_sync_bfly(part, 1 << offset)
                    stream_top = o0 if st == 0 else o1
                    if stream_top != -f32.inf:
                        total += part * cute.math.exp2(stream_top - top, fastmath=True)
                if lane == 0:
                    denom[split, column] = total
                    maxima[split, column] = top


import torch
import triton
import triton.language as tl
from cutlass.cute.runtime import from_dlpack


@triton.jit
def _q1_merge(
    partial,
    maxima,
    denom,
    lengths,
    out,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    NP2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    length = tl.load(lengths)
    steps = tl.cdiv(length, 256)
    spc = tl.cdiv(steps, NSPLIT)
    active = tl.cdiv(steps, spc)
    c = tl.arange(0, NP2)
    valid = c < active
    m = tl.load(maxima + c * ROWS + row, mask=valid, other=float("-inf"))
    w = tl.where(valid, tl.exp2(m - tl.max(m, 0)), 0.0)
    total = tl.sum(tl.load(denom + c * ROWS + row, mask=valid, other=0.0) * w, 0)
    d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    part = tl.load(
        partial + (c[:, None] * ROWS + row) * 256 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    o = tl.sum(part.to(tl.float32) * w[:, None], 0)
    # Denominators are sums of P x 448; partials are O / 448.
    tl.store(out + row * 256 + d, (o / (total * (1.0 / 448.0))).to(tl.bfloat16))


class Q1SplitPlan:
    """Reusable storage and a compiled producer for one fixed input layout, plus the fp32 merge."""

    def __init__(self, query, kv, table, lengths, *, nsplit=None, stages=2, slots=2):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Paged FP8 q1 decode storage must be prepared before graph capture"
            )
        qlen, self.heads, _ = query.shape
        assert qlen == 1
        self.nsplit = (
            nsplit
            or torch.cuda.get_device_properties(query.device).multi_processor_count
        )
        self.partial = torch.empty(
            (self.nsplit, self.heads, 256), dtype=torch.bfloat16, device=query.device
        )
        self.maximum = torch.empty(
            (self.nsplit, self.heads), dtype=torch.float32, device=query.device
        )
        self.denom = torch.empty_like(self.maximum)
        kernel = Q1SplitDecode(self.heads, self.nsplit, stages=stages, slots=slots)
        views = self._views(query, kv, table, lengths)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled = cute.compile(
            kernel, *views, cutlass.Float32(1.0), cutlass.Float32(1.0), stream
        )

    def _views(self, query, kv, table, lengths):
        k, v = kv
        tensors = (
            query.view(-1, 256),
            k.view(-1, 256),
            v.view(-1, 256).T,
            table,
            lengths,
            self.partial,
            self.maximum,
            self.denom,
        )
        return tuple(from_dlpack(t) for t in tensors)

    def run(self, query, kv, table, lengths, *, qk_scale, v_scale, out):
        views = self._views(query, kv, table, lengths)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled(
            *views,
            cutlass.Float32(qk_scale * 1.4426950408889634),
            cutlass.Float32(v_scale / 448.0),
            stream,
        )
        _q1_merge[(self.heads, 256 // 32, 1)](
            self.partial,
            self.maximum,
            self.denom,
            lengths,
            out,
            self.heads,
            self.nsplit,
            triton.next_power_of_2(self.nsplit),
            32,
            num_warps=4,
        )
        return out


def skv2_q1_decode(query, kv, table, lengths, plan, *, qk_scale, v_scale, out):
    """q1 decode for one request through a prepared plan; same contract as trtllm_batch_decode_with_kv_cache."""
    return plan.run(
        query, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out
    )
