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

"""Single-request paged FP8 attention with caller-owned output and scratch."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as ex
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.experimental import primitives as prim


@cute.experimental.jit
def load_keys(
    k: cute.Tensor,
    table: cute.Tensor,
    sk: cute.Tensor,
    barrier: cute.Pointer,
    block: cutlass.Int32,
    stage: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    length: cutlass.Int32,
):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tile * 256)
    for half in cutlass.range_constexpr(tile // 64):
        page_index = block * (tile // 64) + half
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
    block: cutlass.Int32,
    stage: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    length: cutlass.Int32,
):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tile * 256)
    for half in cutlass.range_constexpr(tile // 64):
        page_index = block * (tile // 64) + half
        page = cutlass.Int32(0)
        if page_index < cute.ceil_div(length, 64):
            page = table[0, page_index]
        source = cute.local_tile(v, (256, 64), (0, page))
        dest = cute.local_tile(sv[None, None, stage], (256, 64), (0, half))
        ex.tma_load(source, dest, barrier.value, update_expect_tx=False)


class DecoupledPagedDecode:
    def __init__(
        self,
        heads=4,
        qlen=7,
        splits=152,
        stages=2,
        dual_tma=False,
        pack_probabilities=False,
        tile=128,
        tma_output=False,
        diagnostic=False,
        slots=2,
        parallel_denom=False,
        compensate_probabilities=False,
    ):
        assert stages >= 1
        assert tile in (128, 256)
        self.heads, self.qlen, self.splits, self.stages = heads, qlen, splits, stages
        self.rows = heads * qlen
        self.width = 32 if self.rows <= 32 else 64
        self.dual_tma = dual_tma
        self.pack_probabilities = pack_probabilities
        self.tile = tile
        self.subtiles = tile // 128
        self.tma_output = tma_output
        self.diagnostic = diagnostic
        assert slots in (2, 3, 4)
        assert self.width * slots * (tile // 128 + 2) <= 512
        self.slots = slots
        self.parallel_denom = parallel_denom
        self.compensate_probabilities = compensate_probabilities

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
        stamps: cute.Tensor,
        qk_scale: cutlass.Float32,
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
            q, k, v, table, lengths, partial, maxima, denom, stamps, qk_scale, kq, vp
        ).launch(grid=(self.splits, 1, 1), block=(384, 1, 1), stream=stream)

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
        stamps: cute.Tensor,
        qk_scale: cutlass.Float32,
        kq: cute.TiledMma,
        vp: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.diagnostic):
            if tid == 0:
                stamps[split, 0] = cute.arch.globaltimer()
        local_tid = tid % 128
        width = self.width
        dtype, f32 = cutlass.Float8E4M3FN, cutlass.Float32
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, dtype
        )
        v_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, dtype
        )
        p_kind = (
            tcgen05.SmemLayoutAtomKind.MN_SW32
            if width == 32
            else tcgen05.SmemLayoutAtomKind.MN_SW64
        )
        p_atom = tcgen05.make_smem_layout_atom(p_kind, dtype)
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
        if cutlass.const_expr(self.compensate_probabilities):
            sp_lo = ex.allocate(
                dtype,
                cutlass.AddressSpace.smem,
                cute.tile_to_shape(
                    p_atom, (width, self.tile, self.slots), order=(0, 1, 2)
                ),
                alignment=128,
            )
        if cutlass.const_expr(self.tma_output):
            o_atom = tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, f32
            )
            so = ex.allocate(
                f32,
                cutlass.AddressSpace.smem,
                cute.tile_to_shape(o_atom, (width, 256), order=(1, 0)),
                alignment=128,
            )
        if cutlass.const_expr(self.parallel_denom):
            shared_l = ex.allocate(
                f32,
                cutlass.AddressSpace.smem,
                cute.make_layout((128, width), stride=(width + 1, 1)),
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
            ex.make_tmem_layout_acc(kq, (128, width, 256), self.slots * self.subtiles),
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
        length = lengths[0]
        blocks = cute.ceil_div(length, self.tile)
        begin = split * blocks // self.splits
        end = (split + 1) * blocks // self.splits
        cute.arch.sync_threads()
        if cutlass.const_expr(self.diagnostic):
            if tid == 0:
                stamps[split, 1] = cute.arch.globaltimer()
        if warp < 4:
            scores = cute.make_rmem_tensor((width, self.subtiles), f32)
            running_m = cute.make_rmem_tensor((width // 32,), f32)
            lane_maxima = cute.make_rmem_tensor((width // 32,), f32)
            lane_alphas = cute.make_rmem_tensor((width // 32,), f32)
            running_l = cute.make_rmem_tensor((self.rows,), f32)
            packed = cute.make_rmem_tensor((width, self.subtiles), dtype)
            if cutlass.const_expr(self.compensate_probabilities):
                packed_lo = cute.make_rmem_tensor((width, self.subtiles), dtype)
            running_m.fill(-f32.inf)
            running_l.fill(0)
            # Matrix padding stays zero in both probability slots.
            if cutlass.const_expr(not self.pack_probabilities):
                for j in cutlass.range_constexpr(self.rows, width):
                    for slot in cutlass.range_constexpr(self.slots):
                        for sm in cutlass.range_constexpr(self.subtiles):
                            sp[j, local_tid + sm * 128, slot] = dtype(0)
                            if cutlass.const_expr(self.compensate_probabilities):
                                sp_lo[j, local_tid + sm * 128, slot] = dtype(0)
            for block in cutlass.range(begin, end):
                it = block - begin
                slot = it % self.slots
                cute.arch.mbarrier_wait(bs + slot, phase=(it // self.slots) % 2)
                for sm in cutlass.range_constexpr(self.subtiles):
                    current_s = ts[(None, None), 0, 0, slot * self.subtiles + sm]
                    current_copy = tcgen05.make_tmem_copy(
                        copy_atom, current_s
                    ).get_slice(local_tid)
                    sub_scores = cute.make_tensor(scores.iterator + sm * width, rlayout)
                    ex.partition_and_copy(current_copy, current_s, sub_scores)
                    cute.arch.fence_view_async_tmem_load()
                prim.barrier_cta_sync(1, thread_count=128)
                if warp == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bsfree + slot)
                for j in cutlass.range_constexpr(self.rows):
                    local_max = -f32.inf
                    for sm in cutlass.range_constexpr(self.subtiles):
                        score = scores[j, sm] * qk_scale
                        if (
                            block * self.tile + local_tid + sm * 128
                            > length - self.qlen + j // self.heads
                        ):
                            score = -f32.inf
                        scores[j, sm] = score
                        local_max = cute.arch.fmax(local_max, score)
                    maximum = cute.arch.warp_redux_sync(
                        local_max, kind="fmax", nan=True
                    )
                    if lane == j % 32:
                        shared[warp, j] = maximum
                prim.barrier_cta_sync(1, thread_count=128)
                # The value consumer releases both the probability and alpha slot.
                cute.arch.mbarrier_wait(bfree + slot, phase=(it // self.slots + 1) % 2)
                for chunk in cutlass.range_constexpr(width // 32):
                    column = lane + 32 * chunk
                    maximum = running_m[chunk]
                    if column < self.rows:
                        for w in cutlass.range_constexpr(4):
                            maximum = cute.arch.fmax(maximum, shared[w, column])
                    else:
                        maximum = f32(0)
                    # An entirely causal-masked split contributes zero mass.
                    if maximum == -f32.inf:
                        maximum = f32(0)
                    lane_maxima[chunk] = maximum
                    lane_alphas[chunk] = cute.math.exp2(
                        running_m[chunk] - maximum, fastmath=True
                    )
                    running_m[chunk] = maximum
                    if warp == 0:
                        alphas[column, slot] = lane_alphas[chunk]
                for j in cutlass.range_constexpr(self.rows):
                    maximum = cute.arch.shuffle_sync(lane_maxima[j // 32], j % 32)
                    alpha = cute.arch.shuffle_sync(lane_alphas[j // 32], j % 32)
                    total = running_l[j] * alpha
                    for sm in cutlass.range_constexpr(self.subtiles):
                        probability = cute.math.exp2(
                            scores[j, sm] - maximum, fastmath=True
                        )
                        total += probability
                        if cutlass.const_expr(self.pack_probabilities):
                            scores[j, sm] = probability * 256
                        else:
                            high = dtype(probability * 256)
                            sp[j, local_tid + sm * 128, slot] = high
                            if cutlass.const_expr(self.compensate_probabilities):
                                sp_lo[j, local_tid + sm * 128, slot] = dtype(
                                    probability * 256 - f32(high)
                                )
                    running_l[j] = total
                if cutlass.const_expr(self.pack_probabilities):
                    for sm in cutlass.range_constexpr(self.subtiles):
                        for j in cutlass.range_constexpr(self.rows, width):
                            scores[j, sm] = f32(0)
                        packed[None, sm].store(scores[None, sm].load().to(dtype))
                        cute.autovec_copy(
                            packed[None, sm], sp[None, local_tid + sm * 128, slot]
                        )
                        if cutlass.const_expr(self.compensate_probabilities):
                            residual = scores[None, sm].load() - packed[
                                None, sm
                            ].load().to(f32)
                            packed_lo[None, sm].store(residual.to(dtype))
                            cute.autovec_copy(
                                packed_lo[None, sm],
                                sp_lo[None, local_tid + sm * 128, slot],
                            )
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(bp + slot)
            if cutlass.const_expr(self.diagnostic):
                if tid == 0:
                    stamps[split, 2] = cute.arch.globaltimer()
            if cutlass.const_expr(self.parallel_denom):
                for j in cutlass.range_constexpr(self.rows):
                    shared_l[local_tid, j] = running_l[j]
                cute.arch.mbarrier_arrive(bdenom)
            else:
                prim.barrier_cta_sync(1, thread_count=128)
                for j in cutlass.range_constexpr(self.rows):
                    total = running_l[j]
                    for offset in cutlass.range_constexpr(5):
                        total += cute.arch.shuffle_sync_bfly(total, 1 << offset)
                    if lane == j % 32:
                        shared[warp, j] = total
                prim.barrier_cta_sync(1, thread_count=128)
                for j in cutlass.range_constexpr(self.rows):
                    if j < self.rows and local_tid == j:
                        total = f32(0)
                        for w in cutlass.range_constexpr(4):
                            total += shared[w, j]
                        denom[split, j] = total
            if warp == 0:
                for chunk in cutlass.range_constexpr(width // 32):
                    column = lane + 32 * chunk
                    if column < self.rows:
                        maxima[split, column] = running_m[chunk]
            if cutlass.const_expr(self.diagnostic):
                if tid == 0:
                    stamps[split, 3] = cute.arch.globaltimer()
        elif warp < 8:
            value = cute.make_rmem_tensor(rlayout, f32)
            accum = cute.make_rmem_tensor((self.rows, 2), f32)
            accum.fill(0)
            atom_vp = cute.make_mma_atom(vp.op)
            for block in cutlass.range(begin, end):
                it = block - begin
                slot = it % self.slots
                cute.arch.mbarrier_wait(bo + slot, phase=(it // self.slots) % 2)
                for dm in cutlass.range_constexpr(2):
                    o_plane = to[(None, None), 0, 0, 2 * slot + dm]
                    copy_o = tcgen05.make_tmem_copy(copy_atom, o_plane).get_slice(
                        local_tid
                    )
                    ex.partition_and_copy(copy_o, o_plane, value)
                    cute.arch.fence_view_async_tmem_load()
                    for j in cutlass.range_constexpr(self.rows):
                        accum[j, dm] = accum[j, dm] * alphas[j, slot] + value[j]
                prim.barrier_cta_sync(2, thread_count=128)
                if warp == 4:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bfree + slot)
            if cutlass.const_expr(self.diagnostic):
                if tid == 128:
                    stamps[split, 4] = cute.arch.globaltimer()
            for j in cutlass.range_constexpr(self.rows):
                for dm in cutlass.range_constexpr(2):
                    if cutlass.const_expr(self.tma_output):
                        so[j, local_tid + dm * 128] = accum[j, dm] / 256
                    else:
                        partial[split, j, local_tid + dm * 128] = accum[j, dm] / 256
            if cutlass.const_expr(self.tma_output):
                for j in cutlass.range_constexpr(self.rows, width):
                    for dm in cutlass.range_constexpr(2):
                        so[j, local_tid + dm * 128] = f32(0)
                cute.arch.fence_view_async_shared()
            if cutlass.const_expr(self.diagnostic):
                if tid == 128:
                    stamps[split, 5] = cute.arch.globaltimer()
        else:
            if warp == 8:
                q_frag = kq.get_slice(0).partition_B(sq)
                atom_kq = cute.make_mma_atom(kq.op)
                cute.arch.mbarrier_wait(bq, phase=0)
                for block in cutlass.range(begin, end):
                    it = block - begin
                    slot, stage = it % self.slots, it % self.stages
                    cute.arch.mbarrier_wait(
                        bsfree + slot, phase=(it // self.slots + 1) % 2
                    )
                    cute.arch.mbarrier_wait(bk + stage, phase=(it // self.stages) % 2)
                    for sm in cutlass.range_constexpr(self.subtiles):
                        ktile = cute.local_tile(
                            sk[None, None, stage], (128, 256), (sm, 0)
                        )
                        k_frag = kq.get_slice(0).partition_A(ktile)
                        atom_kq.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range_constexpr(cute.size(k_frag, mode=[2])):
                            ex.dot(
                                atom_kq,
                                k_frag[None, None, kk],
                                q_frag[None, None, kk],
                                ts[None, None, None, slot * self.subtiles + sm],
                            )
                            atom_kq.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(bs + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bkfree + stage, None, tcgen05.CtaGroup.ONE)
                if cutlass.const_expr(self.diagnostic):
                    if lane == 0:
                        stamps[split, 6] = cute.arch.globaltimer()
            elif warp == 9:
                atom_vp = cute.make_mma_atom(vp.op)
                for block in cutlass.range(begin, end):
                    it = block - begin
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
                        if cutlass.const_expr(self.compensate_probabilities):
                            p_lo_frag = vp.get_slice(0).partition_B(
                                sp_lo[None, None, slot]
                            )
                            for kk in cutlass.range_constexpr(
                                cute.size(v_frag, mode=[2])
                            ):
                                ex.dot(
                                    atom_vp,
                                    v_frag[None, None, kk],
                                    p_lo_frag[None, None, kk],
                                    to[None, None, None, 2 * slot + dm],
                                )
                    with cute.arch.elect_one():
                        tcgen05.commit(bo + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bvfree + stage, None, tcgen05.CtaGroup.ONE)
                if cutlass.const_expr(self.diagnostic):
                    if lane == 0:
                        stamps[split, 7] = cute.arch.globaltimer()
            elif warp == 10:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(bq, width * 256)
                ex.tma_load(
                    cute.local_tile(q, (width, 256), (0, 0)),
                    sq,
                    bq.value,
                    update_expect_tx=False,
                )
                if cutlass.const_expr(self.dual_tma):
                    for block in cutlass.range(begin, end):
                        it = block - begin
                        stage = it % self.stages
                        cute.arch.mbarrier_wait(
                            bkfree + stage, phase=(it // self.stages + 1) % 2
                        )
                        load_keys(
                            k, table, sk, bk + stage, block, stage, self.tile, length
                        )
                else:
                    for stage in cutlass.range_constexpr(self.stages):
                        if begin + stage < end:
                            load_keys(
                                k,
                                table,
                                sk,
                                bk + stage,
                                begin + stage,
                                stage,
                                self.tile,
                                length,
                            )
                    for block in cutlass.range(begin, end):
                        it = block - begin
                        stage = it % self.stages
                        cute.arch.mbarrier_wait(
                            bvfree + stage, phase=(it // self.stages + 1) % 2
                        )
                        load_values(
                            v, table, sv, bv + stage, block, stage, self.tile, length
                        )
                        if block + self.stages < end:
                            cute.arch.mbarrier_wait(
                                bkfree + stage, phase=(it // self.stages) % 2
                            )
                            load_keys(
                                k,
                                table,
                                sk,
                                bk + stage,
                                block + self.stages,
                                stage,
                                self.tile,
                                length,
                            )
                if cutlass.const_expr(self.diagnostic):
                    if lane == 0:
                        stamps[split, 8] = cute.arch.globaltimer()
            elif warp == 11:
                if cutlass.const_expr(self.dual_tma):
                    for block in cutlass.range(begin, end):
                        it = block - begin
                        stage = it % self.stages
                        cute.arch.mbarrier_wait(
                            bvfree + stage, phase=(it // self.stages + 1) % 2
                        )
                        load_values(
                            v, table, sv, bv + stage, block, stage, self.tile, length
                        )
        if cutlass.const_expr(self.parallel_denom):
            cute.arch.mbarrier_wait(bdenom, phase=0)
        else:
            cute.arch.sync_threads()
        if cutlass.const_expr(self.parallel_denom):
            for chunk in cutlass.range_constexpr(cute.ceil_div(self.rows, 12)):
                column = warp + 12 * chunk
                if column < self.rows:
                    total = (shared_l[lane, column] + shared_l[lane + 32, column]) + (
                        shared_l[lane + 64, column] + shared_l[lane + 96, column]
                    )
                    for offset in cutlass.range_constexpr(5):
                        total += cute.arch.shuffle_sync_bfly(total, 1 << offset)
                    if lane == 0:
                        denom[split, column] = total
        if cutlass.const_expr(self.tma_output):
            cute.arch.sync_threads()
            if warp == 10:
                ex.tma_store(
                    so,
                    cute.local_tile(partial[split, None, None], (width, 256), (0, 0)),
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.diagnostic):
            if tid == 0:
                stamps[split, 9] = cute.arch.globaltimer()


import torch
import triton
import triton.language as tl
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.split_kv_verify import _merge_tile


@triton.jit
def _merge_scaled(
    Partial,
    Maxima,
    Denom,
    Lengths,
    Out,
    VScale,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    value, dims = _merge_tile(
        Partial, Maxima, Denom, Lengths, ROWS, SPLITS, BS, 64, 128, True, True
    )
    row = tl.program_id(0)
    tl.store(Out + row * 256 + dims, value * VScale, dims < 256)


class PagedFP8DecodePlan:
    """Own reusable storage and a compiled producer for one fixed input layout.

    Construct during eager warmup. Calls write into ``output`` without allocating
    device storage or changing its address, including during graph capture.
    """

    def __init__(self, query, kv, table, lengths, *, splits=152):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Paged FP8 storage must be prepared before graph capture"
            )
        self.qlen, self.heads, _ = query.shape
        self.splits = splits
        rows = self.qlen * self.heads
        self.partial = torch.empty(
            (splits, rows, 256), dtype=torch.float32, device=query.device
        )
        self.maximum = torch.empty(
            (splits, rows), dtype=torch.float32, device=query.device
        )
        self.denom = torch.empty_like(self.maximum)
        self.stamps = torch.empty((splits, 10), dtype=torch.int64, device=query.device)
        self.output = torch.empty(
            query.shape, dtype=torch.bfloat16, device=query.device
        )
        kernel = DecoupledPagedDecode(
            self.heads,
            self.qlen,
            splits,
            stages=2,
            dual_tma=False,
            pack_probabilities=True,
            slots=2,
            parallel_denom=True,
            compensate_probabilities=True,
        )
        views = self._views(query, kv, table, lengths)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled = cute.compile(kernel, *views, cutlass.Float32(1.0), stream)

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
            self.stamps,
        )
        return tuple(from_dlpack(t) for t in tensors)

    def run(self, query, kv, table, lengths, *, qk_scale, v_scale, out=None):
        views = self._views(query, kv, table, lengths)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled(*views, cutlass.Float32(qk_scale * math.log2(math.e)), stream)
        return paged_fp8_decode_merge(self, lengths, v_scale=v_scale, out=out)


def paged_fp8_decode_merge(plan, lengths, *, v_scale, out=None):
    """Merge the producer's partials into a stable caller-owned BF16 output."""
    output = plan.output if out is None else out
    rows = plan.qlen * plan.heads
    _merge_scaled[(rows, 4, 1)](
        plan.partial,
        plan.maximum,
        plan.denom,
        lengths,
        output,
        float(v_scale),
        rows,
        plan.splits,
        triton.next_power_of_2(plan.splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def paged_fp8_decode(
    query, kv, table, lengths, plan, *, qk_scale=0.0625, v_scale=1.0, out=None
):
    """Run q1/q7, one-KV-head, D256 attention with a prepared plan."""
    return plan.run(
        query, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out
    )
