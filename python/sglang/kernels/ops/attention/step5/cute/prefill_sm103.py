# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM103 port of vLLM Step5's grouped-union sparse paged prefill.

Source: token_sparse_attn/prefill_union_paged_sm90_gqa.py. Preserve the
512-thread producer/two-consumer topology and shared 64-token union tiles.
The target TP8 geometry is 16 queries, eight heads, and dimension 192.
"""

import math
import operator

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import tcgen05

from . import cute_utils as utils
from .decode_sm103_utils import (
    elem_pointer,
    elem_pointer_i64_offset,
    tcgen_after_sync,
    tcgen_before_sync,
    vector_copy_with_explicit_width,
)


class SparsePrefillSm103:
    """Exact union attention over native token slots; no per-query KV gathers.

    Each group owns a disjoint packed-query range of at most 16 rows. Union
    metadata carries physical/logical eight-token region IDs and membership
    bits. Query positions are explicit, independent of packed row indices.
    All buffers and group boundaries are caller-owned device inputs.
    """

    HEADS = 8
    DIM = 192
    QUERY_GROUP = 16
    MMA_ROWS = 64
    BLOCK_N = 64
    STAGES = 2
    COMPUTE_GROUPS = 2
    PRODUCER_THREADS = 256
    THREADS = 512
    TMEM_COLUMNS = 512
    TMEM_GROUP_COLUMNS = 256

    def __init__(self, union_capacity: int, *, logical_q_heads: int = 8):
        if union_capacity < 1 or union_capacity > 16 * 513:
            raise ValueError("Step5 prefill union capacity must be in [1,8208]")
        if logical_q_heads not in (8, 16):
            raise ValueError("Step5 prefill Q heads must be 8 or 16")
        self.logical_q_heads = logical_q_heads
        self.union_capacity = union_capacity
        self.dtype = cutlass.BFloat16

    def _mma(self):
        qk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, 64),
        )
        pv = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, 64),
        )
        return qk, pv

    def _layouts(self, qk, pv):
        return (
            sm100_utils.make_smem_layout_a(qk, (64, 64, 192), self.dtype, 2),
            sm100_utils.make_smem_layout_b(qk, (64, 64, 192), self.dtype, 2),
            sm100_utils.make_smem_layout_b(pv, (64, 64, 64), self.dtype, 6),
            sm100_utils.make_smem_layout_a(pv, (64, 64, 64), self.dtype, 2),
        )

    def _storage(self, layouts):
        q_storage, k_storage, v_storage, p_storage = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024
            ]
            for layout in layouts
        ]
        barriers = cute.struct.MemRange[cutlass.Int64, 2]
        metadata = cute.struct.MemRange[Int32, 128]
        heads = cute.struct.MemRange[Float32, 128]
        score_storage = cute.struct.MemRange[Float32, 2 * 64 * 64]
        scalar = cute.struct.MemRange[Int32, 1]

        @cute.struct
        class Storage:
            free: barriers
            ready_k: barriers
            ready_v: barriers
            ready_metadata: barriers
            mma: barriers
            tmem: scalar
            q: q_storage
            k: k_storage
            v: v_storage
            p: p_storage
            tokens: metadata
            masks: metadata
            maxima: heads
            sums: heads
            scales: heads
            scores: score_storage

        return Storage

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        physical: cute.Tensor,
        logical: cute.Tensor,
        counts: cute.Tensor,
        membership: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        positions: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        scale: Float32,
        stream,
    ):
        qk, pv = self._mma()
        layouts = self._layouts(qk, pv)
        storage = self._storage(layouts)
        self.kernel(
            q,
            k,
            v,
            physical,
            logical,
            counts,
            membership,
            starts,
            ends,
            positions,
            out,
            lse,
            scale * Float32(math.log2(math.e)),
            qk,
            pv,
            *layouts,
            storage,
        ).launch(
            grid=(counts.shape[0], self.logical_q_heads // self.HEADS, 1),
            block=(512, 1, 1),
            smem=storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        physical: cute.Tensor,
        logical: cute.Tensor,
        counts: cute.Tensor,
        membership: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        positions: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        scale: Float32,
        qk: cute.TiledMma,
        pv: cute.TiledMma,
        q_layout,
        k_layout,
        v_layout,
        p_layout,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        group, head_tile, _ = cute.arch.block_idx()
        # Preserve the reference eight-head MMA tile. Independent CTAs own
        # disjoint output heads; union metadata and KV are read-only and shared.
        head_offset = head_tile * Int32(self.HEADS)
        q = cute.domain_offset((0, head_offset, 0), q)
        out = cute.domain_offset((0, head_offset, 0), out)
        lse = cute.domain_offset((0, head_offset), lse)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        sq = storage.q.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        sk = storage.k.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        sv = storage.v.get_tensor(v_layout.outer, swizzle=v_layout.inner)
        sp = storage.p.get_tensor(p_layout.outer, swizzle=p_layout.inner)
        tokens = storage.tokens.get_tensor(cute.make_layout((64, 2), stride=(1, 64)))
        masks = storage.masks.get_tensor(cute.make_layout((64, 2), stride=(1, 64)))
        maxima = storage.maxima.get_tensor(cute.make_layout((64, 2), stride=(1, 64)))
        sums = storage.sums.get_tensor(cute.make_layout((64, 2), stride=(1, 64)))
        scales = storage.scales.get_tensor(cute.make_layout((64, 2), stride=(1, 64)))
        scores = storage.scores.get_tensor(
            cute.make_layout((64, 64, 2), stride=(1, 64, 4096))
        )
        free = storage.free.data_ptr()
        ready_k = storage.ready_k.data_ptr()
        ready_v = storage.ready_v.data_ptr()
        ready_metadata = storage.ready_metadata.data_ptr()
        mma = storage.mma.data_ptr()
        tmem = storage.tmem.data_ptr()
        if tid < 2:
            # Each of the two 128-thread consumers retires its own reads.
            cute.arch.mbarrier_init(free + tid, Int32(256))
            # 256 asynchronous-copy arrivals plus one ordinary release that
            # publishes shared metadata and synchronous zero-padding stores.
            cute.arch.mbarrier_init(ready_k + tid, Int32(257))
            cute.arch.mbarrier_init(ready_v + tid, Int32(257))
            cute.arch.mbarrier_init(ready_metadata + tid, Int32(256))
            cute.arch.mbarrier_init(mma + tid, Int32(1))
        if tid < 128:
            row, wg = tid % 64, tid // 64
            maxima[row, wg] = -Float32.inf
            sums[row, wg] = Float32(0)
            scales[row, wg] = Float32(0)
        if warp == 8:
            cute.arch.alloc_tmem(self.TMEM_COLUMNS, tmem, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        count = Int32(counts[group])
        count = cutlass.min(cutlass.max(count, Int32(0)), Int32(self.union_capacity))
        tiles = (count + Int32(7)) // Int32(8)
        if warp < 8:
            cute.arch.warpgroup_reg_dealloc(24)
            # A partial current region contains unwritten native KV slots.
            # Zero softmax probability does not suppress NaN V in the PV MMA.
            last_position = Int32(-1)
            for offset in cutlass.range_constexpr(16):
                query = starts[group] + Int32(offset)
                if query >= 0 and query < ends[group] and query < positions.shape[0]:
                    last_position = cutlass.max(last_position, positions[query])
            self._produce(
                k,
                v,
                physical,
                logical,
                membership,
                sk,
                sv,
                tokens,
                masks,
                free,
                ready_k,
                ready_v,
                ready_metadata,
                k_layout,
                v_layout,
                group,
                count,
                tiles,
                tid,
                last_position,
            )
        else:
            cute.arch.warpgroup_reg_alloc(232)
            wg = cute.arch.make_warp_uniform((tid - Int32(256)) // Int32(128))
            thread = tid - Int32(256) - wg * Int32(128)
            self._consume(
                q,
                positions,
                out,
                lse,
                sq,
                sk,
                sv,
                sp,
                tokens,
                masks,
                maxima,
                sums,
                scales,
                scores,
                free,
                ready_k,
                ready_v,
                ready_metadata,
                mma,
                tmem,
                qk,
                pv,
                q_layout,
                p_layout,
                starts[group],
                ends[group],
                scale,
                tiles,
                wg,
                thread,
            )
            # Both consumers retire TMEM before the allocating warp frees it.
            cute.arch.barrier(barrier_id=7, number_of_threads=Int32(256))
            if warp == 8:
                tcgen_after_sync()
                pointer = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem)
                cute.arch.dealloc_tmem(pointer, self.TMEM_COLUMNS, is_two_cta=False)

    @cute.jit
    def _produce(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        physical: cute.Tensor,
        logical: cute.Tensor,
        membership: cute.Tensor,
        sk: cute.Tensor,
        sv: cute.Tensor,
        tokens: cute.Tensor,
        masks: cute.Tensor,
        free,
        ready_k,
        ready_v,
        ready_metadata,
        k_layout,
        v_layout,
        group: Int32,
        count: Int32,
        tiles: Int32,
        tid: Int32,
        last_position: Int32,
    ):
        # Source topology: 32 eight-lane producer groups, two token rows each.
        subgroup, lane = tid // Int32(8), tid % Int32(8)
        col = lane * Int32(8)
        physical_tokens = cute.make_rmem_tensor((2,), Int32)
        vector_layout = cute.make_layout((8,), stride=(1,))
        for tile in cutlass.range(tiles, unroll=1):
            stage = tile % Int32(2)
            if tile >= Int32(2):
                # Every writer acquires completion of all consumer reads
                # before reusing this metadata and K/V stage.
                cute.arch.mbarrier_wait(free + stage, ((tile - 2 - stage) // 2) & 1)
            for local in cutlass.range_constexpr(2):
                row = subgroup + Int32(local * 32)
                region = tile * Int32(8) + row // Int32(8)
                physical_token = Int32(-1)
                if lane == 0:
                    logical_token, mask = Int32(-1), Int32(0)
                    if region < count:
                        phys = physical[group, region]
                        log = logical[group, region]
                        if phys >= 0 and phys < Int32(k.shape[0]) // Int32(8):
                            if log >= 0:
                                logical_token = log * Int32(8) + row % Int32(8)
                                if logical_token <= last_position:
                                    physical_token = phys * Int32(8) + row % Int32(8)
                                    mask = membership[group, region]
                    tokens[row, stage] = logical_token
                    masks[row, stage] = mask
                physical_token = cute.arch.shuffle_sync(
                    physical_token, (tid % Int32(32)) - lane
                )
                physical_tokens[local] = physical_token
                for slab in cutlass.range_constexpr(3):
                    dim = col + Int32(slab * 64)
                    pointer = elem_pointer(sk, ((row, dim % 16), 0, dim // 16, stage))
                    pointer = cute.make_ptr(
                        self.dtype, pointer.toint(), sk.memspace, assumed_align=16
                    )
                    pointer = cute.recast_ptr(pointer, k_layout.inner)
                    dest = cute.make_tensor(pointer, vector_layout)
                    if physical_token >= 0:
                        source = cute.make_tensor(
                            elem_pointer_i64_offset(k, (physical_token, 0, dim)),
                            vector_layout,
                        )
                        vector_copy_with_explicit_width(
                            source, dest, num_copy_elems=8, is_async=True
                        )
                    else:
                        for i in cutlass.range_constexpr(8):
                            dest[i] = self.dtype(0)
            cute.arch.cp_async_commit_group()
            # Ordinary metadata stores have their own release/acquire edge,
            # independent of asynchronous K/V completion. Every writer arrives.
            cute.arch.mbarrier_arrive(ready_metadata + stage)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=5, number_of_threads=Int32(256))
            if tid == 0:
                cute.arch.mbarrier_arrive(ready_k + stage)
            cute.arch.cp_async_mbarrier_arrive_noinc(ready_k + stage)
            for local in cutlass.range_constexpr(2):
                row = subgroup + Int32(local * 32)
                physical_token = physical_tokens[local]
                for slab in cutlass.range_constexpr(3):
                    pointer = elem_pointer(
                        sv, ((col, row % 16), 0, row // 16, stage + Int32(slab * 2))
                    )
                    pointer = cute.make_ptr(
                        self.dtype, pointer.toint(), sv.memspace, assumed_align=16
                    )
                    pointer = cute.recast_ptr(pointer, v_layout.inner)
                    dest = cute.make_tensor(pointer, vector_layout)
                    if physical_token >= 0:
                        source = cute.make_tensor(
                            elem_pointer_i64_offset(
                                v, (physical_token, 0, col + Int32(slab * 64))
                            ),
                            vector_layout,
                        )
                        vector_copy_with_explicit_width(
                            source, dest, num_copy_elems=8, is_async=True
                        )
                    else:
                        for i in cutlass.range_constexpr(8):
                            dest[i] = self.dtype(0)
            cute.arch.cp_async_commit_group()
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=5, number_of_threads=Int32(256))
            if tid == 0:
                cute.arch.mbarrier_arrive(ready_v + stage)
            cute.arch.cp_async_mbarrier_arrive_noinc(ready_v + stage)

    @cute.jit
    def _softmax(
        self,
        scores: cute.Tensor,
        maxima: cute.Tensor,
        sums: cute.Tensor,
        scales: cute.Tensor,
        scale: Float32,
        wg: Int32,
        thread: Int32,
    ):
        lane, warp = thread % Int32(32), thread // Int32(32)
        for slot in cutlass.range_constexpr(16):
            row = warp * Int32(16) + Int32(slot)
            maximum = cute.arch.fmax(scores[lane, row, wg], scores[lane + 32, row, wg])
            maximum = cute.arch.warp_reduction(maximum, cute.arch.fmax)
            old_max = maxima[row, wg]
            maximum = cute.arch.fmax(old_max, maximum)
            # Zero is only an exponent guard. Persisting it for a masked tile
            # can erase valid negative-logit state or underflow a later tile.
            safe_max = Float32(0) if maximum == -Float32.inf else maximum
            alpha = utils.exp2f((old_max - safe_max) * scale)
            total = Float32(0)
            for half in cutlass.range_constexpr(2):
                tok = lane + Int32(half * 32)
                probability = utils.exp2f(
                    scores[tok, row, wg] * scale - safe_max * scale
                )
                scores[tok, row, wg] = probability
                total += probability
            total = cute.arch.warp_reduction(total, operator.add)
            # Every lane has consumed the old shared maximum before lane zero
            # publishes the next one. Shuffle reductions are not memory fences.
            cute.arch.sync_warp()
            if lane == 0:
                maxima[row, wg] = maximum
                scales[row, wg] = alpha
                sums[row, wg] = sums[row, wg] * alpha + total
        cute.arch.barrier(barrier_id=8 + wg, number_of_threads=Int32(128))

    @cute.jit
    def _consume(
        self,
        q: cute.Tensor,
        positions: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        sq: cute.Tensor,
        sk: cute.Tensor,
        sv: cute.Tensor,
        sp: cute.Tensor,
        tokens: cute.Tensor,
        masks: cute.Tensor,
        maxima: cute.Tensor,
        sums: cute.Tensor,
        scales: cute.Tensor,
        scores: cute.Tensor,
        free,
        ready_k,
        ready_v,
        ready_metadata,
        mma,
        tmem,
        qk: cute.TiledMma,
        pv: cute.TiledMma,
        q_layout,
        p_layout,
        first: Int32,
        end: Int32,
        scale: Float32,
        tiles: Int32,
        wg: Int32,
        thread: Int32,
    ):
        q_tile = sq[None, None, None, wg]
        p_tile = sp[None, None, None, wg]
        vector_layout = cute.make_layout((8,), stride=(1,))
        for index in cutlass.range(thread, 64 * 192 // 8, 128, unroll=1):
            row, dim = index // Int32(24), index % Int32(24) * Int32(8)
            query = first + wg * Int32(8) + row // Int32(8)
            head = row % Int32(8)
            pointer = elem_pointer(q_tile, ((row, dim % 16), 0, dim // 16))
            pointer = cute.make_ptr(
                self.dtype, pointer.toint(), q_tile.memspace, assumed_align=16
            )
            pointer = cute.recast_ptr(pointer, q_layout.inner)
            dest = cute.make_tensor(pointer, vector_layout)
            if query >= 0 and query < end and query < Int32(q.shape[0]):
                source = cute.make_tensor(
                    elem_pointer_i64_offset(q, (query, head, dim)), vector_layout
                )
                vector_copy_with_explicit_width(source, dest, num_copy_elems=8)
            else:
                for i in cutlass.range_constexpr(8):
                    dest[i] = self.dtype(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=8 + wg, number_of_threads=Int32(128))
        # A dynamic consumer index must retain the allocation's alignment in
        # CuTe's type system; each consumer owns a 256-column TMEM interval.
        offset = cute.assume(wg * Int32(256), divby=256)
        pointer = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem) + offset
        mma_thread = qk.get_slice(0)
        shape = mma_thread.partition_shape_C((64, 64))
        layout = mma_thread.make_fragment_C(shape).layout
        acc = cute.make_tensor(pointer, layout)
        acc_mn = acc[((None, None), 0, 0)]
        t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld16x256bOp(), Float32), acc_mn
        )
        copy = t2r.get_slice(thread)
        coords = copy.partition_D(cute.make_identity_tensor((64, 64)))
        r_score = cute.make_rmem_tensor(coords.shape, Float32)
        r_value = cute.make_rmem_tensor(coords.shape, Float32)
        r_out = cute.make_rmem_tensor(cute.append(coords.shape, 3), Float32)
        r_out.fill(Float32(0))
        columns = const_expr(cute.size(acc_mn.shape[1]))
        assert 3 * columns <= self.TMEM_GROUP_COLUMNS
        fq, fk = qk.make_fragment_A(q_tile), qk.make_fragment_B(sk)
        fp, fv = pv.make_fragment_A(p_tile), pv.make_fragment_B(sv)
        phase = Int32(0)
        for tile in cutlass.range(tiles, unroll=1):
            stage = tile % Int32(2)
            ready_phase = ((tile - stage) // Int32(2)) & Int32(1)
            cute.arch.mbarrier_wait(ready_metadata + stage, ready_phase)
            cute.arch.mbarrier_wait(ready_k + stage, ready_phase)
            cute.arch.fence_view_async_shared()
            if thread // Int32(32) == 0:
                tcgen_after_sync()
                qk.set(tcgen05.Field.ACCUMULATE, False)
                for kk in cutlass.range_constexpr(12):
                    cute.gemm(
                        qk, acc, fq[None, None, kk], fk[None, None, kk, stage], acc
                    )
                    qk.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mma + wg)
            cute.arch.mbarrier_wait(mma + wg, phase)
            phase ^= Int32(1)
            tcgen_after_sync()
            cute.copy(t2r, copy.partition_S(acc_mn), r_score)
            cute.arch.fence_view_async_tmem_load()
            tcgen_before_sync()
            for i in cutlass.range_constexpr(cute.size(r_score)):
                row, tok = coords[i]
                slot = wg * Int32(8) + row // Int32(8)
                query = first + slot
                value = -Float32.inf
                if query >= 0 and query < end and query < Int32(q.shape[0]):
                    logical_token = tokens[tok, stage]
                    selected = (masks[tok, stage] & (Int32(1) << slot)) != Int32(0)
                    if (
                        selected
                        and logical_token >= 0
                        and logical_token <= positions[query]
                    ):
                        value = r_score[i]
                scores[tok, row, wg] = value
            cute.arch.barrier(barrier_id=8 + wg, number_of_threads=Int32(128))
            self._softmax(scores, maxima, sums, scales, scale, wg, thread)
            for index in cutlass.range(thread, 64 * 64, 128, unroll=1):
                row, tok = index // Int32(64), index % Int32(64)
                p_tile[((row, tok % 16), 0, tok // 16)] = self.dtype(
                    scores[tok, row, wg]
                )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=8 + wg, number_of_threads=Int32(128))
            cute.arch.mbarrier_wait(ready_v + stage, ready_phase)
            if thread // Int32(32) == 0:
                tcgen_after_sync()
                for slab in cutlass.range_constexpr(3):
                    acc_pv = cute.make_tensor(pointer + slab * columns, layout)
                    pv.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(4):
                        cute.gemm(
                            pv,
                            acc_pv,
                            fp[None, None, kk],
                            fv[None, None, kk, stage + Int32(slab * 2)],
                            acc_pv,
                        )
                        pv.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mma + wg)
            cute.arch.mbarrier_wait(mma + wg, phase)
            phase ^= Int32(1)
            tcgen_after_sync()
            for slab in cutlass.range_constexpr(3):
                tile_pv = cute.make_tensor(pointer + slab * columns, acc_mn.layout)
                cute.copy(t2r, copy.partition_S(tile_pv), r_value)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(cute.size(r_value)):
                    row, _ = coords[i]
                    prior = r_out[None, None, None, slab]
                    prior[i] = prior[i] * scales[row, wg] + r_value[i]
            tcgen_before_sync()
            cute.arch.barrier(barrier_id=8 + wg, number_of_threads=Int32(128))
            if tile + Int32(2) < tiles:
                cute.arch.mbarrier_arrive(free + stage)
        if thread < 64:
            denom = sums[thread, wg]
            valid_sum = denom != Float32(0) and denom == denom
            scales[thread, wg] = cute.arch.rcp_approx(
                denom if valid_sum else Float32(1)
            )
            query = first + wg * Int32(8) + thread // Int32(8)
            if query >= 0 and query < end and query < Int32(q.shape[0]):
                value = -Float32.inf
                if valid_sum:
                    value = (
                        maxima[thread, wg] * scale + utils.log2f(denom)
                    ) * math.log(2.0)
                lse[query, thread % Int32(8)] = value
        cute.arch.barrier(barrier_id=8 + wg, number_of_threads=Int32(128))
        for slab in cutlass.range_constexpr(3):
            prior = r_out[None, None, None, slab]
            for i in cutlass.range_constexpr(cute.size(r_score)):
                row, dim = coords[i]
                query = first + wg * Int32(8) + row // Int32(8)
                if query >= 0 and query < end and query < Int32(q.shape[0]):
                    out[query, row % Int32(8), dim + Int32(slab * 64)] = self.dtype(
                        prior[i] * scales[row, wg]
                    )
