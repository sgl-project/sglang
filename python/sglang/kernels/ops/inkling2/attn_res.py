# ruff: noqa: SIM102

import operator
from functools import cache
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr, range_constexpr
from cutlass.cute.arch.nvvm_wrappers import warp_reduction
from cutlass.cute.runtime import make_fake_compact_tensor

from sglang.kernels.ops.attention.flash_attn.cute.copy_utils import (
    store_shared_remote_fp32x4,
)
from sglang.kernels.ops.inkling2.cute_utils import (
    domain_offset_raw_aligned,
)
from sglang.kernels.ops.inkling2.primitives import (
    fadd_rn_f32,
    fma_rn_f32,
    fmul_rn_f32,
)

_TORCH_TO_CUTE = {
    torch.bfloat16: cute.BFloat16,
    torch.float32: cute.Float32,
    torch.int64: cute.Int64,
}


class AttnResFwdKernel:
    def __init__(
        self,
        D: int,
        N_Q: int,
        *,
        dtype: torch.dtype,
        normalize_values: bool,
        has_partial_addend: bool,
        num_compute_threads: int,
        num_g2s_stages: int,
        vec_size: int,
        reload_X: bool,
        cluster_size: int = 1,
        max_N: int = 12,
    ):
        assert cluster_size in (1, 2), f"cluster_size={cluster_size!r} must be 1 or 2"
        assert D % cluster_size == 0, (
            f"D={D!r} must be divisible by cluster_size={cluster_size!r}"
        )
        D_local = D // cluster_size
        assert D_local % num_compute_threads == 0, (
            f"D/{cluster_size}={D_local} must be divisible by num_compute_threads={num_compute_threads!r}"
        )
        if cluster_size > 1:
            assert D_local * _TORCH_TO_CUTE[dtype].width // 8 % 16 == 0, (
                f"D/{cluster_size}={D_local} rows of {dtype} must be 16-byte aligned"
            )
        assert 1 <= N_Q <= 3, f"N_Q={N_Q!r} must be in [1, 3]"
        assert dtype in (torch.bfloat16, torch.float32), f"unsupported dtype={dtype!r}"
        self.D = D
        self.D_local = D_local
        self.cluster_size = cluster_size
        self.D_rcp = 1 / D
        self.N_Q = N_Q
        self.num_g2s_stages = num_g2s_stages
        assert num_g2s_stages >= 2, f"num_g2s_stages={num_g2s_stages!r} must be >= 2"
        self.num_compute_threads = num_compute_threads
        self.num_load_warps = 1
        self.num_compute_warps = num_compute_threads // 32
        self.load_warp_idx = self.num_compute_warps
        self.num_warps = self.num_load_warps + self.num_compute_warps
        self.num_threads = 32 * self.num_warps
        self.elem_per_thread = D_local // num_compute_threads
        self.vec_size = vec_size
        self.vec_chunks = D_local // (num_compute_threads * self.vec_size)
        self.tail_elems = self.elem_per_thread - self.vec_chunks * self.vec_size
        self.tail_base = self.vec_chunks * self.vec_size
        self.dtype = _TORCH_TO_CUTE[dtype]
        self.X_load_bytes = D_local * self.dtype.width // 8
        self.NORMALIZE_VALUES = normalize_values
        self.HAS_PARTIAL_ADDEND = has_partial_addend
        self.VECTORIZE_PARTIAL_ADDEND = has_partial_addend
        self.reload_X = reload_X
        self.MAX_N = max_N
        self.xchg_floats = self.MAX_N * (self.N_Q + 1)
        self.xchg_vecs = self.xchg_floats // 4
        self.xchg_tx_bytes = self.xchg_floats * 4
        if cluster_size > 1:
            assert self.xchg_floats % 4 == 0, (
                f"exchange payload of {self.xchg_floats} floats must be a multiple of 4 (v4 sends)"
            )
            assert self.xchg_vecs <= 32, "exchange must fit in one warp's lanes"

    @cute.jit
    def local_row(self, row: cute.Tensor) -> cute.Tensor:
        if const_expr(self.cluster_size == 1):
            return row
        return cute.flat_divide(row, (self.D_local,))[None, 0]

    @cute.jit
    def block_offset(self, offsets: tuple, n: Int32) -> Int64:
        offset = Int64(offsets[0])
        for i in range_constexpr(1, len(offsets)):
            if n == i:
                offset = Int64(offsets[i])
        return offset

    @cute.jit
    def g2s_load_X_row(
        self,
        mBlockBase: cute.Tensor,
        block_offset: Int64,
        sX: cute.Tensor,
        bid: Int32,
        half_off: Int64,
        pipeline: cutlass.pipeline.PipelineTmaAsync,
        producer_state: cutlass.pipeline.PipelineState,
        copy_atom: cute.CopyAtom,
    ) -> cutlass.pipeline.PipelineState:
        block_off = Int64(block_offset) + half_off
        mXBlock = domain_offset_raw_aligned(block_off, mBlockBase, 16)
        mXRow = self.local_row(mXBlock[bid, None])
        sXRow = sX[None, producer_state.index]
        mbar_ptr = pipeline.producer_get_barrier(producer_state)
        pipeline.producer_acquire(producer_state)
        with cute.arch.elect_one():
            cute.copy(copy_atom, mXRow, sXRow, mbar_ptr=mbar_ptr)
        pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def g2s_load_A(
        self,
        mPartialAddend: cute.Tensor,
        sX: cute.Tensor,
        bid: Int32,
        pipeline: cutlass.pipeline.PipelineTmaAsync,
        producer_state: cutlass.pipeline.PipelineState,
        copy_atom: cute.CopyAtom,
    ) -> cutlass.pipeline.PipelineState:
        mARow = self.local_row(mPartialAddend[bid, None])
        sARow = sX[None, producer_state.index]
        mbar_ptr = pipeline.producer_get_barrier(producer_state)
        pipeline.producer_acquire(producer_state)
        with cute.arch.elect_one():
            cute.copy(copy_atom, mARow, sARow, mbar_ptr=mbar_ptr)
        pipeline.producer_commit(producer_state)
        producer_state.advance()
        return producer_state

    @cute.jit
    def compute_qk(
        self,
        sX_vec: cute.Tensor,
        sX_unvec: cute.Tensor,
        rQ_vec: cute.Tensor,
        rQ: cute.Tensor,
        sRed: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineTmaAsync,
        consumer_state: cutlass.pipeline.PipelineState,
        tidx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        n: Int32,
        use_addend: cutlass.Constexpr[bool],
    ):
        rXVec = cute.make_rmem_tensor(self.vec_size, sX_vec.element_type)
        rSumSq = Float32(0.0)
        rDot = cute.make_rmem_tensor(self.N_Q, Float32)
        rDot.fill(0.0)
        rPartialAddendVec = None
        if const_expr(use_addend):
            rPartialAddendVec = cute.make_rmem_tensor(
                self.vec_size, sX_vec.element_type
            )
        pipeline.consumer_wait(consumer_state)
        addend_consumer_state = consumer_state.clone()
        if const_expr(use_addend):
            addend_consumer_state.advance()
            pipeline.consumer_wait(addend_consumer_state)
        for chunk in range_constexpr(self.vec_chunks):
            cute.autovec_copy(sX_vec[None, tidx, chunk, consumer_state.index], rXVec)
            if const_expr(self.VECTORIZE_PARTIAL_ADDEND and use_addend):
                cute.autovec_copy(
                    sX_vec[None, tidx, chunk, addend_consumer_state.index],
                    rPartialAddendVec,
                )
            for v in range_constexpr(self.vec_size):
                x = Float32(rXVec[v])
                if const_expr(use_addend):
                    if const_expr(self.VECTORIZE_PARTIAL_ADDEND):
                        x = fadd_rn_f32(x, Float32(rPartialAddendVec[v]))
                    else:
                        x = fadd_rn_f32(
                            x,
                            Float32(
                                sX_vec[v, tidx, chunk, addend_consumer_state.index]
                            ),
                        )
                rSumSq = fma_rn_f32(x, x, rSumSq)
                for q in range_constexpr(self.N_Q):
                    q_val = rQ_vec[v, chunk, q]
                    rDot[q] = fma_rn_f32(x, q_val, rDot[q])
        for t in range_constexpr(self.tail_elems):
            x = sX_unvec[tidx, self.tail_base + t, consumer_state.index]
            if const_expr(use_addend):
                x = fadd_rn_f32(
                    x, sX_unvec[tidx, self.tail_base + t, addend_consumer_state.index]
                )
            rSumSq = fma_rn_f32(x, x, rSumSq)
            for q in range_constexpr(self.N_Q):
                q_val = rQ[self.tail_base + t, q]
                rDot[q] = fma_rn_f32(x, q_val, rDot[q])
        if const_expr(self.reload_X):
            cute.arch.sync_warp()
            cute.arch.nvvm_wrappers.fence_view_async_shared()
            pipeline.consumer_release(consumer_state)
            if const_expr(use_addend):
                pipeline.consumer_release(addend_consumer_state)
        if const_expr(use_addend):
            consumer_state = addend_consumer_state
        consumer_state.advance()
        warp_sum_sq = warp_reduction(rSumSq, operator.add)
        if lane_idx == 0:
            sRed[warp_idx, 0, n] = warp_sum_sq
        for q in range_constexpr(self.N_Q):
            warp_dot = warp_reduction(rDot[q], operator.add)
            if lane_idx == 0:
                sRed[warp_idx, q + 1, n] = warp_dot
        return consumer_state

    @cute.jit
    def compute_pv(
        self,
        sX_vec: cute.Tensor,
        sX_unvec: cute.Tensor,
        sExp: cute.Tensor,
        sStat: cute.Tensor,
        rOut: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineTmaAsync,
        consumer_state: cutlass.pipeline.PipelineState,
        tidx: Int32,
        n: Int32,
        use_addend: cutlass.Constexpr[bool],
    ) -> cutlass.pipeline.PipelineState:
        rXVec = cute.make_rmem_tensor(self.vec_size, sX_vec.dtype)
        rPartialAddendVec = None
        if const_expr(use_addend):
            assert const_expr(self.HAS_PARTIAL_ADDEND)
            rPartialAddendVec = cute.make_rmem_tensor(self.vec_size, sX_vec.dtype)
        addend_consumer_state = consumer_state.clone()
        if const_expr(self.reload_X):
            pipeline.consumer_wait(consumer_state)
        if const_expr(use_addend):
            addend_consumer_state.advance()
            if const_expr(self.reload_X):
                pipeline.consumer_wait(addend_consumer_state)
        for chunk in range_constexpr(self.vec_chunks):
            cute.autovec_copy(sX_vec[None, tidx, chunk, consumer_state.index], rXVec)
            if const_expr(self.VECTORIZE_PARTIAL_ADDEND and use_addend):
                cute.autovec_copy(
                    sX_vec[None, tidx, chunk, addend_consumer_state.index],
                    rPartialAddendVec,
                )
            for v in range_constexpr(self.vec_size):
                idx = chunk * self.vec_size + v
                x = Float32(rXVec[v])
                if const_expr(use_addend):
                    if const_expr(self.VECTORIZE_PARTIAL_ADDEND):
                        x = fadd_rn_f32(x, Float32(rPartialAddendVec[v]))
                    else:
                        x = fadd_rn_f32(
                            x, sX_vec[v, tidx, chunk, addend_consumer_state.index]
                        )
                for q in range_constexpr(self.N_Q):
                    prob = sExp[n, q]
                    if const_expr(self.NORMALIZE_VALUES):
                        prob = fmul_rn_f32(prob, sStat[n, 0])
                    rOut[q, idx] = fma_rn_f32(prob, x, rOut[q, idx])
        for t in range_constexpr(self.tail_elems):
            idx = self.tail_base + t
            x = Float32(sX_unvec[tidx, self.tail_base + t, consumer_state.index])
            if const_expr(self.HAS_PARTIAL_ADDEND and use_addend):
                x = fadd_rn_f32(
                    x, sX_unvec[tidx, self.tail_base + t, addend_consumer_state.index]
                )
            for q in range_constexpr(self.N_Q):
                prob = sExp[n, q]
                if const_expr(self.NORMALIZE_VALUES):
                    prob = fmul_rn_f32(prob, sStat[n, 0])
                rOut[q, idx] = fma_rn_f32(prob, x, rOut[q, idx])
        cute.arch.sync_warp()
        cute.arch.nvvm_wrappers.fence_view_async_shared()
        pipeline.consumer_release(consumer_state)
        if const_expr(use_addend):
            pipeline.consumer_release(addend_consumer_state)
            consumer_state = addend_consumer_state
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def __call__(
        self,
        mBlockBase,
        mBlockOffs,
        mPartialAddend,
        mPseudoQuery,
        mMergedOutput,
        eps: Float32,
        T: Int32,
        N: Int32,
        stream,
    ):
        self.X_smem_layout_staged = cute.make_layout(
            (self.D_local, self.num_g2s_stages), stride=(1, self.D_local)
        )
        self.reduction_smem_layout = cute.make_layout(
            (self.num_compute_warps, self.N_Q + 1, self.MAX_N),
            stride=(1, self.num_compute_warps, (self.N_Q + 1) * self.num_compute_warps),
        )
        self.stat_smem_layout = cute.make_layout(
            (self.MAX_N, self.N_Q + 1), stride=(self.N_Q + 1, 1)
        )
        self.inv_sum_smem_layout = cute.make_layout((self.N_Q,))
        self.exp_smem_layout = cute.make_layout(
            (self.MAX_N, self.N_Q), stride=(self.N_Q, 1)
        )
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            X_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.num_g2s_stages]
            xchg_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            sRed: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(self.reduction_smem_layout)
            ]
            sStat: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(self.stat_smem_layout)
                ],
                16,
            ]
            sXchg: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.xchg_floats], 16
            ]
            sInvSum: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(self.inv_sum_smem_layout)
            ]
            sExp: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(self.exp_smem_layout)
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype, max(cute.cosize(self.X_smem_layout_staged), 0)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        shared_storage_size = self.shared_storage.size_in_bytes()
        self.kernel(
            mBlockBase,
            mBlockOffs,
            mPartialAddend,
            mPseudoQuery,
            mMergedOutput,
            self.X_smem_layout_staged,
            self.reduction_smem_layout,
            self.stat_smem_layout,
            self.inv_sum_smem_layout,
            self.exp_smem_layout,
            eps,
            N,
        ).launch(
            grid=[T * self.cluster_size * 1, 1, 1],
            block=[self.num_threads, 1, 1],
            cluster=[self.cluster_size, 1, 1] if self.cluster_size > 1 else None,
            smem=shared_storage_size,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockBase: cute.Tensor,
        mBlockOffs: tuple,
        mPartialAddend: Optional[cute.Tensor],
        mPseudoQuery: cute.Tensor | tuple[cute.Tensor, ...],
        mMergedOutput: Optional[cute.Tensor],
        X_smem_layout_staged: cute.Layout,
        reduction_smem_layout: cute.Layout,
        stat_smem_layout: cute.Layout,
        inv_sum_smem_layout: cute.Layout,
        exp_smem_layout: cute.Layout,
        eps: Float32,
        N: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        cta_rank = Int32(0)
        if const_expr(self.cluster_size > 1):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        peer_rank = cta_rank ^ 1
        half_off = Int64(cta_rank) * Int64(self.D_local)
        bid = cute.arch.make_warp_uniform(
            Int32(cute.arch.block_idx()[0]) // self.cluster_size
        )
        mPartialAddendL = mPartialAddend
        mPseudoQueryL = mPseudoQuery
        mMergedOutputL = mMergedOutput
        if const_expr(self.cluster_size > 1):
            if const_expr(mPartialAddend is not None):
                mPartialAddendL = domain_offset_raw_aligned(
                    half_off, mPartialAddend, 16
                )
            mPseudoQueryL = domain_offset_raw_aligned(half_off, mPseudoQuery, 16)
            if const_expr(mMergedOutput is not None):
                mMergedOutputL = domain_offset_raw_aligned(half_off, mMergedOutput, 16)
        if const_expr(self.HAS_PARTIAL_ADDEND):
            assert mPartialAddend is not None
        else:
            assert mPartialAddend is None
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sX = storage.sX.get_tensor(X_smem_layout_staged, dtype=mBlockBase.element_type)
        sX_chunks = cute.flat_divide(sX, (self.num_compute_threads * self.vec_size,))
        sX_vec = cute.flat_divide(sX_chunks, (self.vec_size,))
        sX_unvec = cute.flat_divide(sX, (self.num_compute_threads,))
        sRed = storage.sRed.get_tensor(reduction_smem_layout, dtype=cutlass.Float32)
        sStat = storage.sStat.get_tensor(stat_smem_layout, dtype=cutlass.Float32)
        sStatFlat = cute.make_tensor(
            sStat.iterator, cute.make_layout((self.xchg_floats,))
        )
        sXchg = storage.sXchg.get_tensor(
            cute.make_layout((self.xchg_floats,)), dtype=cutlass.Float32
        )
        xchg_mbar_ptr = storage.xchg_mbar_ptr.data_ptr()
        sInvSum = storage.sInvSum.get_tensor(inv_sum_smem_layout, dtype=cutlass.Float32)
        sExp = storage.sExp.get_tensor(exp_smem_layout, dtype=cutlass.Float32)
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_compute_warps
        )
        pipeline_load_X = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.X_mbar_ptr.data_ptr(),
            num_stages=self.num_g2s_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.X_load_bytes,
            defer_sync=True,
        )
        cta_layout_vmnk = cute.make_layout((self.cluster_size, 1, 1, 1))
        if const_expr(self.cluster_size > 1):
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(xchg_mbar_ptr, 1)
        cutlass.pipeline.pipeline_init_arrive(
            cluster_shape_mn=cta_layout_vmnk, is_relaxed=True
        )
        cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        rQ = cute.make_rmem_tensor(
            cute.make_layout(
                (self.elem_per_thread, self.N_Q), stride=(1, self.elem_per_thread)
            ),
            Float32,
        )
        rQ_vec = cute.flat_divide(rQ, (self.vec_size,))
        rOut = cute.make_rmem_tensor((self.N_Q, self.elem_per_thread), Float32)
        rOut.fill(0.0)
        rMergedVec = cute.make_rmem_tensor(self.vec_size, Float32)
        if warp_idx == self.load_warp_idx:
            copy_atom_X = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyBulkG2SOp(), self.dtype
            )
            pipeline_load_X_producer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_g2s_stages
            )
            for n in cutlass.range(0, N, 1, unroll=1):
                pipeline_load_X_producer_state = self.g2s_load_X_row(
                    mBlockBase,
                    self.block_offset(mBlockOffs, n),
                    sX,
                    bid,
                    half_off,
                    pipeline_load_X,
                    pipeline_load_X_producer_state,
                    copy_atom_X,
                )
            if const_expr(self.HAS_PARTIAL_ADDEND):
                pipeline_load_X_producer_state = self.g2s_load_A(
                    mPartialAddendL,
                    sX,
                    bid,
                    pipeline_load_X,
                    pipeline_load_X_producer_state,
                    copy_atom_X,
                )
            if const_expr(self.reload_X):
                for n in cutlass.range(0, N, 1, unroll=1):
                    pipeline_load_X_producer_state = self.g2s_load_X_row(
                        mBlockBase,
                        self.block_offset(mBlockOffs, n),
                        sX,
                        bid,
                        half_off,
                        pipeline_load_X,
                        pipeline_load_X_producer_state,
                        copy_atom_X,
                    )
                if const_expr(self.HAS_PARTIAL_ADDEND):
                    pipeline_load_X_producer_state = self.g2s_load_A(
                        mPartialAddendL,
                        sX,
                        bid,
                        pipeline_load_X,
                        pipeline_load_X_producer_state,
                        copy_atom_X,
                    )
        elif warp_idx < self.num_compute_warps:
            pipeline_load_X_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.num_g2s_stages
            )
            for q in range_constexpr(self.N_Q):
                if const_expr(isinstance(mPseudoQueryL, tuple)):
                    gQ_query = self.local_row(mPseudoQueryL[q])
                else:
                    gQ_query = self.local_row(mPseudoQueryL[q, None])
                gQ_chunks = cute.flat_divide(
                    gQ_query, (self.num_compute_threads * self.vec_size,)
                )
                gQ_vec = cute.flat_divide(gQ_chunks, (self.vec_size,))
                gQ_unvec = cute.flat_divide(gQ_query, (self.num_compute_threads,))
                for chunk in range_constexpr(self.vec_chunks):
                    cute.autovec_copy(gQ_vec[None, tidx, chunk], rQ_vec[None, chunk, q])
                for t in range_constexpr(self.tail_elems):
                    rQ[self.tail_base + t, q] = gQ_unvec[tidx, self.tail_base + t]
            max_n = N - 1 if const_expr(self.HAS_PARTIAL_ADDEND) else N
            for n in cutlass.range(0, max_n, 1, unroll=1):
                pipeline_load_X_consumer_state = self.compute_qk(
                    sX_vec,
                    sX_unvec,
                    rQ_vec,
                    rQ,
                    sRed,
                    pipeline_load_X,
                    pipeline_load_X_consumer_state,
                    tidx,
                    lane_idx,
                    warp_idx,
                    n,
                    False,
                )
            if const_expr(self.HAS_PARTIAL_ADDEND):
                pipeline_load_X_consumer_state = self.compute_qk(
                    sX_vec,
                    sX_unvec,
                    rQ_vec,
                    rQ,
                    sRed,
                    pipeline_load_X,
                    pipeline_load_X_consumer_state,
                    tidx,
                    lane_idx,
                    warp_idx,
                    N - 1,
                    True,
                )
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_compute_threads)
            for stat_pass in range_constexpr(
                (self.xchg_floats + self.num_compute_threads - 1)
                // self.num_compute_threads
            ):
                stat_slot = tidx + stat_pass * self.num_compute_threads
                if stat_slot < N * (self.N_Q + 1):
                    n_slot = stat_slot // (self.N_Q + 1)
                    q_slot = stat_slot % (self.N_Q + 1)
                    acc = Float32(0.0)
                    for w in range_constexpr(self.num_compute_warps):
                        acc = fadd_rn_f32(acc, sRed[w, q_slot, n_slot])
                    if const_expr(self.cluster_size == 1):
                        if q_slot == 0:
                            acc = cute.math.rsqrt(acc * Float32(self.D_rcp) + eps)
                    sStat[n_slot, q_slot] = acc
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_compute_threads)
            if const_expr(self.cluster_size > 1):
                if warp_idx == 0:
                    if lane_idx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            xchg_mbar_ptr,
                            self.xchg_tx_bytes,
                            peer_cta_rank_in_cluster=peer_rank,
                        )
                    if lane_idx < Int32(self.xchg_vecs):
                        k = lane_idx * 4
                        store_shared_remote_fp32x4(
                            Float32(sStatFlat[k]),
                            Float32(sStatFlat[k + 1]),
                            Float32(sStatFlat[k + 2]),
                            Float32(sStatFlat[k + 3]),
                            sXchg.iterator + k,
                            xchg_mbar_ptr,
                            peer_rank,
                        )
                if tidx < N * (self.N_Q + 1):
                    cute.arch.mbarrier_wait(xchg_mbar_ptr, Int32(0))
                cute.arch.barrier(
                    barrier_id=1, number_of_threads=self.num_compute_threads
                )
                for stat_pass in range_constexpr(
                    (self.xchg_floats + self.num_compute_threads - 1)
                    // self.num_compute_threads
                ):
                    stat_slot = tidx + stat_pass * self.num_compute_threads
                    if stat_slot < N * (self.N_Q + 1):
                        n_slot = stat_slot // (self.N_Q + 1)
                        q_slot = stat_slot % (self.N_Q + 1)
                        acc = fadd_rn_f32(
                            Float32(sStat[n_slot, q_slot]),
                            Float32(sXchg[n_slot * (self.N_Q + 1) + q_slot]),
                        )
                        if q_slot == 0:
                            acc = cute.math.rsqrt(acc * Float32(self.D_rcp) + eps)
                        sStat[n_slot, q_slot] = acc
                cute.arch.barrier(
                    barrier_id=1, number_of_threads=self.num_compute_threads
                )
            if tidx < Int32(self.N_Q):
                q = tidx
                max_logit = -Float32.inf
                for n in cutlass.range(0, N, 1, unroll=1):
                    logit = sStat[n, q + 1] * sStat[n, 0]
                    max_logit = cute.arch.fmax(max_logit, logit)
                exp_sum = Float32(0.0)
                for n in cutlass.range(0, N, 1, unroll=1):
                    logit = sStat[n, q + 1] * sStat[n, 0]
                    exp = cute.math.exp(logit - max_logit, fastmath=False)
                    sExp[n, q] = exp
                    exp_sum = fadd_rn_f32(exp_sum, exp)
                inv_sum = Float32(0.0)
                inv_sum = Float32(1.0) / exp_sum
                sInvSum[q] = inv_sum
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_compute_threads)
            for n in cutlass.range(0, max_n, 1, unroll=1):
                pipeline_load_X_consumer_state = self.compute_pv(
                    sX_vec,
                    sX_unvec,
                    sExp,
                    sStat,
                    rOut,
                    pipeline_load_X,
                    pipeline_load_X_consumer_state,
                    tidx,
                    n,
                    False,
                )
            if const_expr(self.HAS_PARTIAL_ADDEND):
                pipeline_load_X_consumer_state = self.compute_pv(
                    sX_vec,
                    sX_unvec,
                    sExp,
                    sStat,
                    rOut,
                    pipeline_load_X,
                    pipeline_load_X_consumer_state,
                    tidx,
                    N - 1,
                    True,
                )
            for q in range_constexpr(self.N_Q):
                inv_sum = sInvSum[q]
                if const_expr(mMergedOutput is not None):
                    gOut_row = self.local_row(mMergedOutputL[q, bid, None])
                    gOut_chunks = cute.flat_divide(
                        gOut_row, (self.vec_size * self.num_compute_threads,)
                    )
                    gOut_vec = cute.flat_divide(gOut_chunks, (self.vec_size,))
                    gOut_unvec = cute.flat_divide(gOut_row, (self.num_compute_threads,))
                for chunk in range_constexpr(self.vec_chunks):
                    for v in range_constexpr(self.vec_size):
                        idx = chunk * self.vec_size + v
                        out = fmul_rn_f32(rOut[q, idx], inv_sum)
                        rMergedVec[v] = out
                    if const_expr(mMergedOutput is not None):
                        cute.autovec_copy(rMergedVec, gOut_vec[None, tidx, chunk])
                for t in range_constexpr(self.tail_elems):
                    idx = self.tail_base + t
                    out = fmul_rn_f32(rOut[q, idx], inv_sum)
                    if const_expr(mMergedOutput is not None):
                        gOut_unvec[tidx, self.tail_base + t] = out


_FWD_TUNED = {
    (1152, 1): (1, 64, "full", 4),
    (1536, 1): (1, 64, "full", 4),
    (1920, 1): (1, 64, 3, 4),
    (2304, 1): (1, 64, 3, 4),
    (2688, 1): (1, 128, 3, 4),
    (2880, 1): (1, 96, 3, 4),
    (3072, 1): (1, 128, 3, 4),
    (3456, 1): (1, 128, 3, 4),
    (4096, 1): (1, 64, 3, 4),
    (4608, 1): (1, 256, 4, 4),
    (6144, 1): (1, 128, 6, 4),
    (8192, 1): (1, 128, 6, 4),
    (1152, 3): (1, 64, 4, 4),
    (1536, 3): (1, 96, 4, 4),
    (1920, 3): (1, 96, 4, 4),
    (2304, 3): (1, 96, 4, 4),
    (2688, 3): (1, 64, 4, 4),
    (2880, 3): (1, 64, 4, 4),
    (3072, 3): (1, 64, 4, 4),
    (3456, 3): (1, 96, 4, 1),
    (4096, 3): (1, 128, 4, 4),
    (4608, 3): (1, 96, 4, 4),
    (6144, 3): (2, 128, 6, 1),
    (8192, 3): (2, 128, 6, 1),
}


def _attn_res_fwd_cutedsl_config(
    D: int,
    N_Q: int,
    has_partial_addend: bool,
    N: int,
    dtype: torch.dtype,
    cluster_size: int | None = None,
) -> tuple[int, int, int, int]:
    vec_size = 4
    desired_g2s_stages = N + (1 if has_partial_addend else 0)

    tuned = _FWD_TUNED.get((D, N_Q))
    if tuned is not None and cluster_size in (None, tuned[0]):
        tuned_cluster_size, num_compute_threads, stages, vec_size = tuned
        num_g2s_stages = (
            desired_g2s_stages if stages == "full" else min(stages, desired_g2s_stages)
        )
        return num_compute_threads, max(2, num_g2s_stages), vec_size, tuned_cluster_size

    if cluster_size is None:
        cluster_size = 1
    D_local = D // cluster_size
    target = (
        64
        if (N_Q == 1 and D_local < 2688) or D_local < 1536
        else 128
        if D_local >= 4096
        else 96
    )
    cands = [t for t in range(64, 513, 32) if D_local % t == 0]
    if cands:
        num_compute_threads = min(cands, key=lambda t: (abs(t - target), t))
    else:
        num_compute_threads = next(
            (t for t in range(128, 32, -32) if D_local % t == 0), 64
        )

    if N_Q == 1 and N <= 5:
        num_g2s_stages = desired_g2s_stages
    elif D >= 6144:
        num_g2s_stages = 6
    elif N_Q > 1 or D >= 4608:
        num_g2s_stages = 4
    else:
        num_g2s_stages = 3
    num_g2s_stages = max(2, min(num_g2s_stages, desired_g2s_stages))

    return num_compute_threads, num_g2s_stages, vec_size, cluster_size


@cache
def _compile_attn_res(width, queries, sources, dtype, normalize_values, has_addend):
    threads, stages, vector, cluster = _attn_res_fwd_cutedsl_config(
        width, queries, has_addend, sources, dtype
    )
    reload = not (stages == sources + has_addend and (queries == 1 or cluster == 2))
    kernel = AttnResFwdKernel(
        width,
        queries,
        dtype=dtype,
        normalize_values=normalize_values,
        has_partial_addend=has_addend,
        num_compute_threads=threads,
        num_g2s_stages=stages,
        vec_size=vector,
        reload_X=reload,
        cluster_size=cluster,
        max_N=(sources + 3) // 4 * 4,
    )
    tokens = cute.sym_int64()

    def tensor(dtype, shape):
        return make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=16,
        )

    x = tensor(_TORCH_TO_CUTE[dtype], (tokens, width))
    offsets = tuple(Int64(0) for _ in range(sources))
    q = tensor(Float32, (queries, width))
    y = tensor(Float32, (queries, tokens, width))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        offsets,
        x if has_addend else None,
        q,
        y,
        Float32(0.0),
        Int32(0),
        Int32(0),
        stream,
        options="--enable-tvm-ffi",
    )


def attn_res(blocks, partial_addend, queries, *, normalize_values=False):
    base = blocks[0]
    assert base.is_cuda and base.is_contiguous() and base.ndim == 2
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    assert base.dtype in (torch.bfloat16, torch.float32)
    for block in blocks:
        assert block.shape == base.shape and block.dtype == base.dtype
        assert block.device == base.device and block.is_contiguous()
        assert block.data_ptr() % 16 == 0
    if partial_addend is not None:
        assert partial_addend.shape == base.shape and partial_addend.dtype == base.dtype
        assert partial_addend.device == base.device and partial_addend.is_contiguous()
    tokens, width = base.shape
    n_queries = queries.shape[0]
    assert queries.shape[1] == width and queries.device == base.device
    sources = len(blocks)
    output = torch.empty(
        (n_queries, tokens, width), dtype=torch.float32, device=base.device
    )
    if not tokens:
        return tuple(output.unbind(0))
    offsets = tuple(
        (block.data_ptr() - base.data_ptr()) // base.element_size() for block in blocks
    )
    compiled = _compile_attn_res(
        width,
        n_queries,
        sources,
        base.dtype,
        normalize_values,
        partial_addend is not None,
    )
    compiled(
        base,
        offsets,
        partial_addend,
        queries,
        output,
        1e-6,
        tokens,
        sources,
    )
    return tuple(output.unbind(0))
