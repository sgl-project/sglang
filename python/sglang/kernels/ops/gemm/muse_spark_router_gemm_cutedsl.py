"""Muse Spark router logits on SM10x: TF32 tcgen05 GEMM with a cluster split-K.

Computes ``logits[M, E] = x[M, H] @ W[E, H]^T`` (fp32 in, fp32 accumulate, fp32 out)
in one launch; cuBLAS's TF32 path runs a split-K GEMM plus a separate reduce kernel.

Both operands are TMA-loaded through a TF32 tensor map,
so the copy engine rounds them fp32 -> TF32 (round to nearest even) on the way into
shared memory: the MMA sees exactly the operands of a round-to-nearest TF32 GEMM,
and the weight is used as stored.

The problem is swapped so the experts fill the MMA M dimension and the few tokens
sit in N. In kernel coordinates:
  A:  (E, K, L)  K-major  -- router weight
  B:  (M, K, L)  K-major  -- activations
  C:  (E, M, L)  E-major  -- view of the row-major [M, E] logits
E and K are static (one compile per router shape); M is dynamic; L = 1.

## Grid and clusters
  grid    = (E / cta_m, token_tiles * split_k, 1), cluster = (1, split_k, 1)
  The ``split_k`` CTAs of a cluster share one (cta_m, cta_n) output tile and each
  accumulates an equal, contiguous K slice; the epilogue reduces the slices over
  DSMEM. Larger M runs several token tiles (grid-y = token tile * split_k + rank).
  There is no TMA multicast: each CTA loads its own K slice.

## Warp specialization (8 warps, 256 threads/CTA)
  Warps 0-3  EPILOG  TMEM alloc/free; TMEM -> RMEM, split-K reduce-scatter over
                     DSMEM, st.global
  Warp 4     MMA     tcgen05.mma into TMEM; releases each stage (tcgen05.commit)
  Warp 5     LOAD_W  TMA-loads weight tiles into sA[..., stage]; with PDL it does
                     not wait for the previous kernel and launches the dependent
                     grid once its loads are issued
  Warp 6     LOAD_X  griddepcontrol.wait (PDL), then TMA-loads activation tiles
                     into sB[..., stage]
  Warp 7     idle

## Synchronization
  full[stage]       2 arrivals (LOAD_W + LOAD_X arrive_and_expect_tx) + TMA bytes
  empty[stage]      1 arrival (MMA tcgen05.commit)
  acc pipeline      PipelineUmmaAsync, 1 stage: MMA -> EPILOG ("accumulator ready")
  tmem barrier      named barrier (MMA + EPILOG) for retrieving the TMEM pointer
  mbar_reduce       1 arrival (EPILOG thread 0 arrive_and_expect_tx) + the
                    (S - 1) * SEG * 4 bytes the peers st.async into sSlots
  All mbarriers are initialized before the cluster arrive/wait, so no peer can
  credit mbar_reduce before it exists.

## Split-K reduce-scatter epilogue
  Rank g of the cluster owns segment g of the output tile: SEG = cta_m * cta_n / S
  floats, i.e. whole token rows (or part of one row for small tiles).
    1. registers -> sStage (cta_m x cta_n, token rows), one st.shared.f32 per
       register; 16 B chunks XOR-swizzled per row (column ^ 4 * (row % 8)), so
       every 8-thread phase of an ld.shared.v4.f32 and 16-thread phase of an
       ld.shared.v2.f32 hits 8 distinct 16 B chunks; named barrier;
    2. every thread copies 16 B pieces of the other ranks' segments:
       ld.shared.v4.f32 from sStage -> st.async.v4 into the owner's
       sSlots[:, slot];
    3. owner: wait for the peer bytes; each thread takes 2 consecutive floats of
       its segment: ld.shared.v2.f32 of its own sStage + (S - 1) of sSlots in
       rank order, packed f32x2 adds, st.global.v2.f32 -> each warp writes 256
       contiguous bytes of one logits row.
  The summation order is fixed (own partial, then peers in rank order), so the
  result is deterministic.

## Tile shapes (any tactic, S = split_k)
  Mma instruction:  Mma_M = cta_m (64 or 128), Mma_N = cta_n, Mma_K = 8 (TF32);
                    cta_k / 8 MMAs per stage, K / cta_k / S K tiles per rank
  tcgen05.ld:       16dp256b (32dp32b for cta_m = 128); cta_m * cta_n / 128 fp32
                    per epilogue thread
  SMEM:             (cta_m + cta_n) * cta_k * 4 B per stage; sStage cta_m * cta_n * 4 B,
                    sSlots (S - 1) / S of sStage
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from sglang.kernels.jit.utils import is_arch_support_pdl

# Tactic space. cta_m = experts per CTA, cta_n = tokens per CTA, split_k = cluster
# size along K (16 is the non-portable cluster maximum), cta_k = K per pipeline stage.
_MMA_M = (64, 128)
_MMA_N = (8, 16, 32, 64, 128)
_SPLIT_K = (2, 4, 8, 16)
_CTA_K = (32, 64, 128)
# Pipeline depth: as deep as shared memory allows, capped at _MAX_STAGES by default;
# the SM107 serve table raises the cap (_SM107_STAGES) so shared memory binds there too.
_MIN_STAGES = 2
_MAX_STAGES = 12

_FP32_BYTES = 4
_AB_BUFFER_ALIGN_BYTES = 1024  # 128 B swizzle atoms of sA / sB
_MAILBOX_ALIGN_BYTES = 128
_MBARRIER_BYTES = 8

_EPILOG_THREADS = 128
_EPILOG_SYNC_BAR_ID = 1  # staging tile
_TMEM_ALLOC_SYNC_BAR_ID = 2

_COMPILE_CACHE: dict = {}


@dataclasses.dataclass(frozen=True, slots=True)
class RouterTactic:
    """One specialization: expert tile, token tile, cluster split-K, stages, K tile."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int
    cta_k: int = 64
    # Programmatic dependent launch (when the device supports it): the weight
    # loads start before the previous kernel finishes, and the next kernel can
    # launch as soon as this one has issued its weight loads.
    pdl: bool = True


class RouterGemmKernel:
    """Router logits: TF32 tcgen05 GEMM, K split over a cluster of ``split_k`` CTAs whose
    partial tiles are summed by a DSMEM reduce-scatter in the epilogue."""

    def __init__(self, tactic: RouterTactic, use_pdl: bool):
        self.use_pdl = use_pdl
        self.acc_dtype = Float32
        self.mma_tiler = (tactic.mma_m, tactic.mma_n, tactic.cta_k)
        self.cta_m, self.cta_n, self.cta_k = self.mma_tiler
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.cta_group = tcgen05.CtaGroup.ONE

        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        # The weight does not depend on the previous
        # kernel, the activations do (with PDL only the activation warp waits).
        self.weight_load_warp_id = 5
        self.act_load_warp_id = 6
        # Warp 7 has no role: 8 warps keep the CTA at 256 threads.
        self.idle_warp_id = 7
        self.threads_per_cta = 32 * 8

        # Each rank owns 1 / split_k of the output tile (seg floats = whole token
        # rows) and receives the other ranks' partials of it: its own partial tile
        # is staged in sStage, the peers' partials of its segment land in sSlots,
        # and the peers' st.async writes credit exactly the sSlots bytes on the
        # rank's reduce mbarrier.
        s = tactic.split_k
        self.seg = tactic.mma_m * tactic.mma_n // s
        self.expected_transaction_bytes = (s - 1) * self.seg * _FP32_BYTES

    @cute.jit
    def __call__(
        self,
        w: cute.Tensor,  # (E, K) fp32, static
        x: cute.Tensor,  # (M, K) fp32, M dynamic
        out: cute.Tensor,  # (M, E) fp32, M dynamic
        stream: cuda.CUstream,
    ):
        # Kernel views with a unit batch mode: A (E, K, 1), B (M, K, 1), C (E, M, 1).
        unit = cute.make_layout(1)
        mA = cute.make_tensor(w.iterator, cute.append(w.layout, unit))
        mB = cute.make_tensor(x.iterator, cute.append(x.layout, unit))
        mC = cute.make_tensor(
            out.iterator, cute.append(cute.select(out.layout, mode=[1, 0]), unit)
        )

        # TF32 MMA on fp32 operands: Mma_M = cta_m, Mma_N = cta_n, Mma_K = 8.
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            cutlass.TFloat32,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        # The cluster runs along K only: the TMA and the pipelines see one CTA.
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.TFloat32, self.num_ab_stage
        )
        b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.TFloat32, self.num_ab_stage
        )
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # internal_type=TFloat32: the TMA rounds fp32 -> TF32 on load.
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cta_layout_vmnk.shape,
            internal_type=cutlass.TFloat32,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cta_layout_vmnk.shape,
            internal_type=cutlass.TFloat32,
        )
        self.a_load_bytes = cute.size_in_bytes(cutlass.TFloat32, a_smem_layout)
        self.b_load_bytes = cute.size_in_bytes(cutlass.TFloat32, b_smem_layout)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mC,
            cta_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ).launch(
            grid=(
                cute.ceil_div(cute.size(mC, mode=[0]), self.cta_m),
                cute.ceil_div(cute.size(mC, mode=[1]), self.cta_n) * self.split_k,
                1,
            ),
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, self.split_k, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (E, K, 1)
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (M, K, 1)
        mC_mnl: cute.Tensor,  # (E, M, 1)
        cta_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        if warp_idx == self.weight_load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
        if warp_idx == self.act_load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)

        # ---- Work tile: expert tile, token tile, and this rank's K slice ----
        e_idx = bidx
        n_idx = bidy // self.split_k
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        k_tile_count = cute.size(mA_mkl, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        # ---- Shared memory ----
        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            reduce_mbar: cutlass.Int64
            tmem_holding_buf: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ab stages: full[stage] = 2 producer arrivals (weight, activations) + their
        # TMA bytes; empty[stage] = 1 MMA tcgen05.commit.
        bar_full = storage.ab_mbar.data_ptr()
        bar_empty = bar_full + self.num_ab_stage
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=_TMEM_ALLOC_SYNC_BAR_ID,
                num_threads=32 * (len(self.epilogue_warp_id) + 1),
            ),
            allocator_warp_id=self.epilogue_warp_id[0],
        )
        mbar_reduce = storage.reduce_mbar.ptr
        if warp_idx == self.epilogue_warp_id[0]:
            with cute.arch.elect_one():
                # One local arrival (arrive_and_expect_tx) + the peers' tx bytes.
                for i in range(self.num_ab_stage):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(mbar_reduce, 1)
        # Fence the mbarrier inits and arrive on the cluster barrier; the wait
        # below (after the tile setup) guarantees every peer's mbar_reduce exists
        # before any st.async credits it.
        pipeline_init_arrive(
            cluster_shape_mn=cute.make_layout((1, self.split_k)), is_relaxed=True
        )

        # (MMA, MMA_M, MMA_K, STAGE) / (MMA, MMA_N, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=cutlass.TFloat32,
            layout=a_smem_layout_staged.outer,
            byte_alignment=_AB_BUFFER_ALIGN_BYTES,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=cutlass.TFloat32,
            layout=b_smem_layout_staged.outer,
            byte_alignment=_AB_BUFFER_ALIGN_BYTES,
            swizzle=b_smem_layout_staged.inner,
        )
        # (cta_m, cta_n): this rank's partial tile, token rows of cta_m experts.
        # 16 B chunks XOR-swizzled per token row (expert ^ 4 * (token % 8)): the
        # fragment st.shared and the ld.shared.v4 / .v2 row reads are bank-conflict
        # free, and every 4-expert chunk stays contiguous.
        sStage = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            byte_alignment=_MAILBOX_ALIGN_BYTES,
            swizzle=cute.make_swizzle(3, 4, self.cta_m.bit_length() - 3),
        )
        # (seg, split_k - 1): the peers' partials of this rank's segment.
        sSlots = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((self.seg, self.split_k - 1)),
            byte_alignment=_MAILBOX_ALIGN_BYTES,
        )

        # ---- Global tiles and TMA partitions ----
        # (bM, bK, RestM, RestK, RestL) / (bN, bK, RestN, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL) / (MMA, MMA_N, MMA_K, ...)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        # ((atom_v, rest_v), STAGE) / ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        # (MMA, MMA_M, MMA_K, STAGE) / (MMA, MMA_N, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

        pipeline_init_wait(cluster_shape_mn=cute.make_layout((1, self.split_k)))

        # ---- Load warps ----
        if warp_idx == self.weight_load_warp_id:
            self.load(
                tma_atom_a,
                tAgA[(None, e_idx, None, 0)],
                tAsA,
                bar_full,
                bar_empty,
                self.a_load_bytes,
                k_tile_start,
                k_tile_count,
                True,
            )
        if warp_idx == self.act_load_warp_id:
            self.load(
                tma_atom_b,
                tBgB[(None, n_idx, None, 0)],
                tBsB,
                bar_full,
                bar_empty,
                self.b_load_bytes,
                k_tile_start,
                k_tile_count,
                False,
            )

        # ---- MMA warp ----
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(self.acc_dtype), tCtAcc_fake.layout
            )[(None, None, None, 0)]
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            acc_pipeline.producer_acquire(acc_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            full_phase = Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.num_ab_stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                for kblk in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                    kblk_crd = (None, None, kblk, stage)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Release the stage once these MMAs have read it (one lane commits).
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, None, self.cta_group)
                if stage == self.num_ab_stage - 1:
                    full_phase = full_phase ^ 1
            acc_pipeline.producer_commit(acc_state)
            acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ---- Epilogue warps ----
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N) -> (cta_m, cta_n)
            tAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[
                ((None, None), 0, 0, 0)
            ]
            epi_tile = (self.cta_m, self.cta_n)
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.mma_tiler,
                utils.LayoutEnum.from_tensor(mC_mnl),
                Float32,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc)
            # (expert, token) coordinate of every fragment register
            tTR_cC = thr_copy_t2r.partition_D(cute.make_identity_tensor(epi_tile))
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)

            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            acc_pipeline.consumer_wait(acc_state)
            cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
            # tcgen05.ld is async: make the registers visible, then free TMEM.
            cute.arch.fence_view_async_tmem_load()
            # One arrival per epilogue warp (the acc pipeline's consumer count).
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_state)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
            #  Split K
            self.reduce_scatter(
                tTR_rAcc,
                tTR_cC,
                mC_mnl,
                sStage,
                sSlots,
                mbar_reduce,
                tidx,
                e_idx,
                n_idx,
                split_rank,
            )

    @cute.jit
    def load(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,  # ((atom_v, rest_v), RestK) of this CTA's tile
        tXsX: cute.Tensor,  # ((atom_v, rest_v), STAGE)
        bar_full: cute.Pointer,
        bar_empty: cute.Pointer,
        load_bytes: cutlass.Constexpr,
        k_tile_start: Int32,
        k_tile_count: cutlass.Constexpr,
        is_weight: cutlass.Constexpr,
    ):
        """One TMA producer: this rank's K tiles of one operand into the stages."""
        stages = self.num_ab_stage
        if cutlass.const_expr(self.use_pdl and not is_weight):
            # The activations come from the previous kernel.
            cute.arch.griddepcontrol_wait()
        # A fresh mbarrier is in phase 0, so waiting on phase 1 passes at once: the
        # first `stages` tiles go out without waiting for the MMA warp.
        empty_phase = Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, load_bytes)
            cute.copy(
                tma_atom,
                tXgX[(None, k_tile_start + k_tile)],
                tXsX[(None, stage)],
                tma_bar_ptr=bar_full + stage,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        if cutlass.const_expr(self.use_pdl and is_weight):
            cute.arch.griddepcontrol_launch_dependents()
        # Keep the CTA alive until the MMA warp has released every stage.
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.jit
    def reduce_scatter(
        self,
        tTR_rAcc: cute.Tensor,  # fp32 fragment
        tTR_cC: cute.Tensor,  # (expert, token) coordinates of the fragment
        mC_mnl: cute.Tensor,  # (E, M, 1)
        sStage: cute.Tensor,  # (cta_m, cta_n) swizzled: this rank's partial tile
        sSlots: cute.Tensor,  # (seg, split_k - 1): peers' partials of my segment
        mbar_reduce: cute.Pointer,
        epi_tid: Int32,
        e_idx: Int32,
        n_idx: Int32,
        split_rank: Int32,
    ):
        RW = self.cta_m  # experts per token row
        SEG = self.seg
        S = self.split_k
        # 4- and 2-float vector views ((V, 1), (rest...)): the swizzle keeps each
        # 4-expert chunk contiguous, so copies through them are ld.shared.v4 / .v2.
        sStage4 = cute.zipped_divide(sStage, (4, 1))
        sStage2 = cute.zipped_divide(sStage, (2, 1))
        sSlots4 = cute.zipped_divide(sSlots, (4, 1))
        sSlots2 = cute.zipped_divide(sSlots, (2, 1))
        r2 = cute.make_rmem_tensor((2,), Float32)
        r4 = cute.make_rmem_tensor((4,), Float32)

        # 1. Fragment -> sStage (one st.shared.f32 per register).
        for r in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            sStage[tTR_cC[r]] = tTR_rAcc[r]
        pipeline.NamedBarrier(
            barrier_id=_EPILOG_SYNC_BAR_ID, num_threads=_EPILOG_THREADS
        ).arrive_and_wait()

        # 2. Send the other ranks' segments, 16 B per st.async: piece j of rank g's
        #    segment goes to sSlots[4 j : 4 j + 4, slot] of rank g.
        if epi_tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_reduce, self.expected_transaction_bytes
            )
        n4 = SEG // 4  # 16 B pieces per segment
        for it in cutlass.range_constexpr(
            (S * n4 + _EPILOG_THREADS - 1) // _EPILOG_THREADS
        ):
            idx = it * _EPILOG_THREADS + epi_tid
            owner = idx // n4
            if idx < S * n4 and owner != split_rank:
                j = idx % n4
                flat = owner * SEG + 4 * j  # row-major position in the tile
                cute.autovec_copy(
                    sStage4[(None, 0), ((flat % RW) // 4, flat // RW)], r4
                )
                # Slot of this rank in the owner's sSlots: ranks above the owner
                # shift down by one (the owner has no slot of its own).
                slot = split_rank - 1
                if split_rank < owner:
                    slot = split_rank
                # st.async to the owner's sSlots, crediting 16 B on its mbar_reduce
                # (both pointers are local; store_async_dsmem maps them to the peer).
                cute.arch.store_async_dsmem(
                    sSlots4[(None, 0), (j, slot)].iterator,
                    tuple(r4[i].bitcast(Int32) for i in range(4)),
                    mbar_reduce,
                    owner,
                )
        cute.arch.mbarrier_wait(mbar_reduce, 0)

        # 3. Reduce and store my segment, 2 floats per thread per iteration.
        rows_left = cute.size(mC_mnl, mode=[1]) - n_idx * self.cta_n
        # ((2, 1), (cta_m / 2, cta_n)): expert pairs of this CTA's output tile. Built
        # with local_tile so the compiler can prove each pair 8 B aligned (one
        # st.global.v2 per pair); indexing mC with a hand-computed offset loses that
        # and splits the store.
        gC2 = cute.zipped_divide(
            cute.local_tile(mC_mnl, (self.cta_m, self.cta_n), (e_idx, n_idx, 0)),
            (2, 1),
        )
        n2 = SEG // 2
        for it in cutlass.range_constexpr(
            (n2 + _EPILOG_THREADS - 1) // _EPILOG_THREADS
        ):
            idx = it * _EPILOG_THREADS + epi_tid
            if idx < n2:
                flat = split_rank * SEG + 2 * idx
                row = flat // RW
                col = flat % RW
                acc = cute.make_rmem_tensor((2,), Float32)
                cute.autovec_copy(sStage2[(None, 0), (col // 2, row)], acc)
                for p in cutlass.range_constexpr(S - 1):
                    cute.autovec_copy(sSlots2[(None, 0), (idx, p)], r2)
                    acc.store(acc.load() + r2.load())
                if row < rows_left:
                    cute.autovec_copy(acc, gC2[(None, 0), (col // 2, row)])


# =====================================================================
# Host side: SMEM sizing, tactics, compile cache, entry points.
# =====================================================================
def _smem_capacity(device=None) -> int:
    """Shared memory per block of the current device."""
    if torch.cuda.get_device_capability(device) == (10, 7):
        # The device query does not report the sm_107 capacity; CuTe DSL's does.
        return cutlass.memory.get_smem_capacity_in_bytes("sm_107")
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin


def _smem_bytes(tactic: RouterTactic, stages: int) -> int:
    """Upper bound of the kernel's shared memory (allocation order and alignment)."""

    def align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    # SharedStorage: ab pipeline (2 per stage), acc pipeline (2), mbar_reduce, TMEM slot.
    cursor = (2 * stages + 3) * _MBARRIER_BYTES + 4
    stage_bytes = tactic.cta_k * _FP32_BYTES * stages
    cursor = align(cursor, _AB_BUFFER_ALIGN_BYTES) + tactic.mma_m * stage_bytes  # sA
    cursor = align(cursor, _AB_BUFFER_ALIGN_BYTES) + tactic.mma_n * stage_bytes  # sB
    tile = tactic.mma_m * tactic.mma_n
    cursor = align(cursor, _MAILBOX_ALIGN_BYTES) + tile * _FP32_BYTES  # sStage
    slots = (tactic.split_k - 1) * (tile // tactic.split_k) * _FP32_BYTES
    return align(cursor, _MAILBOX_ALIGN_BYTES) + slots  # sSlots


def _max_stages(
    tactic: RouterTactic, smem_capacity: int, limit: int = _MAX_STAGES
) -> int:
    return next(
        (
            s
            for s in range(limit, _MIN_STAGES - 1, -1)
            if _smem_bytes(tactic, s) <= smem_capacity
        ),
        0,
    )


def default_tactic(
    m: int,
    e: int,
    k: int,
    split_k: int = 0,
    mma_m: int = 64,
    stages: int = 0,
    cta_k: int = 64,
    mma_n: int = 0,
    pdl: bool = True,
    smem_capacity: int | None = None,
    max_stages_limit: int = _MAX_STAGES,
) -> RouterTactic:
    smem_capacity = smem_capacity or _smem_capacity()
    # Token tile: the smallest covering M, or an explicit one (larger M runs
    # several token tiles).
    mma_n = mma_n or next((n for n in _MMA_N if n >= m), _MMA_N[-1])
    # Largest split whose sStage + sSlots still leave room for 4 pipeline stages.
    for split in (split_k,) if split_k else reversed(_SPLIT_K):
        tactic = RouterTactic(mma_m, mma_n, split, _MIN_STAGES, cta_k, pdl)
        max_stages = _max_stages(tactic, smem_capacity, max_stages_limit)
        if max_stages >= (_MIN_STAGES if split_k else 4) and (k // cta_k) % split == 0:
            break
    tactic = dataclasses.replace(
        tactic, ab_stages=min(stages, max_stages) if stages else max_stages
    )
    validate(tactic, m, e, k, smem_capacity)
    return tactic


def validate(
    tactic: RouterTactic, m: int, e: int, k: int, smem_capacity: int | None = None
) -> None:
    """Reject a tactic that cannot serve ``x[m, k] @ w[e, k]^T``."""
    smem_capacity = smem_capacity or _smem_capacity()
    if tactic.mma_m not in _MMA_M or tactic.mma_n not in _MMA_N:
        raise ValueError(f"unsupported tile {tactic}")
    if tactic.split_k not in _SPLIT_K or tactic.cta_k not in _CTA_K:
        raise ValueError(f"unsupported split_k / cta_k in {tactic}")
    if m < 1 or e % tactic.mma_m:
        raise ValueError(f"M={m}, E={e} do not fit {tactic}")
    if k % tactic.cta_k or (k // tactic.cta_k) % tactic.split_k:
        raise ValueError(f"K={k} does not split into {tactic.split_k} x {tactic.cta_k}")
    if (
        tactic.ab_stages < _MIN_STAGES
        or _smem_bytes(tactic, tactic.ab_stages) > smem_capacity
    ):
        raise ValueError(f"tactic {tactic} does not fit in {smem_capacity} B of SMEM")
    tile = tactic.mma_m * tactic.mma_n
    if (tile // tactic.split_k) % 4:
        raise ValueError(f"tile {tactic} gives a rank segment of partial 16 B pieces")


def _cute_operands(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    # W is fully static; x and out have a dynamic token count M.
    return (
        from_dlpack(w, assumed_align=16),
        from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=x.dim_order()
        ),
        from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=out.dim_order()
        ),
    )


def _compile(tactic: RouterTactic, e: int, k: int, use_pdl: bool):
    # One binary per tactic and router shape (E, K); it serves every M.
    key = (tactic, e, k, use_pdl)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        x = torch.empty((tactic.mma_n, k), device="cuda")
        w = torch.empty((e, k), device="cuda")
        out = torch.empty((tactic.mma_n, e), device="cuda")
        compiled = cute.compile(
            RouterGemmKernel(tactic, use_pdl),
            *_cute_operands(x, w, out),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        _COMPILE_CACHE[key] = compiled
    return compiled


def router_logits_tf32(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    tactic: RouterTactic | None = None,
) -> torch.Tensor:
    """``x[M, H]`` fp32 @ ``w[E, H]^T`` fp32 -> fp32 ``[M, E]`` (TF32 MMA, fp32 accumulate)."""
    m, h = x.shape
    e, k = w.shape
    assert k == h and x.dtype == w.dtype == torch.float32
    assert x.is_contiguous() and w.is_contiguous()
    smem_capacity = _smem_capacity(x.device)
    if tactic is None:
        tactic = serve_tactic(m, e, k, smem_capacity, _capability(x.device.index))
    validate(tactic, m, e, k, smem_capacity)
    if out is None:
        out = torch.empty((m, e), dtype=torch.float32, device=x.device)
    _compile(tactic, e, k, tactic.pdl and is_arch_support_pdl())(
        *_cute_operands(x, w, out),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return out


# Serving tactics per device, keyed by compute capability: (max M, overrides for
# default_tactic) in increasing M. A device without a table keeps cuBLAS; add or
# replace one with register_serve_table().
#
# SM103 (GB300): picked in context, not in isolation: TP4 / TP8 decode MoE front (projection
# slice -> gather + norm + NVFP4 quant on the main stream, router + top-8 routing on the side stream),
# CUDA graph.
#
# SM107: picked in TP4 / TP8 serving (DFlash decode, nsys, expert start after the attention
# boundary, routing metadata on the router stream), not in isolation: beside the pooler GEMM a
# 16-CTA cluster waits for SMs, which a standalone run does not show. M ranges where no tactic
# beat cuBLAS at both TP sizes keep cuBLAS.
#
# A None entry keeps cuBLAS for its M range.
_SM107_STAGES = 24
_SERVE_TABLES = {
    (10, 3): (
        (16, dict(split_k=8, cta_k=128)),
        (64, dict(split_k=4, cta_k=64, mma_n=32)),
        (192, dict(split_k=8, cta_k=64, mma_n=64)),
        (256, dict(split_k=4, cta_k=64, mma_n=32)),
        (512, dict(split_k=4, cta_k=64, mma_n=64, pdl=False)),
        (1024, dict(split_k=4, cta_k=32, mma_m=128, mma_n=64, pdl=False)),
    ),
    (10, 7): (
        (
            32,
            dict(split_k=8, cta_k=32, mma_n=64, max_stages_limit=_SM107_STAGES),
        ),
        (
            64,
            dict(split_k=16, cta_k=128, mma_n=32, max_stages_limit=_SM107_STAGES),
        ),
        (
            128,
            dict(split_k=16, cta_k=32, mma_n=64, max_stages_limit=_SM107_STAGES),
        ),
        (256, None),
        (
            512,
            dict(
                split_k=4,
                cta_k=32,
                mma_m=128,
                mma_n=64,
                max_stages_limit=_SM107_STAGES,
            ),
        ),
        (1024, None),
    ),
}
SERVE_MAX_TOKENS = 1024


def register_serve_table(capability: tuple[int, int], table) -> None:
    """Add or replace the serving tactics of a compute capability: a sequence of
    (max M, default_tactic overrides or None for cuBLAS) in increasing M, covering M
    up to SERVE_MAX_TOKENS."""
    _SERVE_TABLES[tuple(capability)] = tuple(table)
    serve_tactic.cache_clear()


@functools.lru_cache(maxsize=None)
def _capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _serve_overrides(m: int, capability: tuple) -> dict | None:
    for max_m, overrides in _SERVE_TABLES[capability]:
        if m <= max_m:
            return overrides
    raise ValueError(f"M={m} is above the serving table of {capability}")


@functools.lru_cache(maxsize=None)
def serve_tactic(
    m: int, e: int, k: int, smem_capacity: int = 0, capability: tuple = (10, 3)
) -> RouterTactic:
    smem_capacity = smem_capacity or _smem_capacity()
    overrides = _serve_overrides(m, capability)
    if overrides is None:
        raise ValueError(f"M={m} runs cuBLAS on {capability}")
    return default_tactic(m, e, k, smem_capacity=smem_capacity, **overrides)


def router_tf32_eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    """FP32 router logits for 2 <= M <= SERVE_MAX_TOKENS on a device with a serving
    table (M = 1 keeps the GEMV), only with TF32 matmuls allowed: under "highest"
    cuBLAS keeps the logits FP32."""
    return (
        x.is_cuda
        and torch.get_float32_matmul_precision() != "highest"
        and x.dim() == 2
        and 2 <= x.shape[0] <= SERVE_MAX_TOKENS
        and _capability(x.device.index) in _SERVE_TABLES
        and _serve_overrides(x.shape[0], _capability(x.device.index)) is not None
        and x.dtype == torch.float32
        and w.dtype == torch.float32
        and x.shape[1] == w.shape[1]
        and x.shape[1] % 128 == 0
        and w.shape[0] % 128 == 0
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 32 == 0
        and w.data_ptr() % 32 == 0
    )


def router_tf32(
    x: torch.Tensor, w: torch.Tensor, tactic: RouterTactic | None = None
) -> torch.Tensor:
    """Router logits with the device's serving tactic, or an explicit one."""
    return router_logits_tf32(x, w, tactic=tactic)
