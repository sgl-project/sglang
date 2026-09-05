# SPDX-License-Identifier: BSD-3-Clause
"""SM100 FP8 GEMM derived from kernel_sm100_bf16.py.

Each 128-element K partial is drained from tensor memory, scaled in FP32,
and accumulated in registers. Software scaling preserves checkpoint values
and permits 8-wide token tiles.

Scale layouts after swap_ab: weights [N/128, K/128, E], tokens [M, K/128, E|1].
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from sglang.srt.lora.moe.kernels.cutedsl.kernel_sm100_bf16 import (
    GroupedGemmKernelSm100Bf16,
)
from sglang.srt.lora.moe.kernels.cutedsl.scheduler import (
    create_moe_tile_scheduler,
    resolve_scheduler_params_and_grid,
)


class GroupedGemmKernelSm100Fp8(GroupedGemmKernelSm100Bf16):
    NUM_ACC_STAGES = 2
    USE_TMA_STORE = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.swap_ab:
            raise ValueError("the software-scaled FP8 kernel ships swap_ab only")
        if self.use_2cta_instrs or self.cluster_shape_mn != (1, 1):
            raise ValueError(
                "the software-scaled FP8 kernel ships (1, 1) single-CTA only"
            )
        if self.mma_tiler_mn[0] != 128 or self.mma_tiler_mn[1] > 128:
            raise ValueError(
                "the software-scaled FP8 kernel indexes weight scales per 128-wide "
                "output tile and stages one column scale per epilogue thread, so "
                f"mma_tiler_mn must be (128, <=128); got {self.mma_tiler_mn}"
            )

    def _compute_stages(self, *args, **kwargs):
        num_acc_stage, num_ab_stage, num_c_stage = (
            GroupedGemmKernelSm100Bf16._compute_stages(*args, **kwargs)
        )
        return type(self).NUM_ACC_STAGES, num_ab_stage, num_c_stage

    def check_supported_dtypes(
        self,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ):
        if a_dtype is not cutlass.Float8E4M3FN or b_dtype is not cutlass.Float8E4M3FN:
            raise testing.CantImplementError(
                f"the software-scaled GEMM takes e4m3 A/B; got {a_dtype}, {b_dtype}"
            )
        if c_dtype is not cutlass.BFloat16:
            raise testing.CantImplementError(f"C must be BF16; got {c_dtype}")
        if self.acc_dtype is not cutlass.Float32:
            raise testing.CantImplementError(
                f"the accumulator must be FP32; got {self.acc_dtype}"
            )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sf_a: cute.Tensor,  # weight scales [NT, KT, E] fp32
        sf_b: cute.Tensor,  # token scales [M, KT, E|1] fp32
        c: cute.Tensor,
        group_m: cute.Tensor,
        direct_schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()
        # One k_tile must cover exactly one 128-element scale group.
        if cutlass.const_expr(self.mma_tiler[2] != 128):
            raise ValueError(
                f"k_tile must equal the 128-element scale group; got "
                f"{self.mma_tiler[2]} (mma_inst_tile_k={self.mma_inst_tile_k})"
            )

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = a_copy_size + b_copy_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = resolve_scheduler_params_and_grid(
            a=a,
            c=c,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            swap_ab=self.swap_ab,
            contiguous_segments=self.contiguous_segments,
            persistent_clusters=self.persistent_clusters,
            max_active_clusters=max_active_clusters,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            c,
            sf_a,
            sf_b,
            group_m,
            direct_schedule,
            schedule_tiles,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mC_raw: cute.Tensor,
        sf_a: cute.Tensor,
        sf_b: cute.Tensor,
        group_m: cute.Tensor,
        direct_schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params,
        epilogue_op: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        tidx, _, _ = cute.arch.thread_idx()
        cta_n = self.cta_tile_shape_mnk[1]

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            col_scale: cute.struct.MemRange[cutlass.Float32, self.cta_tile_shape_mnk[1]]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.epilogue_warp_id)
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sColScale = cute.make_tensor(
            storage.col_scale.data_ptr(), cute.make_layout(cta_n)
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = create_moe_tile_scheduler(
            tile_sched_params=tile_sched_params,
            direct_schedule=direct_schedule,
            schedule_tiles=schedule_tiles,
            swap_ab=self.swap_ab,
        )
        work_tile = tile_sched.initial_work_tile_info()

        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                mma_tile_coord_mnl = (
                    work_tile.tile_n_idx,
                    work_tile.tile_m_idx,
                    work_tile.expert_idx,
                )
                a_rest_l_coord = mma_tile_coord_mnl[2]
                if cutlass.const_expr(self.contiguous_segments):
                    seg_base_tile = (
                        group_m[work_tile.expert_idx] // self.cta_tile_shape_mnk[1]
                    )
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        seg_base_tile + mma_tile_coord_mnl[1],
                        cutlass.Int32(0),
                    )
                    a_rest_l_coord = work_tile.expert_idx

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, a_rest_l_coord)]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, work_tile.k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < work_tile.k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                work_tile = tile_sched.advance_to_next_work()

            ab_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                ab_consumer.reset()
                peek_ab_full_status = ab_consumer.try_wait()

                # Hand each 128-K partial to the epilogue warps for scaling.
                for k_tile in range(work_tile.k_tile_cnt):
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    acc_pipeline.producer_acquire(acc_producer_state)

                    handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kblk_crd],
                            tCrB[kblk_crd],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    handle.release()

                    peek_ab_full_status = cutlass.Boolean(1)
                    if handle.count + 1 < work_tile.k_tile_cnt:
                        peek_ab_full_status = ab_consumer.try_wait()

                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                work_tile = tile_sched.advance_to_next_work()

            acc_pipeline.producer_tail(acc_producer_state)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )

            # Match accumulator elements to token columns for scale lookup.
            tCgC_t = utils.gemm.sm100.transform_partitioned_tensor_layout(tCgC)
            tCtAcc_t = utils.gemm.sm100.transform_partitioned_tensor_layout(tCtAcc_base)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                utils.gemm.sm100.epilogue_tmem_copy_and_partition(
                    self, tidx, tCtAcc_t, tCgC_t, epi_tile, False
                )
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            cC = cute.make_identity_tensor(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
            )
            cC_epi = cute.flat_divide(cC, epi_tile)
            tTR_cC_epi = thr_copy_t2r.partition_D(cC_epi)
            # ((frag), (subtile)): linear element and subtile indices.
            tTR_cC = cute.group_modes(
                cute.group_modes(tTR_cC_epi, 3, cute.rank(tTR_cC_epi)), 0, 3
            )

            gC_raw = cute.local_tile(
                mC_raw,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            tCgC_raw = utils.gemm.sm100.transform_partitioned_tensor_layout(
                thr_mma.partition_C(gC_raw)
            )
            tCgC_epi_t2r = cute.flat_divide(tCgC_raw, epi_tile)
            tTR_gC_partitioned = thr_copy_t2r.partition_D(tCgC_epi_t2r)
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = (
                utils.gemm.sm100.epilogue_smem_copy_and_partition(
                    self, tiled_copy_t2r, tTR_rC, tidx, sC
                )
            )

            tCgC_epi = cute.flat_divide(tCgC_t, epi_tile)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                cute.group_modes(sC, 0, 2),
                cute.group_modes(tCgC_epi, 0, 2),
            )

            subtile_cnt = cute.size(tTR_cC.shape, mode=[1])
            frag_size = cute.size(tTR_rAcc.shape)
            tTR_rAcc_flat = cute.group_modes(tTR_rAcc, 0, cute.rank(tTR_rAcc))
            tTR_rTotal = cute.make_rmem_tensor(
                cute.make_layout((frag_size, subtile_cnt)), cutlass.Float32
            )

            num_tiles_executed = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                mma_tile_coord_mnl = (
                    work_tile.tile_n_idx,
                    work_tile.tile_m_idx,
                    work_tile.expert_idx,
                )
                weight_expert = work_tile.expert_idx
                if cutlass.const_expr(self.contiguous_segments):
                    seg_base_tile = (
                        group_m[work_tile.expert_idx] // self.cta_tile_shape_mnk[1]
                    )
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        seg_base_tile + mma_tile_coord_mnl[1],
                        cutlass.Int32(0),
                    )
                k_tile_cnt = work_tile.k_tile_cnt
                nt_idx = mma_tile_coord_mnl[0]
                token_base = mma_tile_coord_mnl[1] * cta_n
                token_rest_l = mma_tile_coord_mnl[2]
                work_tile = tile_sched.advance_to_next_work()

                for i in cutlass.range_constexpr(frag_size):
                    for st in cutlass.range_constexpr(subtile_cnt):
                        tTR_rTotal[(i, st)] = cutlass.Float32(0.0)

                for k_tile in range(k_tile_cnt):
                    sw = sf_a[(nt_idx, k_tile, weight_expert)]
                    if tidx < cta_n:
                        sColScale[tidx] = (
                            sw * sf_b[(token_base + tidx, k_tile, token_rest_l)]
                        )
                    epilog_sync_barrier.arrive_and_wait()

                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    for st in cutlass.range_constexpr(subtile_cnt):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, st)],
                            tTR_rAcc,
                        )
                        for i in cutlass.range_constexpr(frag_size):
                            col = tTR_cC[(i, st)][1]
                            tTR_rTotal[(i, st)] = (
                                tTR_rTotal[(i, st)] + tTR_rAcc_flat[i] * sColScale[col]
                            )
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    epilog_sync_barrier.arrive_and_wait()

                num_prev_subtiles = num_tiles_executed * subtile_cnt
                if cutlass.const_expr(self.USE_TMA_STORE):
                    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    for st in cutlass.range_constexpr(subtile_cnt):
                        for i in cutlass.range_constexpr(frag_size):
                            tTR_rAcc_flat[i] = tTR_rTotal[(i, st)]
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tRS_rC.store(acc_vec)

                        c_buffer = (num_prev_subtiles + st) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilogue_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, st)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        epilog_sync_barrier.arrive_and_wait()
                else:
                    tTR_gC = tTR_gC_partitioned[
                        (None, None, None, None, None, *mma_tile_coord_mnl)
                    ]
                    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
                    for st in cutlass.range_constexpr(subtile_cnt):
                        for i in cutlass.range_constexpr(frag_size):
                            tTR_rAcc_flat[i] = tTR_rTotal[(i, st)]
                        acc_vec = tTR_rAcc.load()
                        tTR_rC.store(epilogue_op(acc_vec.to(self.c_dtype)))
                        cute.autovec_copy(tTR_rC, tTR_gC[(None, None, None, st)])
                num_tiles_executed = num_tiles_executed + 1

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
