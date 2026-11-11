# SPDX-License-Identifier: BSD-3-Clause
"""SM90 FP8 GEMM with one scale per weight row and per token.

Derived from kernel_sm90_fp8.py: the WGMMA mainloop accumulates the whole
K extent in registers (as the BF16 kernel does) and one promotion applies
weight_scale[row] * token_scale[token] before the epilogue.

Scale layouts after swap_ab: weights [N, E], tokens [M, E|1].
"""

import cutlass
import cutlass.cute as cute
from cutlass import pipeline

from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm90_fp8 import (
    GroupedGemmKernelSm90Fp8,
)
from sglang.kernels.ops.lora.moe.cutedsl.scheduler import (
    MoESchedulerParams,
    create_moe_tile_scheduler,
)


class GroupedGemmKernelSm90Fp8Rowwise(GroupedGemmKernelSm90Fp8):
    # One accumulator set: no partial-tile pipeline to select.
    SUPPORTS_SINGLE_K_PIPELINE = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.single_k_pipeline:
            raise ValueError("the row-scaled FP8 kernel has no single-K pipeline")

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,  # weight-row scales (N, E)
        mSFB_nkl: cute.Tensor,  # token scales (M, E|1)
        group_m: cute.Tensor,
        direct_schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: MoESchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * 4
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
        )

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # (bM, bK, RestM, RestK, RestL). The L mode is the expert index.
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mnl)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        # Match accumulator elements to (weight row, token) for scale lookup.
        idC = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        tCcC = thr_mma.partition_C(idC)

        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        tile_sched = create_moe_tile_scheduler(
            tile_sched_params=tile_sched_params,
            direct_schedule=direct_schedule,
            schedule_tiles=schedule_tiles,
            swap_ab=self.swap_ab,
        )
        work_tile = tile_sched.initial_work_tile_info()

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            while work_tile.is_valid_tile:
                mma_tile_coord_mnl = (
                    work_tile.tile_n_idx,
                    work_tile.tile_m_idx,
                    work_tile.expert_idx,
                )
                a_rest_l_coord = mma_tile_coord_mnl[2]
                if cutlass.const_expr(self.contiguous_segments):
                    # Flat rows use slot 0 plus seg_offsets; weights keep the expert.
                    seg_base_tile = (
                        group_m[work_tile.expert_idx] // self.tile_shape_mnk[1]
                    )
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        seg_base_tile + mma_tile_coord_mnl[1],
                        cutlass.Int32(0),
                    )
                    a_rest_l_coord = work_tile.expert_idx
                tAgA_mkl = tAgA[(None, mma_tile_coord_mnl[0], None, a_rest_l_coord)]
                tBgB_nkl = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                mainloop_producer_state.reset_count()
                for k_tile in cutlass.range(0, work_tile.k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_mkl[(None, mainloop_producer_state.count)],
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_nkl[(None, mainloop_producer_state.count)],
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                work_tile = tile_sched.advance_to_next_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            num_k_blocks = cute.size(tCrA, mode=[2])

            # N=8 has only four values/thread; the helper's fixed st.matrix x4
            # needs eight. Scale the matrix count with the tile instead.
            num_store_matrices = min(4, max(1, self.tile_shape_mnk[1] // 4))
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(), num_store_matrices
                ),
                self.c_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(), num_store_matrices
                ),
                self.c_dtype,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_threads
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            # The warpgroup partition holds its base thread's coordinates; add
            # this lane's row (16 per warp, 1 per 4 lanes) and column (2 per lane).
            lane = tidx % 32
            lane_row_base = ((tidx % self.num_threads_per_warp_group) // 32) * 16 + (
                lane // 4
            )
            lane_col_base = (lane % 4) * 2

            # Rotate epilogue buffers by executed tiles, not global work index.
            num_tiles_executed = cutlass.Int32(0)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            while work_tile.is_valid_tile:
                coord = (
                    work_tile.tile_n_idx,
                    work_tile.tile_m_idx,
                    work_tile.expert_idx,
                )
                sfa_l = coord[2]
                if cutlass.const_expr(self.contiguous_segments):
                    # Use the same flat segment offset as the DMA warp.
                    seg_base_tile = (
                        group_m[work_tile.expert_idx] // self.tile_shape_mnk[1]
                    )
                    coord = (coord[0], seg_base_tile + coord[1], cutlass.Int32(0))
                    sfa_l = work_tile.expert_idx
                gC_mnl_slice = gC_mnl[(None, None, *coord)]
                row_base = coord[0] * self.tile_shape_mnk[0]
                token_base = coord[1] * self.tile_shape_mnk[1]
                sfb_l = coord[2]

                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                cute.nvgpu.warpgroup.fence()

                # Positive, aligned K guarantees one prologue tile.
                k_pipe_mmas = 1
                for k_tile in cutlass.range(0, k_pipe_mmas, 1, unroll=1):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()

                for k_tile in cutlass.range(
                    k_pipe_mmas, work_tile.k_tile_cnt, 1, unroll=1
                ):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in cutlass.range(0, k_pipe_mmas, 1, unroll=1):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                # One promotion: the scale is K-invariant.
                # Nested coordinates pair layouts correctly; flat iteration does not.
                for v in cutlass.range_constexpr(cute.size(accumulators, mode=[0])):
                    for im in cutlass.range_constexpr(
                        cute.size(accumulators, mode=[1])
                    ):
                        for in_ in cutlass.range_constexpr(
                            cute.size(accumulators, mode=[2])
                        ):
                            crd = (v, im, in_)
                            sfa_s = mSFA_mkl[
                                (row_base + lane_row_base + tCcC[crd][0], sfa_l)
                            ]
                            sfb_s = mSFB_nkl[
                                (token_base + lane_col_base + tCcC[crd][1], sfb_l)
                            ]
                            accumulators[crd] = accumulators[crd] * (sfa_s * sfb_s)

                tCgC_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition,
                )
                epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )
                num_prev_epi_tiles = num_tiles_executed * epi_tile_num

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                    acc_vec = epilogue_op(tRS_rD.load())
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                        tRS_sD, mode=[3]
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                num_tiles_executed += cutlass.Int32(1)
                work_tile = tile_sched.advance_to_next_work()

            tma_store_pipeline.producer_tail()
