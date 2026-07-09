# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils


class Sm120Fp8BlockwiseGemmKernel:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int, int],
        scale_gran_mn: Tuple[int, int],
    ):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.scale_gran_m, self.scale_gran_n = scale_gran_mn
        assert self.tile_shape_mnk[0] % self.scale_gran_m == 0
        assert self.tile_shape_mnk[1] % self.scale_gran_n == 0
        self.scale_m_per_tile = self.tile_shape_mnk[0] // self.scale_gran_m
        self.scale_n_per_tile = self.tile_shape_mnk[1] // self.scale_gran_n

        self.tiled_mma = None
        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        self.atom_layout = (2, 2, 1) if self.tile_shape_mnk[1] >= 16 else (4, 1, 1)
        self.mma_inst_mnk = (16, 8, 32)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.mainloop_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        op = cute.nvgpu.warp.MmaFP8Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        n_dup = 2 if self.tile_shape_mnk[1] >= 32 else 1
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * n_dup,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.ab_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.scale_m_per_tile,
            self.scale_n_per_tile,
            self.smem_capacity,
            self.occupancy,
        )

        if self.ab_stage <= 1:
            raise RuntimeError("Not enough shared memory for this tile shape.")

        self.a_smem_layout_staged, self.b_smem_layout_staged = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.MemRange[
                self.sf_dtype, self.scale_m_per_tile * self.ab_stage
            ]
            sSFB: cute.struct.MemRange[
                self.sf_dtype, self.scale_n_per_tile * self.ab_stage
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            sfa,
            sfb,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mC_mn: cute.Tensor,
        mSFA_gk: cute.Tensor,
        mSFB_gk: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1 + self.num_threads_per_warp
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(
            cute.make_layout((self.scale_m_per_tile, self.ab_stage))
        )
        sSFB = storage.sSFB.get_tensor(
            cute.make_layout((self.scale_n_per_tile, self.ab_stage))
        )

        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None),
        )
        gB_nk = cute.local_tile(
            mB_nk,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None),
        )
        gC_mn = cute.local_tile(
            mC_mn,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None),
        )
        cC_mn = cute.make_identity_tensor(mC_mn.shape)
        cC_local = cute.local_tile(
            cC_mn,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None),
        )
        gSFA_gk = cute.local_tile(mSFA_gk, (self.scale_m_per_tile,), (None, None))
        gSFB_gk = cute.local_tile(mSFB_gk, (self.scale_n_per_tile,), (None, None))
        cSFA_gk = cute.make_identity_tensor(mSFA_gk.shape)
        cSFB_gk = cute.make_identity_tensor(mSFB_gk.shape)
        cSFA = cute.local_tile(cSFA_gk, (self.scale_m_per_tile,), (None, None))
        cSFB = cute.local_tile(cSFB_gk, (self.scale_n_per_tile,), (None, None))
        scale_m_bound = mSFA_gk.shape[0]
        scale_n_bound = mSFB_gk.shape[0]

        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2),
        )

        scale_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            self.sf_dtype,
            num_bits_per_copy=self.sf_dtype.width,
        )
        scale_copy_a = cute.make_tiled_copy_tv(
            scale_copy_atom, cute.make_layout((32,)), cute.make_layout((1,))
        )
        scale_copy_b = cute.make_tiled_copy_tv(
            scale_copy_atom, cute.make_layout((32,)), cute.make_layout((1,))
        )
        thr_scale_copy_a = scale_copy_a.get_slice(lane_idx)
        thr_scale_copy_b = scale_copy_b.get_slice(lane_idx)
        tAgSFA = thr_scale_copy_a.partition_S(gSFA_gk)
        tAcSFA = thr_scale_copy_a.partition_S(cSFA)
        tAsSFA = thr_scale_copy_a.partition_D(sSFA)
        tBgSFB = thr_scale_copy_b.partition_S(gSFB_gk)
        tBcSFB = thr_scale_copy_b.partition_S(cSFB)
        tBsSFB = thr_scale_copy_b.partition_D(sSFB)
        num_rest_a = cute.size(tAsSFA, mode=[1])
        num_rest_b = cute.size(tBsSFB, mode=[1])

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mn)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        tmp_accum = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        sSFA_view_as_C_layout = cute.make_layout(
            (
                (self.scale_gran_m, self.scale_m_per_tile),
                self.tile_shape_mnk[1],
                self.ab_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB_view_as_C_layout = cute.make_layout(
            (
                self.tile_shape_mnk[0],
                (self.scale_gran_n, self.scale_n_per_tile),
                self.ab_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
        sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)
        tCsScaleA = thr_mma.partition_C(sSFA_view_as_C)
        tCsScaleB = thr_mma.partition_C(sSFB_view_as_C)

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mk, mode=[3])

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            tCrScaleA = cute.make_rmem_tensor_like(tCsScaleA[(None, None, None, 0)])
            tCrScaleB = cute.make_rmem_tensor_like(tCsScaleB[(None, None, None, 0)])

            while work_tile.is_valid_tile:
                tile_coord_mn = work_tile.tile_idx
                gC_mn_slice = gC_mn[(None, None, *tile_coord_mn[:2])]
                cC_slice = cC_local[(None, None, *tile_coord_mn[:2])]
                accumulators.fill(0.0)
                tmp_accum.fill(0.0)

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )

                cute.autovec_copy(
                    tCsScaleA[(None, None, None, mainloop_consumer_state.index)],
                    tCrScaleA,
                )
                cute.autovec_copy(
                    tCsScaleB[(None, None, None, mainloop_consumer_state.index)],
                    tCrScaleB,
                )

                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            self.mainloop_sync_barrier.arrive_and_wait()
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            tmp_accum,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            tmp_accum,
                        )

                        if k_block_idx == num_k_blocks - 1:
                            for i in cutlass.range_constexpr(cute.size(accumulators)):
                                s = (
                                    tCrScaleA[i % cute.size(tCrScaleA)]
                                    * tCrScaleB[i % cute.size(tCrScaleB)]
                                )
                                accumulators[i] += tmp_accum[i] * s
                                tmp_accum[i] = 0.0
                            cute.autovec_copy(
                                tCsScaleA[
                                    (None, None, None, mainloop_consumer_state.index)
                                ],
                                tCrScaleA,
                            )
                            cute.autovec_copy(
                                tCsScaleB[
                                    (None, None, None, mainloop_consumer_state.index)
                                ],
                                tCrScaleB,
                            )
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == num_k_blocks - 1:
                        self.mainloop_sync_barrier.arrive_and_wait()
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        tmp_accum,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        tmp_accum,
                    )
                for i in cutlass.range_constexpr(cute.size(accumulators)):
                    s = (
                        tCrScaleA[i % cute.size(tCrScaleA)]
                        * tCrScaleB[i % cute.size(tCrScaleB)]
                    )
                    accumulators[i] += tmp_accum[i] * s
                    tmp_accum[i] = 0.0

                tCgC_out = thr_mma.partition_C(gC_mn_slice)
                tCrC_out = cute.make_rmem_tensor_like(tCgC_out, self.c_dtype)
                acc_vec = accumulators.load()
                tCrC_out.store(acc_vec.to(self.c_dtype))

                tCcC_out = thr_mma.partition_C(cC_slice)
                for i in cutlass.range_constexpr(cute.size(tCgC_out)):
                    coord = tCcC_out[i]
                    if cute.elem_less(coord, mC_mn.shape):
                        tCgC_out[i] = tCrC_out[i]

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mn = work_tile.tile_idx
                tAgA_mk = tAgA[(None, tile_coord_mn[0], None)]
                tBgB_nk = tBgB[(None, tile_coord_mn[1], None)]
                tAgSFA_g = tAgSFA[(None, None, tile_coord_mn[0], None)]
                tAcSFA_g = tAcSFA[(None, None, tile_coord_mn[0], None)]
                tBgSFB_g = tBgSFB[(None, None, tile_coord_mn[1], None)]
                tBcSFB_g = tBcSFB[(None, None, tile_coord_mn[1], None)]

                tApA = cute.make_rmem_tensor(
                    cute.shape(tAsSFA[(None, None, 0)]), cutlass.Boolean
                )
                tBpB = cute.make_rmem_tensor(
                    cute.shape(tBsSFB[(None, None, 0)]), cutlass.Boolean
                )
                for r in cutlass.range_constexpr(num_rest_a):
                    tApA[(0, r)] = cute.elem_less(tAcSFA_g[(0, r, 0)][0], scale_m_bound)
                for r in cutlass.range_constexpr(num_rest_b):
                    tBpB[(0, r)] = cute.elem_less(tBcSFB_g[(0, r, 0)][0], scale_n_bound)

                mainloop_producer_state.reset_count()

                for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    bar_ptr = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )

                    tAgA_k = tAgA_mk[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    tBgB_k = tBgB_nk[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=bar_ptr,
                        mcast_mask=0,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=bar_ptr,
                        mcast_mask=0,
                    )

                    tAgSFA_k = tAgSFA_g[(None, None, mainloop_producer_state.count)]
                    tAsSFA_pipe = tAsSFA[(None, None, mainloop_producer_state.index)]
                    cute.copy(scale_copy_a, tAgSFA_k, tAsSFA_pipe, pred=tApA)
                    tBgSFB_k = tBgSFB_g[(None, None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, None, mainloop_producer_state.index)]
                    cute.copy(scale_copy_b, tBgSFB_k, tBsSFB_pipe, pred=tBpB)

                    cute.arch.cp_async_mbarrier_arrive_noinc(bar_ptr)

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk,
        a_dtype,
        b_dtype,
        scale_m_per_tile,
        scale_n_per_tile,
        smem_capacity,
        occupancy,
    ):
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        scale_bytes_per_stage = (scale_m_per_tile + scale_n_per_tile) * 4
        mbar_helpers_bytes = 2048

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes
        ) // (ab_bytes_per_stage + scale_bytes_per_stage)
        return ab_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage,
    ):
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage
        )
        return a_smem_layout_staged, b_smem_layout_staged

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, max_active_clusters):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None))].shape + (1,)
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


def _make_ab_scale_tensors(m, n, k, device):
    import torch

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    a_fp32 = (torch.rand(m, k, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32_nk = (
        (torch.rand(n, k, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    )
    b_fp8_nk = b_fp32_nk.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp8_kn = b_fp8_nk.t()

    kt = -(-k // 128)
    nt = -(-n // 128)
    import os

    if os.environ.get("DEBUG_UNIT_SCALE"):
        scale_a = torch.ones(kt, m, dtype=torch.float32, device=device).t()
        scale_b = torch.ones(nt, kt, dtype=torch.float32, device=device).t()
    else:
        scale_a = torch.randn(kt, m, dtype=torch.float32, device=device).t() * 0.01
        scale_b = torch.randn(nt, kt, dtype=torch.float32, device=device).t() * 0.01
    return a_fp8, b_fp8_kn, scale_a, scale_b


def _torch_ref(a_fp8, b_fp8_kn, scale_a, scale_b):
    import torch

    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    b_fp8_nk = b_fp8_kn.t()
    sa = group_broadcast(scale_a, a_fp8.shape)
    sb = group_broadcast(scale_b.t(), b_fp8_nk.shape)
    return torch.mm(
        sa * a_fp8.to(torch.float32), (sb * b_fp8_nk.to(torch.float32)).t()
    ).to(torch.bfloat16)


def _build_cute_ab(torch_tensor, cutlass_dtype):
    import cutlass.torch as cutlass_torch

    cute_tensor, torch_gpu = cutlass_torch.cute_tensor_like(
        torch_tensor, cutlass_dtype, True, 16
    )
    f32_tensor = torch_tensor.to(dtype=__import__("torch").float32)
    cute_tensor = cutlass_torch.convert_cute_tensor(
        f32_tensor, cute_tensor, cutlass_dtype, is_dynamic_layout=True
    )
    return cute_tensor, torch_gpu


def _build_cute_plain(torch_tensor, cutlass_dtype, assumed_align=16):
    import cutlass.torch as cutlass_torch

    cute_tensor, torch_gpu = cutlass_torch.cute_tensor_like(
        torch_tensor, cutlass_dtype, True, assumed_align
    )
    torch_gpu.copy_(torch_tensor)
    return cute_tensor, torch_gpu


def run_one(m, n, k, tile_shape_mnk, scale_gran_mn, swap_ab, tag):
    import cutlass.torch as cutlass_torch
    import torch

    device = "cuda"
    a_fp8, b_fp8_kn, scale_a, scale_b = _make_ab_scale_tensors(m, n, k, device)
    out_torch = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    if not swap_ab:
        assert n % tile_shape_mnk[1] == 0, (
            f"main path (swap_ab=False) requires N (={n}) to be a multiple of "
            f"tile N (={tile_shape_mnk[1]}); partial N tiles are not supported "
            "here yet"
        )
        b_nk = b_fp8_kn.t()
        a_cute, _ = _build_cute_ab(a_fp8, cutlass.Float8E4M3FN)
        b_cute, _ = _build_cute_ab(b_nk, cutlass.Float8E4M3FN)
        c_cute, c_torch = _build_cute_plain(out_torch, cutlass.BFloat16)
        sfa_cute, _ = _build_cute_plain(scale_a, cutlass.Float32, assumed_align=4)
        sfb_cute, _ = _build_cute_plain(scale_b.t(), cutlass.Float32, assumed_align=4)
    else:
        weight_nk = b_fp8_kn.t()
        act_mk = a_fp8
        a_cute, _ = _build_cute_ab(weight_nk, cutlass.Float8E4M3FN)
        b_cute, _ = _build_cute_ab(act_mk, cutlass.Float8E4M3FN)
        c_cute, c_torch = _build_cute_plain(
            torch.empty(n, m, dtype=torch.bfloat16, device=device), cutlass.BFloat16
        )
        sfa_cute, _ = _build_cute_plain(scale_b.t(), cutlass.Float32, assumed_align=4)
        sfb_cute, _ = _build_cute_plain(scale_a, cutlass.Float32, assumed_align=4)

    gemm = Sm120Fp8BlockwiseGemmKernel(cutlass.Float32, tile_shape_mnk, scale_gran_mn)
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()

    compiled_gemm = cute.compile(
        gemm, a_cute, b_cute, c_cute, sfa_cute, sfb_cute, max_active_clusters, stream
    )
    compiled_gemm(a_cute, b_cute, c_cute, sfa_cute, sfb_cute, stream)
    torch.cuda.synchronize()

    ref = _torch_ref(a_fp8, b_fp8_kn, scale_a, scale_b)
    result = c_torch.t() if swap_ab else c_torch

    torch.testing.assert_close(result.float(), ref.float(), atol=0.5, rtol=0.05)
    print(f"[PASS] {tag}: m={m} n={n} k={k} tile={tile_shape_mnk} swap_ab={swap_ab}")


if __name__ == "__main__":
    import torch

    torch.cuda.set_device(1)
    torch.manual_seed(0)

    run_one(256, 256, 256, (128, 128, 128), (1, 128), swap_ab=False, tag="main")
    run_one(32, 256, 256, (128, 32, 128), (128, 1), swap_ab=True, tag="swapAB")
    run_one(16, 256, 256, (128, 16, 128), (128, 1), swap_ab=True, tag="swapAB-N16")
    run_one(8, 256, 256, (128, 8, 128), (128, 1), swap_ab=True, tag="swapAB-N8")
    run_one(
        12, 256, 256, (128, 16, 128), (128, 1), swap_ab=True, tag="swapAB-N16-partial"
    )
    run_one(5, 256, 256, (128, 8, 128), (128, 1), swap_ab=True, tag="swapAB-N8-partial")
    run_one(
        5,
        256,
        384,
        (128, 8, 128),
        (128, 1),
        swap_ab=True,
        tag="swapAB-N8-partial-mismatchK",
    )
    run_one(
        20, 256, 256, (128, 32, 128), (128, 1), swap_ab=True, tag="swapAB-N32-partial"
    )
    run_one(
        100,
        4096,
        4096,
        (128, 32, 128),
        (128, 1),
        swap_ab=True,
        tag="swapAB-real-shape-partial",
    )

    print("ALL PASS")
