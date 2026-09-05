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

import math
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from sglang.srt.lora.moe.kernels.cutedsl.scheduler import (
    MoESchedulerParams,
    create_moe_tile_scheduler,
    resolve_scheduler_params_and_grid,
)


class GroupedGemmKernelSm90Fp8:
    """Hopper FP8 GEMM with software FP32 scale promotion per 128-K tile.

    swap_ab puts weights on M, giving one weight scale per tile and one
    activation scale per token. Partial sums are scaled before accumulation.
    """

    SUPPORTS_SINGLE_K_PIPELINE = True

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        mma_inst_tile_k: int = 4,
        persistent_clusters: Optional[int] = None,
        swap_ab: bool = False,
        contiguous_segments: bool = False,
        single_k_pipeline: bool = False,
    ):
        if use_2cta_instrs:
            raise ValueError("2-CTA MMA is a tcgen05 feature; SM90 has none")
        if not swap_ab:
            raise ValueError("the FP8 grouped GEMM implements swap_ab only")
        if mma_inst_tile_k != 4:
            # 4 x k32 FP8 WGMMA steps = one 128-element scale group per tile.
            raise ValueError("FP8 promotion pins mma_inst_tile_k to 4")
        self.acc_dtype = acc_dtype
        self.single_k_pipeline = single_k_pipeline
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_inst_tile_k = mma_inst_tile_k
        self.persistent_clusters = persistent_clusters
        self.swap_ab = swap_ab
        self.contiguous_segments = contiguous_segments
        if contiguous_segments and not swap_ab:
            raise ValueError("contiguous_segments requires swap_ab")
        if contiguous_segments and cluster_shape_mn != (1, 1):
            raise ValueError("contiguous_segments requires a (1, 1) cluster")

        # The K extent here is a placeholder. _setup_attributes computes it.
        self.tile_shape_mnk = (*mma_tiler_mn, 1)
        # Upstream uses a second math warp group only for large tiles.
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )

        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * 4
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        self.buffer_align_bytes = 1024

        self.tiled_mma = None
        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None

    def _setup_attributes(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        group_m: cute.Tensor,
        direct_schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c, self.epi_smem_layout_staged, self.epi_tile
        )

        self.tile_sched_params, grid = resolve_scheduler_params_and_grid(
            a=a,
            c=c,
            cta_tile_shape_mnk=self.tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            swap_ab=self.swap_ab,
            contiguous_segments=self.contiguous_segments,
            persistent_clusters=self.persistent_clusters,
            max_active_clusters=max_active_clusters,
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
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            sfa,
            sfb,
            group_m,
            direct_schedule,
            schedule_tiles,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
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

        smem = utils.SmemAllocator()
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
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

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
        acc_partial = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        # Match accumulator elements to token columns for scale lookup.
        idC = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        tCcC = thr_mma.partition_C(idC)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

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
                    work_tile.tile_m_idx,
                    work_tile.tile_n_idx,
                    work_tile.expert_idx,
                )
                if cutlass.const_expr(self.swap_ab):
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
            if cutlass.const_expr(self.c_dtype.width == 16):
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(), num_store_matrices
                    ),
                    self.c_dtype,
                )
            else:
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
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

            # Rotate epilogue buffers by executed tiles, not global work index.
            num_tiles_executed = cutlass.Int32(0)

            # These helpers inline at trace time.

            def _tile_ctx(self, work, gC_mnl, group_m):
                coord = (work.tile_m_idx, work.tile_n_idx, work.expert_idx)
                if cutlass.const_expr(self.swap_ab):
                    coord = (work.tile_n_idx, work.tile_m_idx, work.expert_idx)
                rest_l = coord[2]
                if cutlass.const_expr(self.contiguous_segments):
                    # Use the same flat segment offset as the DMA warp.
                    seg_base_tile = group_m[work.expert_idx] // self.tile_shape_mnk[1]
                    coord = (coord[0], seg_base_tile + coord[1], cutlass.Int32(0))
                    rest_l = work.expert_idx
                gC_slice = gC_mnl[(None, None, *coord)]
                sfa_row_blk = (coord[0] * self.tile_shape_mnk[0]) // 128
                token_base = coord[1] * self.tile_shape_mnk[1]
                return gC_slice, sfa_row_blk, token_base, rest_l, coord[2]

            # Add each lane's column offset to the warpgroup partition.
            lane_col_base = (tidx % 32) % 4 * 2

            def _load_scales(
                sfa_row_blk,
                token_base,
                sfa_l,
                sfb_l,
                k_tile,
                sfb_frag,
                mSFA_mkl,
                mSFB_nkl,
                tCcC,
                lane_col_base,
            ):
                sfa_s = mSFA_mkl[(sfa_row_blk, k_tile, sfa_l)]
                for v in cutlass.range_constexpr(cute.size(sfb_frag, mode=[0])):
                    for im in cutlass.range_constexpr(cute.size(sfb_frag, mode=[1])):
                        for in_ in cutlass.range_constexpr(
                            cute.size(sfb_frag, mode=[2])
                        ):
                            crd = (v, im, in_)
                            sfb_frag[crd] = mSFB_nkl[
                                (
                                    token_base + lane_col_base + tCcC[crd][1],
                                    k_tile,
                                    sfb_l,
                                )
                            ]
                return sfa_s

            def _issue_mma(
                acc_frag,
                mainloop_pipeline,
                read_state,
                tiled_mma,
                tCrA,
                tCrB,
                num_k_blocks,
            ):
                """Issue one K tile without draining its WGMMA group."""
                mainloop_pipeline.consumer_wait(read_state)
                acc_frag.fill(0.0)
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (
                        None,
                        None,
                        k_block_idx,
                        read_state.index,
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_frag,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        acc_frag,
                    )
                cute.nvgpu.warpgroup.commit_group()
                read_state.advance()
                return read_state

            def _release_stage(mainloop_pipeline, release_state):
                mainloop_pipeline.consumer_release(release_state)
                release_state.advance()
                return release_state

            def _promote(acc_frag, sfa_s, sfb_frag, accumulators, tCcC):
                # Nested coordinates pair layouts correctly; flat iteration does not.
                # Single-K promotion overwrites the total.
                for v in cutlass.range_constexpr(cute.size(accumulators, mode=[0])):
                    for im in cutlass.range_constexpr(
                        cute.size(accumulators, mode=[1])
                    ):
                        for in_ in cutlass.range_constexpr(
                            cute.size(accumulators, mode=[2])
                        ):
                            crd = (v, im, in_)
                            accumulators[crd] = acc_frag[crd] * (sfa_s * sfb_frag[crd])

            def _epilogue(
                self,
                gC_slice,
                tiles_executed,
                tma_atom_c,
                sC,
                tRS_sD,
                tRS_rAcc,
                tRS_rD,
                tRS_rD_out,
                tiled_copy_r2s,
                tma_store_pipeline,
                warp_idx,
                size_tRS_rD,
                epilogue_op,
            ):
                tCgC_for_tma_partition = cute.zipped_divide(gC_slice, self.epi_tile)
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
                num_prev_epi_tiles = tiles_executed * epi_tile_num

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

            # Set outside helpers: handle updates cannot cross staged regions.
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            if cutlass.const_expr(self.single_k_pipeline):
                # Keep these buffers out of the multi-K path's register live ranges.
                acc_partial2 = cute.make_fragment_like(acc_partial)
                sfb_frag = cute.make_fragment_like(acc_partial)
                sfb_frag2 = cute.make_fragment_like(acc_partial)
                # Promote/store t-1 while tile t's WGMMA drains. Fixed fragment
                # sets plus parity avoid spilling memrefs through the loop.
                have_pending = cutlass.Boolean(False)
                parity = cutlass.Boolean(False)
                pend_sfa = cutlass.Float32(0.0)
                pend_gC = gC_mnl[
                    (None, None, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))
                ]
                while work_tile.is_valid_tile:
                    pend2_gC, sfa_blk0, tok0, sfal0, sfbl0 = _tile_ctx(
                        self, work_tile, gC_mnl, group_m
                    )
                    pend2_sfa = cutlass.Float32(0.0)
                    mainloop_consumer_read_state.reset_count()
                    mainloop_consumer_release_state.reset_count()
                    if parity:
                        pend2_sfa = _load_scales(
                            sfa_blk0,
                            tok0,
                            sfal0,
                            sfbl0,
                            0,
                            sfb_frag2,
                            mSFA_mkl,
                            mSFB_nkl,
                            tCcC,
                            lane_col_base,
                        )
                        mainloop_consumer_read_state = _issue_mma(
                            acc_partial2,
                            mainloop_pipeline,
                            mainloop_consumer_read_state,
                            tiled_mma,
                            tCrA,
                            tCrB,
                            num_k_blocks,
                        )
                    else:
                        pend2_sfa = _load_scales(
                            sfa_blk0,
                            tok0,
                            sfal0,
                            sfbl0,
                            0,
                            sfb_frag,
                            mSFA_mkl,
                            mSFB_nkl,
                            tCcC,
                            lane_col_base,
                        )
                        mainloop_consumer_read_state = _issue_mma(
                            acc_partial,
                            mainloop_pipeline,
                            mainloop_consumer_read_state,
                            tiled_mma,
                            tCrA,
                            tCrB,
                            num_k_blocks,
                        )
                    if have_pending:
                        # Leave only the just-issued WGMMA group outstanding.
                        cute.nvgpu.warpgroup.wait_group(1)
                        mainloop_consumer_release_state = _release_stage(
                            mainloop_pipeline, mainloop_consumer_release_state
                        )
                        if parity:
                            _promote(
                                acc_partial, pend_sfa, sfb_frag, accumulators, tCcC
                            )
                        else:
                            _promote(
                                acc_partial2, pend_sfa, sfb_frag2, accumulators, tCcC
                            )
                        _epilogue(
                            self,
                            pend_gC,
                            num_tiles_executed,
                            tma_atom_c,
                            sC,
                            tRS_sD,
                            tRS_rAcc,
                            tRS_rD,
                            tRS_rD_out,
                            tiled_copy_r2s,
                            tma_store_pipeline,
                            warp_idx,
                            size_tRS_rD,
                            epilogue_op,
                        )
                        num_tiles_executed += cutlass.Int32(1)
                    pend_sfa = pend2_sfa
                    pend_gC = pend2_gC
                    have_pending = cutlass.Boolean(True)
                    parity = ~parity
                    work_tile = tile_sched.advance_to_next_work()
                if have_pending:
                    cute.nvgpu.warpgroup.wait_group(0)
                    mainloop_consumer_release_state = _release_stage(
                        mainloop_pipeline, mainloop_consumer_release_state
                    )
                    # Drain the buffer selected before the final parity flip.
                    if parity:
                        _promote(acc_partial, pend_sfa, sfb_frag, accumulators, tCcC)
                    else:
                        _promote(acc_partial2, pend_sfa, sfb_frag2, accumulators, tCcC)
                    _epilogue(
                        self,
                        pend_gC,
                        num_tiles_executed,
                        tma_atom_c,
                        sC,
                        tRS_sD,
                        tRS_rAcc,
                        tRS_rD,
                        tRS_rD_out,
                        tiled_copy_r2s,
                        tma_store_pipeline,
                        warp_idx,
                        size_tRS_rD,
                        epilogue_op,
                    )
                    num_tiles_executed += cutlass.Int32(1)
            else:
                while work_tile.is_valid_tile:
                    gC_mnl_slice, sfa_row_blk, token_base, sfa_l, sfb_l = _tile_ctx(
                        self, work_tile, gC_mnl, group_m
                    )
                    mainloop_consumer_read_state.reset_count()
                    mainloop_consumer_release_state.reset_count()
                    accumulators.fill(0.0)

                    # Hoisting scale loads ahead of the wait regresses multi-K tiles.
                    for k_tile in cutlass.range(0, work_tile.k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                        acc_partial.fill(0.0)
                        cute.nvgpu.warpgroup.fence()
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (
                                None,
                                None,
                                k_block_idx,
                                mainloop_consumer_read_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                acc_partial,
                                tCrA[k_block_coord],
                                tCrB[k_block_coord],
                                acc_partial,
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(0)
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_release_state
                        )
                        mainloop_consumer_release_state.advance()
                        mainloop_consumer_read_state.advance()

                        sfa_s = mSFA_mkl[(sfa_row_blk, k_tile, sfa_l)]
                        # Nested coordinates pair layouts correctly; flat iteration does not.
                        for v in cutlass.range_constexpr(
                            cute.size(accumulators, mode=[0])
                        ):
                            for im in cutlass.range_constexpr(
                                cute.size(accumulators, mode=[1])
                            ):
                                for in_ in cutlass.range_constexpr(
                                    cute.size(accumulators, mode=[2])
                                ):
                                    crd = (v, im, in_)
                                    sfb_s = mSFB_nkl[
                                        (
                                            token_base + lane_col_base + tCcC[crd][1],
                                            k_tile,
                                            sfb_l,
                                        )
                                    ]
                                    accumulators[crd] += acc_partial[crd] * (
                                        sfa_s * sfb_s
                                    )

                    _epilogue(
                        self,
                        gC_mnl_slice,
                        num_tiles_executed,
                        tma_atom_c,
                        sC,
                        tRS_sD,
                        tRS_rAcc,
                        tRS_rD,
                        tRS_rD_out,
                        tiled_copy_r2s,
                        tma_store_pipeline,
                        warp_idx,
                        size_tRS_rD,
                        epilogue_op,
                    )
                    num_tiles_executed += cutlass.Int32(1)
                    work_tile = tile_sched.advance_to_next_work()

            tma_store_pipeline.producer_tail()

    # The host helpers below come from the upstream Hopper persistent example.

    @staticmethod
    def _compute_stages(
        tile_shape_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        smem_capacity,
        occupancy,
    ):
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage
        mbar_helpers_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk, element_type, is_cooperative=False, epi_tile_override=None
    ):
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        n_perf = 64 if element_type.width == 8 else 32
        tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
        tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk,
        epi_tile,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage,
        c_dtype,
        c_layout,
        epi_stage,
    ):
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_layout_staged, epi_tile):
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim
        )

    def check_supported_dtypes(self, a_dtype, b_dtype, c_dtype):
        if a_dtype is not cutlass.Float8E4M3FN or b_dtype is not cutlass.Float8E4M3FN:
            raise TypeError(f"the FP8 grouped GEMM takes e4m3 A/B; got {a_dtype}")
        if c_dtype is not cutlass.BFloat16:
            raise TypeError(f"C must be BF16; got {c_dtype}")

    def check_mma_tiler_and_cluster_shape(self):
        if self.tile_shape_mnk[0] not in (64, 128):
            raise ValueError("SM90 CTA tile M must be 64/128")
        if self.tile_shape_mnk[1] not in (8, 16, 32, 64, 128, 256):
            raise ValueError(
                "this port validates CTA tile N in {8, 16, 32, 64, 128, "
                "256}; WGMMA itself accepts 8..256 step 8 -- extend the "
                "validated set before requesting other widths"
            )
        if math.prod(self.cluster_shape_mn) > 4:
            raise ValueError("SM90 cluster size must be <= 4")
        for extent in self.cluster_shape_mn:
            if extent <= 0 or extent & (extent - 1):
                raise ValueError(
                    "cluster extents must be positive powers of two; got "
                    f"{self.cluster_shape_mn}"
                )

    def check_tensor_alignment(self, problem_shape, a_major, b_major, c_major):
        if a_major != "k" or b_major != "k":
            raise ValueError("SM90 port requires K-major A and B")
        m, n, k, _length = problem_shape
        for extent, name in ((m, "M"), (n, "N"), (k, "K")):
            if extent % 8:
                raise ValueError(f"{name}={extent} breaks 16-byte TMA alignment")

    def can_implement(
        self, problem_shape, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
    ) -> bool:
        try:
            self.check_supported_dtypes(a_dtype, b_dtype, c_dtype)
            self.check_mma_tiler_and_cluster_shape()
            self.check_tensor_alignment(problem_shape, a_major, b_major, c_major)
        except (TypeError, ValueError):
            return False
        return True
