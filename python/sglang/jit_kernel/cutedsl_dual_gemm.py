# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
CuteDSL SM90 Dual GEMM kernel: fuses SiLU(X @ W_gate) * (X @ W_up) into a
single kernel using TMA + WGMMA on Hopper GPUs.

Architecture:
  - Producer warpgroup: TMA loads A, B0, B1 into SMEM (A shared between GEMMs)
  - Consumer warpgroup: 2x WGMMA per K-tile (reusing A descriptor)
  - Epilogue: fused SiLU*mul in registers, then R2S + TMA store

Supports optional FP8 quantized mode:
  - FP8 (e4m3fn) inputs with per-tensor x_scale and w_scale
  - FP8 (e4m3fn) output with o_scale for requantization
"""

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from sglang.srt.utils.custom_op import register_custom_op

# Monkey-patch PersistentTileSchedulerParams to fix attribute name mismatches
# between JIT-traced __init__ (which creates _raster_along_m, cluster_shape_m_fdd,
# cluster_shape_n_fdd) and __extract_mlir_values__ (which expects raster_along_m,
# cluster_shape_major_fdd, cluster_shape_minor_fdd).
_PTSP_ATTR_FIXES = {
    "raster_along_m": "_raster_along_m",
    "cluster_shape_major_fdd": "cluster_shape_m_fdd",
    "cluster_shape_minor_fdd": "cluster_shape_n_fdd",
}
_orig_extract = utils.PersistentTileSchedulerParams.__extract_mlir_values__


def _patched_extract(self):
    for expected, actual in _PTSP_ATTR_FIXES.items():
        if not hasattr(self, expected) and hasattr(self, actual):
            setattr(self, expected, getattr(self, actual))
    return _orig_extract(self)


utils.PersistentTileSchedulerParams.__extract_mlir_values__ = _patched_extract

_COMPILED_KERNELS = {}
_WEIGHT_CACHE = {}  # id(w) -> (w_ref, w_transposed, cute_tensor)
_KERNEL_OBJ_CACHE = (
    {}
)  # (acc_dtype, tile_mn, cluster_mn, use_fp8) -> HopperDualGemmKernel
_DUMMY_SCALE_CACHE = {}  # device -> (dummy_tensor, cute_dummy)
_RAW_SCALE_CACHE = {}  # data_ptr -> (scale_f32, cute_tensor)
_STREAM_CACHE = {}  # cuda_stream_int -> CUstream


class HopperDualGemmKernel:
    """SM90 persistent dual GEMM: computes SiLU(A @ B0) * (A @ B1).

    Uses warp specialization with dedicated DMA and MMA warp groups.
    DMA warp group loads A, B0, B1 via TMA; MMA warp groups compute
    both GEMMs and fused SiLU*mul epilogue. Persistent tile scheduler
    keeps CTAs alive across multiple output tiles.
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        use_fp8_scales: bool = False,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.use_fp8_scales = use_fp8_scales
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )

        # Warp specialization: 1 DMA warp group + N MMA warp groups
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.occupancy = 1
        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.tiled_mma = None
        self.shared_storage = None
        self.buffer_align_bytes = 128

        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

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
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
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

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout,
            self.tile_shape_mnk,
            self.a_dtype,
            self.ab_stage,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout,
            self.tile_shape_mnk,
            self.b_dtype,
            self.ab_stage,
        )
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.epi_stage,
        )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        epi_stage = 2
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        # 3 buffers per stage: A + B0 + B1
        bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + 2 * cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // bytes_per_stage
        return ab_stage, epi_stage

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b0: cute.Tensor,
        b1: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        x_scale: cute.Tensor = None,
        w_scale_gate: cute.Tensor = None,
        w_scale_up: cute.Tensor = None,
        o_scale: cute.Tensor = None,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b0.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b0)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        # TMA atoms for A, B0, B1 (loads)
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b0, tma_tensor_b0 = self._make_tma_atoms_and_tensors(
            b0,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_b1, tma_tensor_b1 = self._make_tma_atoms_and_tensors(
            b1,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        # TMA atom for C (store)
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
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
            sB0: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b0,
            tma_tensor_b0,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            x_scale,
            w_scale_gate,
            w_scale_up,
            o_scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b0: cute.CopyAtom,
        mB0_nkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB1_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        m_x_scale: cute.Tensor = None,
        m_w_scale_gate: cute.Tensor = None,
        m_w_scale_up: cute.Tensor = None,
        m_o_scale: cute.Tensor = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b0)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # Multicast masks
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
        ) + 2 * cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # =====================================================================
        #  Allocate SMEM and create pipeline
        # =====================================================================
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # =====================================================================
        #  SMEM tensors
        # =====================================================================
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB0 = storage.sB0.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB1 = storage.sB1.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # Separate sC buffer for persistent kernel (DMA may load next tile's A
        # while MMA is still writing epilogue)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # =====================================================================
        #  Partition global tensors (all tiles)
        # =====================================================================
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB0_nkl = cute.local_tile(
            mB0_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gB1_nkl = cute.local_tile(
            mB1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # =====================================================================
        #  TMA load partitions
        # =====================================================================
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tB0sB0, tB0gB0 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b0,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB0, 0, 2),
            cute.group_modes(gB0_nkl, 0, 2),
        )
        tB1sB1, tB1gB1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b1,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB1, 0, 2),
            cute.group_modes(gB1_nkl, 0, 2),
        )

        # =====================================================================
        #  MMA thread partitions
        # =====================================================================
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
        tCsB0 = thr_mma.partition_B(sB0)
        tCsB1 = thr_mma.partition_B(sB1)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB0 = tiled_mma.make_fragment_B(tCsB0)
        tCrB1 = tiled_mma.make_fragment_B(tCsB1)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        acc_gate = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        acc_up = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        num_k_blocks = cute.size(tCrA, mode=[2])

        # =====================================================================
        #  Cluster wait
        # =====================================================================
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups

        # =====================================================================
        #  DMA warp group: persistent TMA loader
        # =====================================================================
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tB0gB0_nkl = tB0gB0[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                tB1gB1_nkl = tB1gB1[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    tB0gB0_k = tB0gB0_nkl[(None, mainloop_producer_state.count)]
                    tB0sB0_pipe = tB0sB0[(None, mainloop_producer_state.index)]
                    tB1gB1_k = tB1gB1_nkl[(None, mainloop_producer_state.count)]
                    tB1sB1_pipe = tB1sB1[(None, mainloop_producer_state.index)]

                    bar_ptr = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=bar_ptr,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b0,
                        tB0gB0_k,
                        tB0sB0_pipe,
                        tma_bar_ptr=bar_ptr,
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b1,
                        tB1gB1_k,
                        tB1sB1_pipe,
                        tma_bar_ptr=bar_ptr,
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        # =====================================================================
        #  MMA warp group: persistent compute + epilogue
        # =====================================================================
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            # Epilogue setup
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            if cutlass.const_expr(self.c_dtype.width == 8):
                copy_atom_C = cute.make_copy_atom(CopyUniversalOp(), self.c_dtype)
            else:
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    self.c_dtype,
                )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc_gate = tiled_copy_r2s.retile(acc_gate)
            tRS_rAcc_up = tiled_copy_r2s.retile(acc_up)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            # TMA store pipeline
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            if cutlass.const_expr(self.use_fp8_scales):
                input_scale_gate_val = m_x_scale[0] * m_w_scale_gate[0]
                input_scale_up_val = m_x_scale[0] * m_w_scale_up[0]
                output_scale_inv_val = cutlass.Float32(1.0) / m_o_scale[0]
                fp8_max = cutlass.Float32(448.0)
                fp8_min = cutlass.Float32(-448.0)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]

                # ============= MAINLOOP =============
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                acc_gate.fill(0.0)
                acc_up.fill(0.0)

                # --- Prologue: dual GEMM ---
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for k_tile in range(prologue_mma_cnt):
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
                            acc_gate,
                            tCrA[k_block_coord],
                            tCrB0[k_block_coord],
                            acc_gate,
                        )
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_up,
                            tCrA[k_block_coord],
                            tCrB1[k_block_coord],
                            acc_up,
                        )
                    cute.nvgpu.warpgroup.commit_group()

                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_consumer_read_state.advance()

                # --- Steady state: dual GEMM ---
                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)

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
                            acc_gate,
                            tCrA[k_block_coord],
                            tCrB0[k_block_coord],
                            acc_gate,
                        )
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_up,
                            tCrA[k_block_coord],
                            tCrB1[k_block_coord],
                            acc_up,
                        )
                    cute.nvgpu.warpgroup.commit_group()

                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                # ============= EPILOGUE: fused SiLU + multiply =============
                tCgC_for_tma = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma,
                )

                epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
                epi_tile_shape = tCgC_for_tma.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    if cutlass.const_expr(self.use_fp8_scales):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            idx = epi_idx * size_tRS_rD + epi_v
                            g = tRS_rAcc_gate[idx] * input_scale_gate_val
                            u = tRS_rAcc_up[idx] * input_scale_up_val
                            neg_g = -g
                            exp_neg_g = cute.exp(neg_g)
                            sigmoid_g = cutlass.Float32(1.0) / (
                                cutlass.Float32(1.0) + exp_neg_g
                            )
                            result = g * sigmoid_g * u * output_scale_inv_val
                            tRS_rD[epi_v] = cutlass.max(
                                cutlass.min(result, fp8_max), fp8_min
                            )
                    else:
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            idx = epi_idx * size_tRS_rD + epi_v
                            g = tRS_rAcc_gate[idx]
                            u = tRS_rAcc_up[idx]
                            neg_g = -g
                            exp_neg_g = cute.exp(neg_g)
                            sigmoid_g = cutlass.Float32(1.0) / (
                                cutlass.Float32(1.0) + exp_neg_g
                            )
                            tRS_rD[epi_v] = g * sigmoid_g * u

                    acc_vec = tRS_rD.load()
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

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()

    # =====================================================================
    #  Static helper methods
    # =====================================================================
    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c


def _select_tile_config(M):
    # (64,64) tiles give many more pipeline stages (8 vs 2-3 for 128x128)
    # due to the 3-buffer-per-stage design (A+B0+B1). More stages = better
    # latency hiding for TMA loads.
    return (64, 64), (1, 1)


def _to_3d_k_major(t2d: torch.Tensor) -> torch.Tensor:
    """Convert 2D row-major (R, C) tensor to 3D (R, C, 1) with strides (C, 1, R*C).

    This matches the dense_gemm convention for K-major tensors where the
    contiguous (stride-1) dimension is dim 1.
    """
    return t2d.unsqueeze(0).permute(1, 2, 0)


def _make_cute_tensor(t3d: torch.Tensor, cutlass_dtype=None) -> "cute.Tensor":
    """Create a cute.Tensor from a 3D torch tensor, with optional dtype override for FP8.

    DLPack doesn't support float8 types, so FP8 tensors are viewed as uint8
    and the element_type is overridden on the cute.Tensor.
    """
    is_fp8 = t3d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8:
        t3d_for_dlpack = t3d.view(torch.uint8)
    else:
        t3d_for_dlpack = t3d

    ct = from_dlpack(t3d_for_dlpack, assumed_align=16)
    if is_fp8 and cutlass_dtype is not None:
        ct.element_type = cutlass_dtype
    ct = ct.mark_layout_dynamic(leading_dim=1)
    return ct


def _get_cached_weight(w: torch.Tensor, fp8_dtype) -> "cute.Tensor":
    """Cache weight transpose and cute tensor creation.

    Weights are static in inference, so we cache the transposed (N, K)
    tensor and its cute.Tensor wrapper keyed by object identity.
    This avoids GPU transpose kernels being captured in CUDA graphs.
    """
    key = id(w)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None and cached[0] is w:
        return cached[2]
    w_t = w.t().contiguous()
    w_3d = _to_3d_k_major(w_t)
    ct = _make_cute_tensor(w_3d, fp8_dtype)
    # Store strong ref to w to prevent id reuse while cached
    _WEIGHT_CACHE[key] = (w, w_t, ct)
    return ct


def _get_cached_dummy_scale(device) -> "cute.Tensor":
    """Cache dummy scale tensor for non-FP8 mode."""
    cached = _DUMMY_SCALE_CACHE.get(device)
    if cached is not None:
        return cached[1]
    dummy = torch.zeros(1, device=device, dtype=torch.float32)
    ct = from_dlpack(dummy, assumed_align=4)
    _DUMMY_SCALE_CACHE[device] = (dummy, ct)
    return ct


def _get_cached_raw_scale(scale: torch.Tensor, idx: int = 0) -> "cute.Tensor":
    """Cache from_dlpack wrapper for a scalar extracted from a scale tensor.

    Args:
        scale: Scale tensor (may be scalar or per-channel).
        idx: Element index to extract (default 0).
    """
    key = (scale.data_ptr(), idx)
    cached = _RAW_SCALE_CACHE.get(key)
    if cached is not None:
        return cached[1]
    scale_f32 = scale.flatten()[idx].float().reshape(1).contiguous()
    ct = from_dlpack(scale_f32, assumed_align=4)
    _RAW_SCALE_CACHE[key] = (scale_f32, ct)
    return ct


def _get_cached_stream(torch_stream) -> cuda.CUstream:
    """Cache CUstream wrapper."""
    stream_int = torch_stream.cuda_stream
    cached = _STREAM_CACHE.get(stream_int)
    if cached is not None:
        return cached
    stream = cuda.CUstream(stream_int)
    _STREAM_CACHE[stream_int] = stream
    return stream


def _cutedsl_dual_gemm_impl(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    out: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
):
    """Internal implementation — takes pre-split gate/up weights.

    Weights are expected to be persistent tensors (not transient views from
    torch.split) so that _get_cached_weight can cache by id() reliably.
    """
    M, K = x.shape
    _, N = w_gate.shape
    assert out.shape == (M, N), f"out shape mismatch: {out.shape} vs ({M}, {N})"

    use_fp8_scales = x_scale is not None and w_scale is not None and o_scale is not None

    tile_shape_mn, cluster_shape_mn = _select_tile_config(M)
    acc_dtype = cutlass.Float32

    # Cache kernel object to avoid re-creating it every call
    kern_key = (acc_dtype, tile_shape_mn, cluster_shape_mn, use_fp8_scales)
    kernel = _KERNEL_OBJ_CACHE.get(kern_key)
    if kernel is None:
        kernel = HopperDualGemmKernel(
            acc_dtype, tile_shape_mn, cluster_shape_mn, use_fp8_scales=use_fp8_scales
        )
        _KERNEL_OBJ_CACHE[kern_key] = kernel

    # Determine cutlass dtype for FP8 override
    fp8_ab_dtype = cutlass.Float8E4M3FN if use_fp8_scales else None
    fp8_c_dtype = cutlass.Float8E4M3FN if use_fp8_scales else None

    # Cache weight transposes to avoid GPU copy ops in CUDA graph
    mB0 = _get_cached_weight(w_gate, fp8_ab_dtype)
    mB1 = _get_cached_weight(w_up, fp8_ab_dtype)

    # x and out change each call — create fresh cute tensors (CPU-only, no GPU ops)
    x_3d = _to_3d_k_major(x)
    out_3d = _to_3d_k_major(out)
    mA = _make_cute_tensor(x_3d, fp8_ab_dtype)
    mC = _make_cute_tensor(out_3d, fp8_c_dtype)

    stream = _get_cached_stream(torch.cuda.current_stream())

    # Prepare scale tensors for the kernel
    if use_fp8_scales:
        m_x_scale = _get_cached_raw_scale(x_scale)
        m_o_scale = _get_cached_raw_scale(o_scale)
        # Support per-tensor, fused per-tensor, and per-channel w_scale:
        #   numel()==1 : single per-tensor scale for both gate & up
        #   numel()==2 : fused per-tensor from MergedColumnParallelLinear
        #                (w_scale[0] for gate, w_scale[1] for up)
        #   numel()>2  : per-channel (2*N,) — gate at index 0, up at index N
        if w_scale.numel() == 1:
            m_w_scale_gate = _get_cached_raw_scale(w_scale)
            m_w_scale_up = m_w_scale_gate
        elif w_scale.numel() == 2:
            m_w_scale_gate = _get_cached_raw_scale(w_scale, 0)
            m_w_scale_up = _get_cached_raw_scale(w_scale, 1)
        else:
            m_w_scale_gate = _get_cached_raw_scale(w_scale, 0)
            m_w_scale_up = _get_cached_raw_scale(w_scale, N)
    else:
        dummy = _get_cached_dummy_scale(x.device)
        m_x_scale = dummy
        m_w_scale_gate = dummy
        m_w_scale_up = dummy
        m_o_scale = dummy

    # max_active_clusters: use device SM count for optimal persistent occupancy
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count

    compile_key = (
        x.dtype,
        out.dtype,
        tile_shape_mn,
        cluster_shape_mn,
        K,
        N,
        use_fp8_scales,
        sm_count,
    )

    if compile_key not in _COMPILED_KERNELS:
        _COMPILED_KERNELS[compile_key] = cute.compile(
            kernel,
            mA,
            mB0,
            mB1,
            mC,
            sm_count,
            stream,
            m_x_scale,
            m_w_scale_gate,
            m_w_scale_up,
            m_o_scale,
        )

    compiled_kernel = _COMPILED_KERNELS[compile_key]
    # max_active_clusters is Constexpr — baked into compiled kernel, not passed at runtime
    compiled_kernel(
        mA, mB0, mB1, mC, stream, m_x_scale, m_w_scale_gate, m_w_scale_up, m_o_scale
    )


def _cutedsl_dual_gemm_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake impl for torch.compile shape inference."""
    return out


@register_custom_op(
    op_name="cutedsl_dual_gemm",
    mutates_args=["out"],
    fake_impl=_cutedsl_dual_gemm_fake,
)
def cutedsl_dual_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute out = SiLU(x @ w_gate) * (x @ w_up) using SM90 CuteDSL kernel.

    Args:
        x: Input tensor (M, K), row-major, BF16/FP16 or FP8 (e4m3fn)
        w: Combined weight [w_gate | w_up] (K, 2*N), row-major
        out: Output tensor (M, N), row-major, BF16/FP16 or FP8 (e4m3fn)
        x_scale: Per-tensor input scale (scalar float32), required for FP8
        w_scale: Weight scale tensor, required for FP8. Supports:
            numel()==1 : single per-tensor scale for both gate & up
            numel()==2 : fused per-tensor from MergedColumnParallelLinear
                         (w_scale[0] for gate, w_scale[1] for up)
            numel()>2  : per-channel (2*N,) — gate at index 0, up at index N
        o_scale: Per-tensor output scale (scalar float32), required for FP8
    """
    N = w.shape[1] // 2
    w_gate, w_up = torch.split(w, N, dim=1)
    _cutedsl_dual_gemm_impl(x, w_gate, w_up, out, x_scale, w_scale, o_scale)
    return out


def _run_test(M, K, N, dtype=torch.bfloat16):
    torch.manual_seed(42)
    # Use smaller values to reduce BF16 accumulation error
    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
    w_gate = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    w_up = torch.randn(K, N, device="cuda", dtype=dtype) * 0.1
    w = torch.cat([w_gate, w_up], dim=1)
    out = torch.empty(M, N, device="cuda", dtype=dtype)

    cutedsl_dual_gemm(x, w, out)
    torch.cuda.synchronize()

    ref = torch.nn.functional.silu(x @ w_gate) * (x @ w_up)
    torch.testing.assert_close(out, ref, atol=1.0, rtol=0.03)
    print(f"  PASS (M={M}, K={K}, N={N}, dtype={dtype})")


def _run_fp8_test(M, K, N):
    torch.manual_seed(42)
    # Create FP8 inputs by quantizing random data
    x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1
    w_gate_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.1
    w_up_fp32 = torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.1

    # Compute per-tensor scales
    x_scale = x_fp32.abs().max() / 448.0
    w_scale = max(w_gate_fp32.abs().max(), w_up_fp32.abs().max()) / 448.0
    o_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    # Quantize to FP8
    x_fp8 = (x_fp32 / x_scale).to(torch.float8_e4m3fn)
    w_gate_fp8 = (w_gate_fp32 / w_scale).to(torch.float8_e4m3fn)
    w_up_fp8 = (w_up_fp32 / w_scale).to(torch.float8_e4m3fn)
    w_fp8 = torch.cat([w_gate_fp8, w_up_fp8], dim=1)
    out_fp8 = torch.empty(M, N, device="cuda", dtype=torch.float8_e4m3fn)

    x_scale_t = x_scale.reshape(1).to(torch.float32)
    w_scale_t = w_scale.reshape(1).to(torch.float32)

    cutedsl_dual_gemm(x_fp8, w_fp8, out_fp8, x_scale_t, w_scale_t, o_scale)
    torch.cuda.synchronize()

    # Reference: dequantize, compute in FP32, requantize
    x_deq = x_fp8.float() * x_scale
    wg_deq = w_gate_fp8.float() * w_scale
    wu_deq = w_up_fp8.float() * w_scale
    ref_fp32 = torch.nn.functional.silu(x_deq @ wg_deq) * (x_deq @ wu_deq)
    ref_fp8 = torch.clamp(ref_fp32 / o_scale, -448.0, 448.0).to(torch.float8_e4m3fn)

    # Compare in float (relaxed tolerance for FP8 WGMMA vs float32 matmul differences)
    torch.testing.assert_close(out_fp8.float(), ref_fp8.float(), atol=8.0, rtol=0.15)
    print(f"  PASS FP8 (M={M}, K={K}, N={N})")


if __name__ == "__main__":
    print("Testing CuteDSL Dual GEMM kernel...")
    # Test various shapes
    for M, K, N in [
        (128, 4096, 11008),
        (256, 4096, 11008),
        (64, 4096, 4096),
    ]:
        _run_test(M, K, N, torch.bfloat16)
    # Test FP16
    _run_test(128, 4096, 4096, torch.float16)
    # Test FP8
    print("Testing FP8 mode...")
    _run_fp8_test(128, 4096, 4096)
    print("All tests passed!")
