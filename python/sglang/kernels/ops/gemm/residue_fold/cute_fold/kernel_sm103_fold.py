# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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
#
# This file is ported from CUTLASS's sm103_dense_blockscaled_gemm_persistent.py
# with modifications for FlashInfer integration (alpha scaling, PDL support, wrapper method).
# Original: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py
#
# Adapted from FlashInfer's dense_blockscaled_gemm_sm103.py for residue fold.

from dataclasses import dataclass, field
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm103_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# ResInfer MExt R1 fork (see fold_epilogue.py): the direct-store epilogue is
# replaced by a row-pair folding store, and wrapper() gains a fold_pairs mode
# that builds the pair-collapsed output view. Forked from FlashInfer's
# dense_blockscaled_gemm_sm103.py (BSD-3-Clause, header retained above).
from .fold_epilogue import mext_fold_epilogue, mext_fold_epilogue_tma


class Sm103BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for FP4 data types
    and architectural features specific to Blackwell SM103 GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]


    :note: In current version, A and B tensor must have the same data type
        - i.e., Float4E2M1FN for A and Float4E2M1FN for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm103BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 256),
        ...     cluster_shape_mn=(2, 4)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        enable_pdl: bool = True,
        fold_pairs_tma: bool = False,
    ):
        """Initializes the configuration for a Blackwell SM103 3xFP4 GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Whether TMA store is enabled.
        :type use_tma_store: bool
        :param enable_pdl: Whether Programmatic Dependent Launch is enabled.
        :type enable_pdl: bool
        """
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.enable_pdl = enable_pdl
        # ResInfer MExt: TMA-store fold mode. The accumulator side keeps
        # stock m2-wide geometry; the C/TMA side uses half-width tiles over
        # the compact D_t[n_w, m_tok] (see fold_epilogue.mext_fold_epilogue_tma).
        self.fold_pairs_tma = fold_pairs_tma and use_tma_store
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_ab_warp_id = 5
        self.tma_sf_warp_id = 6
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_ab_warp_id,
                self.tma_sf_warp_id,
                *self.epilogue_warp_id,
            )
        )
        # Set barrier id for epilogue sync and tmem ptr sync
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_103")
        # SM103 TMEM capacity is 512 columns (same as SM100).
        # This replaces cute.arch.get_max_tmem_alloc_cols("sm_103") which
        # may not be available in older cutlass-dsl versions.
        SM103_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM103_TMEM_CAPACITY_COLUMNS
        self.sf_buffers_per_tile_k = 4 if self.sf_vec_size == 16 else 2

    def _setup_attributes(self):
        """Set up kernel attributes that depend on runtime tensor inputs.

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])

        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        dummy_tiled_mma_sfb = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            768,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_layout_vmnk.shape[0]),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        blk_mn = 128
        self.cta_n_sf = cute.round_up(cute.size(self.cta_tile_shape_mnk[1]), blk_mn)
        self.mma_sf_tiler = (
            self.cta_tile_shape_mnk[0],
            self.cta_n_sf,
            self.cta_tile_shape_mnk[2] // self.sf_buffers_per_tile_k,
        )

        self.sf_atom = self.Sm103BlockScaledBasicChunk(
            self.sf_vec_size, tiled_mma.op.a_major_mode
        ).layout

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (dummy_tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = (self.cta_tile_shape_mnk[0], 64)
        # ResInfer MExt: half-width C-side epi tile for the TMA fold path
        # (accumulator subtiles stay m2-wide; folded output is m_tok-wide,
        # same subtile count, 2x narrower).
        self.epi_tile_c = (self.epi_tile[0], self.epi_tile[1] // 2)

        self.num_acc_stage, self.num_ab_stage, self.num_sf_stage, self.num_c_stage = (
            self._compute_stages(
                tiled_mma,
                self.mma_tiler,
                self.epi_tile,
                self.c_dtype,
                self.c_layout,
                self.sf_dtype,
                self.sf_vec_size,
                self.smem_capacity,
                self.occupancy,
                self.use_tma_store,
            )
        )

        # Overlapping accumulator (ported from kernel_sm100_fold, measured
        # +6.8-8.3% on B200 tp1 fold m>=384): at N=256 only one honest acc
        # stage fits TMEM, so two pseudo-buffers overlap by the SF column
        # footprint; phase 0 drains its subtiles in reverse so the shared
        # columns are consumed first and the pipeline is released early.
        # Scoped to OUR fold epilogues (fold_epilogue.py); the plain path
        # uses the stock library epilogue, which drains strictly serially.
        self.overlapping_accum = self.num_acc_stage == 1 and (
            self.fold_pairs_tma or not self.use_tma_store
        )
        # Set at trace time in the kernel body (needs the SF TMEM layouts);
        # consumed by fold_epilogue.py as a Constexpr attribute.
        self.iter_acc_early_release_in_epilogue = None

        # Compute A/B/SFA/SFB/C shared memory layout
        # ((CTA_MMA_M,16bytes),1,8,num_ab_stage)
        self.a_smem_layout_staged = self.sm103_make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.num_ab_stage,
        )

        # ((CTA_MMA_M,16bytes),1,8,3)
        self.a_smem_layout_staged_tma = self.sm103_make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            3,
        )

        # ((CTA_MMA_N,16bytes),1,8,num_ab_stage)
        self.b_smem_layout_staged = self.sm103_make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.num_ab_stage,
        )

        # ((CTA_MMA_N,16bytes),1,8,3)
        self.b_smem_layout_staged_tma = self.sm103_make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            3,
        )

        # (((8,4,4),(sf_vec_size,4)),1,3,num_sf_stage)
        self.sfa_smem_layout_staged = self.sm103_make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_sf_stage,
        )

        # (((32,4,2),(sf_vec_size,4)),1,3,num_sf_stage)
        self.sfb_smem_layout_staged = self.sm103_make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_sf_stage,
        )
        self.c_smem_layout_staged = None
        if self.use_tma_store:
            if self.fold_pairs_tma:
                # Transposed half-width staging (n_w contiguous): the TMA box
                # maps directly onto row-major D[m_tok, n_w], eliminating the
                # host-side transpose. Threads write their 32 m_tok values at
                # stride epi_m (scalar STS; instruction count is not the
                # epilogue bottleneck -- SASS-verified). Half-width staging
                # frees half the stock smem budget: spend it on 2x stages to
                # cut TMA-drain waits in producer_acquire.
                self.num_c_stage *= 2
                epi_m, epi_n2 = self.epi_tile_c
                self.c_smem_layout_staged = cute.make_layout(
                    (epi_m, epi_n2, self.num_c_stage),
                    stride=(1, epi_m, epi_m * epi_n2),
                )
            else:
                self.c_smem_layout_staged = sm103_utils.make_smem_layout_epi(
                    self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
                )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_alias_tensor: cute.Tensor = None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param alpha: Single-element tensor containing alpha scaling value
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        if cutlass.const_expr(self.fold_pairs_tma):
            # The physical C is the transposed D view (m-major); the TMEM
            # load atom must still be chosen for the m2-contiguous (n-major)
            # fragment orientation the fold pairing relies on. Pin the enum
            # rather than deriving it from the transposed tensor.
            self.c_layout = utils.LayoutEnum.ROW_MAJOR
        else:
            self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)
        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        sfa_layout = cute.tile_to_shape(self.sf_atom, a_tensor.shape, (2, 1, 3))
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        sfb_layout = cute.tile_to_shape(self.sf_atom, b_tensor.shape, (2, 1, 3))
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        dummy_tiled_mma_sfb = self.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm103_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        # casting layout as uint8 for multicast
        a_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(
            self.a_smem_layout_staged_tma
        )
        a_tensor_uint8 = cute.recast_tensor(a_tensor, cutlass.Uint8)
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            a_op,
            a_tensor_uint8,
            a_smem_layout_tma_ready,
            # 384 corresponds to the number of uint8 elements along the K dimension processed in a single MMA mainloop iteration.
            (cute.size(tiled_mma.tv_layout_A[1][0]), 384),
            self.cluster_shape_mn[1],
            internal_type=cutlass.Uint8,
        )

        # Setup TMA load for B
        b_op = sm103_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        # casting layout as uint8 for multicast
        b_smem_layout_tma_ready = self.adapt_layout_for_tma_ab(
            self.b_smem_layout_staged_tma
        )
        b_tensor_uint8 = cute.recast_tensor(b_tensor, cutlass.Uint8)
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            b_op,
            b_tensor_uint8,
            b_smem_layout_tma_ready,
            (cute.size(tiled_mma.tv_layout_B[1][0]), 384),
            self.cluster_shape_mn[0] // cute.size(tiled_mma.thr_id.shape),
            internal_type=cutlass.Uint8,
        )

        # Setup TMA load for SFA
        sfa_op = sm103_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfa_smem_layout_tma_ready = self.adapt_layout_for_tma_sf(sfa_smem_layout)
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.cpasync.make_tiled_tma_atom(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout_tma_ready,
            (self.mma_sf_tiler[0], self.mma_sf_tiler[2]),
            self.cluster_shape_mn[1],
            internal_type=cutlass.Uint8,
        )

        # Setup TMA load for SFB
        sfb_op = sm103_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout_tma_ready = self.adapt_layout_for_tma_sf(sfb_smem_layout)
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.cpasync.make_tiled_tma_atom(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout_tma_ready,
            (self.mma_sf_tiler[1], self.mma_sf_tiler[2]),
            self.cluster_shape_mn[0] // cute.size(dummy_tiled_mma_sfb.thr_id),
            internal_type=cutlass.Uint8,
        )

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_tensor,
                epi_smem_layout,
                self.epi_tile_c if self.fold_pairs_tma else self.epi_tile,
            )

        a_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.a_smem_layout_staged_tma, (None, None, None, 0)),
        )
        b_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.b_smem_layout_staged_tma, (None, None, None, 0)),
        )
        sfa_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        sfb_copy_size = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )
        self.num_tma_load_bytes_ab = (a_copy_size + b_copy_size) * atom_thr_size
        self.num_tma_load_bytes_sf = (sfa_copy_size + sfb_copy_size) * atom_thr_size

        # Compute grid size. For the TMA fold path the compact C is half the
        # GEMM's logical N, so tiling must come from the m2-congruent alias.
        self.tile_sched_params, grid = self._compute_grid(
            c_alias_tensor if self.fold_pairs_tma else c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            sf_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sf_stage]
            sf_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_sf_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            (
                c_alias_tensor
                if self.fold_pairs_tma
                else (tma_tensor_c if self.use_tma_store else c_tensor)
            ),
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
            tma_tensor_c if self.fold_pairs_tma else None,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        mC_fold_tma: cute.Tensor = None,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        # Keep alpha in FP32 for precision: the accumulator is in FP32 and alpha
        # may be a very small scaling factor. Converting to c_dtype (e.g., FP16)
        # before multiplication could cause overflow when acc values are large.
        alpha_value = alpha[0].to(cutlass.Float32)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_ab_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)
        if warp_idx == self.tma_sf_warp_id:
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, sfa+sfb full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_producer and ab_consumer
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes_ab,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize mainloop sf_producer and sf_consumer
        sf_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_sf_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        sf_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_sf_tma_producer
        )
        sf_producer, sf_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.sf_full_mbar_ptr.data_ptr(),
            num_stages=self.num_sf_stage,
            producer_group=sf_producer_group,
            consumer_group=sf_consumer_group,
            tx_count=self.num_tma_load_bytes_sf,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
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
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (BLK_M, BLK_K, m, k, l)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((self.mma_tiler[0], self.mma_tiler[1], 384), (None, 0, None)),
            (None, None, None),
        )
        # (BLK_N, BLK_K, n, k, l)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_((self.mma_tiler[0], self.mma_tiler[1], 384), (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.mma_sf_tiler, (None, 0, None)),
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_sf_tiler, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # TMA fold path: tile the compact D_t by the half-width CTA tile so
        # its tile grid matches the m2-wide accumulator tile grid 1:1.
        gC_fold_mnl = None
        if cutlass.const_expr(self.fold_pairs_tma):
            gC_fold_mnl = cute.local_tile(
                mC_fold_tma,
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1] // 2),
                (None, None, None),
            )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        # create tCgA_tmp
        tCgA_mkl_tmp = thr_mma.partition_A(gA_mkl)
        tCgA_layout = self.append_coalesce_layout(tCgA_mkl_tmp.layout)
        cta_tCgA = cute.make_tensor(tCgA_mkl_tmp.iterator, tCgA_layout)
        # ((CTA_MMA_M,256),Rest_MMA_M,Rest_MMA_K, m, k, l)
        tCgA = cute.make_tensor(
            cta_tCgA.iterator,
            cute.tiled_divide(
                cta_tCgA.layout, (cute.size(tiled_mma.tv_layout_A[1][0]), 128)
            ),
        )

        tCgB_nkl_tmp = thr_mma.partition_B(gB_nkl)
        tCgB_layout = self.append_coalesce_layout(tCgB_nkl_tmp.layout)
        cta_tCgB = cute.make_tensor(tCgB_nkl_tmp.iterator, tCgB_layout)
        # ((CTA_MMA_N,256),Rest_MMA_N, Rest_MMA_K, n, k, l)
        tCgB = cute.make_tensor(
            cta_tCgB.iterator,
            cute.tiled_divide(
                cta_tCgB.layout, (cute.size(tiled_mma.tv_layout_B[1][0]), 128)
            ),
        )

        tCgSFA = cute.make_tensor(
            gSFA_mkl.iterator,
            cute.tiled_divide(
                gSFA_mkl.layout, (self.mma_sf_tiler[0], self.mma_sf_tiler[2])
            ),
        )

        tCgSFB = cute.make_tensor(
            gSFB_nkl.iterator,
            cute.tiled_divide(
                gSFB_nkl.layout, (self.mma_sf_tiler[1], self.mma_sf_tiler[2])
            ),
        )
        tCgC = thr_mma.partition_C(gC_mnl)

        # Create identity tensor for C to use in epilogue predication
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCcC = thr_mma.partition_C(cC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 1),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 1),
        )

        # TMA partition for scale factor A
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA_compact = cute.filter_zeros(tAsSFA)

        # TMA partition for scale factor B
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB_compact = cute.filter_zeros(tBsSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        if cutlass.const_expr(self.overlapping_accum):
            # Two pseudo-buffers overlapping by the SF TMEM footprint. The SF
            # column count is a trace-time constant derived from the same
            # layouts the MMA warp builds; the shared region equals the last
            # (phase 0) / first (phase 1) sf_cols-wide slice of subtiles, so
            # the epilogue releases after draining ceil-ish sf_cols/EPI_N
            # subtile columns (index sf_cols // EPI_N, donor formula).
            _sf_fp = cute.make_ptr(self.sf_dtype, 0, mem_space=cute.AddressSpace.tmem)
            _sfa_cols = tcgen05.find_tmem_tensor_col_offset(
                cute.make_tensor(
                    _sf_fp,
                    blockscaled_utils.make_tmem_layout_sfa(
                        tiled_mma,
                        self.mma_tiler,
                        self.sf_vec_size,
                        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                    ),
                )
            )
            _sfb_cols = tcgen05.find_tmem_tensor_col_offset(
                cute.make_tensor(
                    _sf_fp,
                    blockscaled_utils.make_tmem_layout_sfb(
                        tiled_mma,
                        self.mma_tiler,
                        self.sf_vec_size,
                        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                    ),
                )
            )
            _sf_cols = _sfa_cols + _sfb_cols
            assert _sf_cols < self.cta_tile_shape_mnk[1], (
                f"overlapping accum needs sf cols ({_sf_cols}) < tile N "
                f"({self.cta_tile_shape_mnk[1]})"
            )
            self.iter_acc_early_release_in_epilogue = _sf_cols // self.epi_tile[1]
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (self.cta_tile_shape_mnk[1] - _sf_cols)
                        * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # PDL bookend: always emitted, actual behavior controlled by use_pdl= in .launch()
        griddepcontrol_wait()

        #
        # Construct the scheduler
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp for A/B tensors
        #
        if warp_idx == self.tma_ab_warp_id:
            #
            # Persistent tile scheduling loop for AB loads
            #
            buffers_per_k_tile = 3

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                tAgA_slice = tAgA[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        None,
                        mma_tile_coord_mnl[2],
                    )
                ]
                tBgB_slice = tBgB[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[1],
                        None,
                        mma_tile_coord_mnl[2],
                    )
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = cutlass.Boolean(1)
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # TMA load loop for A/B tensors
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Load buffers_per_k_tile buffers
                    for buffer in cutlass.range(buffers_per_k_tile, unroll_full=True):
                        # Acquire next empty AB buffer
                        ab_empty = ab_producer.acquire_and_advance(peek_ab_empty_status)

                        # TMA load A/B
                        cute.copy(
                            tma_atom_a,
                            cute.group_modes(
                                tAgA_slice[(None, None, buffer, k_tile)], 0, 2
                            ),
                            tAsA[(None, ab_empty.index)],
                            tma_bar_ptr=ab_empty.barrier,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            cute.group_modes(
                                tBgB_slice[(None, None, buffer, k_tile)], 0, 2
                            ),
                            tBsB[(None, ab_empty.index)],
                            tma_bar_ptr=ab_empty.barrier,
                            mcast_mask=b_full_mcast_mask,
                        )

                        # Peek (try_wait) AB buffer empty for next buffer
                        peek_ab_empty_status = cutlass.Boolean(1)
                        # Check if we're not at the last buffer of the last k_tile
                        if not (
                            (k_tile == k_tile_cnt - 1)
                            and (buffer == buffers_per_k_tile - 1)
                        ):
                            peek_ab_empty_status = ab_producer.try_acquire()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Signal end of AB loads
            ab_producer.tail()

        #
        # Specialized TMA load warp for scale factor tensors
        #
        if warp_idx == self.tma_sf_warp_id:
            #
            # Persistent tile scheduling loop for SF loads
            #
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0],
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) SF buffer empty
                sf_producer.reset()
                peek_sf_empty_status = cutlass.Boolean(1)
                peek_sf_empty_status = sf_producer.try_acquire()

                #
                # TMA load loop for scale factors
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Load SF stages based on sf_buffers_per_tile_k
                    for sf_stage in cutlass.range(
                        self.sf_buffers_per_tile_k, unroll_full=True
                    ):
                        # Acquire next empty SF buffer
                        sf_empty = sf_producer.acquire_and_advance(peek_sf_empty_status)

                        tAgSFA_compact = cute.filter_zeros(
                            tAgSFA_slice[
                                (None, k_tile * self.sf_buffers_per_tile_k + sf_stage)
                            ]
                        )
                        tBgSFB_compact = cute.filter_zeros(
                            tBgSFB_slice[
                                (None, k_tile * self.sf_buffers_per_tile_k + sf_stage)
                            ]
                        )

                        # TMA load SFA/SFB for this SF stage
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_compact,
                            tAsSFA_compact[(None, sf_empty.index)],
                            tma_bar_ptr=sf_empty.barrier,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_compact,
                            tBsSFB_compact[(None, sf_empty.index)],
                            tma_bar_ptr=sf_empty.barrier,
                            mcast_mask=sfb_full_mcast_mask,
                        )

                        # Peek (try_wait) SF buffer empty for next stage
                        peek_sf_empty_status = cutlass.Boolean(1)
                        # Check if we're not at the last stage of the last k_tile
                        if not (
                            k_tile == k_tile_cnt - 1
                            and sf_stage == self.sf_buffers_per_tile_k - 1
                        ):
                            peek_sf_empty_status = sf_producer.try_acquire()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Signal end of SF loads
            sf_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )

            MMA_M = self.cta_tile_shape_mnk[0]
            MMA_N_SF = self.cta_n_sf
            MMA_K_SF = self.cta_tile_shape_mnk[2] // 2
            mnBasicBlockShape = (32, 4)
            kBasicBlockShape_single = (self.sf_vec_size, 1)
            mma_iter_SFA_shape = (
                (mnBasicBlockShape, MMA_M // 128),
                kBasicBlockShape_single,
            )
            sSFA_iter_shape = (mma_iter_SFA_shape, 1, MMA_K_SF // self.sf_vec_size)
            sSFA_iter_layout = cute.make_layout(sSFA_iter_shape)
            mma_iter_SFB_shape = (
                (mnBasicBlockShape, MMA_N_SF // 128),
                kBasicBlockShape_single,
            )
            sSFB_iter_shape = (mma_iter_SFB_shape, 1, MMA_K_SF // self.sf_vec_size)
            sSFB_iter_layout = cute.make_layout(sSFB_iter_shape)

            tCtSFA_layout_mma = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, self.sf_vec_size, sSFA_iter_layout
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            tCtSFA_mma = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout_mma)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_layout_mma = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, self.sf_vec_size, sSFB_iter_layout
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tCtSFB_mma = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_mma)

            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            MmasPerSfBuffer = 8 // self.sf_buffers_per_tile_k
            sf_stride = 6 if self.sf_vec_size == 16 else 3

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                if cutlass.const_expr(self.overlapping_accum):
                    # Pseudo-buffer parity: producer phase starts at 1, so
                    # phase ^ 1 gives buffer 0 for the first tile (matches
                    # the consumer, whose phase starts at 0).
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, 0, 0, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                # Peek (try_wait) SF buffer full
                sf_consumer.reset()
                peek_sf_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_sf_full_status = sf_consumer.try_wait()

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                is_first_iteration = True

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        # Conditionally load SFA/SFB for MMA0/MMA1 depending on sf_vec_size
                        if 0 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()

                        # Wait for A/B data to be ready(MMA0, MMA1, part of MMA2)
                        ab_full0 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # peek for next stage (MMA2, MMA3, MMA4, part of MMA5)
                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = ab_consumer.try_wait()

                        # delay the acc acquire to ublock tmem
                        if is_first_iteration:
                            acc_pipeline.producer_acquire(acc_producer_state)
                            is_first_iteration = False

                        # MMA0
                        k_block_coord_cur = (None, 0, 0, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full0.index)
                        sf_kblock_coord = (None, None, 0 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # MMA1
                        k_block_coord_cur = (None, 0, 3, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full0.index)
                        sf_kblock_coord = (None, None, 1 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # Conditionally load SFA/SFB for MMA2/MMA3
                        if 2 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()

                        # Wait for A/B data to be ready(MMA2, MMA3, MMA4, part of MMA5)
                        ab_full1 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # peek for next stage (part of MMA5, MMA6, MMA7)
                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = ab_consumer.try_wait()

                        # MMA2
                        k_block_coord_cur = (None, 0, 6, ab_full0.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 2 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # Release stage_ab_0 as it is no longer needed
                        ab_full0.release()

                        # MMA3
                        k_block_coord_cur = (None, 0, 1, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 3 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # Conditionally load SFA/SFB for MMA4/MMA5
                        if 4 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            peek_sf_full_status = sf_consumer.try_wait()

                        # MMA4
                        k_block_coord_cur = (None, 0, 4, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full1.index)
                        sf_kblock_coord = (None, None, 4 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # Wait for A/B data to be ready(part of MMA5, MMA6, MMA7)
                        ab_full2 = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # peek for next loop's first stage (MMA0, MMA1, part of MMA2)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        # MMA5
                        k_block_coord_cur = (None, 0, 7, ab_full1.index)
                        k_block_coord_next = (None, 0, 0, ab_full2.index)
                        sf_kblock_coord = (None, None, 5 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # Conditionally load SFA/SFB for MMA6/MMA7
                        if 6 % MmasPerSfBuffer == 0:
                            sf_full = sf_consumer.wait_and_advance(peek_sf_full_status)
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                sf_full.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            sf_full.release()
                            peek_sf_full_status = cutlass.Boolean(1)
                            if k_tile + 1 < k_tile_cnt:
                                peek_sf_full_status = sf_consumer.try_wait()

                        ab_full1.release()

                        # MMA6
                        k_block_coord_cur = (None, 0, 2, ab_full2.index)
                        k_block_coord_next = (None, 0, 0, ab_full2.index)
                        sf_kblock_coord = (None, None, 6 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        # MMA7
                        k_block_coord_cur = (None, 0, 5, ab_full2.index)
                        k_block_coord_next = (None, 0, 0, ab_full2.index)
                        sf_kblock_coord = (None, None, 7 % MmasPerSfBuffer * sf_stride)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_mma[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        self.make_desc_and_call_mma(
                            tiled_mma,
                            tCtAcc,
                            sA[k_block_coord_cur],
                            sA[k_block_coord_next],
                            sB[k_block_coord_cur],
                            sB[k_block_coord_next],
                            tCtAcc,
                        )

                        ab_full2.release()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            if cutlass.const_expr(self.fold_pairs_tma):
                # Plain half-width staging layout (no swizzle/ComposedLayout).
                sC = smem.allocate_tensor(
                    element_type=self.c_dtype,
                    layout=c_smem_layout_staged,
                    byte_alignment=128,
                )
            else:
                # (EPI_TILE_M, EPI_TILE_N, STAGE)
                sC = smem.allocate_tensor(
                    element_type=self.c_dtype,
                    layout=c_smem_layout_staged.outer,
                    byte_alignment=128,
                    swizzle=c_smem_layout_staged.inner,
                )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
            # Wrap epilogue_op with alpha scaling. Multiply in f32 and
            # convert back to the output dtype inside the wrapper:
            # cutlass-dsl >= 4.6 rejects implicit f32 -> bf16 narrowing at
            # store time, which the original dtype-promoting lambda hits.
            alpha_epilogue_op = lambda x: epilogue_op(
                (alpha_value * x.to(cutlass.Float32)).to(self.c_dtype)
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # Pre-advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.fold_pairs_tma):
                    # fold gC tiles are CTA-tile sized (128 rows): slice with
                    # the CTA-level coord. For 2-CTA MMA the mma-level coord
                    # is in 256-row units and would skip every peer CTA's
                    # half (the historical NaN of the 2SM tactic).
                    cta_tile_coord_mnl = (
                        cur_tile_coord[0],
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )
                    acc_consumer_state = mext_fold_epilogue_tma(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        gC_fold_mnl,
                        epi_tile,
                        num_tiles_executed,
                        alpha_epilogue_op,
                        cta_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                elif cutlass.const_expr(self.use_tma_store):
                    acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        epi_tile,
                        num_tiles_executed,
                        alpha_epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = mext_fold_epilogue(
                        self,
                        tidx,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        alpha_epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        tCcC_base=tCcC,
                        mC_mnl=mC_mnl,
                    )

            if cutlass.const_expr(self.use_tma_store):
                # Wait for C store complete
                c_pipeline.producer_tail()
            else:
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

            cute.arch.mbarrier_init_fence()

        griddepcontrol_launch_dependents()

    @staticmethod
    def make_desc_and_call_mma(
        tiled_mma: cute.TiledMma,
        d: cute.Tensor,
        sA_cur: cute.Tensor,
        sA_next: cute.Tensor,
        sB_cur: cute.Tensor,
        sB_next: cute.Tensor,
        c: cute.Tensor,
    ) -> None:
        """Specialized GEMM for circular-buffered A/B from SMEM.

        Performs D <- A * B + C where A and B are described by circular SMEM
        descriptors constructed from the (current, next) buffers. C and D may alias.

        Some tcgen05 MMAs require explicitly toggling an accumulate field outside of
        this routine; the caller is responsible for that.

        All tensors must already be partitioned for the provided tiled MMA.

        For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
        election internally. Manual thread selection is not required in such cases.

        :param atom: MMA atom
        :type atom: cute.MmaAtom
        :param d: Destination tensor
        :type d: cute.Tensor
        :param sA_cur: Current shared memory tensor for operand A
        :type sA_cur: cute.Tensor
        :param sA_next: Next shared memory tensor for operand A, used for circular buffering
        :type sA_next: cute.Tensor
        :param sB_cur: Current shared memory tensor for operand B
        :type sB_cur: cute.Tensor
        :param sB_next: Next shared memory tensor for operand B, used for circular buffering
        :type sB_next: cute.Tensor
        :param c: Third source tensor
        :type c: cute.Tensor
        :return: None
        :rtype: None
        """
        a_desc = tcgen05.make_umma_smem_desc(
            sA_cur.iterator,
            sA_cur.layout,
            "k" if tiled_mma.op.a_major_mode.name == "K" else "mn",
            next_src=sA_next.iterator,
        )
        b_desc = tcgen05.make_umma_smem_desc(
            sB_cur.iterator,
            sB_cur.layout,
            "k" if tiled_mma.op.b_major_mode.name == "K" else "mn",
            next_src=sB_next.iterator,
        )

        view_layout = cute.make_layout(1, stride=0)
        a_tensor = cute.make_tensor(a_desc, view_layout)
        b_tensor = cute.make_tensor(b_desc, view_layout)
        return cute.mma_atom_call(tiled_mma, d, a_tensor, b_tensor, c)

    @staticmethod
    def sm103_make_blockscaled_trivial_tiled_mma(
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        cta_group: tcgen05.CtaGroup,
        mma_tiler_mn: Tuple[int, int],
        a_source: tcgen05.OperandSource = tcgen05.OperandSource.SMEM,
    ) -> cute.TiledMma:
        """Create a blockscaled trivial tiled MMA for SM103 (3xFP4), K fixed to 96.

        Returns a tcgen05 MMA configured for the given (M, N) tiler and CTA group.

        :param sf_dtype: Data type of the scale factor (typically 8-bit)
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param cta_group: The CTA group configuration
        :type cta_group: tcgen05.CtaGroup
        :param mma_tiler_mn: The MMA tiler dimensions (M, N)
        :type mma_tiler_mn: Tuple[int, int]
        :param a_source: Source location for operand A (SMEM by default)
        :type a_source: tcgen05.OperandSource

        :return: A tiled MMA atom configured for SM103 blockscaled operations
        :rtype: cute.TiledMma

        :raises TypeError: If the data type is not supported.
        :raises ValueError: If the sf_vec_size is not supported.
        """
        if sf_vec_size == 32:
            mma_op = tcgen05.SM103MmaMXF4Op(
                (*mma_tiler_mn, 96),
                cta_group,
                a_source,
            )
        elif sf_vec_size == 16:
            mma_op = tcgen05.SM103MmaMXF4NVF4Op(
                sf_dtype,
                (*mma_tiler_mn, 96),
                cta_group,
                a_source,
            )
        else:
            raise ValueError(
                f"Unsupported sf_vec_size: {sf_vec_size}. Expected 16 or 32."
            )
        return cute.make_tiled_mma(cute.make_mma_atom(mma_op))

    # Utils
    @staticmethod
    def sm103_make_smem_layout_a(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: cute.Tile,
        num_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """
        Create the SMEM layout for operand A using K_SW128 and Uint8.

        This function creates a SMEM layout for operand A using the make_smem_layout_atom function with K_SW128 kind and Uint8 element type.

        :param tiled_mma: The tiled MMA atom
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The mma tiler shape (M, N, K)
        :type mma_tiler_mnk: cute.Tile
        :param num_stages: The number of stages
        :type num_stages: int

        :return: SMEM layout for operand A
        :rtype: cute.Layout
        """
        is_k_major = tiled_mma.op.a_major_mode == tcgen05.OperandMajorMode.K
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8
            ),
            cute.append(
                (
                    (
                        mma_tiler_mnk[0]
                        // cute.size(tiled_mma.thr_layout_vmnk.shape[0]),
                        16,
                    ),
                    1,
                    8,
                ),
                num_stages,
            ),
            order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        )

        return a_smem_layout_staged

    @staticmethod
    def sm103_make_smem_layout_b(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: cute.Tile,
        num_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        """
        Create the SMEM layout for operand B using K_SW128 and Uint8.

        This function creates a SMEM layout for operand B using the make_smem_layout_atom function with K_SW128 kind and Uint8 element type.

        :param tiled_mma: The tiled MMA atom
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The mma tiler shape (M, N, K)
        :type mma_tiler_mnk: cute.Tile
        :param num_stages: The number of stages
        :type num_stages: int

        :return: SMEM layout for operand B
        :rtype: cute.Layout
        """
        is_k_major = tiled_mma.op.b_major_mode == tcgen05.OperandMajorMode.K
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8
            ),
            cute.append(
                ((mma_tiler_mnk[1] // cute.size(tiled_mma.thr_id.shape), 16), 1, 8),
                num_stages,
            ),
            order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        )
        return b_smem_layout_staged

    @dataclass(frozen=True)
    class Sm103BlockScaledBasicChunk:
        """
        Basic scale-factor atom layout decided by tcgen05 BlockScaled MMA Ops on SM103.

        Represents the fixed layout pattern for scale factors used by tcgen05
        BlockScaled MMA Ops on SM103. The layout is determined by the instruction
        specification and is not configurable.
        """

        sf_vec_size: int
        major_mode: tcgen05.OperandMajorMode = tcgen05.OperandMajorMode.K
        _layout: cute.Layout = field(init=False, repr=False)

        def __post_init__(self) -> None:
            if self.major_mode == tcgen05.OperandMajorMode.K:
                atom_shape = ((8, 4, 4), (self.sf_vec_size, 4))
                atom_stride = ((16, 128, 4), (0, 1))
            else:
                atom_shape = ((self.sf_vec_size, 4), (8, 4, 4))  # type: ignore[assignment]
                atom_stride = ((0, 1), (16, 128, 4))  # type: ignore[assignment]

            object.__setattr__(
                self, "_layout", cute.make_layout(shape=atom_shape, stride=atom_stride)
            )

        @property
        def layout(self) -> cute.Layout:
            return self._layout

    @staticmethod
    def sm103_make_smem_layout_sfa(
        tiled_mma: cute.TiledMma,
        mma_tiler: cute.Tile,
        sf_vec_size: int,
        num_stages: int,
    ) -> cute.Layout:
        """
        Make SMEM layout for SFA based on:
        1) Sm103BlockScaledBasicChunk, 2) MMA tiler, 3) sf_vec_size, 4) stages.

        :param tiled_mma: The tiled MMA
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The mma tiler shape
        :type mma_tiler: cute.Tile
        :param sf_vec_size: The scale factor vector size
        :type sf_vec_size: int
        :param num_stages: The number of stages
        :type num_stages: int

        :return: Smem layout for SFA
        :rtype: cute.Layout
        """
        mma_shape_mk = tiled_mma.partition_shape_A((mma_tiler[0], mma_tiler[2]))
        sf_atom = Sm103BlockScaledPersistentDenseGemmKernel.Sm103BlockScaledBasicChunk(
            sf_vec_size, tiled_mma.op.a_major_mode
        ).layout
        k_divisor = 4 if sf_vec_size == 16 else 2
        mma_sfa_tiler = (
            mma_shape_mk[0][0] * mma_shape_mk[1],
            mma_shape_mk[0][1] * mma_shape_mk[2] // k_divisor,
        )
        sfa_smem_atom_layout = cute.tiled_product(
            sf_atom,
            cute.make_layout(
                cute.shape_div(mma_sfa_tiler, cute.product_each(sf_atom.shape))
            ),
        )
        sfa_smem_layout_staged = cute.make_layout(
            shape=cute.append(sfa_smem_atom_layout.shape, num_stages),
            stride=cute.append(
                sfa_smem_atom_layout.stride,
                cute.size(cute.filter_zeros(sfa_smem_atom_layout)),
            ),
        )
        return sfa_smem_layout_staged

    @staticmethod
    def sm103_make_smem_layout_sfb(
        tiled_mma: cute.TiledMma,
        mma_tiler: cute.Tile,
        sf_vec_size: int,
        num_stages: int,
    ) -> cute.Layout:
        """
        Make SMEM layout for SFB based on the basic chunk, MMA tiler, sf_vec_size, stages.

        :param tiled_mma: The tiled MMA
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The mma tiler shape
        :type mma_tiler: cute.Tile
        :param sf_vec_size: The scale factor vector size
        :type sf_vec_size: int
        :param num_stages: The number of stages
        :type num_stages: int

        :return: Smem layout for SFB
        :rtype: cute.Layout
        """
        sf_atom = Sm103BlockScaledPersistentDenseGemmKernel.Sm103BlockScaledBasicChunk(
            sf_vec_size, tiled_mma.op.a_major_mode
        ).layout
        k_divisor = 4 if sf_vec_size == 16 else 2
        mma_sfb_tiler = (mma_tiler[1], mma_tiler[2] // k_divisor)
        if mma_sfb_tiler[0] == 128:
            sfb_smem_atom_layout = cute.tiled_product(
                sf_atom,
                cute.make_layout(
                    cute.shape_div(mma_sfb_tiler, cute.product_each(sf_atom.shape))
                ),
            )
        else:
            sf_k_major_atom256 = cute.make_layout(
                shape=(
                    (32, 4, 2),
                    (sf_vec_size, 4),
                ),
                stride=(
                    (16, 4, mma_sfb_tiler[1] // sf_vec_size // 4 * 512),
                    (0, 1),
                ),
            )
            sfb_smem_atom_layout = cute.tiled_product(
                sf_k_major_atom256,
                cute.make_layout(
                    cute.shape_div(
                        mma_sfb_tiler, cute.product_each(sf_k_major_atom256.shape)
                    )
                ),
            )

        sfb_smem_layout_staged = cute.make_layout(
            shape=cute.append(sfb_smem_atom_layout.shape, num_stages),
            stride=cute.append(
                sfb_smem_atom_layout.stride,
                cute.size(cute.filter_zeros(sfb_smem_atom_layout)),
            ),
        )
        return sfb_smem_layout_staged

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)
        tCtSF_compact_copy = cute.make_tensor(
            tCtSF_compact.iterator,
            cute.append(
                cute.append(tCtSF_compact[(None, 0, 0)].layout, cute.make_layout((1))),
                cute.make_layout(1),
            ),
        )
        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact_copy)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        use_tma_store: bool,
    ) -> Tuple[int, int, int, int]:
        """Computes the number of stages for A/B and SF operands based on heuristics.

        SM103 requires separate stage counts for AB and SF pipelines.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler: tuple[int, int, int]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int
        :param use_tma_store: Whether TMA store is enabled.
        :type use_tma_store: bool

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, SF stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages - same as SM100 dense blockscaled gemm
        num_acc_stage = 1 if mma_tiler[1] == 256 else 2

        # Default C stages
        num_c_stage = 2 if use_tma_store else 0

        # Calculate smem layout and size for one stage of A, B, SFA, SFB
        a_smem_layout_stage_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_a(
                tiled_mma,
                mma_tiler,
                1,
            )
        )
        b_smem_layout_staged_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_b(
                tiled_mma,
                mma_tiler,
                1,
            )
        )
        sfa_smem_layout_staged_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_sfa(
                tiled_mma,
                mma_tiler,
                sf_vec_size,
                1,
            )
        )
        sfb_smem_layout_staged_one = (
            Sm103BlockScaledPersistentDenseGemmKernel.sm103_make_smem_layout_sfb(
                tiled_mma,
                mma_tiler,
                sf_vec_size,
                1,
            )
        )

        c_smem_layout_staged_one = sm103_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        ab_bytes_per_stage = cute.size_in_bytes(
            cutlass.Uint8, a_smem_layout_stage_one
        ) + cute.size_in_bytes(cutlass.Uint8, b_smem_layout_staged_one)
        sf_bytes_per_stage = cute.size_in_bytes(
            sf_dtype, sfa_smem_layout_staged_one
        ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)

        mbar_helpers_bytes = 1024

        num_ab_stage = (
            smem_capacity // occupancy
            - (mbar_helpers_bytes + sf_bytes_per_stage + c_bytes)
        ) // ab_bytes_per_stage

        num_sf_stage = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * mbar_helpers_bytes
            - occupancy * c_bytes
        ) // (occupancy * sf_bytes_per_stage)

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if use_tma_store:
            # xinyu TODO: not sure if aligned with c++
            num_c_stage += (
                smem_capacity
                - occupancy * ab_bytes_per_stage * num_ab_stage
                - occupancy * sf_bytes_per_stage * num_sf_stage
                - occupancy * mbar_helpers_bytes
                - occupancy * c_bytes
            ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_sf_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype != cutlass.Float4E2M1FN:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        # ResInfer probe: PTX allows N down to 8 for kind::mxf4nvf4; the
        # upstream kernel only wired 128/256 (SFB pipeline rounds N up to
        # 128). N=64 admitted experimentally -- correctness gated by tests.
        if mma_tiler_mn[1] not in [64, 128, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        _is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_alignment(
            dtype, is_mode0_major, tensor_shape, alignment_bytes
        ):
            """Check if tensor satisfies the required byte alignment.

            :param dtype: Data type of the tensor
            :param is_mode0_major: Whether mode 0 is the major (contiguous) mode
            :param tensor_shape: Shape of the tensor (mode0, mode1, batch)
            :param alignment_bytes: Required alignment in bytes (e.g., 16 or 32)
            :return: True if alignment is satisfied
            """
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            # Calculate number of contiguous elements needed for alignment
            # alignment_bytes * 8 (bits per byte) / dtype.width (bits per element)
            num_contiguous_elements = alignment_bytes * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        # Check A/B tensors for 16B alignment
        # Check C tensor for 32B alignment
        if (
            not check_contigous_alignment(ab_dtype, a_major == "m", (m, k, l), 16)
            or not check_contigous_alignment(ab_dtype, b_major == "n", (n, k, l), 16)
            or not check_contigous_alignment(c_dtype, c_major == "m", (m, n, l), 32)
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        use_tma_store: bool,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm103BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement

    # Helper function for append and coalesce layout
    @staticmethod
    def append_coalesce_layout(layout):
        # coalesce is like: cutlass/python/pycute/layout.py:coalesce
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(part1, part2)
        result = cute.append(result, layout[3])
        result = cute.append(result, layout[4])
        result = cute.append(result, layout[5])
        return result

    @staticmethod
    def adapt_layout_for_tma_ab(composed_layout):
        # input:  S<3,4,3> o 0 o ((128,16),1,8,3):((128,1),0,16,16384)
        # output: S<3,4,3> o 0 o (128,(128,3)):(128,(1,16384))
        # for ctaValueMap: (128,384):(1@0,1@1)
        layout = composed_layout.outer
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        part3 = cute.append(part2, layout[3])
        result = cute.append(part1, part3)
        return cute.make_composed_layout(
            composed_layout.inner, composed_layout.offset, result
        )

    @staticmethod
    def adapt_layout_for_tma_sf(layout):
        # TODO: need ethan check this
        # input: (((8,4,4),(16,4)),1,3):(((16,128,4),(0,1)),0,512)
        # output: ((32,4),(16,4,3)):((16,4),(0,1,512))
        # for ctaValueMap: ((8,4,4),(16,4,3)):((1@0@0@0,1@1@0@0,1@2@0@0),(1@0@0@1,1@1@0@1,1@1@1))
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        result = cute.append(cute.group_modes(part1, 0, cute.rank(part1)), part2)
        return result

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        fold_pairs: cutlass.Constexpr = False,
    ):
        """Execute the wrapped GEMM kernel with dynamically shaped tensors.

        Uses TVM-FFI for efficient tensor passing: A, B, C, and alpha are passed
        as cute.Tensor directly (torch tensors at runtime via TVM-FFI's C-level
        dlpack). Scale factor tensors remain as pointers (complex 6D layout).

        Args:
            mA (cute.Tensor): Input A, shape (m, k_packed), Uint8 (FP4 packed).
            mB (cute.Tensor): Input B, shape (n, k_packed), Uint8 (FP4 packed).
            mC (cute.Tensor): Output C, shape (m, n).
            sf_m/sf_n/sf_k: Scale factor dimensions.
            l: Batch dimension.
            a_sf_ptr/b_sf_ptr: Scale factor pointers (6D layout).
            alpha_tensor: Alpha scaling factor, shape (1,), float32.
            max_active_clusters: Max active clusters.
            current_stream: CUDA stream (TVM-FFI fake stream).
            swap_ab: Whether A/B are swapped (controls C layout).
            epilogue_op: Elementwise epilogue function.
        """
        # A/B come in as Uint8 (FP4 packed as uint8 in torch). Recast to FP4.
        m = cute.size(mA, mode=[0])
        k_packed = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])
        k = k_packed * 2  # 2 FP4 values per uint8 byte

        # Recast Uint8 -> Float4E2M1FN and reshape to (m, k, l) K-major.
        # Row strides come from the INCOMING tensors, not from (m, k): they
        # match for a compact operand, but an ext-K residue weight arrives as a
        # `weight_ext[:, :k_base_packed]` view (width k_base, rows k_ext apart)
        # and make_ordered_layout would rebuild it compact, silently reading the
        # wrong elements. Strides are in ELEMENTS, so the uint8 -> FP4 recast
        # (which doubled k) doubles them too.
        lda = cute.size(mA.layout.stride[0]) * 2
        ldb = cute.size(mB.layout.stride[0]) * 2
        a_fp4_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
        a_tensor = cute.make_tensor(
            a_fp4_ptr,
            layout=cute.make_layout((m, k, l), stride=(lda, 1, m * lda)),
        )
        b_fp4_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        b_tensor = cute.make_tensor(
            b_fp4_ptr,
            layout=cute.make_layout((n, k, l), stride=(ldb, 1, n * ldb)),
        )
        # C: swap_ab is constexpr, determines layout at compile time
        if cutlass.const_expr(fold_pairs):
            # MExt R1 fold: A = weight [n_w=m, K], B = row-pair interleaved
            # activation [m2=n, K] with m2 = 2*m_tok. The physical output is
            # the transposed compact D[n_w, m_tok]; logical C (m, (2, m_tok))
            # aliases each (base, residue) pair onto one address (stride-0
            # pair mode) and keeps the m2 direction memory-contiguous so the
            # fold epilogue sees pairs adjacent in its fragment value mode.
            # Row pitch comes from the physical mC width, which the host may
            # pad (e.g. to 8 elements) so vectorized epilogue stores stay
            # 16B-aligned; the logical extent stays n // 2 for predication.
            if cutlass.const_expr(self.fold_pairs_tma):
                # TMA fold: mC is row-major D[m_tok, n_w]; the kernel-facing C
                # is its transposed affine view (n_w contiguous), so the TMA
                # box lands on D directly (no host transpose). The alias is
                # partition-geometry metadata only -- never dereferenced on
                # this path -- so it keeps the m2-contiguous stride fiction
                # that selects the pair-in-value-mode TMEM load atom.
                c_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        (m, n // 2, l), stride=(1, m, m * (n // 2))
                    ),
                )
                c_alias_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        (m, (2, n // 2), l),
                        stride=(n // 2, (0, 1), m * (n // 2)),
                    ),
                )
            else:
                mtok_phys = cute.size(mC, mode=[1])
                c_alias_tensor = cute.make_tensor(
                    mC.iterator,
                    layout=cute.make_layout(
                        (m, (2, n // 2), l),
                        stride=(mtok_phys, (0, 1), m * mtok_phys),
                    ),
                )
                c_tensor = c_alias_tensor
        elif cutlass.const_expr(swap_ab):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )
        # Scale factor tensors: 6D BlockScaledBasicChunk layout from pointers
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )

        if cutlass.const_expr(self.fold_pairs_tma):
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
                c_alias_tensor,
            )
        else:
            self(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                c_tensor,
                alpha_tensor,
                max_active_clusters,
                current_stream,
                epilogue_op,
            )
