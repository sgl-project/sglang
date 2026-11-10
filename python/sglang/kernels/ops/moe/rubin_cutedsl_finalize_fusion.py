# Vendored: FlashInfer v0.7.2 (d84335d8c5) flashinfer/fused_moe/cute_dsl/rubin/
# blockscaled_contiguous_grouped_gemm_finalize_fusion.py with FlashInfer #6307 (91b201a670,
# 'wait for Rubin finalize reductions before shared-buffer reuse') applied as is, and FlashInfer #6950
# (5240384d8f, 'unfused (deterministic) finalize path for the Rubin CuTe DSL fused MoE') adapted to v0.7.2's
# epilogue (v0.7.2 has no locality-domain column bound, so the full-tile copy size is kept). BSD-3-Clause.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Rubin (SM107) Contiguous Grouped Blockscaled GEMM Kernel with Finalize Fusion

This module implements a contiguous grouped GEMM kernel for Rubin architecture
with fused MoE finalize (scatter-add) operation.

Key features:
- Rubin-specific MMA features (B-reuse pattern, CollectorOp, etc.)
- Tile scheduling logic for contiguous grouped GEMM
- Fused finalize operation with atomic add for MoE scatter-add

The finalize fusion performs:
1. GEMM: C_permuted = alpha * A * SFA * B * SFB
2. Scatter-add: c[token_idx] += token_scale * C_permuted[permuted_row]


"""

from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

# ============================================================================
# Inline utility functions
# ============================================================================


@dsl_user_op
def blk_copy(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block copy using cp.async.bulk (plain store, no reduction)."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block reduce for BF16 using cp.reduce.async.bulk."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp32(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block reduce for FP32 using cp.reduce.async.bulk."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Block reduce for FP16 using cp.reduce.async.bulk."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


class Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel:
    """Rubin (SM107) Contiguous Grouped Blockscaled GEMM Kernel with Finalize Fusion.

    This kernel implements batched matrix multiplication (c = scatter_add(alpha * A x SFA x B x SFB * token_scale))
    with contiguous grouped GEMM support and fused MoE finalize for Rubin GPUs.

    Key features:
    - Persistent tile scheduling with dedicated scheduler warp
    - Warp specialization (scheduler, TMA, MMA, epilogue warps)
    - Support for B-reuse pattern (Bkeep-Breuse)
    - Per-group alpha scaling
    - Fused finalize with atomic add scatter

    :param sf_vec_size: Scale factor vector size (16 or 32)
    :param mma_inst_shape: Shape of MMA instruction (M, N, K)
    :param mma_tiler: Shape of MMA tiler (M, N, K)
    :param cluster_shape_mn: Cluster dimensions (M, N)
    :param raster_along_m: If True, raster tiles along M dimension first
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        raster_along_m: bool = False,
        topK: int = 1,
        enable_pdl: Optional[bool] = None,
        use_fused_finalize: bool = True,
    ):
        # flashinfer dispatcher compatibility: the pre-sync kernel took
        # `enable_pdl`; the TRT-LLM version has no PDL path, so the value is
        # accepted and ignored.
        del enable_pdl
        self.sf_vec_size = sf_vec_size
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        self.topK = topK
        self.use_fused_finalize = use_fused_finalize

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1

        # Warp IDs for warp specialization
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
            )
        )

        # Set barriers for synchronization
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id
        self.epilog_sync_bar_id = self.epilog_sync_barrier.barrier_id

        # B-reuse pattern control
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        cta_tile: Tuple[int, int, int],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        with_breuse: bool,
    ) -> Tuple[int, int, int, int]:
        """Compute the number of stages for A/B/C/tile_info operands."""
        # ACC stages
        # Note that here we have assumed the kernel have access to all TMEM capacity
        # associated with sm_107 architecture.
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages and tile info stages (5 elements now: bidx, bidy, bidz, valid, mn_limit)
        num_c_stage = 1  # Thinking about it
        num_tile_stage = 2

        # Calculate smem layout and size for one stage
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )

        # Shared memory for epilogue block reduce (if enabled)
        swizzled_pad = 16 // (c_dtype.width // 8)
        c_smem_layout_staged_one = cute.make_layout(
            (cta_tile[0], cta_tile[1]), stride=(cta_tile[1] + swizzled_pad, 1)
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs."""
        # Compute mma instruction shapes
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm107_utils.compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup stage counts
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.cta_tile_shape_mnk,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        # C smem layout for block reduce (if enabled)
        swizzled_pad = 16 // (self.c_dtype.width // 8)
        self.c_smem_layout_staged = cute.make_layout(
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], self.num_c_stage),
            stride=(
                self.cta_tile_shape_mnk[1] + swizzled_pad,
                1,
                self.cta_tile_shape_mnk[0] * (self.cta_tile_shape_mnk[1] + 8),
            ),
        )

        # Compute TMEM layouts for SFA/SFB
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Compute TMEM column counts
        # Each column entry in TMEM is 32-bit wide, and so we recast the TMEM layout
        # from its original data type to a 32-bit wide data type. Moreover, TMEM
        # addresses are expressed as (row << 16) | col, which in CUTE are expressed
        # as an affine transformation row * (1<<16) + col, which can be seen as a CUTE
        # layout of (row, col):(1<<16, 1). As a result, by masking out the upper 16 bits
        # (keeping only the lower 16 bits), we extract the cosize corresponding
        # to only the columns.
        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage * (2 if self.enable_breuse else 1)
        )

        # Epilogue vectorization config
        if cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
            self.element_offset = 8
            self.epi_loop_size = self.epi_tile_n // 8
        elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
            self.element_offset = 2
            self.epi_loop_size = self.epi_tile_n // 2
        else:
            self.element_offset = 1
            self.epi_loop_size = self.epi_tile_n

        # copy_size is in bytes for cp.reduce.async.bulk instruction
        self.copy_size = self.cta_tile_shape_mnk[1] * (self.c_dtype.width // 8)

    def _is_interleaved_utccp(self) -> bool:
        """Enable interleaving UTCCP for Bkeep-Breuse case for 4xFP4 kernel."""
        return self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1]))
            layout = cute.append(cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _mainloop_s2t_copies(
        self,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Copy SFA/SFB from smem to tmem."""
        s2t_stage_coord = (None, None, None, None, stage_idx)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfa_s2t_bundle.tSF_compact,
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Interleaved UTCCP for Bkeep-Breuse pattern."""
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)

        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with finalize fusion.

        :param a: Input tensor A (permuted_m, k, 1)
        :param b: Input tensor B (n, k, l)
        :param c: Output tensor (seq_len, n, 1)
        :param sfa: Scale factor tensor A
        :param sfb: Scale factor tensor B
        :param tile_idx_to_group_idx: Mapping from tile index to group ID
        :param num_non_exiting_tiles: Number of valid tiles
        :param tile_idx_to_mn_limit: M limit for each tile
        :param alpha: Alpha tensor for each group
        :param max_active_clusters: Maximum number of active clusters
        :param stream: CUDA stream
        :param permuted_idx_to_expanded_idx: Mapping from permuted row to expanded index
        :param token_final_scales: Final scales for each token (seq_len, topK)
        :param epilogue_op: Optional epilogue operation
        """
        # Setup static attributes
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.final_scale_dtype = cutlass.Float32
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.ROW_MAJOR  # Always N-major for GEMM output

        # Check data types
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes
        self._setup_attributes()

        # Setup sfa/sfb tensor
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Handle cta_tile_shape_n=192 case
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Compute grid size based on GEMM shape
        gemm_m = a.shape[0]
        gemm_n = b.shape[0]
        gemm_l = a.shape[2]
        gemm_shape = (gemm_m, gemm_n, gemm_l)

        self.tile_sched_params, grid = self._compute_grid(
            gemm_shape,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            # (bidx, bidy, expert_idx, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        token_final_scales: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel for contiguous grouped GEMM with finalize fusion."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize pipelines
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize tile info pipeline
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Initialize tensor memory allocator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensors
        sC = storage.sC.get_tensor(c_smem_layout_staged)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        # Compute multicast masks
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

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        # Partition global tensors for TiledMMA
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition for TMA load
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

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
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Partition for MMA
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized Scheduler warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_valid_tiles = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)

                    tile_idx = mma_tile_coord_m

                    if tile_idx < num_valid_tiles:
                        expert_idx = tile_idx_to_group_idx[tile_idx]
                        mn_limit = tile_idx_to_mn_limit[tile_idx]

                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)

                    tile_idx = mma_tile_coord_m

                    if tile_idx < num_valid_tiles:
                        expert_idx = tile_idx_to_group_idx[tile_idx]
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            # Signal end of work
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, 0)]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                # Get next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # TMEM pointer offset for cta_tile_shape_n=192 or 64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                # MMA mainloop
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            self._mainloop_s2t_copies(
                                ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                            )

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (
                                    None,
                                    0,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                a_kblk_crd_reuse = (
                                    None,
                                    1,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)

                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block,
                                        ab_consumer_state.index,
                                        sfa_s2t_bundle,
                                        sfb_s2t_bundle,
                                    )

                                # Bkeep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Breuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (
                                    None,
                                    None,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                sf_kblk_crd = (None, None, k_block)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Get next tile (4 elements)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps with finalize fusion
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Layout transformation for tCtAcc_base and tCgC
            # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
            # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest)
            tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_transformed = transform_partitioned_tensor_layout(tCgC)

            # Partition for epilogue
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = epilogue_tmem_copy_and_partition(
                self, epi_tidx, tCtAcc, tCgC_transformed, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

            # Setup smem copy for block reduce
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(atom, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            token_idx = cutlass.Int32(0)
            token_scale = self.final_scale_dtype(0.0)

            # Get first tile info (5 elements)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                # Compute base row index for this tile
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]

                # Get accumulator stage index
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Group tRS_sC modes 1 and 2 (m_iter, n_subtile) into a single 2D mode
                # Before: ((1,32), 2, 4, 1) -> After: ((1,32), (2,4), 1)
                # This allows using tuple indexing like tTR_tAcc
                tRS_sC_grouped = cute.group_modes(tRS_sC, 1, 3)

                # Get m-iteration count and n-subtile count from tTR_tAcc shape
                # In B-reuse (cta_tile_m=256): mode-3 is (2,4), m_iter_cnt=2, n_subtile_cnt=4
                # In non-B-reuse (cta_tile_m=128): mode-3 is (1,4), m_iter_cnt=1, n_subtile_cnt=4
                m_iter_cnt = cute.size(tTR_tAcc.shape[3], mode=[0])
                n_subtile_cnt = cute.size(tTR_tAcc.shape[3], mode=[1])

                # TODO:How could reduction be done with a better perf?
                # Process all m-iterations and n-subtiles
                for m_iter_idx in cutlass.range(m_iter_cnt):
                    # Compute row indices for this m-iteration
                    # Each thread handles row: tile_m_start + m_iter_idx * 128 + epi_tidx
                    permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                    expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                    is_valid_row = permuted_row < tile_info[4]

                    # Compute token info and scaled alpha for this row
                    token_idx = cutlass.Int32(0)
                    alpha_val_iter = alpha_val
                    if cutlass.const_expr(self.use_fused_finalize):
                        if is_valid_row:
                            token_idx = expanded_idx // self.topK
                            topk_idx = expanded_idx % self.topK
                            token_scale = token_final_scales[(token_idx, topk_idx)]
                            alpha_val_iter = alpha_val * token_scale

                    for n_iter_idx in cutlass.range(n_subtile_cnt):
                        # Load accumulator from tensor memory buffer to register
                        # Index with (m_iter_idx, subtile_idx) for the composite mode-3
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, (m_iter_idx, n_iter_idx))]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # For block reduce: retile then load, compute, store
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec_final = (alpha_val_iter * acc_vec).to(self.c_dtype)
                        tRS_rC.store(acc_vec_final)
                        if is_valid_row:
                            # Use grouped tRS_sC with tuple indexing to preserve rank
                            # tRS_sC_grouped: ((1,32), (2,4), 1)
                            # Index: (None, (m_iter_idx, subtile_idx), None) -> ((1,32), 1, 1) rank 3
                            cute.copy(
                                tiled_copy_r2s,
                                # TODO: check if there is a better way to index and make same rank
                                tRS_rC[None, 0, 0],
                                tRS_sC_grouped[(None, (m_iter_idx, n_iter_idx), 0)],
                            )

                        cute.arch.fence_proxy("async.shared", space="cta")

                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # TODO:Currently finish all sts and do all reduce, do we have a better way for epilogue overlapping?
                # Block reduce for all m-iterations
                for m_iter_idx in cutlass.range(m_iter_cnt):
                    permuted_row = tile_m_start + m_iter_idx * 128 + epi_tidx
                    is_valid_row = permuted_row < tile_info[4]

                    if is_valid_row:
                        expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                        # Fused: accumulate into the token's row. Unfused: each
                        # expanded slot is written by exactly one permuted row.
                        output_idx = expanded_idx
                        if cutlass.const_expr(self.use_fused_finalize):
                            output_idx = expanded_idx // self.topK
                        coord_n = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                        scatter_out_offset = cute.domain_offset((output_idx, coord_n, 0), c)

                        # sC row index: m_iter_idx * 128 + epi_tidx
                        sC_row = m_iter_idx * 128 + epi_tidx

                        if cutlass.const_expr(not self.use_fused_finalize):
                            blk_copy(
                                scatter_out_offset,
                                sC[sC_row, None, 0],
                                cutlass.Int32(self.copy_size),
                            )
                        elif cutlass.const_expr(self.c_dtype == cutlass.BFloat16):
                            blk_reduce_bf16(
                                scatter_out_offset,
                                sC[sC_row, None, 0],
                                cutlass.Int32(self.copy_size),
                            )
                        elif cutlass.const_expr(self.c_dtype == cutlass.Float32):
                            blk_reduce_fp32(
                                scatter_out_offset,
                                sC[sC_row, None, 0],
                                cutlass.Int32(self.copy_size),
                            )
                        elif cutlass.const_expr(self.c_dtype == cutlass.Float16):
                            blk_reduce_fp16(
                                scatter_out_offset,
                                sC[sC_row, None, 0],
                                cutlass.Int32(self.copy_size),
                            )
                # The next tile reuses sC. Finish the async reductions' reads
                # before any epilogue thread overwrites its shared rows.
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()
                # ============================================================
                # END OF NEW CODE
                # ============================================================

                # Get next tile (5 elements)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Dealloc tensor memory
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    @staticmethod
    def _compute_grid(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid size based on GEMM shape."""
        (m, n, l) = gemm_shape  # noqa: E741

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l

        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Check if the dtypes and sf_vec_size are valid combinations."""
        valid_combinations = {
            # 4xFP4
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
            (cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 32),
            # 2xFP8
            (cutlass.Float8E5M2, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
        }

        current_combination = (a_dtype, b_dtype, sf_dtype, sf_vec_size)
        if current_combination not in valid_combinations:
            return False

        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            return False

        return True

    @staticmethod
    def is_valid_layouts(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if layouts and dtypes are valid combinations."""
        if (
            a_dtype is cutlass.Float4E2M1FN
            and b_dtype is cutlass.Float4E2M1FN
            and not (a_major == "k" and b_major == "k")
        ):
            return False

        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            return False

        return True

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        # Check valid mma_inst_shape
        if mma_inst_shape[0] not in [128, 256]:
            return False
        if mma_inst_shape[1] not in [64, 128, 192, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [128, 256, 512]:
            return False
        if mma_tiler[1] not in [64, 128, 192, 256]:
            return False

        # Check MMA tiler vs MMA instruction relationship
        b_reuse = mma_tiler[0] // mma_inst_shape[0] == 2
        if mma_tiler[0] != mma_inst_shape[0] and not b_reuse:
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # Check K-dimension constraints
        if a_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2} and b_dtype in {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            if mma_tiler[2] != 128 or mma_inst_shape[2] != 64:
                return False
        else:
            if mma_tiler[2] != 256 or mma_inst_shape[2] != 128:
                return False

        # Check cluster shape
        if cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
            return False

        # Check cluster shape validity
        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the tensor alignment is valid (16B alignment)."""

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            return False

        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_stride_row: cutlass.Int64 = cutlass.Int64(0),
    ):
        scale_k = k // scaling_vector_size
        num_tiles = m // tile_size
        a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)))
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        actual_c_stride_row = n if c_stride_row == 0 else c_stride_row
        output_rows = num_tokens
        if cutlass.const_expr(not self.use_fused_finalize):
            output_rows = num_tokens * top_k
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout(
                (output_rows, n, 1),
                stride=(actual_c_stride_row, 1, output_rows * actual_c_stride_row),
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        permuted_idx_to_expanded_idx = cute.make_tensor(
            permuted_idx_to_expanded_idx_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
        )

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            epilogue_op=epilogue_op,
        )

    @staticmethod
    def can_implement(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Check if the GEMM can be implemented."""
        # Check data types
        if not Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, b_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            return False

        # Check layouts
        if not Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_layouts(
            a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        # Check MMA tiler and cluster shape
        if not Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        # Check tensor alignment
        if not Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        return True


# ============================================================================
# Run utilities
# ============================================================================


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification layout."""
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]
