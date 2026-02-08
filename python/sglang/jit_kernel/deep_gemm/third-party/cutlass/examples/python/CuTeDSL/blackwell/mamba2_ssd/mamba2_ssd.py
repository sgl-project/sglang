# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
from typing import List, Type, Tuple, Optional
import cuda.bindings.driver as cuda

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from mamba2_ssd_reference import (
    ssd_reference_fp32_all,
    ssd_reference_lowprecision_intermediates,
    analyze_relative_diffs,
)
from mamba2_ssd_tile_scheduler import (
    Mamba2SSDTileSchedulerParams,
    Mamba2SSDTileScheduler,
)


class SSDKernel:
    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        cumsum_delta_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        L: int,
        D: int,
        N: int,
        has_d: bool,
        d_has_hdim: bool,
    ):
        self.io_dtype: Type[cutlass.Numeric] = io_dtype
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.cumsum_delta_dtype: Type[cutlass.Numeric] = cumsum_delta_dtype
        # has_d means epilog warp performs Y += X*D fusion
        self.has_d: bool = has_d
        # d_has_hdim = True means D is (D, EH) shape and loaded by TMA
        # d_has_hdim = False means D is (1, EH) shape and loaded directly to register
        self.d_has_hdim: bool = d_has_hdim
        self.tile_shape = (L, D, N)

        assert io_dtype in {
            cutlass.Float16,
            cutlass.BFloat16,
        }, "Do not support other I/O types."
        assert acc_dtype in {cutlass.Float32}, "Do not support other ACC types."
        assert cumsum_delta_dtype in {
            cutlass.Float32
        }, "Do not support other cumsum types."
        assert not (not has_d and d_has_hdim), "D cannot have Hdim if has_d is False"

        # Hardcode default setting
        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.epi_tile = (128, 32)

        # Setup mma tile shapes
        self.tile_shape_mnk_intra1 = (L, L, N)
        self.tile_shape_mnk_intra2 = (L, D, L)
        self.tile_shape_mnk_inter1 = (N, D, L)
        self.tile_shape_mnk_inter2 = (L, D, N)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        # Launch config
        self.occupancy = 1
        self.mma_inter_warp_id = 0
        self.mma_intra_warp_id = 1
        self.tma_b_c_warp_id = 2
        self.tma_deltas_x_d_warp_id = 3
        self.pre_inter_warp_id = [4, 5, 6, 7]
        self.pre_intra_warp_id = [8, 9, 10, 11]
        self.epilog_warp_id = [12, 13, 14, 15]
        self.threads_per_cta = 32 * len(
            (
                self.mma_inter_warp_id,
                self.mma_intra_warp_id,
                self.tma_b_c_warp_id,
                self.tma_deltas_x_d_warp_id,
                *self.pre_inter_warp_id,
                *self.pre_intra_warp_id,
                *self.epilog_warp_id,
            )
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        # Named barriers
        self.pre_inter_sync_bar_id = 1
        self.epilog_sync_bar_id = 2
        self.pre_intra_sync_bar_id = 3
        self.tmem_dealloc_sync_bar_id = 4

        # Number of registers used by each warp
        self.num_regs_uniform_warps = 24
        self.num_regs_pre_inter_warps = 168
        self.num_regs_pre_intra_warps = 208
        self.num_regs_epilogue_warps = 112

        # Shared storage
        self.shared_storage = None

        # TMEM buffer offsets
        self.tmem_intra1_acc_offset = 0
        self.tmem_intra2_q_offset = 0
        self.tmem_intra2_acc_offset = 0
        self.tmem_inter1_acc_offset = 0
        self.tmem_inter2_acc_offset = 0
        self.num_tmem_cols_total = 0

    def _setup_attributes(self):
        (
            tiled_mma_intra1,
            tiled_mma_intra2,
            tiled_mma_inter1,
            tiled_mma_inter2,
        ) = self.make_tiled_mmas(
            self.io_dtype,
            self.acc_dtype,
            self.cta_group,
            self.tile_shape_mnk_intra1,
            self.tile_shape_mnk_intra2,
            self.tile_shape_mnk_inter1,
            self.tile_shape_mnk_inter2,
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_intra1.thr_id.shape,),
        )

        # Setup stages
        (
            self.input_stages,
            self.output_stages,
            self.internal_stages,
            self.intra1_acc_stages,
        ) = self._compute_stages(
            self.smem_capacity,
        )

        # Setup smem layouts
        # X is B operand (from smem) of INTRA2_MMA and INTER1_MMA
        self.x_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_intra2,
            self.tile_shape_mnk_intra2,
            self.io_dtype,
            self.input_stages,
        )
        self.num_x_load_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.x_smem_layout, (None, None, None, 0))
        )

        # XT is same shape as ACC operand of INTER2_MMA, before postprocessing by EPILOG
        self.xt_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            self.tile_shape_mnk_intra2[:2],
            self.input_stages,
        )

        # B is B operand (from smem) of INTRA1_MMA
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_intra1,
            self.tile_shape_mnk_intra1,
            self.io_dtype,
            self.input_stages,
        )
        self.num_b_load_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.b_smem_layout, (None, None, None, 0))
        )

        # B_INTERNAL is also A operand (from smem) of INTER1_MMA, after preprocessed by PRE_INTER
        self.bt_internal_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_inter1,
            self.tile_shape_mnk_inter1,
            self.io_dtype,
            self.internal_stages,
        )

        # B needs to be proprocessed to be used as A operand of INTER1_MMA
        self.bt_smem_layout = cute.coalesce(
            sm100_utils.make_smem_layout_epi(
                self.io_dtype,
                utils.LayoutEnum.ROW_MAJOR,
                (self.tile_shape_mnk_inter1[0], self.tile_shape_mnk_inter1[2]),
                self.input_stages,
            ),
            target_profile=(1, 1, 1),
        )

        # C is A operand (from smem) of INTRA1_MMA and INTER2_MMA
        self.c_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_intra1,
            self.tile_shape_mnk_intra1,
            self.io_dtype,
            self.input_stages,
        )
        self.num_c_load_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.c_smem_layout, (None, None, None, 0))
        )

        # P is B operand (from smem) of INTER2_MMA, after preprocessed by PRE_INTER
        self.p_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_inter2,
            self.tile_shape_mnk_inter2,
            self.io_dtype,
            self.internal_stages,
        )

        # PT is ACC operand (from tmem) of INTER1_MMA, after postprocessed by PRE_INTER
        self.pt_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            self.tile_shape_mnk_inter1[:2],
            self.internal_stages,
        )

        # Q is A operand (from tmem) of INTRA2_MMA, after preprocessed by PRE_INTRA
        self.q_tmem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_intra2,
            self.tile_shape_mnk_intra2,
            self.io_dtype,
            self.internal_stages,
        )

        # P is ACC operand (from tmem) of INTER1_MMA, to be TMA stored by PRE_INTER
        self.p_smem_layout_store = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.tile_shape_mnk_inter2[1:],
            self.internal_stages,
        )

        # Y is ACC operand (from smem) of INTER2_MMA and INTRA2_MMA, after postprocessed and TMA stored by EPILOG
        self.y_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            self.epi_tile,
            self.output_stages,
        )

        # Delta is linear smem layouts for pre/post processing
        self.delta_linear_smem_layout = cute.make_layout(
            (self.tile_shape_mnk_inter1[2], self.input_stages)
        )
        self.num_delta_load_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.delta_linear_smem_layout, (None, 0))
        )

        # Cumsum delta is linear smem layouts for pre/post processing
        self.cumsum_delta_linear_smem_layout = cute.make_layout(
            (self.tile_shape_mnk_inter1[2], self.input_stages)
        )
        self.num_cumsum_delta_load_bytes = cute.size_in_bytes(
            self.cumsum_delta_dtype,
            cute.slice_(self.cumsum_delta_linear_smem_layout, (None, 0)),
        )

        # D is linear smem layouts when d_has_hdim is True
        self.d_linear_smem_layout = (
            cute.make_layout((self.tile_shape_mnk_inter2[1], self.input_stages))
            if self.d_has_hdim
            else None
        )
        self.num_d_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype,
                cute.slice_(self.d_linear_smem_layout, (None, 0)),
            )
            if self.d_has_hdim
            else 0
        )

        # Setup tmem offsets
        (
            self.tmem_intra1_acc_offset,
            self.tmem_intra2_q_offset,
            self.tmem_intra2_acc_offset,
            self.tmem_inter1_acc_offset,
            self.tmem_inter2_acc_offset,
            self.num_tmem_cols_total,
        ) = self._plan_tmem_offsets(
            tiled_mma_intra1,
            self.tile_shape_mnk_intra1,
            tiled_mma_intra2,
            self.tile_shape_mnk_intra2,
            tiled_mma_inter1,
            self.tile_shape_mnk_inter1,
            tiled_mma_inter2,
            self.tile_shape_mnk_inter2,
            self.internal_stages,
            self.q_tmem_layout,
            self.io_dtype,
            self.internal_stages,
            self.intra1_acc_stages,
        )

        return

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        cumsum_delta: cute.Tensor,
        delta: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        y: cute.Tensor,
        fstate: cute.Tensor,
        d: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        (
            tiled_mma_intra1,
            tiled_mma_intra2,
            tiled_mma_inter1,
            tiled_mma_inter2,
        ) = self.make_tiled_mmas(
            self.io_dtype,
            self.acc_dtype,
            self.cta_group,
            self.tile_shape_mnk_intra1,
            self.tile_shape_mnk_intra2,
            self.tile_shape_mnk_inter1,
            self.tile_shape_mnk_inter2,
        )

        # Setup TMA atoms and convert TMA tensors
        # TMA load for A
        x_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, tiled_mma_intra2.thr_id
        )
        tma_atom_x, tma_tensor_x = cute.nvgpu.make_tiled_tma_atom_B(
            x_op,
            x,
            cute.slice_(self.x_smem_layout, (None, None, None, 0)),
            self.tile_shape_mnk_intra2,
            tiled_mma_intra2,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if x.element_type is cutlass.Float32 else None
            ),
        )

        # TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, tiled_mma_intra1.thr_id
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            cute.slice_(self.b_smem_layout, (None, None, None, 0)),
            self.tile_shape_mnk_intra1,
            tiled_mma_intra1,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        # TMA load for C
        c_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mnk, tiled_mma_intra1.thr_id
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.make_tiled_tma_atom_A(
            c_op,
            c,
            cute.slice_(self.c_smem_layout, (None, None, None, 0)),
            self.tile_shape_mnk_intra1,
            tiled_mma_intra1,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if c.element_type is cutlass.Float32 else None
            ),
        )

        # TMA load for delta
        # TODO: use bulkcp instead of tma
        delta_cta_v_layout = cute.slice_(
            cute.make_identity_layout(delta.shape), (None, 0, 0, 0)
        )
        delta_linear_smem_layout = cute.slice_(self.delta_linear_smem_layout, (None, 0))
        tma_atom_delta, tma_tensor_delta = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            delta,
            delta_linear_smem_layout,
            delta_cta_v_layout,
        )

        # TMA load for cumsum_delta
        cumsum_delta_cta_v_layout = cute.slice_(
            cute.make_identity_layout(cumsum_delta.shape), (None, 0, 0, 0)
        )
        cumsum_delta_linear_smem_layout = cute.slice_(
            self.cumsum_delta_linear_smem_layout, (None, 0)
        )
        (
            tma_atom_cumsum_delta,
            tma_tensor_cumsum_delta,
        ) = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cumsum_delta,
            cumsum_delta_linear_smem_layout,
            cumsum_delta_cta_v_layout,
        )

        tma_atom_d = None
        tma_tensor_d = d
        # TMA load for D
        if cutlass.const_expr(self.d_has_hdim):
            d_cta_v_layout = cute.slice_(cute.make_identity_layout(d.shape), (None, 0))
            d_linear_smem_layout = cute.slice_(self.d_linear_smem_layout, (None, 0))
            (
                tma_atom_d,
                tma_tensor_d,
            ) = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                d,
                d_linear_smem_layout,
                d_cta_v_layout,
            )

        # TMA store for y
        y_cta_v_layout = cute.composition(
            cute.make_identity_layout(y.shape), self.epi_tile
        )
        y_smem_layout = cute.slice_(self.y_smem_layout, (None, None, 0))
        tma_atom_y, tma_tensor_y = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            y,
            y_smem_layout,
            y_cta_v_layout,
        )

        # TMA store for fstate(p)
        p_cta_v_layout = cute.slice_(
            cute.make_identity_layout(fstate.shape), (None, None, 0, 0)
        )
        p_smem_layout_store = cute.slice_(self.p_smem_layout_store, (None, None, 0))
        tma_atom_p, tma_tensor_p = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            fstate,
            p_smem_layout_store,
            p_cta_v_layout,
        )

        # Compute grid size
        tile_sched_params, grid = self._compute_grid(y, b, max_active_clusters)

        # Plan shared memory storage
        swizzle_buffer_align_bytes = 1024
        nonswizzle_buffer_align_bytes = 128

        @cute.struct
        class SharedStorage:
            # Input stage barriers
            x_full: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            x_empty: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            b_full: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            b_empty: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            c_full: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            c_empty: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            deltas_full: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            deltas_empty: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            d_full: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            d_empty: cute.struct.MemRange[cutlass.Int64, self.input_stages]  # type: ignore
            # Intra1 acc stage barriers
            intra1_acc_full: cute.struct.MemRange[cutlass.Int64, self.intra1_acc_stages]  # type: ignore
            intra1_acc_empty: cute.struct.MemRange[cutlass.Int64, self.intra1_acc_stages]  # type: ignore
            # Internal stage barriers
            intra2_q_full: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            intra2_q_empty: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            intra2_acc_full: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            intra2_acc_empty: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter1_b_full: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter1_b_empty: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter1_acc_full: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter1_acc_empty: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter2_p_full: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter2_p_empty: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter2_acc_full: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            inter2_acc_empty: cute.struct.MemRange[cutlass.Int64, self.internal_stages]  # type: ignore
            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32
            # Smem tensors
            smem_x: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.x_smem_layout)],
                swizzle_buffer_align_bytes,
            ]
            smem_b: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.b_smem_layout)],
                swizzle_buffer_align_bytes,
            ]
            smem_bt_internal: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(self.bt_internal_smem_layout)
                ],
                swizzle_buffer_align_bytes,
            ]
            smem_c: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.c_smem_layout)],
                swizzle_buffer_align_bytes,
            ]
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.p_smem_layout)],
                swizzle_buffer_align_bytes,
            ]
            smem_y: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.y_smem_layout)],
                swizzle_buffer_align_bytes,
            ]
            smem_cumsum_delta: cute.struct.Align[
                cute.struct.MemRange[
                    self.cumsum_delta_dtype,
                    cute.cosize(self.cumsum_delta_linear_smem_layout),
                ],
                nonswizzle_buffer_align_bytes,
            ]
            smem_delta: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(self.delta_linear_smem_layout)
                ],
                nonswizzle_buffer_align_bytes,
            ]
            smem_d: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype,
                    cute.cosize(self.d_linear_smem_layout) if self.d_has_hdim else 0,
                ],
                nonswizzle_buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"SharedStorage size {self.shared_storage.size_in_bytes()} exceeds smem_capacity {self.smem_capacity}"
            )

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_x,
            tma_tensor_x,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_p,
            tma_tensor_p,
            tma_atom_y,
            tma_tensor_y,
            tma_atom_delta,
            tma_tensor_delta,
            tma_atom_cumsum_delta,
            tma_tensor_cumsum_delta,
            tma_atom_d,
            tma_tensor_d,
            self.cluster_layout_vmnk,
            self.x_smem_layout,
            self.xt_smem_layout,
            self.b_smem_layout,
            self.bt_smem_layout,
            self.bt_internal_smem_layout,
            self.c_smem_layout,
            self.pt_smem_layout,
            self.p_smem_layout,
            self.q_tmem_layout,
            self.p_smem_layout_store,
            self.y_smem_layout,
            self.delta_linear_smem_layout,
            self.cumsum_delta_linear_smem_layout,
            self.d_linear_smem_layout,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            min_blocks_per_mp=1,
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_x: cute.CopyAtom,
        tma_tensor_x: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        tma_tensor_b: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        tma_tensor_c: cute.Tensor,
        tma_atom_p: cute.CopyAtom,
        tma_tensor_p: cute.Tensor,
        tma_atom_y: cute.CopyAtom,
        tma_tensor_y: cute.Tensor,
        tma_atom_delta: cute.CopyAtom,
        tma_tensor_delta: cute.Tensor,
        tma_atom_cumsum_delta: cute.CopyAtom,
        tma_tensor_cumsum_delta: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        tma_tensor_d: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        x_smem_layout: cute.ComposedLayout,
        xt_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        bt_smem_layout: cute.ComposedLayout,
        bt_internal_smem_layout: cute.ComposedLayout,
        c_smem_layout: cute.ComposedLayout,
        pt_smem_layout: cute.ComposedLayout,
        p_smem_layout: cute.ComposedLayout,
        q_tmem_layout: cute.ComposedLayout,
        p_smem_layout_store: cute.ComposedLayout,
        y_smem_layout: cute.ComposedLayout,
        delta_linear_smem_layout: cute.Layout,
        cumsum_delta_linear_smem_layout: cute.Layout,
        d_linear_smem_layout: Optional[cute.Layout],
        epi_tile: cute.Tile,
        tile_sched_params: Mamba2SSDTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            tma_atoms = [
                tma_atom_x,
                tma_atom_b,
                tma_atom_c,
                tma_atom_p,
                tma_atom_y,
                tma_atom_delta,
                tma_atom_cumsum_delta,
            ]
            if cutlass.const_expr(self.d_has_hdim):
                tma_atoms.append(tma_atom_d)
            for tma_atom in tma_atoms:
                cpasync.prefetch_descriptor(tma_atom)

        # Static consts
        D = cute.size(tma_tensor_x, mode=[0])
        L = cute.size(tma_tensor_x, mode=[1])
        N = cute.size(tma_tensor_b, mode=[1])
        # Dynamic values
        C = cute.size(tma_tensor_x, mode=[2])
        EH = cute.size(tma_tensor_x, mode=[3])
        B = cute.size(tma_tensor_x, mode=[4])
        G = cute.size(tma_tensor_b, mode=[3])
        NGROUP_RATIO = EH // G

        # Make TiledMma
        (
            tiled_mma_intra1,
            tiled_mma_intra2,
            tiled_mma_inter1,
            tiled_mma_inter2,
        ) = self.make_tiled_mmas(
            self.io_dtype,
            self.acc_dtype,
            self.cta_group,
            self.tile_shape_mnk_intra1,
            self.tile_shape_mnk_intra2,
            self.tile_shape_mnk_inter1,
            self.tile_shape_mnk_inter2,
        )

        # Setup cta/thread coordinates
        # Block coord
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_intra1.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Workload coord
        tile_sched = Mamba2SSDTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Thread/warp coord
        tidx, _, _ = cute.arch.thread_idx()
        # Thread coord inside specialized warps
        local_tidx = tidx % 128
        local_warp_idx = cute.arch.make_warp_uniform(local_tidx // 32)

        # Alloc and init smem tensors and pipelines
        smem = utils.SmemAllocator()
        smem_storage = smem.allocate(self.shared_storage)

        # Setup smem tensors
        smem_x = smem_storage.smem_x.get_tensor(
            x_smem_layout.outer, swizzle=x_smem_layout.inner
        )
        smem_xt = smem_storage.smem_x.get_tensor(
            xt_smem_layout.outer, swizzle=xt_smem_layout.inner
        )
        smem_b = smem_storage.smem_b.get_tensor(
            b_smem_layout.outer, swizzle=b_smem_layout.inner
        )
        smem_bt = smem_storage.smem_b.get_tensor(
            bt_smem_layout.outer, swizzle=bt_smem_layout.inner
        )
        smem_bt_internal = smem_storage.smem_bt_internal.get_tensor(
            bt_internal_smem_layout.outer, swizzle=bt_internal_smem_layout.inner
        )
        smem_c = smem_storage.smem_c.get_tensor(
            c_smem_layout.outer, swizzle=c_smem_layout.inner
        )
        smem_p = smem_storage.smem_p.get_tensor(
            p_smem_layout.outer, swizzle=p_smem_layout.inner
        )
        smem_pt = smem_storage.smem_p.get_tensor(
            pt_smem_layout.outer, swizzle=pt_smem_layout.inner
        )
        smem_p_store = smem_storage.smem_p.get_tensor(
            p_smem_layout_store.outer, swizzle=p_smem_layout_store.inner
        )
        smem_y = smem_storage.smem_y.get_tensor(
            y_smem_layout.outer, swizzle=y_smem_layout.inner
        )
        smem_cumsum_delta = smem_storage.smem_cumsum_delta.get_tensor(
            cumsum_delta_linear_smem_layout
        )
        smem_delta = smem_storage.smem_delta.get_tensor(delta_linear_smem_layout)
        smem_d = None
        if cutlass.const_expr(self.d_has_hdim):
            smem_d = smem_storage.smem_d.get_tensor(d_linear_smem_layout)

        # Init mbarrier for pipeline
        x_pipeline = self.make_and_init_x_pipeline(smem_storage.x_full.data_ptr())
        b_pipeline = self.make_and_init_b_pipeline(smem_storage.b_full.data_ptr())
        c_pipeline = self.make_and_init_c_pipeline(smem_storage.c_full.data_ptr())
        deltas_pipeline = self.make_and_init_deltas_pipeline(
            smem_storage.deltas_full.data_ptr()
        )
        d_pipeline = self.make_and_init_d_pipeline(smem_storage.d_full.data_ptr())
        intra1_acc_pipeline = self.make_and_init_intra1_acc_pipeline(
            smem_storage.intra1_acc_full.data_ptr()
        )
        intra2_q_pipeline = self.make_and_init_intra2_q_pipeline(
            smem_storage.intra2_q_full.data_ptr()
        )
        intra2_acc_pipeline = self.make_and_init_intra2_acc_pipeline(
            smem_storage.intra2_acc_full.data_ptr()
        )
        inter1_b_pipeline = self.make_and_init_inter1_b_pipeline(
            smem_storage.inter1_b_full.data_ptr()
        )
        inter1_acc_pipeline = self.make_and_init_inter1_acc_pipeline(
            smem_storage.inter1_acc_full.data_ptr()
        )
        inter2_p_pipeline = self.make_and_init_inter2_p_pipeline(
            smem_storage.inter2_p_full.data_ptr()
        )
        inter2_acc_pipeline = self.make_and_init_inter2_acc_pipeline(
            smem_storage.inter2_acc_full.data_ptr()
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Cluster wait before tmem alloc
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()

        # Alloc tmem buffer
        if warp_idx == self.epilog_warp_id[0]:
            cute.arch.alloc_tmem(
                self.num_tmem_cols_total,
                smem_storage.tmem_holding_buf,
                is_two_cta=self.use_2cta_instrs,
            )

        # Bar sync before retrieving tmem ptr from shared mem
        cute.arch.barrier()

        # Retrieve tmem ptr
        tmem_ptr_base = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            alignment=16,
            ptr_to_buffer_holding_addr=smem_storage.tmem_holding_buf,
        )

        # Specialized TMA load Delta/CumsumDelta/X warp
        if warp_idx == self.tma_deltas_x_d_warp_id:
            # Dealloc regs for pre-inter/pre-intra warps
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), 1, 1, C, EH, B)
            tXsX, tXgX_pre_slice = self.tma_partition_for_mma_b_operand(
                tma_atom_x,
                tma_tensor_x,
                smem_x,
                tiled_mma_intra2,
                cluster_layout_vmnk,
                mma_tile_coord_v,
                block_in_cluster_coord_vmnk,
            )

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), 1, C, EH, B)
            tDeltasDelta, tDeltagDelta_pre_slice = self.tma_partition_with_shape(
                tma_atom_delta,
                tma_tensor_delta,
                smem_delta,
                (self.tile_shape_mnk_inter1[2],),
            )
            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), 1, C, EH, B)
            (
                tDeltasCumsumDelta,
                tDeltagCumsumDelta_pre_slice,
            ) = self.tma_partition_with_shape(
                tma_atom_cumsum_delta,
                tma_tensor_cumsum_delta,
                smem_cumsum_delta,
                (self.tile_shape_mnk_inter1[2],),
            )

            tDsD = None
            tDgD_pre_slice = None
            if cutlass.const_expr(self.d_has_hdim):
                # Partition global/shared tensor for D
                # ((ATOM_V, REST_V), INPUT_STAGE)
                # ((ATOM_V, REST_V), 1, EH)
                tDsD, tDgD_pre_slice = self.tma_partition_with_shape(
                    tma_atom_d, tma_tensor_d, smem_d, (self.tile_shape_mnk_inter2[1],)
                )

            # Pipeline X/Delta/CumsumDelta/D producer state
            x_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            deltas_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            d_producer_state = None
            if cutlass.const_expr(self.d_has_hdim):
                # D is loaded by TMA only when d_has_hdim is True
                d_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.input_stages
                )

            while work_tile.is_valid_tile:
                b_idx, eh_idx, g_idx = work_tile.tile_idx

                # Slice global tensor to current tile idx
                # ((ATOM_V, REST_V), C)
                tXgX = tXgX_pre_slice[None, 0, 0, None, eh_idx, b_idx]
                tDeltagDelta = tDeltagDelta_pre_slice[None, 0, None, eh_idx, b_idx]
                tDeltagCumsumDelta = tDeltagCumsumDelta_pre_slice[
                    None, 0, None, eh_idx, b_idx
                ]
                tDgD = None
                if cutlass.const_expr(self.d_has_hdim):
                    # ((ATOM_V, REST_V))
                    tDgD = tDgD_pre_slice[None, 0, eh_idx]

                # Reset count for pipeline state
                x_producer_state.reset_count()
                deltas_producer_state.reset_count()
                if cutlass.const_expr(self.d_has_hdim):
                    d_producer_state.reset_count()

                # Peek (try_wait) X/deltas buffer empty status
                peek_x_empty_status = self.conditional_producer_try_acquire(
                    x_producer_state, x_pipeline, C
                )
                peek_deltas_empty_status = self.conditional_producer_try_acquire(
                    deltas_producer_state, deltas_pipeline, C
                )

                if cutlass.const_expr(self.d_has_hdim):
                    # Wait for D buffer empty
                    d_pipeline.producer_acquire(d_producer_state)
                    # TMA load D
                    cute.copy(
                        tma_atom_d,
                        tDgD,
                        tDsD[None, d_producer_state.index],
                        tma_bar_ptr=d_pipeline.producer_get_barrier(d_producer_state),
                    )
                    # Advance D producer state
                    d_producer_state.advance()

                # Batched load over C dimension
                for chunk_idx in cutlass.range(C, unroll=1):
                    # Conditionally wait for X buffer empty
                    x_pipeline.producer_acquire(x_producer_state, peek_x_empty_status)

                    # TMA load X
                    cute.copy(
                        tma_atom_x,
                        tXgX[None, x_producer_state.count],
                        tXsX[None, x_producer_state.index],
                        tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
                    )

                    # Conditionally wait for deltas buffer empty
                    deltas_pipeline.producer_acquire(
                        deltas_producer_state, peek_deltas_empty_status
                    )

                    # TMA load Delta/CumsumDelta
                    cute.copy(
                        tma_atom_delta,
                        tDeltagDelta[None, deltas_producer_state.count],
                        tDeltasDelta[None, deltas_producer_state.index],
                        tma_bar_ptr=deltas_pipeline.producer_get_barrier(
                            deltas_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_cumsum_delta,
                        tDeltagCumsumDelta[None, deltas_producer_state.count],
                        tDeltasCumsumDelta[None, deltas_producer_state.index],
                        tma_bar_ptr=deltas_pipeline.producer_get_barrier(
                            deltas_producer_state
                        ),
                    )

                    # Advance X/deltas producer state
                    x_producer_state.advance()
                    deltas_producer_state.advance()

                    # Peek (try_wait) X/deltas buffer empty status
                    peek_x_empty_status = self.conditional_producer_try_acquire(
                        x_producer_state, x_pipeline, C
                    )
                    peek_deltas_empty_status = self.conditional_producer_try_acquire(
                        deltas_producer_state, deltas_pipeline, C
                    )
                # END of for chunk_idx in cutlass.range(C, unroll=1)

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # END of while work_tile.is_valid_tile

            # Producer tail for X/Deltas/D
            x_pipeline.producer_tail(x_producer_state)
            deltas_pipeline.producer_tail(deltas_producer_state)
            if cutlass.const_expr(self.d_has_hdim):
                d_pipeline.producer_tail(d_producer_state)
        # END of specialized tma load X/Deltas/D warp

        # Specialized TMA load B/C warp
        elif warp_idx == self.tma_b_c_warp_id:
            # Dealloc regs for pre-inter/pre-intra warps
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), 1, 1, C, G, B)
            tBsB, tBgB_pre_slice = self.tma_partition_for_mma_b_operand(
                tma_atom_b,
                tma_tensor_b,
                smem_b,
                tiled_mma_intra1,
                cluster_layout_vmnk,
                mma_tile_coord_v,
                block_in_cluster_coord_vmnk,
            )

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), 1, 1, C, G, B)
            tCsC, tCgC_pre_slice = self.tma_partition_for_mma_a_operand(
                tma_atom_c,
                tma_tensor_c,
                smem_c,
                tiled_mma_intra1,
                cluster_layout_vmnk,
                mma_tile_coord_v,
                block_in_cluster_coord_vmnk,
            )

            # Pipeline B/C producer state
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            c_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )

            while work_tile.is_valid_tile:
                b_idx, eh_idx, g_idx = work_tile.tile_idx

                # Slice global tensor to current tile idx
                # ((ATOM_V, REST_V), C)
                tBgB = tBgB_pre_slice[None, 0, 0, None, g_idx, b_idx]
                tCgC = tCgC_pre_slice[None, 0, 0, None, g_idx, b_idx]

                # Reset count for pipeline state
                b_producer_state.reset_count()
                c_producer_state.reset_count()

                # Peek (try_wait) B/C buffer empty status
                peek_b_empty_status = self.conditional_producer_try_acquire(
                    b_producer_state, b_pipeline, C
                )
                peek_c_empty_status = self.conditional_producer_try_acquire(
                    c_producer_state, c_pipeline, C
                )

                # Batched load over C dimension
                for chunk_idx in cutlass.range(C, unroll=1):
                    # Conditionally wait for B buffer empty
                    b_pipeline.producer_acquire(b_producer_state, peek_b_empty_status)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB[None, b_producer_state.count],
                        tBsB[None, b_producer_state.index],
                        tma_bar_ptr=b_pipeline.producer_get_barrier(b_producer_state),
                    )

                    # Conditionally wait for C buffer empty
                    c_pipeline.producer_acquire(c_producer_state, peek_c_empty_status)

                    # TMA load C
                    cute.copy(
                        tma_atom_c,
                        tCgC[None, c_producer_state.count],
                        tCsC[None, c_producer_state.index],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                    )

                    # Advance B/C producer state
                    b_producer_state.advance()
                    c_producer_state.advance()

                    # Peek (try_wait) B/C buffer empty status
                    peek_b_empty_status = self.conditional_producer_try_acquire(
                        b_producer_state, b_pipeline, C
                    )
                    peek_c_empty_status = self.conditional_producer_try_acquire(
                        c_producer_state, c_pipeline, C
                    )
                # END of for chunk_idx in cutlass.range(C, unroll=1)

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # END of while work_tile.is_valid_tile

            # Producer tail for B/C
            b_pipeline.producer_tail(b_producer_state)
            c_pipeline.producer_tail(c_producer_state)
        # END of specialized tma load B/C warp

        # Specialized MMA Intra warp
        elif warp_idx == self.mma_intra_warp_id:
            # Dealloc regs for pre-inter/pre-intra warps
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            # Make shared/tmem fragments for INTRA_MMA1 B/C/ACC
            # (MMA, MMA_N, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_N, INTRA1_ACC_STAGE)
            tCrC, tCrB, tCtAccIntra1 = self.mma_partition_ss(
                tiled_mma_intra1,
                self.tile_shape_mnk_intra1,
                smem_c,
                smem_b,
                tmem_ptr_base + self.tmem_intra1_acc_offset,
                self.intra1_acc_stages,
            )

            # Make shared/tmem fragments for INTRA_MMA2 X/Q/ACC
            # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
            # (MMA, MMA_N, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
            tCrQ, tCrX, tCtAccIntra2 = self.mma_partition_ts(
                tiled_mma_intra2,
                self.tile_shape_mnk_intra2,
                q_tmem_layout,
                smem_x,
                tmem_ptr_base + self.tmem_intra2_q_offset,
                tmem_ptr_base + self.tmem_intra2_acc_offset,
                self.internal_stages,
            )

            # Pipeline B/C/X/INTRA2_Q consumer state
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            c_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            intra2_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.internal_stages
            )

            # Pipeline INTRA1_ACC/INTRA2_ACC producer state
            intra1_acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.intra1_acc_stages
            )
            intra2_acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.internal_stages
            )

            while work_tile.is_valid_tile:
                # Reset count for pipeline state
                b_consumer_state.reset_count()
                c_consumer_state.reset_count()
                intra1_acc_producer_state.reset_count()
                x_consumer_state.reset_count()
                intra2_q_consumer_state.reset_count()
                intra2_acc_producer_state.reset_count()

                # Peek (try_wait) B/C/X/INTRA1_ACC buffer full/full/full/empty status
                peek_b_full_status = self.conditional_consumer_try_wait(
                    b_consumer_state, b_pipeline, C
                )
                peek_c_full_status = self.conditional_consumer_try_wait(
                    c_consumer_state, c_pipeline, C
                )
                peek_wr_intra1_acc_empty_status = self.conditional_producer_try_acquire(
                    intra1_acc_producer_state, intra1_acc_pipeline, C
                )
                peek_x_full_status = self.conditional_consumer_try_wait(
                    x_consumer_state, x_pipeline, C
                )

                # Manual pipeline: unrolled INTRA_MMA1 chunk_idx = 0 loop
                # Conditionally wait for B/C/INTRA1_ACC buffer full/full/empty
                b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                c_pipeline.consumer_wait(c_consumer_state, peek_c_full_status)
                intra1_acc_pipeline.producer_acquire(
                    intra1_acc_producer_state, peek_wr_intra1_acc_empty_status
                )

                # INTRA_MMA1
                tiled_mma_intra1 = self.exec_mma(
                    tiled_mma_intra1,
                    tCtAccIntra1,
                    tCrC,
                    tCrB,
                    intra1_acc_producer_state,
                    c_consumer_state,
                    b_consumer_state,
                )

                # Async arrive B/C/INTRA1_ACC buffer empty/empty/full
                b_pipeline.consumer_release(
                    b_consumer_state, pipeline.PipelineOp.TCGen05Mma
                )
                c_pipeline.consumer_release(c_consumer_state)
                intra1_acc_pipeline.producer_commit(intra1_acc_producer_state)

                # Advance B/C/INTRA1_ACC state
                b_consumer_state.advance()
                c_consumer_state.advance()
                intra1_acc_producer_state.advance()

                # Peek (try_wait) B/C/INTRA1_ACC buffer full/full/empty for chunk_idx = chunk_idx + 1
                peek_b_full_status = self.conditional_consumer_try_wait(
                    b_consumer_state, b_pipeline, C
                )
                peek_c_full_status = self.conditional_consumer_try_wait(
                    c_consumer_state, c_pipeline, C
                )
                peek_wr_intra1_acc_empty_status = self.conditional_producer_try_acquire(
                    intra1_acc_producer_state, intra1_acc_pipeline, C
                )

                # Manual pipeline: batched gemm over C-1 dimension
                for chunk_idx in cutlass.range(C - 1, unroll=1):
                    # Conditionally wait for B/C/INTRA1_ACC buffer full/full/empty
                    b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                    c_pipeline.consumer_wait(c_consumer_state, peek_c_full_status)
                    intra1_acc_pipeline.producer_acquire(
                        intra1_acc_producer_state, peek_wr_intra1_acc_empty_status
                    )

                    # INTRA_MMA1
                    tiled_mma_intra1 = self.exec_mma(
                        tiled_mma_intra1,
                        tCtAccIntra1,
                        tCrC,
                        tCrB,
                        intra1_acc_producer_state,
                        c_consumer_state,
                        b_consumer_state,
                    )

                    # Async arrive B/C/INTRA1_ACC buffer empty/empty/full
                    b_pipeline.consumer_release(
                        b_consumer_state, pipeline.PipelineOp.TCGen05Mma
                    )
                    c_pipeline.consumer_release(c_consumer_state)
                    intra1_acc_pipeline.producer_commit(intra1_acc_producer_state)

                    # Conditionally wait for X/INTRA2_Q/INTRA2_ACC buffer full/full/empty
                    x_pipeline.consumer_wait(x_consumer_state, peek_x_full_status)
                    intra2_q_pipeline.consumer_wait(intra2_q_consumer_state)
                    intra2_acc_pipeline.producer_acquire(intra2_acc_producer_state)

                    # INTRA_MMA2
                    tiled_mma_intra2 = self.exec_mma(
                        tiled_mma_intra2,
                        tCtAccIntra2,
                        tCrQ,
                        tCrX,
                        intra2_acc_producer_state,
                        intra2_q_consumer_state,
                        x_consumer_state,
                    )

                    # Async arrive X/INTRA2_Q/INTRA2_ACC buffer empty/empty/full
                    if cutlass.const_expr(self.has_d):
                        x_pipeline.consumer_release(
                            x_consumer_state, pipeline.PipelineOp.TCGen05Mma
                        )
                    else:
                        x_pipeline.consumer_release(x_consumer_state)
                    intra2_q_pipeline.consumer_release(intra2_q_consumer_state)
                    intra2_acc_pipeline.producer_commit(intra2_acc_producer_state)

                    # Advance B/C/INTRA1_ACC cstate
                    b_consumer_state.advance()
                    c_consumer_state.advance()
                    intra1_acc_producer_state.advance()

                    # Peek (try_wait) B/C/INTRA1_ACC buffer full/full/empty for chunk_idx = chunk_idx + 1
                    peek_b_full_status = self.conditional_consumer_try_wait(
                        b_consumer_state, b_pipeline, C
                    )
                    peek_c_full_status = self.conditional_consumer_try_wait(
                        c_consumer_state, c_pipeline, C
                    )
                    peek_wr_intra1_acc_empty_status = (
                        self.conditional_producer_try_acquire(
                            intra1_acc_producer_state, intra1_acc_pipeline, C
                        )
                    )

                    # Advance X/INTRA2_Q/INTRA2_ACC state
                    x_consumer_state.advance()
                    intra2_q_consumer_state.advance()
                    intra2_acc_producer_state.advance()

                    # Peek (try_wait) X buffer full for chunk_idx = chunk_idx + 1
                    peek_x_full_status = self.conditional_consumer_try_wait(
                        x_consumer_state, x_pipeline, C
                    )
                # END of for chunk_idx in cutlass.range(C-1, unroll=1)

                # Manual pipeline: unrolled INTRA_MMA2 chunk_idx = C-1 loop
                # Conditionally wait for X/INTRA2_Q/INTRA2_ACC buffer full/full/empty
                x_pipeline.consumer_wait(x_consumer_state, peek_x_full_status)
                intra2_q_pipeline.consumer_wait(intra2_q_consumer_state)
                intra2_acc_pipeline.producer_acquire(intra2_acc_producer_state)

                # INTRA_MMA2
                tiled_mma_intra2 = self.exec_mma(
                    tiled_mma_intra2,
                    tCtAccIntra2,
                    tCrQ,
                    tCrX,
                    intra2_acc_producer_state,
                    intra2_q_consumer_state,
                    x_consumer_state,
                )

                # Async arrive X/INTRA2_Q/INTRA2_ACC buffer empty/empty/full
                if cutlass.const_expr(self.has_d):
                    x_pipeline.consumer_release(
                        x_consumer_state, pipeline.PipelineOp.TCGen05Mma
                    )
                else:
                    x_pipeline.consumer_release(x_consumer_state)
                intra2_q_pipeline.consumer_release(intra2_q_consumer_state)
                intra2_acc_pipeline.producer_commit(intra2_acc_producer_state)

                # Advance X/INTRA2_Q/INTRA2_ACC state
                x_consumer_state.advance()
                intra2_q_consumer_state.advance()
                intra2_acc_producer_state.advance()

                # Peek (try_wait) X buffer full for chunk_idx = chunk_idx + 1
                peek_x_full_status = self.conditional_consumer_try_wait(
                    x_consumer_state, x_pipeline, C
                )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # END of while work_tile.is_valid_tile

            # Producer tail for INTRA1_ACC/INTRA2_ACC
            intra1_acc_pipeline.producer_tail(intra1_acc_producer_state)
            intra2_acc_pipeline.producer_tail(intra2_acc_producer_state)
        # END of specialized mma-intra warp

        # Specialized MMA Inter warp
        elif warp_idx == self.mma_inter_warp_id:
            # Dealloc regs for pre-inter/pre-intra warps
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            # Make shared/tmem fragments for INTER_MMA1 X/B/ACC
            # (MMA, MMA_N, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
            # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
            tCrB, tCrX, tCtAccInter1 = self.mma_partition_ss(
                tiled_mma_inter1,
                self.tile_shape_mnk_inter1,
                smem_bt_internal,
                smem_x,
                tmem_ptr_base + self.tmem_inter1_acc_offset,
                self.internal_stages,
            )

            # Make shared/tmem fragments for INTER_MMA2 C/P/ACC
            # (MMA, MMA_M, MMA_K, INPUT_STAGE)
            # (MMA, MMA_N, MMA_K, INTERNAL_STAGE)
            # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
            tCrC, tCrP, tCtAccInter2 = self.mma_partition_ss(
                tiled_mma_inter2,
                self.tile_shape_mnk_inter2,
                smem_c,
                smem_p,
                tmem_ptr_base + self.tmem_inter2_acc_offset,
                self.internal_stages,
            )

            # Pipeline X/C/INTER1_B/INTER2_P consumer state
            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            c_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            inter1_b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.internal_stages
            )
            inter2_p_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.internal_stages
            )

            # Pipeline INTER1_ACC/INTER2_ACC producer state
            inter1_acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.internal_stages
            )
            inter2_acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.internal_stages
            )

            while work_tile.is_valid_tile:
                # Reset count for pipeline state
                x_consumer_state.reset_count()
                c_consumer_state.reset_count()
                inter1_acc_producer_state.reset_count()
                inter1_b_consumer_state.reset_count()
                inter2_p_consumer_state.reset_count()
                inter2_acc_producer_state.reset_count()

                # Peek (try_wait) C/INTER2_P/INTER2_ACC buffer full/full/empty status
                peek_c_full_status = self.conditional_consumer_try_wait(
                    c_consumer_state, c_pipeline, C
                )
                peek_inter2_p_full_status = self.conditional_consumer_try_wait(
                    inter2_p_consumer_state, inter2_p_pipeline, C
                )
                peek_inter2_acc_empty_status = self.conditional_producer_try_acquire(
                    inter2_acc_producer_state, inter2_acc_pipeline, C
                )

                # Batched gemm over C dimension
                for chunk_idx in cutlass.range(C, unroll=1):
                    # Conditionally wait for C/INTER2_P/INTER2_ACC buffer full/full/empty
                    c_pipeline.consumer_wait(c_consumer_state, peek_c_full_status)
                    inter2_p_pipeline.consumer_wait(
                        inter2_p_consumer_state, peek_inter2_p_full_status
                    )
                    inter2_acc_pipeline.producer_acquire(
                        inter2_acc_producer_state, peek_inter2_acc_empty_status
                    )

                    # INTER MMA2
                    tiled_mma_inter2 = self.exec_mma(
                        tiled_mma_inter2,
                        tCtAccInter2,
                        tCrC,
                        tCrP,
                        inter2_acc_producer_state,
                        c_consumer_state,
                        inter2_p_consumer_state,
                    )

                    # Async arrive C/INTER2_P/INTER2_ACC buffer empty/empty/full
                    c_pipeline.consumer_release(c_consumer_state)
                    inter2_p_pipeline.consumer_release(inter2_p_consumer_state)
                    inter2_acc_pipeline.producer_commit(inter2_acc_producer_state)

                    # Wait for X/INTER1_B/INTER1_ACC buffer full/full/empty
                    x_pipeline.consumer_wait(x_consumer_state)
                    inter1_b_pipeline.consumer_wait(inter1_b_consumer_state)
                    inter1_acc_pipeline.producer_acquire(inter1_acc_producer_state)

                    # INTER MMA1
                    tiled_mma_inter1 = self.exec_mma(
                        tiled_mma_inter1,
                        tCtAccInter1,
                        tCrB,
                        tCrX,
                        inter1_acc_producer_state,
                        inter1_b_consumer_state,
                        x_consumer_state,
                    )

                    # Async arrive X/INTER1_B/INTER1_ACC buffer empty/empty/full
                    if cutlass.const_expr(self.has_d):
                        x_pipeline.consumer_release(
                            x_consumer_state, pipeline.PipelineOp.TCGen05Mma
                        )
                    else:
                        x_pipeline.consumer_release(x_consumer_state)
                    inter1_b_pipeline.consumer_release(inter1_b_consumer_state)
                    inter1_acc_pipeline.producer_commit(inter1_acc_producer_state)

                    # Advance X/C/INTER1_B/INTER1_ACC/INTER2_P/INTER2_ACC state
                    x_consumer_state.advance()
                    c_consumer_state.advance()
                    inter1_b_consumer_state.advance()
                    inter1_acc_producer_state.advance()
                    inter2_p_consumer_state.advance()
                    inter2_acc_producer_state.advance()

                    # Peek (try_wait) C/INTER2_P/INTER2_ACC buffer full/full/empty for chunk_idx = chunk_idx + 1
                    peek_c_full_status = self.conditional_consumer_try_wait(
                        c_consumer_state, c_pipeline, C
                    )
                    peek_inter2_p_full_status = self.conditional_consumer_try_wait(
                        inter2_p_consumer_state, inter2_p_pipeline, C
                    )
                    peek_inter2_acc_empty_status = (
                        self.conditional_producer_try_acquire(
                            inter2_acc_producer_state, inter2_acc_pipeline, C
                        )
                    )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Producer tail for INTER1_ACC/INTER2_ACC
            inter1_acc_pipeline.producer_tail(inter1_acc_producer_state)
            inter2_acc_pipeline.producer_tail(inter2_acc_producer_state)

        # Specialized Pre-Inter warp
        elif (
            warp_idx == self.pre_inter_warp_id[0]
            or warp_idx == self.pre_inter_warp_id[1]
            or warp_idx == self.pre_inter_warp_id[2]
            or warp_idx == self.pre_inter_warp_id[3]
        ):
            # Alloc regs in pre_inter warps
            cute.arch.warpgroup_reg_alloc(self.num_regs_pre_inter_warps)

            # Make tiledCopy and partition smem/register tensor for smem load Bt
            # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
            # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
            tiled_s2r_b, tBsB_s2r, tBrB_s2r = self.pre_inter_smem_load_and_partition_b(
                local_tidx, smem_bt
            )

            # Partition shared tensor for smem store Bt
            smem_bt_internal_ = cute.make_tensor(
                smem_bt_internal.iterator, smem_bt.layout
            )
            # Make tiledCopy and partition register/smem tensor for smem store Bt
            # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N)
            # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N, INTERNAL_STAGE)
            tiled_r2s_b, tBrB_r2s, tBsB_r2s = self.pre_inter_smem_store_and_partition_b(
                local_tidx,
                smem_bt_internal_,
                tiled_s2r_b,
                tBrB_s2r,
            )

            # (MMA, MMA_M, MMA_K, INPUT_STAGE)
            sDelta = self.pre_inter_make_delta(smem_delta, smem_bt.layout)
            sDeltaA = self.pre_inter_make_delta(smem_cumsum_delta, smem_bt.layout)

            # Make copy_atom and partition register/smem tensor for smem load/store of Delta/DeltaA
            # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
            # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
            (
                s2r_atom_delta,
                tBsDelta_s2r,
                tBrDelta_s2r,
            ) = self.smem_load_and_partition_delta_d(
                tiled_s2r_b, local_tidx, sDelta, (None, None, None, 0)
            )
            (
                s2r_atom_cumsum,
                tBsDeltaA_s2r,
                tBrDeltaA_s2r,
            ) = self.smem_load_and_partition_delta_d(
                tiled_s2r_b, local_tidx, sDeltaA, (None, None, None, 0)
            )

            # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N)
            thr_r2s_b = tiled_r2s_b.get_slice(local_tidx)
            tBrDelta_r2s = thr_r2s_b.retile(tBrDelta_s2r)
            tBrDeltaA_r2s = thr_r2s_b.retile(tBrDeltaA_s2r)

            # Make tmem fragment for INTER1_ACC
            # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
            tCtAccInter1 = self.mma_partition_c(
                tiled_mma_inter1,
                self.tile_shape_mnk_inter1,
                tmem_ptr_base + self.tmem_inter1_acc_offset,
                self.internal_stages,
            )
            # (M_PER_MMA, N_PER_MMA, INTERNAL_STAGE)
            tInter1 = tCtAccInter1[((None, None), 0, 0, None)]

            # Make tiledCopy and partition tmem/register tensor for tmem load INTER1_ACC
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, INTERNAL_STAGE)
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            (
                tiled_t2r_inter1,
                tTR_tP,
                tTR_rP,
            ) = self.pre_inter_tmem_load_and_partition_p(local_tidx, tInter1, smem_pt)

            # Make fragment for register to hold P after post-processing (in acc dtype)
            tState = cute.make_fragment(tTR_rP.shape, self.acc_dtype)

            # Make tiledCopy and partition smem/register tensor for smem store INTER2_P
            # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N)
            # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N, INTERNAL_STAGE)
            tiled_r2s_p, tRS_rP, tRS_sP = self.smem_store_and_partition_p_y(
                local_tidx, smem_pt, tiled_t2r_inter1
            )

            # Partition global/shared tensor for P (State)
            # ((ATOM_V, REST_V), INTERNAL_STAGE)
            # ((ATOM_V, REST_V), 1, 1, EH, B)
            bSG_sP, bSG_gP_pre_slice = self.tma_partition_with_shape(
                tma_atom_p,
                tma_tensor_p,
                smem_p_store,
                self.tile_shape_mnk_inter2[1:],
            )

            # Pipeline B/Delta/INTER1_ACC consumer state
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            deltas_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            inter1_acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.internal_stages
            )

            # Pipeline INTER1_B/INTER2_P producer state
            inter1_b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.internal_stages
            )
            inter2_p_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.internal_stages
            )

            # Pipeline TMA store P
            tma_p_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.internal_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.pre_inter_warp_id), 128
                ),
            )

            while work_tile.is_valid_tile:
                b_idx, eh_idx, g_idx = work_tile.tile_idx

                # Slice global tensor to current tile idx
                # ((ATOM_V, REST_V))
                bSG_gP = bSG_gP_pre_slice[(None, 0, 0, eh_idx, b_idx)]

                # Reset count for pipeline state
                b_consumer_state.reset_count()
                deltas_consumer_state.reset_count()
                inter1_b_producer_state.reset_count()
                inter1_acc_consumer_state.reset_count()
                inter2_p_producer_state.reset_count()

                # State (P) init
                tState.fill(0.0)

                # Peek (try_wait) B/Delta/INTER1_B buffer full/full/empty status
                peek_b_full_status = self.conditional_consumer_try_wait(
                    b_consumer_state, b_pipeline, C
                )
                peek_deltas_full_status = self.conditional_consumer_try_wait(
                    deltas_consumer_state, deltas_pipeline, C
                )
                peek_wr_inter1_b_empty_status = self.conditional_producer_try_acquire(
                    inter1_b_producer_state, inter1_b_pipeline, C
                )

                # Prefill INTER2_P with 0
                # Wait for INTER2_P buffer empty
                inter2_p_pipeline.producer_acquire(inter2_p_producer_state)

                tRS_rP.fill(0.0)
                # Copy INTER2_P from register to smem
                inter2_p_coord = (None, None, None, inter2_p_producer_state.index)
                cute.copy(tiled_r2s_p, tRS_rP, tRS_sP[inter2_p_coord])

                # Fence for shared memory
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                # Async arrive INTER2_P buffer full
                inter2_p_pipeline.producer_commit(inter2_p_producer_state)
                # Advance INTER2_P producer state
                inter2_p_producer_state.advance()

                # Batched processing over C dimension
                for chunk_idx in cutlass.range(C, unroll=1):
                    # Conditionally wait for B/Delta/B_TMEM buffer full/full/empty
                    b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                    deltas_pipeline.consumer_wait(
                        deltas_consumer_state, peek_deltas_full_status
                    )
                    inter1_b_pipeline.producer_acquire(
                        inter1_b_producer_state, peek_wr_inter1_b_empty_status
                    )

                    # Load B/Delta/DeltaA/last_column
                    b_coord = (None, None, None, b_consumer_state.index)
                    delta_coord = (None, None, None, deltas_consumer_state.index)
                    cute.copy(tiled_s2r_b, tBsB_s2r[b_coord], tBrB_s2r)
                    cute.copy(s2r_atom_delta, tBsDelta_s2r[delta_coord], tBrDelta_s2r)
                    cute.copy(
                        s2r_atom_cumsum, tBsDeltaA_s2r[delta_coord], tBrDeltaA_s2r
                    )
                    last_column = smem_cumsum_delta[
                        smem_cumsum_delta.shape[0] - 1, deltas_consumer_state.index
                    ]

                    # Fence for shared memory
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # Combine B/Delta/DeltaA/last_column
                    tScaledB = self.pre_inter_scale_bt_with_delta(
                        tBrB_s2r, tBrDelta_r2s, tBrDeltaA_r2s, last_column
                    )

                    # Store scaled B to tBrB_r2s
                    for reg_idx in range(cute.size(tBrB_r2s)):
                        tBrB_r2s[reg_idx] = tScaledB[reg_idx].to(self.io_dtype)

                    # Store tBrB_r2s to bt_smem_internal
                    inter1_b_coord = (None, None, None, inter1_b_producer_state.index)
                    cute.copy(tiled_r2s_b, tBrB_r2s, tBsB_r2s[inter1_b_coord])

                    # Fence for shared memory
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # Async arrive B/Delta/B_TMEM buffer empty/empty/full
                    b_pipeline.consumer_release(
                        b_consumer_state, pipeline.PipelineOp.AsyncThread
                    )
                    deltas_pipeline.consumer_release(deltas_consumer_state)
                    inter1_b_pipeline.producer_commit(inter1_b_producer_state)

                    # Wait for INTER1_ACC/INTER2_P buffer full/empty
                    inter1_acc_pipeline.consumer_wait(inter1_acc_consumer_state)
                    inter2_p_pipeline.producer_acquire(inter2_p_producer_state)

                    # Load INTER1_ACC
                    inter1_acc_coord = (
                        None,
                        None,
                        None,
                        inter1_acc_consumer_state.index,
                    )
                    cute.copy(tiled_t2r_inter1, tTR_tP[inter1_acc_coord], tTR_rP)

                    # Fence for TMEM load
                    cute.arch.fence_view_async_tmem_load()

                    # Combine INTER1_ACC/last_column/State
                    exp_last_column = cute.math.exp(last_column, fastmath=True)
                    for reg_idx in range(0, cute.size(tTR_rP), 2):
                        (
                            tTR_rP[reg_idx],
                            tTR_rP[reg_idx + 1],
                        ) = cute.arch.fma_packed_f32x2(
                            (exp_last_column, exp_last_column),
                            (tState[reg_idx], tState[reg_idx + 1]),
                            (tTR_rP[reg_idx], tTR_rP[reg_idx + 1]),
                        )

                    # Store scaled P to tRS_rP
                    for reg_idx in range(cute.size(tTR_rP)):
                        tRS_rP[reg_idx] = tTR_rP[reg_idx].to(self.io_dtype)

                    # Update old state
                    tState.store(tTR_rP.load())

                    # Store INTER2_P
                    inter2_p_coord = (None, None, None, inter2_p_producer_state.index)
                    cute.copy(tiled_r2s_p, tRS_rP, tRS_sP[inter2_p_coord])

                    # Fence for shared memory
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # Async arrive INTER1_ACC/INTER2_P buffer empty/full
                    inter1_acc_pipeline.consumer_release(inter1_acc_consumer_state)
                    # Last iteration consumer is PRE_INTER warp itself, not MMA_INTER warp
                    if inter2_p_producer_state.count < C:
                        inter2_p_pipeline.producer_commit(inter2_p_producer_state)

                    # Advance B/Delta/INTER1_B/INTER1_ACC state
                    b_consumer_state.advance()
                    deltas_consumer_state.advance()
                    inter1_b_producer_state.advance()
                    inter1_acc_consumer_state.advance()
                    # Peek (try_wait) B/Delta/INTER1_B buffer full/full./empty for chunk_idx = chunk_idx + 1
                    peek_b_full_status = self.conditional_consumer_try_wait(
                        b_consumer_state, b_pipeline, C
                    )
                    peek_deltas_full_status = self.conditional_consumer_try_wait(
                        deltas_consumer_state, deltas_pipeline, C
                    )
                    peek_wr_inter1_b_empty_status = (
                        self.conditional_producer_try_acquire(
                            inter1_b_producer_state, inter1_b_pipeline, C
                        )
                    )

                    # Last iteration producer is PRE_INTER warp itself, not MMA_INTER warp
                    if inter2_p_producer_state.count < C:
                        # Advance INTER2_P producer state
                        inter2_p_producer_state.advance()
                # END of for chunk_idx in cutlass.range(C, unroll=1)

                # Store last INTER2_P (State) from smem to gmem
                # Wait for all previous stores to smem to be done
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.barrier(
                    barrier_id=self.pre_inter_sync_bar_id,
                    number_of_threads=len(self.pre_inter_warp_id) * 32,
                )

                if local_warp_idx == 0:
                    # TMA store P
                    cute.copy(
                        tma_atom_p,
                        bSG_sP[(None, inter2_p_producer_state.index)],
                        bSG_gP,
                    )
                    # Wait for TMA store done
                    tma_p_pipeline.producer_commit()
                    tma_p_pipeline.producer_acquire()

                cute.arch.barrier(
                    barrier_id=self.pre_inter_sync_bar_id,
                    number_of_threads=len(self.pre_inter_warp_id) * 32,
                )
                tma_p_pipeline.producer_tail()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # END of while work_tile.is_valid_tile

            # Producer tail for INTER1_B/INTER2_P/TMA store P
            inter1_b_pipeline.producer_tail(inter1_b_producer_state)
            inter2_p_pipeline.producer_tail(inter2_p_producer_state)
        # END of specialized pre-inter warp

        # Specialized Pre-Intra warp
        elif (
            warp_idx == self.pre_intra_warp_id[0]
            or warp_idx == self.pre_intra_warp_id[1]
            or warp_idx == self.pre_intra_warp_id[2]
            or warp_idx == self.pre_intra_warp_id[3]
        ):
            # Alloc regs in pre_inter warps
            cute.arch.warpgroup_reg_alloc(self.num_regs_pre_intra_warps)

            # Make tmem fragment for INTRA1_ACC
            # (MMA, MMA_M, MMA_N, INTRA1_ACC_STAGE)
            tCtAccIntra1 = self.mma_partition_c(
                tiled_mma_intra1,
                self.tile_shape_mnk_intra1,
                tmem_ptr_base + self.tmem_intra1_acc_offset,
                self.intra1_acc_stages,
            )
            # (M_PER_MMA, N_PER_MMA, INTRA1_ACC_STAGE)
            tIntra1 = tCtAccIntra1[((None, None), 0, 0, None)]

            # Make tiledCopy and partition tmem/register tensor for tensor memory load INTRA1_ACC
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, INTERNAL_STAGE)
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            tiled_t2r_intra1, tTR_tQ, tTR_rQ = self.pre_intra_tmem_load_and_partition_q(
                tIntra1, local_tidx
            )

            # Broadcast delta/delta_cumsum smem tensor from LxINPUT_STAGE to LxLxINPUT_STAGE
            sDeltaA_Row = self.pre_intra_make_delta(smem_cumsum_delta, 0)
            sDeltaA_Col = self.pre_intra_make_delta(smem_cumsum_delta, 1)
            sDelta = self.pre_intra_make_delta(smem_delta, 0)

            # Make tiledCopy and partition smem/register tensor for smem memory load delta/delta_cumsum
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, INPUT_STAGE)
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            (
                s2r_atom_cumsum,
                tQsDeltaA_Row,
                tQrDeltaA_Row,
            ) = self.smem_load_and_partition_delta_d(
                tiled_t2r_intra1, local_tidx, sDeltaA_Row, (None, None, None, 0)
            )
            (
                s2r_atom_cumsum,
                tQsDeltaA_Col,
                tQrDeltaA_Col,
            ) = self.smem_load_and_partition_delta_d(
                tiled_t2r_intra1, local_tidx, sDeltaA_Col, (None, None, None, 0)
            )
            (
                s2r_atom_delta,
                tQsDelta,
                tQrDelta,
            ) = self.smem_load_and_partition_delta_d(
                tiled_t2r_intra1, local_tidx, sDelta, (None, None, None, 0)
            )

            # Make and partition coord tensor for delta_cumsum load
            # (L, L)
            coord_tensor = cute.make_identity_tensor(
                cute.dice(self.tile_shape_mnk_intra1, (1, 1, None))
            )
            thr_t2r_intra1 = tiled_t2r_intra1.get_slice(local_tidx)
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            tCoord = thr_t2r_intra1.partition_D(coord_tensor)

            # Make tmem tensor for INTRA2_Q
            # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
            tCrQ = self.mma_partition_a_tmem(
                tiled_mma_intra2,
                q_tmem_layout,
                tmem_ptr_base + self.tmem_intra2_q_offset,
            )

            # Make tiledCopy and partition tmem/register tensor for tensor memory store INTRA2_Q
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ...)
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ..., INTERNAL_STAGE)
            tiled_r2t_q, tRT_rQ, tRT_tQ = self.pre_intra_tmem_store_and_partition_q(
                local_tidx, tCrQ
            )

            # Pipeline DELTA/INTRA1_ACC consumer state
            deltas_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            intra1_acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.intra1_acc_stages
            )
            # Pipeline INTRA2_Q producer state
            intra2_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.internal_stages
            )

            while work_tile.is_valid_tile:
                # Reset count for pipeline state
                deltas_consumer_state.reset_count()
                intra1_acc_consumer_state.reset_count()
                intra2_q_producer_state.reset_count()

                # Peek (try_wait) DELTA/INTRA1_ACC buffer full
                peek_deltas_full_status = self.conditional_consumer_try_wait(
                    deltas_consumer_state, deltas_pipeline, C
                )
                peek_rd_intra1_acc_full_status = self.conditional_consumer_try_wait(
                    intra1_acc_consumer_state, intra1_acc_pipeline, C
                )

                # Batched processing over C dimension
                for chunk_idx in cutlass.range(C, unroll=1):
                    # Conditionally wait for Delta/INTRA1_ACC buffer full
                    deltas_pipeline.consumer_wait(
                        deltas_consumer_state, peek_deltas_full_status
                    )
                    intra1_acc_pipeline.consumer_wait(
                        intra1_acc_consumer_state, peek_rd_intra1_acc_full_status
                    )

                    # Load Q from tmem
                    intra1_coord = (None, None, None, intra1_acc_consumer_state.index)
                    cute.copy(tiled_t2r_intra1, tTR_tQ[intra1_coord], tTR_rQ)
                    cute.arch.fence_view_async_tmem_load()

                    # Load tQsDeltaA_Row/tQsDeltaA_Col/tQsDelta from smem
                    delta_coord = (None, None, None, deltas_consumer_state.index)
                    cute.copy(
                        s2r_atom_cumsum, tQsDeltaA_Row[delta_coord], tQrDeltaA_Row
                    )
                    cute.copy(
                        s2r_atom_cumsum, tQsDeltaA_Col[delta_coord], tQrDeltaA_Col
                    )
                    cute.copy(s2r_atom_delta, tQsDelta[delta_coord], tQrDelta)

                    # SegSum
                    tRT_rQ = self.pre_intra_segsum(
                        tTR_rQ, tQrDeltaA_Row, tQrDeltaA_Col, tQrDelta, tCoord, tRT_rQ
                    )

                    # Wait for INTRA2_Q buffer empty
                    # Delay producer_acquire to right before data store
                    intra2_q_pipeline.producer_acquire(intra2_q_producer_state)

                    # Store Q from reg to tmem
                    q_coord = (None, None, None, None, intra2_q_producer_state.index)
                    cute.copy(tiled_r2t_q, tRT_rQ, tRT_tQ[q_coord])

                    # Async arrive Delta/INTRA1_ACC buffer empty
                    intra1_acc_pipeline.consumer_release(intra1_acc_consumer_state)
                    deltas_pipeline.consumer_release(deltas_consumer_state)

                    cute.arch.fence_view_async_tmem_store()

                    # Async arrive INTRA2_Q buffer full
                    intra2_q_pipeline.producer_commit(intra2_q_producer_state)

                    # Advance deltas/intra1_acc/intra2_q states
                    deltas_consumer_state.advance()
                    intra1_acc_consumer_state.advance()
                    intra2_q_producer_state.advance()

                    # Peek (try_wait) Delta/INTRA1_ACC buffer full for chunk_idx = chunk_idx + 1
                    peek_deltas_full_status = self.conditional_consumer_try_wait(
                        deltas_consumer_state, deltas_pipeline, C
                    )
                    peek_rd_intra1_acc_full_status = self.conditional_consumer_try_wait(
                        intra1_acc_consumer_state, intra1_acc_pipeline, C
                    )
                # END of for chunk_idx in cutlass.range(C, unroll=1)

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # END of while work_tile.is_valid_tile

            # Producer tail for INTRA2_Q
            intra2_q_pipeline.producer_tail(intra2_q_producer_state)
        # END of specialized pre-intra warp

        # Specialized Epilogue warp
        else:
            # Dealloc regs for pre-inter/pre-intra warps
            cute.arch.warpgroup_reg_dealloc(self.num_regs_epilogue_warps)

            # (L, D, INPUT_STAGE)
            sDeltaA = self.epilog_make_delta(smem_cumsum_delta)

            # Make tmem tensor for INTRA2_ACC/INTER2_ACC
            # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
            tCtAccIntra2 = self.mma_partition_c(
                tiled_mma_intra2,
                self.tile_shape_mnk_intra2,
                tmem_ptr_base + self.tmem_intra2_acc_offset,
                self.internal_stages,
            )
            # (M_PER_MMA, N_PER_MMA, INTERNAL_STAGE)
            tIntra2 = tCtAccIntra2[((None, None), 0, 0, None)]
            # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
            tCtAccInter2 = self.mma_partition_c(
                tiled_mma_inter2,
                self.tile_shape_mnk_inter2,
                tmem_ptr_base + self.tmem_inter2_acc_offset,
                self.internal_stages,
            )
            # (M_PER_MMA, N_PER_MMA, INTERNAL_STAGE)
            tInter2 = tCtAccInter2[((None, None), 0, 0, None)]

            # Subtiling INTRA2_ACC/INTER2_ACC/Delta/Y
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, INTERNAL_STAGE)
            tIntra_epi = cute.flat_divide(tIntra2, epi_tile)
            tInter_epi = cute.flat_divide(tInter2, epi_tile)
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, INPUT_STAGE)
            sDeltaA_epi = cute.flat_divide(sDeltaA, epi_tile)

            # Make tiled copy and partition tmem/reg tensor w.r.t tensor memory load
            # ((T2R_ATOM_V, T2R_REST_V), REST_M, REST_N, EPI_M, EPI_N, INTERNAL_STAGE)
            # ((T2R_ATOM_V, T2R_REST_V), REST_M, REST_N)
            (
                tiled_t2r_intra2,
                tTR_tIntra,
                tTR_rIntra,
            ) = self.epilog_tmem_load_and_partition_acc(local_tidx, tIntra_epi, smem_y)
            (
                tiled_t2r_inter2,
                tTR_tInter2,
                tTR_rInter,
            ) = self.epilog_tmem_load_and_partition_acc(local_tidx, tInter_epi, smem_y)

            # Make tiled copy and partition smem/reg tensor w.r.t smem load Delta
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, EPI_M, EPI_N, INPUT_STAGE)
            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            (
                s2r_atom_delta,
                tTR_sDeltaA,
                tTR_rDeltaA,
            ) = self.smem_load_and_partition_delta_d(
                tiled_t2r_inter2, local_tidx, sDeltaA_epi, (None, None, None, 0, 0, 0)
            )

            # Make tiled copy and Partition smem/register tensor w.r.t smem store Y
            # ((R2S_ATOM_V, R2S_REST_V), REST_M, REST_N, OUTPUT_STAGE)
            # ((R2S_ATOM_V, R2S_REST_V), REST_M, REST_N)
            tiled_r2s_y, tRS_rY, tRS_sY = self.smem_store_and_partition_p_y(
                local_tidx, smem_y, tiled_t2r_inter2
            )

            tRS_rCompute = cute.make_fragment(tRS_rY.shape, self.acc_dtype)

            tiled_s2r_x = None
            tSR_sX = None
            tSR_rX = None
            if cutlass.const_expr(self.has_d):
                # Make TiledCopy/smem/register tensor for smem load X
                # (R2S_ATOM, R2S_M, R2S_N, EPI_M, EPI_N, INPUT_STAGES)
                # (R2S_ATOM, R2S_M, R2S_N)
                tiled_s2r_x, tSR_sX, tSR_rX = self.epilog_smem_load_and_partition_x(
                    tiled_t2r_inter2, local_tidx, smem_xt, epi_tile
                )

            tRS_sD = None
            tRS_rD = None
            s2r_atom_d = None
            if cutlass.const_expr(self.d_has_hdim):
                # (L, D, INPUT_STAGE)
                sD = self.epilog_make_d(smem_d)
                # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, INPUT_STAGE)
                tD_sepi = cute.flat_divide(sD, epi_tile)

                # Make tiled copy and partition smem/reg tensor w.r.t smem load D
                # ((T2R_ATOM_V, T2R_REST_V), REST_M, REST_N, EPI_M, EPI_N, INPUT_STAGE)
                # ((T2R_ATOM_V, T2R_REST_V), REST_M, REST_N)
                s2r_atom_d, tRS_sD, tRS_rD = self.smem_load_and_partition_delta_d(
                    tiled_t2r_inter2, local_tidx, tD_sepi, (None, None, None, 0, 0, 0)
                )

            elif cutlass.const_expr(self.has_d):
                tRS_rD = cutlass.Float32(0.0).to(self.io_dtype)

            # Partition global/shared tensor for TMA store Y
            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), EPI_M, EPI_N, 1, 1, C, EH, B)
            bSG_sY, bSG_gY_pre_slice = self.epilog_tma_partition_y(
                tma_tensor_y, tma_atom_y, smem_y, epi_tile
            )

            # Make TMA store pipeline Y
            tma_y_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.output_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_id), 128
                ),
            )

            # Make consumer pipeline states for Delta/INTRA2_ACC/INTER2_ACC/X/D buffer
            deltas_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            intra2_acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.internal_stages
            )
            inter2_acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.internal_stages
            )
            x_consumer_state = None
            if cutlass.const_expr(self.has_d):
                x_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.input_stages
                )
            d_consumer_state = None
            if cutlass.const_expr(self.d_has_hdim):
                d_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.input_stages
                )

            while work_tile.is_valid_tile:
                b_idx, eh_idx, g_idx = work_tile.tile_idx

                # Slice global tensor to current tile idx
                # ((ATOM_V, REST_V), EPI_M, EPI_N, C)
                bSG_gY = bSG_gY_pre_slice[(None, None, None, 0, 0, None, eh_idx, b_idx)]
                if cutlass.const_expr(self.has_d and not self.d_has_hdim):
                    tRS_rD = tma_tensor_d[0, eh_idx]

                # Reset count for pipeline state
                deltas_consumer_state.reset_count()
                intra2_acc_consumer_state.reset_count()
                inter2_acc_consumer_state.reset_count()
                if cutlass.const_expr(self.has_d):
                    x_consumer_state.reset_count()
                if cutlass.const_expr(self.d_has_hdim):
                    d_consumer_state.reset_count()

                # Peek Delta/INTRA2_ACC/INTER2_ACC buffer status
                peek_deltas_full_status = self.conditional_consumer_try_wait(
                    deltas_consumer_state, deltas_pipeline, C
                )
                peek_rd_intra2_acc_full_status = self.conditional_consumer_try_wait(
                    intra2_acc_consumer_state, intra2_acc_pipeline, C
                )
                peek_rd_inter2_acc_full_status = self.conditional_consumer_try_wait(
                    inter2_acc_consumer_state, inter2_acc_pipeline, C
                )
                peek_rd_x_full_status = None
                if cutlass.const_expr(self.has_d):
                    peek_rd_x_full_status = self.conditional_consumer_try_wait(
                        x_consumer_state, x_pipeline, C
                    )

                if cutlass.const_expr(self.d_has_hdim):
                    d_pipeline.consumer_wait(d_consumer_state)

                # Batched processing over C dimension
                for chunk_idx in cutlass.range(C, unroll=1):
                    # Conditionally wait for Delta/INTRA2_ACC/INTER2_ACC/X buffer full
                    deltas_pipeline.consumer_wait(
                        deltas_consumer_state, peek_deltas_full_status
                    )
                    intra2_acc_pipeline.consumer_wait(
                        intra2_acc_consumer_state, peek_rd_intra2_acc_full_status
                    )
                    inter2_acc_pipeline.consumer_wait(
                        inter2_acc_consumer_state, peek_rd_inter2_acc_full_status
                    )
                    if cutlass.const_expr(self.has_d):
                        x_pipeline.consumer_wait(
                            x_consumer_state, peek_rd_x_full_status
                        )
                    # Loop over EPI_M and EPI_N subtiles
                    for epi_n in range(cute.size(tTR_tIntra, mode=[4])):
                        for epi_m in range(cute.size(tTR_tIntra, mode=[3])):
                            epi_iter_cnt = (
                                epi_n * cute.size(tTR_tIntra, mode=[3]) + epi_m
                            )
                            epi_buffer_idx = epi_iter_cnt % self.output_stages

                            # Load INTRA2_ACC/INTER2_ACC from tmem
                            subtile_coord = (
                                None,
                                None,
                                None,
                                epi_m,
                                epi_n,
                            )
                            intra2_coord = subtile_coord + (
                                intra2_acc_consumer_state.index,
                            )
                            cute.copy(
                                tiled_t2r_intra2,
                                tTR_tIntra[intra2_coord],
                                tTR_rIntra,
                            )
                            inter2_coord = subtile_coord + (
                                inter2_acc_consumer_state.index,
                            )
                            cute.copy(
                                tiled_t2r_inter2,
                                tTR_tInter2[inter2_coord],
                                tTR_rInter,
                            )
                            # Fence for T2R load
                            cute.arch.fence_view_async_tmem_load()

                            # Load Delta from smem
                            delta_coord = subtile_coord + (deltas_consumer_state.index,)
                            cute.copy(
                                s2r_atom_delta, tTR_sDeltaA[delta_coord], tTR_rDeltaA
                            )

                            # Load X from smem
                            if cutlass.const_expr(self.has_d):
                                x_coord = subtile_coord + (x_consumer_state.index,)
                                cute.copy(tiled_s2r_x, tSR_sX[x_coord], tSR_rX)

                            # Load D from smem
                            if cutlass.const_expr(self.d_has_hdim):
                                # Load vector D from smem (d_has_hdim = True)
                                d_coord = subtile_coord + (d_consumer_state.index,)
                                cute.copy(s2r_atom_d, tRS_sD[d_coord], tRS_rD)

                            # Combine INTRA2_ACC/INTER2_ACC/Delta/X/D
                            for reg_idx in range(0, cute.size(tRS_rCompute), 2):
                                (
                                    tRS_rCompute[reg_idx],
                                    tRS_rCompute[reg_idx + 1],
                                ) = cute.arch.fma_packed_f32x2(
                                    (tTR_rInter[reg_idx], tTR_rInter[reg_idx + 1]),
                                    (
                                        cute.math.exp(
                                            tTR_rDeltaA[reg_idx], fastmath=True
                                        ),
                                        cute.math.exp(
                                            tTR_rDeltaA[reg_idx + 1], fastmath=True
                                        ),
                                    ),
                                    (tTR_rIntra[reg_idx], tTR_rIntra[reg_idx + 1]),
                                )
                                # Fuse Y += X * D
                                if cutlass.const_expr(self.d_has_hdim):
                                    (
                                        tRS_rCompute[reg_idx],
                                        tRS_rCompute[reg_idx + 1],
                                    ) = cute.arch.fma_packed_f32x2(
                                        (
                                            tRS_rD[reg_idx].to(self.acc_dtype),
                                            tRS_rD[reg_idx + 1].to(self.acc_dtype),
                                        ),
                                        (
                                            tSR_rX[reg_idx].to(self.acc_dtype),
                                            tSR_rX[reg_idx + 1].to(self.acc_dtype),
                                        ),
                                        (
                                            tRS_rCompute[reg_idx],
                                            tRS_rCompute[reg_idx + 1],
                                        ),
                                    )
                                elif cutlass.const_expr(self.has_d):
                                    (
                                        tRS_rCompute[reg_idx],
                                        tRS_rCompute[reg_idx + 1],
                                    ) = cute.arch.fma_packed_f32x2(
                                        (
                                            tRS_rD.to(self.acc_dtype),
                                            tRS_rD.to(self.acc_dtype),
                                        ),
                                        (
                                            tSR_rX[reg_idx].to(self.acc_dtype),
                                            tSR_rX[reg_idx + 1].to(self.acc_dtype),
                                        ),
                                        (
                                            tRS_rCompute[reg_idx],
                                            tRS_rCompute[reg_idx + 1],
                                        ),
                                    )

                            tRS_rY.store(tRS_rCompute.load().to(self.io_dtype))

                            # Store Y to smem
                            cute.copy(
                                tiled_r2s_y,
                                tRS_rY,
                                tRS_sY[None, None, None, epi_buffer_idx],
                            )

                            # Fence for R2S store
                            cute.arch.fence_proxy(
                                cute.arch.ProxyKind.async_shared,
                                space=cute.arch.SharedSpace.shared_cta,
                            )
                            # Sync before TMA store
                            cute.arch.barrier(
                                barrier_id=self.epilog_sync_bar_id,
                                number_of_threads=len(self.epilog_warp_id) * 32,
                            )

                            # Async arrive Delta/INTRA2_ACC/INTER2_ACC buffer empty
                            if (
                                epi_iter_cnt
                                == cute.size(tTR_tIntra, mode=[4])
                                * cute.size(tTR_tIntra, mode=[3])
                                - 1
                            ):
                                deltas_pipeline.consumer_release(deltas_consumer_state)
                                intra2_acc_pipeline.consumer_release(
                                    intra2_acc_consumer_state
                                )
                                inter2_acc_pipeline.consumer_release(
                                    inter2_acc_consumer_state
                                )
                                if cutlass.const_expr(self.has_d):
                                    x_pipeline.consumer_release(
                                        x_consumer_state,
                                        pipeline.PipelineOp.AsyncThread,
                                    )

                            # TMA store Y to global memory
                            if local_warp_idx == 0:
                                cute.copy(
                                    tma_atom_y,
                                    bSG_sY[None, epi_buffer_idx],
                                    bSG_gY[None, epi_m, epi_n, chunk_idx],
                                )

                                # Commit TMA store
                                tma_y_pipeline.producer_commit()
                                # Wait for TMA store
                                tma_y_pipeline.producer_acquire()
                            # Sync before smem store
                            cute.arch.barrier(
                                barrier_id=self.epilog_sync_bar_id,
                                number_of_threads=len(self.epilog_warp_id) * 32,
                            )

                    # Advance deltas/intra2_acc/inter2_acc consumer states
                    deltas_consumer_state.advance()
                    intra2_acc_consumer_state.advance()
                    inter2_acc_consumer_state.advance()

                    # Peek (try_wait) Delta/INTRA2_ACC/INTER2_ACC buffer full for chunk_idx = chunk_idx + 1
                    peek_deltas_full_status = self.conditional_consumer_try_wait(
                        deltas_consumer_state, deltas_pipeline, C
                    )
                    peek_rd_intra2_acc_full_status = self.conditional_consumer_try_wait(
                        intra2_acc_consumer_state, intra2_acc_pipeline, C
                    )
                    peek_rd_inter2_acc_full_status = self.conditional_consumer_try_wait(
                        inter2_acc_consumer_state, inter2_acc_pipeline, C
                    )

                    if cutlass.const_expr(self.has_d):
                        # Advance x consumer states
                        x_consumer_state.advance()
                        # Peek (try_wait) X buffer full for chunk_idx = chunk_idx + 1
                        peek_rd_x_full_status = self.conditional_consumer_try_wait(
                            x_consumer_state, x_pipeline, C
                        )

                if cutlass.const_expr(self.d_has_hdim):
                    d_pipeline.consumer_release(d_consumer_state)
                    d_consumer_state.advance()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Producer tail for TMA store Y
            tma_y_pipeline.producer_tail()

        # Dealloc tmem buffer
        if warp_idx == self.epilog_warp_id[0]:
            cute.arch.barrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                number_of_threads=self.threads_per_cta,
            )
            cute.arch.dealloc_tmem(
                tmem_ptr_base,
                self.num_tmem_cols_total,
                is_two_cta=self.use_2cta_instrs,
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                number_of_threads=self.threads_per_cta,
            )

        return

    @staticmethod
    def _compute_stages(smem_capacity):
        return 2, 2, 1, 2  # input, output, internal, intra1_acc

    @staticmethod
    def _compute_grid(y, b, max_active_clusters):
        B = cute.size(y, mode=[4])
        EH = cute.size(y, mode=[3])
        G = cute.size(b, mode=[3])
        NGROUP_RATIO = EH // G
        num_blocks = B * EH

        tile_sched_params = Mamba2SSDTileSchedulerParams(num_blocks, EH, NGROUP_RATIO)
        grid = Mamba2SSDTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _plan_tmem_offsets(
        tiled_mma_intra1,
        tile_shape_mnk_intra1,
        tiled_mma_intra2,
        tile_shape_mnk_intra2,
        tiled_mma_inter1,
        tile_shape_mnk_inter1,
        tiled_mma_inter2,
        tile_shape_mnk_inter2,
        acc_stages,
        intra2_a_tmem_layout,
        a_dtype,
        internal_stages,
        intra1_acc_stages,
    ):
        SM100_TMEM_CAPACITY_COLUMNS = 512
        BITS_PER_TMEM_COL = 32
        # (MMA, MMA_M, MMA_N)
        acc_shape_intra1 = tiled_mma_intra1.partition_shape_C(tile_shape_mnk_intra1[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccIntra1_fake = tiled_mma_intra1.make_fragment_C(
            cute.append(acc_shape_intra1, intra1_acc_stages)
        )
        num_intra1_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccIntra1_fake)
        assert tile_shape_mnk_intra1[1] * intra1_acc_stages == num_intra1_acc_cols
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrQ_fake = tiled_mma_intra2.make_fragment_A(intra2_a_tmem_layout.outer.shape)
        num_intra2_a_cols = tcgen05.find_tmem_tensor_col_offset(tCrQ_fake)
        assert (
            tile_shape_mnk_intra2[2]
            * internal_stages
            * a_dtype.width
            // BITS_PER_TMEM_COL
            == num_intra2_a_cols
        )
        # (MMA, MMA_M, MMA_N)
        acc_shape_intra2 = tiled_mma_intra2.partition_shape_C(tile_shape_mnk_intra2[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccIntra2_fake = tiled_mma_intra2.make_fragment_C(
            cute.append(acc_shape_intra2, acc_stages)
        )
        num_intra2_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccIntra2_fake)
        assert tile_shape_mnk_intra2[1] * acc_stages == num_intra2_acc_cols

        # (MMA, MMA_M, MMA_N)
        acc_shape_inter1 = tiled_mma_inter1.partition_shape_C(tile_shape_mnk_inter1[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccInter1_fake = tiled_mma_inter1.make_fragment_C(
            cute.append(acc_shape_inter1, acc_stages)
        )
        num_inter1_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccInter1_fake)
        assert tile_shape_mnk_inter1[1] * acc_stages == num_inter1_acc_cols

        # (MMA, MMA_M, MMA_N)
        acc_shape_inter2 = tiled_mma_inter2.partition_shape_C(tile_shape_mnk_inter2[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccInter2_fake = tiled_mma_inter2.make_fragment_C(
            cute.append(acc_shape_inter2, acc_stages)
        )
        num_inter2_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccInter2_fake)
        assert tile_shape_mnk_inter2[1] * acc_stages == num_inter2_acc_cols

        tmem_intra1_acc_offset = 0
        tmem_intra2_q_offset = tmem_intra1_acc_offset + num_intra1_acc_cols
        tmem_intra2_acc_offset = tmem_intra2_q_offset + num_intra2_a_cols
        tmem_inter1_acc_offset = tmem_intra2_acc_offset + num_intra2_acc_cols
        tmem_inter2_acc_offset = tmem_inter1_acc_offset + num_inter1_acc_cols
        num_tmem_cols_total_tmp = tmem_inter2_acc_offset + num_inter2_acc_cols
        # Turn num_tmem_cols_total to the nearest power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLUMNS

        return (
            tmem_intra1_acc_offset,
            tmem_intra2_q_offset,
            tmem_intra2_acc_offset,
            tmem_inter1_acc_offset,
            tmem_inter2_acc_offset,
            num_tmem_cols_total,
        )

    @staticmethod
    def make_tiled_mmas(
        io_dtype,
        acc_dtype,
        cta_group,
        tile_shape_mnk_intra1,
        tile_shape_mnk_intra2,
        tile_shape_mnk_inter1,
        tile_shape_mnk_inter2,
    ):
        tiled_mma_intra1 = sm100_utils.make_trivial_tiled_mma(
            io_dtype,
            tcgen05.OperandMajorMode("mn"),
            tcgen05.OperandMajorMode("mn"),
            acc_dtype,
            cta_group,
            tile_shape_mnk_intra1[:2],
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma_intra2 = sm100_utils.make_trivial_tiled_mma(
            io_dtype,
            tcgen05.OperandMajorMode("k"),
            tcgen05.OperandMajorMode("k"),
            acc_dtype,
            cta_group,
            tile_shape_mnk_intra2[:2],
            tcgen05.OperandSource.TMEM,
        )
        tiled_mma_inter1 = sm100_utils.make_trivial_tiled_mma(
            io_dtype,
            tcgen05.OperandMajorMode("k"),
            tcgen05.OperandMajorMode("k"),
            acc_dtype,
            cta_group,
            tile_shape_mnk_inter1[:2],
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma_inter2 = sm100_utils.make_trivial_tiled_mma(
            io_dtype,
            tcgen05.OperandMajorMode("mn"),
            tcgen05.OperandMajorMode("k"),
            acc_dtype,
            cta_group,
            tile_shape_mnk_inter2[:2],
            tcgen05.OperandSource.SMEM,
        )
        return tiled_mma_intra1, tiled_mma_intra2, tiled_mma_inter1, tiled_mma_inter2

    def make_and_init_x_pipeline(self, x_full_mbar_ptr):
        x_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.tma_deltas_x_d_warp_id])
        )
        if not self.has_d:
            x_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len([self.mma_intra_warp_id, self.mma_inter_warp_id]),
            )
            return pipeline.PipelineTmaUmma.create(
                num_stages=self.input_stages,
                producer_group=x_producer_group,
                consumer_group=x_consumer_group,
                tx_count=self.num_x_load_bytes,
                barrier_storage=x_full_mbar_ptr,
            )
        else:
            x_consumer_group_umma = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len([self.mma_intra_warp_id, self.mma_inter_warp_id]),
            )
            x_consumer_group_async = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilog_warp_id), 128
            )
            return pipeline.PipelineTmaMultiConsumersAsync.create(
                num_stages=self.input_stages,
                producer_group=x_producer_group,
                consumer_group_umma=x_consumer_group_umma,
                consumer_group_async=x_consumer_group_async,
                tx_count=self.num_x_load_bytes,
                barrier_storage=x_full_mbar_ptr,
            )

    def make_and_init_b_pipeline(self, b_full_mbar_ptr):
        b_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.tma_b_c_warp_id])
        )
        b_consumer_group_umma = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_intra_warp_id])
        )
        b_consumer_group_async = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.pre_inter_warp_id), 128
        )
        return pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.input_stages,
            producer_group=b_producer_group,
            consumer_group_umma=b_consumer_group_umma,
            consumer_group_async=b_consumer_group_async,
            tx_count=self.num_b_load_bytes,
            barrier_storage=b_full_mbar_ptr,
        )

    def make_and_init_c_pipeline(self, c_full_mbar_ptr):
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.tma_b_c_warp_id])
        )
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_intra_warp_id, self.mma_inter_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            num_stages=self.input_stages,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=self.num_c_load_bytes,
            barrier_storage=c_full_mbar_ptr,
        )

    def make_and_init_deltas_pipeline(self, deltas_full_mbar_ptr):
        deltas_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.tma_deltas_x_d_warp_id])
        )
        deltas_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(
                [*self.pre_inter_warp_id, *self.pre_intra_warp_id, *self.epilog_warp_id]
            ),
            len(
                [*self.pre_inter_warp_id, *self.pre_intra_warp_id, *self.epilog_warp_id]
            ),
        )

        return pipeline.PipelineTmaAsync.create(
            num_stages=self.input_stages,
            producer_group=deltas_producer_group,
            consumer_group=deltas_consumer_group,
            tx_count=self.num_delta_load_bytes + self.num_cumsum_delta_load_bytes,
            barrier_storage=deltas_full_mbar_ptr,
        )

    def make_and_init_d_pipeline(self, d_full_mbar_ptr):
        if not self.d_has_hdim:
            return None
        else:
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.tma_deltas_x_d_warp_id])
            )
            d_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.epilog_warp_id),
                len(self.epilog_warp_id),
            )

            return pipeline.PipelineTmaAsync.create(
                num_stages=self.input_stages,
                producer_group=d_producer_group,
                consumer_group=d_consumer_group,
                tx_count=self.num_d_load_bytes,
                barrier_storage=d_full_mbar_ptr,
            )

    def make_and_init_intra1_acc_pipeline(self, intra1_acc_full_mbar_ptr):
        intra1_acc_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_intra_warp_id])
        )
        intra1_acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.pre_intra_warp_id), 128
        )
        return pipeline.PipelineUmmaAsync.create(
            num_stages=self.intra1_acc_stages,
            producer_group=intra1_acc_producer_group,
            consumer_group=intra1_acc_consumer_group,
            barrier_storage=intra1_acc_full_mbar_ptr,
        )

    def make_and_init_intra2_q_pipeline(self, intra2_q_full_mbar_ptr):
        intra2_q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.pre_intra_warp_id), 128
        )
        intra2_q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_intra_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            num_stages=self.internal_stages,
            producer_group=intra2_q_producer_group,
            consumer_group=intra2_q_consumer_group,
            barrier_storage=intra2_q_full_mbar_ptr,
        )

    def make_and_init_intra2_acc_pipeline(self, intra2_acc_full_mbar_ptr):
        intra2_acc_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_intra_warp_id])
        )
        intra2_acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.epilog_warp_id), 128
        )
        return pipeline.PipelineUmmaAsync.create(
            num_stages=self.internal_stages,
            producer_group=intra2_acc_producer_group,
            consumer_group=intra2_acc_consumer_group,
            barrier_storage=intra2_acc_full_mbar_ptr,
        )

    def make_and_init_inter1_b_pipeline(self, inter1_b_full_mbar_ptr):
        inter1_b_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.pre_inter_warp_id), 128
        )
        inter1_b_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_inter_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            num_stages=self.internal_stages,
            producer_group=inter1_b_producer_group,
            consumer_group=inter1_b_consumer_group,
            barrier_storage=inter1_b_full_mbar_ptr,
        )

    def make_and_init_inter1_acc_pipeline(self, inter1_acc_full_mbar_ptr):
        inter1_acc_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_inter_warp_id])
        )
        inter1_acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.pre_inter_warp_id), 128
        )
        return pipeline.PipelineUmmaAsync.create(
            num_stages=self.internal_stages,
            producer_group=inter1_acc_producer_group,
            consumer_group=inter1_acc_consumer_group,
            barrier_storage=inter1_acc_full_mbar_ptr,
        )

    def make_and_init_inter2_p_pipeline(self, inter2_p_full_mbar_ptr):
        inter2_p_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.pre_inter_warp_id), 128
        )
        inter2_p_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_inter_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            num_stages=self.internal_stages,
            producer_group=inter2_p_producer_group,
            consumer_group=inter2_p_consumer_group,
            barrier_storage=inter2_p_full_mbar_ptr,
        )

    def make_and_init_inter2_acc_pipeline(self, inter2_acc_full_mbar_ptr):
        inter2_acc_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_inter_warp_id])
        )
        inter2_acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(self.epilog_warp_id), 128
        )
        return pipeline.PipelineUmmaAsync.create(
            num_stages=self.internal_stages,
            producer_group=inter2_acc_producer_group,
            consumer_group=inter2_acc_consumer_group,
            barrier_storage=inter2_acc_full_mbar_ptr,
        )

    def tma_partition_for_mma_b_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tiled_mma_intra2,
        cluster_layout_vmnk,
        mma_tile_coord_v,
        block_in_cluster_coord_vmnk,
    ):
        # Local_tile partition global tensors
        # (D, L, 1, 1, C, EH, B)
        gX = cute.local_tile(
            tma_tensor_x,
            self.tile_shape_mnk_intra2[1:],
            (None, None, None, None, None),
        )
        # Partition global tensor with regard to TiledMMA
        thr_mma_intra2 = tiled_mma_intra2.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, 1, 1, C, EH, B)
        tCgX = thr_mma_intra2.partition_B(gX)

        # Partition global/shared tensor for X
        x_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), 1, 1, C, EH, B)
        tXsX, tXgX_pre_slice = cpasync.tma_partition(
            tma_atom_x,
            block_in_cluster_coord_vmnk[2],
            x_cta_layout,
            cute.group_modes(smem_x, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX_pre_slice

    def tma_partition_for_mma_a_operand(
        self,
        tma_atom_c,
        tma_tensor_c,
        smem_c,
        tiled_mma_intra1,
        cluster_layout_vmnk,
        mma_tile_coord_v,
        block_in_cluster_coord_vmnk,
    ):
        # Local_tile partition global tensors
        # (L, N, 1, 1, C, G, B)
        gC = cute.local_tile(
            tma_tensor_c,
            cute.slice_(self.tile_shape_mnk_intra1, (None, 0, None)),
            (None, None, None, None, None),
        )
        # Partition global tensor with regard to TiledMMA
        thr_mma_intra1 = tiled_mma_intra1.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M/N, MMA_K, 1, 1, C, G, B)
        tCgC = thr_mma_intra1.partition_A(gC)

        # Partition global/shared tensor for TMA C
        c_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), 1, 1, C, G, B)
        tCsC, tCgC_pre_slice = cpasync.tma_partition(
            tma_atom_c,
            block_in_cluster_coord_vmnk[1],
            c_cta_layout,
            cute.group_modes(smem_c, 0, 3),
            cute.group_modes(tCgC, 0, 3),
        )
        return tCsC, tCgC_pre_slice

    def tma_partition_with_shape(
        self, tma_atom_delta, tma_tensor_delta, smem_delta, shape
    ):
        # Local_tile partition global tensors
        # (L, 1, C, EH, B)
        gDelta = cute.local_tile(
            tma_tensor_delta,
            shape,
            (None,) * cute.rank(tma_tensor_delta),
        )
        # Partition global/shared tensor for DELTA
        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), 1, C, EH, B)
        tDeltasDelta, tDeltagDelta_pre_slice = cpasync.tma_partition(
            tma_atom_delta,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_delta, 0, cute.rank(shape)),
            cute.group_modes(gDelta, 0, cute.rank(shape)),
        )

        return tDeltasDelta, tDeltagDelta_pre_slice

    def mma_partition_ss(
        self,
        tiled_mma,
        tile_shape_mnk,
        smem_a,
        smem_b,
        tmem_acc_ptr,
        acc_stages,
    ):
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        tCrA = tiled_mma.make_fragment_A(smem_a)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrB = tiled_mma.make_fragment_B(smem_b)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCtAcc = self.mma_partition_c(
            tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages
        )
        return tCrA, tCrB, tCtAcc

    def mma_partition_ts(
        self,
        tiled_mma,
        tile_shape_mnk,
        a_tmem_layout,
        smem_b,
        tmem_a_ptr,
        tmem_acc_ptr,
        acc_stages,
    ):
        # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
        tCrA = self.mma_partition_a_tmem(tiled_mma, a_tmem_layout, tmem_a_ptr)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrB = tiled_mma.make_fragment_B(smem_b)
        # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
        tCtAcc = self.mma_partition_c(
            tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages
        )
        return tCrA, tCrB, tCtAcc

    def mma_partition_a_tmem(self, tiled_mma, a_tmem_layout, tmem_a_ptr):
        tCrA_fake = tiled_mma.make_fragment_A(a_tmem_layout.outer.shape)
        tCrA = cute.make_tensor(
            cute.recast_ptr(
                tmem_a_ptr,
                dtype=tCrA_fake.element_type,
            ),
            tCrA_fake.layout,
        )
        return tCrA

    def mma_partition_c(self, tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages):
        acc_shape = tiled_mma.partition_shape_C(tile_shape_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
        tCtAcc = cute.make_tensor(tmem_acc_ptr, tCtAcc_fake.layout)
        return tCtAcc

    @cute.jit
    def exec_mma(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        acc_producer_state,
        a_consumer_state,
        b_consumer_state,
    ):
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_full=True):
            # set accu = 1
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE,
                cutlass.Boolean(kphase_idx != 0),
            )
            cute.gemm(
                tiled_mma,
                tCtAcc[None, None, None, acc_producer_state.index],
                tCrA[None, None, kphase_idx, a_consumer_state.index],
                tCrB[None, None, kphase_idx, b_consumer_state.index],
                tCtAcc[None, None, None, acc_producer_state.index],
            )
        return tiled_mma

    @cute.jit
    def conditional_consumer_try_wait(self, b_consumer_state, b_pipeline, C):
        peek_b_full_status = cutlass.Boolean(1)
        if b_consumer_state.count < C:
            peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)
        return peek_b_full_status

    @cute.jit
    def conditional_producer_try_acquire(
        self, intra1_acc_producer_state, intra1_acc_pipeline, C
    ):
        peek_wr_intra1_acc_empty_status = cutlass.Boolean(1)
        if intra1_acc_producer_state.count < C:
            peek_wr_intra1_acc_empty_status = intra1_acc_pipeline.producer_try_acquire(
                intra1_acc_producer_state
            )
        return peek_wr_intra1_acc_empty_status

    def pre_intra_tmem_load_and_partition_q(self, tIntra1, local_tidx):
        copy_atom_t2r_intra1 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        # (L, L)
        fake_sQ = cute.make_tensor(
            cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            cute.dice(self.tile_shape_mnk_intra1, (1, 1, None)),
        )
        return self.make_tmem_load_and_partition(
            copy_atom_t2r_intra1, tIntra1, (None, None, 0), local_tidx, fake_sQ
        )

    def pre_intra_make_delta(self, smem_delta, extend_on_row_or_col):
        smem_iterator = smem_delta.iterator
        delta_linear_smem_layout = smem_delta.layout
        # extend L linear layout to LxL
        extend_layout = cute.make_layout(delta_linear_smem_layout.shape[0], stride=0)
        if extend_on_row_or_col == 0:
            # (L, L, INPUT_STAGE):(0, 1, L)
            sDelta = cute.make_tensor(
                smem_iterator,
                cute.prepend(
                    delta_linear_smem_layout,
                    extend_layout,
                ),
            )
        else:
            # (L, L, INPUT_STAGE):(1, 0, L)
            sDelta = cute.make_tensor(
                smem_iterator,
                cute.append(
                    cute.append(
                        cute.get(delta_linear_smem_layout, mode=[0]),
                        extend_layout,
                    ),
                    cute.get(delta_linear_smem_layout, mode=[1]),
                ),
            )
        return sDelta

    def pre_intra_tmem_store_and_partition_q(self, local_tidx, tCrQ):
        dtype = tCrQ.element_type
        # Make tiledCopy for tensor memory store INTRA2_Q
        copy_atom_r2t_q = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition(16), tcgen05.Unpack.NONE),
            dtype,
        )
        tiled_r2t_q = tcgen05.make_tmem_copy(copy_atom_r2t_q, tCrQ)
        thr_r2t_q = tiled_r2t_q.get_slice(local_tidx)

        # Partition tmem/register tensor for tensor memory store INTRA2_Q
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ...)
        tRT_rQ = cute.make_fragment(
            cute.slice_(thr_r2t_q.partition_S(tCrQ).shape, (None, None, None, None, 0)),
            dtype,
        )
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ..., INTERNAL_STAGE)
        tRT_tQ = thr_r2t_q.partition_D(tCrQ)

        return tiled_r2t_q, tRT_rQ, tRT_tQ

    @cute.jit
    def pre_intra_segsum(
        self, tTR_rQ, tQrDeltaA_Row, tQrDeltaA_Col, tQrDelta, tCoord, tRT_rQ
    ):
        # Make tmp acc type fragments
        tCrDeltaA_Row = cute.make_fragment(tQrDeltaA_Row.shape, self.acc_dtype)
        tCrDeltaA_Col = cute.make_fragment(tQrDeltaA_Col.shape, self.acc_dtype)
        tCrDelta = cute.make_fragment(tQrDelta.shape, self.acc_dtype)
        tCompute = cute.make_fragment(tRT_rQ.shape, self.acc_dtype)

        # Combine tTR_rQ/tCrDeltaA_Row/tCrDeltaA_Col/tCrDelta
        tCrDeltaA_Row.store(tQrDeltaA_Row.load().to(self.acc_dtype))
        tCrDeltaA_Col.store(tQrDeltaA_Col.load().to(self.acc_dtype))
        tCrDelta.store(tQrDelta.load().to(self.acc_dtype))

        # SegSum
        # fadd2 + fsel + fmul2/mufu + fmul2
        for subtile_idx in cutlass.range(0, cute.size(tTR_rQ), 2, unroll_full=True):
            (
                tCompute[subtile_idx],
                tCompute[subtile_idx + 1],
            ) = cute.arch.add_packed_f32x2(
                (tCrDeltaA_Col[subtile_idx], tCrDeltaA_Col[subtile_idx + 1]),
                (-tCrDeltaA_Row[subtile_idx], -tCrDeltaA_Row[subtile_idx + 1]),
            )
        for subtile_idx in cutlass.range(cute.size(tTR_rQ), unroll_full=True):
            m, n = tCoord[subtile_idx]
            if m < n:
                tCompute[subtile_idx] = cutlass.Float32(-float("inf"))
        LOG2_E = cutlass.Float32(1.4426950408889634)
        for subtile_idx in cutlass.range(0, cute.size(tTR_rQ), 2, unroll_full=True):
            # TODO: use math.exp directly
            tCompute_log2e = cute.arch.mul_packed_f32x2(
                (tCompute[subtile_idx], tCompute[subtile_idx + 1]), (LOG2_E, LOG2_E)
            )
            (
                tCompute[subtile_idx],
                tCompute[subtile_idx + 1],
            ) = cute.arch.mul_packed_f32x2(
                (
                    cute.math.exp2(tCompute_log2e[0], fastmath=True),
                    cute.math.exp2(tCompute_log2e[1], fastmath=True),
                ),
                (tCrDelta[subtile_idx], tCrDelta[subtile_idx + 1]),
            )
            (
                tCompute[subtile_idx],
                tCompute[subtile_idx + 1],
            ) = cute.arch.mul_packed_f32x2(
                (tCompute[subtile_idx], tCompute[subtile_idx + 1]),
                (tTR_rQ[subtile_idx], tTR_rQ[subtile_idx + 1]),
            )

        tRT_rQ.store(tCompute.load().to(self.io_dtype))
        return tRT_rQ

    def pre_inter_smem_load_and_partition_b(self, local_tidx, smem_bt):
        dtype = smem_bt.element_type
        copy_atom_s2r_b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        num_elements_per_thread = 128 // dtype.width
        num_threads_per_row = self.tile_shape_mnk_inter1[2] // num_elements_per_thread
        num_threads_per_col = 128 // num_threads_per_row
        thread_layout = cute.make_layout(
            (num_threads_per_col, num_threads_per_row),
            stride=(num_threads_per_row, 1),
        )
        val_layout = cute.make_layout((1, num_elements_per_thread))
        tiled_s2r_b = cute.make_tiled_copy_tv(
            copy_atom_s2r_b,
            thread_layout,
            val_layout,
        )
        thr_s2r_b = tiled_s2r_b.get_slice(local_tidx)

        # Partition shared tensor for smem load Bt
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
        tBsB_s2r = thr_s2r_b.partition_S(smem_bt)

        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tBrB_s2r = cute.make_fragment(
            cute.slice_(tBsB_s2r.shape, (None, None, None, 0)),
            dtype,
        )
        return tiled_s2r_b, tBsB_s2r, tBrB_s2r

    def pre_inter_smem_store_and_partition_b(
        self, local_tidx, smem_bt_internal, tiled_s2r_b, tBrB_s2r
    ):
        dtype = smem_bt_internal.element_type
        # Make tiledCopy from register to smem store Bt
        copy_atom_r2s_b = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        tiled_r2s_b = cute.make_tiled_copy_S(copy_atom_r2s_b, tiled_s2r_b)
        thr_r2s_b = tiled_r2s_b.get_slice(local_tidx)

        # Partition shared tensor for smem store Bt
        # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N, INTERNAL_STAGE)
        tBsB_r2s = thr_r2s_b.partition_D(smem_bt_internal)

        # Make register fragments for smem load/store Bt
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tBrB_r2s = thr_r2s_b.retile(tBrB_s2r)
        return tiled_r2s_b, tBrB_r2s, tBsB_r2s

    def smem_load_and_partition_delta_d(
        self, tiled_s2r_b, local_tidx, smem_delta, smem_tile_coord
    ):
        dtype = smem_delta.element_type
        s2r_atom_delta = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype)

        thr_s2r_b = tiled_s2r_b.get_slice(local_tidx)
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N, INPUT_STAGE)
        tBsDelta_s2r = thr_s2r_b.partition_D(smem_delta)

        # Make register fragments for smem load/store of Delta/DeltaA
        # ((S2R_ATOM_V, S2R_REST_V), S2R_M, S2R_N)
        tBrDelta_s2r = cute.make_fragment(tBsDelta_s2r[smem_tile_coord].shape, dtype)
        return s2r_atom_delta, tBsDelta_s2r, tBrDelta_s2r

    def pre_inter_tmem_load_and_partition_p(self, local_tidx, tInter1, smem_pt):
        copy_atom_t2r_inter1 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        return self.make_tmem_load_and_partition(
            copy_atom_t2r_inter1,
            tInter1,
            (None, None, 0),
            local_tidx,
            smem_pt[None, None, 0],
        )

    def make_tmem_load_and_partition(
        self, copy_atom_t2r, tmem_tensor, tmem_tile_coord, local_tidx, smem_tensor
    ):
        dtype = tmem_tensor.element_type
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tmem_tensor[tmem_tile_coord])
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        # Partition tmem/shared tensor for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_t = thr_t2r.partition_S(tmem_tensor)
        tTR_s = thr_t2r.partition_D(smem_tensor)
        # Make register fragments for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_r = cute.make_fragment(
            tTR_s.shape,
            dtype,
        )
        return tiled_t2r, tTR_t, tTR_r

    def smem_store_and_partition_p_y(self, local_tidx, smem_pt, tiled_t2r_inter1):
        dtype = smem_pt.element_type
        copy_atom_r2s_p = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
            dtype,
        )
        tiled_r2s_p = cute.make_tiled_copy_D(copy_atom_r2s_p, tiled_t2r_inter1)
        thr_r2s_p = tiled_r2s_p.get_slice(local_tidx)
        # Partition smem/register tensor for smem store INTER2_P
        # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N, INTERNAL_STAGE)
        tRS_sP = thr_r2s_p.partition_D(smem_pt)
        # ((R2S_ATOM_V, R2S_REST_V), R2S_M, R2S_N)
        tRS_rP = cute.make_fragment(
            cute.slice_(tRS_sP.shape, (None, None, None, 0)), self.io_dtype
        )
        return tiled_r2s_p, tRS_rP, tRS_sP

    def pre_inter_make_delta(self, smem_delta, smem_bt_layout):
        # Broadcast Delta/DeltaA to Bt shape on M dimension
        # before: (128,(64,2),2):(64,(1,8192),16384)
        # after : (128,(64,2),2):(0,(1,64),128)
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        sDeltaA = cute.make_tensor(
            smem_delta.iterator,
            cute.make_layout(
                smem_bt_layout.shape,
                stride=(
                    0,
                    (1, cute.get(smem_bt_layout.shape, mode=[1, 0])),
                    smem_delta.layout.stride[1],
                ),
            ),
        )
        return sDeltaA

    def pre_inter_scale_bt_with_delta(
        self, tBrB_s2r, tBrDelta_s2r, tBrDeltaA_s2r, last_column
    ):
        tCompute = cute.make_fragment(tBrB_s2r.shape, self.acc_dtype)
        tBrB_Compute = cute.make_fragment(tBrB_s2r.shape, self.acc_dtype)
        tBrDelta_Compute = cute.make_fragment(tBrDelta_s2r.shape, self.acc_dtype)
        tBrDeltaA_Compute = cute.make_fragment(tBrDeltaA_s2r.shape, self.acc_dtype)

        tBrB_Compute.store(tBrB_s2r.load().to(self.acc_dtype))
        tBrDelta_Compute.store(tBrDelta_s2r.load().to(self.acc_dtype))
        tBrDeltaA_Compute.store(tBrDeltaA_s2r.load().to(self.acc_dtype))

        for reg_idx in range(0, cute.size(tBrB_Compute), 2):
            tCompute[reg_idx], tCompute[reg_idx + 1] = cute.arch.mul_packed_f32x2(
                (
                    cute.math.exp(
                        (last_column - tBrDeltaA_Compute[reg_idx]), fastmath=True
                    ),
                    cute.math.exp(
                        (last_column - tBrDeltaA_Compute[reg_idx + 1]), fastmath=True
                    ),
                ),
                (tBrDelta_Compute[reg_idx], tBrDelta_Compute[reg_idx + 1]),
            )
            tCompute[reg_idx], tCompute[reg_idx + 1] = cute.arch.mul_packed_f32x2(
                (tCompute[reg_idx], tCompute[reg_idx + 1]),
                (tBrB_Compute[reg_idx], tBrB_Compute[reg_idx + 1]),
            )
        return tCompute

    def epilog_make_delta(self, smem_cumsum_delta):
        # Broadcast cumsum delta from LxINPUT_STAGE to LxDxINPUT_STAGE
        sDeltaA = cute.make_tensor(
            smem_cumsum_delta.iterator,
            cute.make_layout(
                (*self.tile_shape_mnk_inter2[:2], self.input_stages),
                stride=(1, 0, smem_cumsum_delta.layout.shape[0]),
            ),
        )
        return sDeltaA

    def epilog_make_d(self, smem_d):
        # Broadcast d from DxINPUT_STAGE to LxDxINPUT_STAGE
        sD = cute.make_tensor(
            smem_d.iterator,
            cute.make_layout(
                (*self.tile_shape_mnk_inter2[:2], self.input_stages),
                stride=(0, 1, smem_d.layout.shape[0]),
            ),
        )
        return sD

    def epilog_tma_partition_y(self, tma_tensor_y, tma_atom_y, smem_y, epi_tile):
        # Local_tile partition global tensors
        # (L, D, 1, 1, C, EH, B)
        gY = cute.local_tile(
            tma_tensor_y,
            cute.slice_(self.tile_shape_mnk_inter2, (None, None, 0)),
            (None, None, None, None, None),
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, 1, 1, C, EH, B)
        gY_epi = cute.flat_divide(gY, epi_tile)
        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, 1, 1, C, EH, B)
        bSG_sY, bSG_gY_pre_slice = cpasync.tma_partition(
            tma_atom_y,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_y, 0, 2),
            cute.group_modes(gY_epi, 0, 2),
        )
        return bSG_sY, bSG_gY_pre_slice

    def epilog_smem_load_and_partition_x(
        self, tiled_t2r_inter2_intra2, local_tidx, smem_xt, epi_tile
    ):
        dtype = smem_xt.element_type
        copy_atom_s2r_x = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            dtype,
        )
        tiled_s2r_x = cute.make_tiled_copy_D(copy_atom_s2r_x, tiled_t2r_inter2_intra2)
        thr_s2r_x = tiled_s2r_x.get_slice(local_tidx)
        # Partition smem/register tensor for smem store INTER2_P
        # (R2S_ATOM, R2S_M, R2S_N, EPI_M, EPI_N, INPUT_STAGES)
        tSR_sX = thr_s2r_x.partition_S(cute.flat_divide(smem_xt, epi_tile))
        # (R2S_ATOM, R2S_M, R2S_N)
        tSR_rX = cute.make_fragment(
            cute.slice_(tSR_sX.shape, (None, None, None, 0, 0, 0)), dtype
        )
        return tiled_s2r_x, tSR_sX, tSR_rX

    def epilog_tmem_load_and_partition_acc(self, local_tidx, tIntra, smem_y):
        copy_atom_t2r_inter2_intra2 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(4), tcgen05.Pack.NONE),
            self.acc_dtype,
        )
        return self.make_tmem_load_and_partition(
            copy_atom_t2r_inter2_intra2,
            tIntra,
            (None, None, 0, 0, 0),
            local_tidx,
            smem_y[None, None, 0],
        )


def run(
    gbehcdln: Tuple[int, int, int, int, int, int, int, int],
    io_dtype: Type[cutlass.Numeric],
    cumsum_delta_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    fuse_scale_d: str,
    tolerance: float,
    print_rtol_stats: bool,
    ref_lower_precision: bool,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    has_d = fuse_scale_d != "none"
    d_has_hdim = fuse_scale_d == "vector"

    print(f"Running B100 Mamba2 SSD with:")
    print(f"GBEHCDLN: {gbehcdln}")
    print(
        f"Input/Output dtype: {io_dtype}, Intermediate delta dtype: {cumsum_delta_dtype}, Acc dtype: {acc_dtype}"
    )
    print(
        f"Has D (True means fuse Y+=X*D): {has_d}, D has Hdim (True means D.shape DxEH, False means 1xEH): {d_has_hdim}"
    )
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Unpack parameters
    G, B, E, H, C, D, L, N = gbehcdln
    EH = E * H

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Match same seed in ssd_reference.py for reference check
    torch.manual_seed(42)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        shape, permute_order, dtype, dt_or_a=0, dynamic_modes=None, ref_tensor=None
    ):
        # Build fp32 reference torch tensor
        if ref_tensor is None:
            ref_tensor = (
                torch.empty(*shape, dtype=torch.float32)
                # .random_(-1, 1)
                .normal_(0, 0.5)
                # .uniform_(-1,1)
                .permute(permute_order)
            )
            if dt_or_a == 1:  # dt:
                ref_tensor = F.softplus(ref_tensor - 4)
            elif dt_or_a == 2:  # A:
                ref_tensor = -torch.exp(ref_tensor)

        # Build torch_dtype torch tensor
        torch_dtype = cutlass_torch.dtype(dtype)

        dst_tensor = ref_tensor.to(torch_dtype).cuda()
        cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
        for mode in dynamic_modes:
            cute_tensor = cute_tensor.mark_compact_shape_dynamic(
                mode=mode, stride_order=dst_tensor.dim_order()
            )

        return ref_tensor, cute_tensor, dst_tensor

    # INPUT tensors
    # x: (D, L, C, EH, B):(C*L, 1, L, D*C*L, EH*D*C*L)
    x_ref, x_tensor, x_torch = create_and_permute_tensor(
        [B, EH, D, C, L], [2, 4, 3, 1, 0], io_dtype, dynamic_modes=[2, 3, 4]
    )
    # delta/delta_a/cumsum_delta: (L, C, EH, B):(1, L, C*L, EH*C*L)
    delta_ref, delta_tensor, delta_torch = create_and_permute_tensor(
        [B, EH, C, L], [3, 2, 1, 0], io_dtype, dt_or_a=1, dynamic_modes=[1, 2, 3]
    )
    # a: (EH):(1)
    a_ref, a_tensor, a_torch = create_and_permute_tensor(
        [EH], [0], io_dtype, dt_or_a=2, dynamic_modes=[0]
    )

    if has_d:
        # d: (D, EH):(1, D) or (1, EH):(0, 1)
        d_ref, d_tensor, d_torch = create_and_permute_tensor(
            [EH, D if d_has_hdim else 1], [1, 0], io_dtype, dynamic_modes=[1]
        )
    else:
        d_ref = None
        d_tensor = None

    # b/c: (L, N, C, G, B):(1, C*L, L, N*C*L, G*N*C*L)
    b_ref, b_tensor, b_torch = create_and_permute_tensor(
        [B, G, N, C, L], [4, 2, 3, 1, 0], io_dtype, dynamic_modes=[2, 3, 4]
    )
    c_ref, c_tensor, c_torch = create_and_permute_tensor(
        [B, G, N, C, L], [4, 2, 3, 1, 0], io_dtype, dynamic_modes=[2, 3, 4]
    )

    # OUTPUT tensors
    # y: (L, D, C, EH, B):(1, C*L, L, D*C*L, EH*D*C*L)
    y_ref, y_tensor, y_torch = create_and_permute_tensor(
        [B, EH, D, C, L], [4, 2, 3, 1, 0], io_dtype, dynamic_modes=[2, 3, 4]
    )
    # fstate: (D, N, EH, B):(N, 1, D*N, EH*D*N)
    fstate_ref, fstate_tensor, fstate_torch = create_and_permute_tensor(
        [B, EH, D, N], [2, 3, 1, 0], io_dtype, dynamic_modes=[2, 3]
    )

    # Call pytorch reference on cpu
    if not ref_lower_precision:
        ssd_reference_fp32_all(
            x_ref,
            a_ref,
            delta_ref,
            b_ref,
            c_ref,
            y_ref,
            fstate_ref,
            d_ref,
            has_d,
            d_has_hdim,
        )
    else:
        ssd_reference_lowprecision_intermediates(
            x_ref,
            a_ref,
            delta_ref,
            b_ref,
            c_ref,
            y_ref,
            fstate_ref,
            cutlass_torch.dtype(io_dtype),
            d_ref,
            has_d,
            d_has_hdim,
        )

    # Compute cumsum with pytorch on cpu
    delta_a_ref = delta_ref * a_ref.view(1, 1, -1, 1)
    cumsum_delta_ref = torch.empty([B, EH, C, L], dtype=torch.float32).permute(
        [3, 2, 1, 0]
    )
    cumsum_delta_ref.copy_(torch.cumsum(delta_a_ref, dim=0).permute([0, 1, 2, 3]))
    # Copy cumsum_delta_ref to cumsum_delta_tensor
    (
        cumsum_delta_ref,
        cumsum_delta_tensor,
        cumsum_delta_torch,
    ) = create_and_permute_tensor(
        [B, EH, C, L],
        [3, 2, 1, 0],
        cumsum_delta_dtype,
        ref_tensor=cumsum_delta_ref,
        dynamic_modes=[1, 2, 3],
    )

    # Call fused ssd kernel
    ssd = SSDKernel(
        io_dtype,
        cumsum_delta_dtype,
        acc_dtype,
        L,
        D,
        N,
        has_d,
        d_has_hdim,
    )

    # Compute max active clusters on current device
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(1)

    stream = cutlass.cuda.default_stream()
    # Compile ssd kernel
    compiled_ssd = cute.compile(
        ssd,
        x_tensor,
        cumsum_delta_tensor,
        delta_tensor,
        b_tensor,
        c_tensor,
        y_tensor,
        fstate_tensor,
        d_tensor,
        max_active_clusters,
        stream,
    )

    # Launch compiled ssd kernel for reference check
    if not skip_ref_check:
        compiled_ssd(
            x_tensor,
            cumsum_delta_tensor,
            delta_tensor,
            b_tensor,
            c_tensor,
            y_tensor,
            fstate_tensor,
            d_tensor,
            stream,
        )

        # Reference check
        if print_rtol_stats:
            print("\nY's Relative diffs:")
            analyze_relative_diffs(
                y_torch.cpu(), y_ref.to(cutlass_torch.dtype(io_dtype))
            )
            print("\nFstate's Relative diffs:")
            analyze_relative_diffs(
                fstate_torch.cpu(), fstate_ref.to(cutlass_torch.dtype(io_dtype))
            )
        torch.testing.assert_close(
            y_torch.cpu(),
            y_ref.to(cutlass_torch.dtype(io_dtype)),
            atol=tolerance,
            rtol=1e-02,
        )
        torch.testing.assert_close(
            fstate_torch.cpu(),
            fstate_ref.to(cutlass_torch.dtype(io_dtype)),
            atol=tolerance,
            rtol=1e-05,
        )

    def generate_tensors():
        # Reuse existing CPU reference tensors and create new GPU tensors from them
        _, x_tensor_new, _ = create_and_permute_tensor(
            [B, EH, D, C, L],
            [2, 4, 3, 1, 0],
            io_dtype,
            ref_tensor=x_ref,
            dynamic_modes=[2, 3, 4],
        )
        _, cumsum_delta_tensor_new, _ = create_and_permute_tensor(
            [B, EH, C, L],
            [3, 2, 1, 0],
            cumsum_delta_dtype,
            ref_tensor=cumsum_delta_ref,
            dynamic_modes=[1, 2, 3],
        )
        _, delta_tensor_new, _ = create_and_permute_tensor(
            [B, EH, C, L],
            [3, 2, 1, 0],
            io_dtype,
            ref_tensor=delta_ref,
            dynamic_modes=[1, 2, 3],
        )
        _, b_tensor_new, _ = create_and_permute_tensor(
            [B, G, N, C, L],
            [4, 2, 3, 1, 0],
            io_dtype,
            ref_tensor=b_ref,
            dynamic_modes=[2, 3, 4],
        )
        _, c_tensor_new, _ = create_and_permute_tensor(
            [B, G, N, C, L],
            [4, 2, 3, 1, 0],
            io_dtype,
            ref_tensor=c_ref,
            dynamic_modes=[2, 3, 4],
        )
        _, y_tensor_new, _ = create_and_permute_tensor(
            [B, EH, D, C, L],
            [4, 2, 3, 1, 0],
            io_dtype,
            ref_tensor=y_ref,
            dynamic_modes=[2, 3, 4],
        )
        _, fstate_tensor_new, _ = create_and_permute_tensor(
            [B, EH, D, N],
            [2, 3, 1, 0],
            io_dtype,
            ref_tensor=fstate_ref,
            dynamic_modes=[2, 3],
        )

        if has_d:
            _, d_tensor_new, _ = create_and_permute_tensor(
                [EH, D if d_has_hdim else 1],
                [1, 0],
                io_dtype,
                ref_tensor=d_ref,
                dynamic_modes=[1],
            )
        else:
            d_tensor_new = d_tensor

        return testing.JitArguments(
            x_tensor_new,
            cumsum_delta_tensor_new,
            delta_tensor_new,
            b_tensor_new,
            c_tensor_new,
            y_tensor_new,
            fstate_tensor_new,
            d_tensor_new,
            stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            x_torch.numel() * x_torch.element_size()
            + cumsum_delta_torch.numel() * cumsum_delta_torch.element_size()
            + delta_torch.numel() * delta_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
            + y_torch.numel() * y_torch.element_size()
            + fstate_torch.numel() * fstate_torch.element_size()
        )
        if has_d:
            one_workspace_bytes += d_torch.numel() * d_torch.element_size()

        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_ssd,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> List[int]:
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of MxNxKxL GEMM on Blackwell."
    )

    parser.add_argument(
        "--gbehcdln",
        type=parse_comma_separated_ints,
        default=[2, 4, 2, 40, 32, 64, 128, 128],
        # default=[2, 3, 2, 2, 8, 64, 128, 128],
        # default=[1, 2, 1, 4, 8, 64, 128, 128],
        help="gbehcdln dimensions (comma-separated)",
    )
    parser.add_argument("--io_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument(
        "--cumsum_delta_dtype", type=cutlass.dtype, default=cutlass.Float32
    )
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32)
    parser.add_argument(
        "--fuse_scale_d",
        type=str,
        choices=["none", "scalar", "vector"],
        default="vector",
        help="Fuse scale type: none (no Y+=X*D fusion), scalar (Y+=X*D fusion with D.shape=1xEH), or vector (Y+=X*D fusion with D.shape=DxEH)",
    )
    parser.add_argument(
        "--ref_lower_precision",
        action="store_true",
        default=True,
        help="Use lower precision for reference check",
    )
    parser.add_argument(
        "--no-ref_lower_precision",
        action="store_false",
        dest="ref_lower_precision",
        default=False,
        help="Disable lower precision for reference check",
    )
    parser.add_argument(
        "--tolerance", type=float, default=5e-02, help="Tolerance for validation"
    )
    parser.add_argument(
        "--print_rtol_stats",
        action="store_true",
        default=True,
        help="Enable print rtol stats",
    )
    parser.add_argument(
        "--no-print_rtol_stats",
        action="store_false",
        dest="print_rtol_stats",
        default=False,
        help="Disable print rtol stats",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.gbehcdln) != 8:
        parser.error("--gbehcdln must contain exactly 8 values")

    run(
        args.gbehcdln,
        args.io_dtype,
        args.cumsum_delta_dtype,
        args.acc_dtype,
        args.fuse_scale_d,
        args.tolerance,
        args.print_rtol_stats,
        args.ref_lower_precision,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    print("PASS")
