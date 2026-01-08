# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

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

import enum
import math
import operator
from functools import partial
from typing import Callable, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.cute_dsl_utils import ParamsBase

# return PipelineStateWAdvance instead of PipelineState
from quack.pipeline import PipelineTmaCpAsync, make_pipeline_state
from quack.reduce import warp_reduce
from quack.sm90_utils import partition_for_epilogue
from quack.tensormap_manager import TensorMapManagerSm90
from quack.tile_scheduler import (
    RasterOrderOption,
    TileSchedulerArguments,
    VarlenMTileSchedulerArguments,
)
from quack.utils import make_acc_tensor_mn_view, sm90_get_smem_load_op

from .tile_scheduler import SonicMoETileScheduler, SonicMoEVarlenMTileScheduler


class NamedBarrierGemm(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    EpilogueLoad = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()
    Prolog = enum.auto()


class HopperWgmma_MoE_kernel:
    def __init__(
        self,
        E: int,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        compute_dz_and_partial_ds_and_y1s: bool = False,
        compute_weight_gradient: bool = False,
        compute_relu: bool = False,
        compute_silu: bool = False,
        compute_gelu: bool = False,
        compute_relu_sq: bool = False,
        compute_swiglu: bool = False,
        compute_reglu: bool = False,
        compute_geglu: bool = False,
        is_normal_act: bool = False,
        is_glu: bool = False,
        is_A_gather: bool = False,
        is_scatter_idx_prefetched: bool = False,
        epi_tile_size: int = 32,
        initial_d_epi_stage: int = 4,
        index_dtype: Type[cutlass.Numeric] = cutlass.Int32,
        prefetch_idx_store_to_smem: int = 2048,
        inference_mode: bool = False,
        L2_group_size: int = 8,
        raster_order: RasterOrderOption = RasterOrderOption.Heuristic,
    ):
        self.epi_tile_size = epi_tile_size
        self.initial_d_epi_stage = initial_d_epi_stage

        self.is_A_gather = is_A_gather
        self.is_scatter_idx_prefetched = is_scatter_idx_prefetched

        self.compute_swiglu = compute_swiglu
        self.compute_geglu = compute_geglu
        self.compute_reglu = compute_reglu

        self.compute_relu = compute_relu
        self.compute_silu = compute_silu
        self.compute_gelu = compute_gelu
        self.compute_relu_sq = compute_relu_sq

        self.is_glu = is_glu or (compute_swiglu or compute_geglu or compute_reglu)
        self.is_normal_act = is_normal_act or (
            compute_gelu or compute_relu_sq or compute_relu or compute_silu
        )

        self.compute_dz_and_partial_ds_and_y1s = compute_dz_and_partial_ds_and_y1s
        self.compute_weight_gradient = compute_weight_gradient

        self.need_adhoc_epilogue_store = (
            self.is_glu or self.is_normal_act or compute_dz_and_partial_ds_and_y1s
        )
        self.need_epilogue_load = compute_dz_and_partial_ds_and_y1s

        self.L2_group_size = L2_group_size
        self.raster_order = raster_order

        self.E = E
        self.acc_dtype = acc_dtype
        assert self.acc_dtype == cutlass.Float32
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"

        self.cluster_shape_mnk = cluster_shape_mnk
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        tile_M, tile_N = tile_shape_mnk[0], tile_shape_mnk[1]
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256)
                    or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(
                    f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}"
                )
        if not self.tile_shape_mnk[2] % 16 == 0:
            raise ValueError("CTA tile shape K must be divisible by 16")

        self.tile_M, self.tile_N, self.tile_K = tile_shape_mnk

        if not self.pingpong:
            if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
                atom_layout_m, atom_layout_n = 1, 2
            elif tile_M == 192:
                if tile_N <= 128:
                    atom_layout_m, atom_layout_n = 3, 1
                else:
                    atom_layout_m, atom_layout_n = 1, 2
            else:
                atom_layout_m = (
                    tile_shape_mnk[0] // 64 if tile_shape_mnk[0] < 256 else 2
                )
                atom_layout_n = 1
            assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        else:
            atom_layout_m, atom_layout_n = 1, 1
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)

        if is_A_gather:
            assert self.cluster_shape_mnk[1] == 1
            self.num_mcast_ctas_a = None
            self.is_a_mcast = False
        else:
            self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
            self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (
            1 if not self.pingpong else 2
        )
        if self.pingpong:
            assert self.mma_warp_groups == 2
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            self.mma_warp_groups + 1
        ) * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.num_mma_threads = (
            self.mma_warp_groups if not self.pingpong else 1
        ) * self.num_threads_per_warp_group
        self.num_epi_threads = (
            self.mma_warp_groups if not self.pingpong else 1
        ) * self.num_threads_per_warp_group
        self.tma_warp_id = self.mma_warp_groups * 4
        self.universal_copy_bits = 128
        # assumed BF16 now
        self.num_load_A_threads = (
            min(
                self.tile_M * self.tile_K // 8,
                self.threads_per_cta - self.tma_warp_id * cute.arch.WARP_SIZE,
            )
            if is_A_gather
            else 0
        )
        if self.compute_weight_gradient and self.is_A_gather:
            if tile_M == 192:  # contiguous dimension
                self.num_load_A_threads = 3 * 32
            assert tile_M in [64, 128, 192, 256]

        self.num_epi_load_threads = 0
        if self.need_epilogue_load:
            # 3 warps to load A, 1 warp to load C, (and 1 warp to load S)
            self.num_load_A_threads = 4 * cute.arch.WARP_SIZE
            self.num_epi_load_threads = self.num_epi_threads

        regs_per_thread = math.prod(self.tile_shape_mnk[:2]) // self.num_mma_threads
        heavy_register_pressure = regs_per_thread >= 208

        if not is_A_gather:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 32, 160
            else:
                heavy_register_pressure = regs_per_thread >= 208
                self.num_regs_load, self.num_regs_mma = (
                    (40, 232) if not heavy_register_pressure else (24, 240)
                )
        else:
            if self.mma_warp_groups == 3:
                self.num_regs_load, self.num_regs_mma = 56, 152
            else:
                self.num_regs_load, self.num_regs_mma = (56, 224)

        self.ab_stage = None
        self.c_epi_stage = None
        self.d_epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.d_epi_smem_layout_staged = None
        self.d_epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.tensormap_update_mode = cutlass.utils.TensorMapUpdateMode.SMEM
        self.bytes_per_tensormap = 128
        self.tensor_memory_management_bytes = 12

        self.inference_mode = inference_mode

        if is_A_gather:
            if self.need_adhoc_epilogue_store:
                if self.inference_mode:
                    self.num_tensormaps = self.mma_warp_groups if self.pingpong else 1
                else:
                    self.num_tensormaps = (
                        2 * self.mma_warp_groups if self.pingpong else 2
                    )
            else:
                self.num_tensormaps = 1 * self.mma_warp_groups if self.pingpong else 1
        else:
            if self.need_adhoc_epilogue_store:
                if self.inference_mode:
                    self.num_tensormaps = (
                        2 * self.mma_warp_groups + 1 if self.pingpong else 3
                    )
                else:
                    self.num_tensormaps = (
                        self.mma_warp_groups + 1 if self.pingpong else 2
                    )
            else:
                self.num_tensormaps = (
                    1 * self.mma_warp_groups + 1 if self.pingpong else 2
                )

        if self.need_epilogue_load:
            self.num_tensormaps += 2 * self.mma_warp_groups if self.pingpong else 1

        if self.compute_weight_gradient:
            if self.is_A_gather:
                self.num_tensormaps = 1
                self.prefetch_token_idx_size = prefetch_idx_store_to_smem
                self.index_dtype = index_dtype

                assert (
                    self.prefetch_token_idx_size % self.tile_K == 0
                    and self.prefetch_token_idx_size >= self.tile_K
                    and self.prefetch_token_idx_size % self.num_load_A_threads == 0
                )
            else:
                self.num_tensormaps = 2
                self.prefetch_token_idx_size = 0
                self.index_dtype = None
        else:
            self.prefetch_token_idx_size = 0
            self.index_dtype = None

        self.tensormap_bytes_total = self.num_tensormaps * self.bytes_per_tensormap

    def _setup_attributes(self):
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.d_epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )
        self.c_epi_tile = self.d_epi_tile
        if const_expr(self.compute_dz_and_partial_ds_and_y1s):
            self.y_epi_tile = self.d_epi_tile
        elif const_expr(self.is_glu):
            self.y_epi_tile = (self.d_epi_tile[0], self.d_epi_tile[1] // 2)
        elif const_expr(self.is_normal_act):
            self.y_epi_tile = self.d_epi_tile
        else:
            self.y_epi_tile = None

        if const_expr(self.use_bias):
            self.bias_epi_tile = self.d_epi_tile
            self.initial_d_epi_stage -= 1  # for safety
        else:
            self.bias_epi_tile = None

        # Compute stage before compute smem layout
        self.ab_stage, self.c_epi_stage, self.d_epi_stage, self.y_epi_stage = (
            self._compute_stages(
                self.tile_shape_mnk,
                self.initial_d_epi_stage,
                # epi_smem will reuse smem ab if not persistent.
                self.d_epi_tile,
                self.c_epi_tile,
                self.y_epi_tile,
                self.a_dtype,
                self.b_dtype,
                self.d_dtype,
                self.c_dtype,
                self.y_dtype,
                self.smem_capacity,
                self.occupancy,
                # epi_smem will reuse smem ab if not persistent.
                overlap_sD_sA=not self.is_persistent,
            )
        )

        if const_expr((not self.inference_mode) and self.need_adhoc_epilogue_store):
            assert self.d_epi_stage == self.y_epi_stage

        self.sched_stage = 2 if self.pingpong else 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_epi_smem_layout_staged,
            self.bias_epi_smem_layout_staged,
            self.d_epi_smem_layout_staged,
            self.y_epi_smem_layout_staged,
            self.s_epi_smem_layout_staged,
            self.prefetch_AIdx_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.c_epi_tile,
            self.bias_epi_tile,
            self.d_epi_tile,
            self.y_epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.prefetch_token_idx_size,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.bias_dtype,
            self.bias_layout,
            self.d_dtype,
            self.d_layout,
            self.y_dtype,
            self.y_layout,
            self.s_dtype,
            self.c_epi_stage,
            self.d_epi_stage,
            self.y_epi_stage,
        )

    @dsl_user_op
    def tanh(self, a: float | Float32, *, loc=None, ip=None) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Float32(a).ir_value(loc=loc, ip=ip)],
                "tanh.approx.f32 $0, $1;",
                "=f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def fma(
        self,
        a: float | Float32,
        b: float | Float32,
        c: float | Float32,
        *,
        loc=None,
        ip=None,
    ) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [
                    Float32(a).ir_value(loc=loc, ip=ip),
                    Float32(b).ir_value(loc=loc, ip=ip),
                    Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def silu(self, a: float | Float32, *, loc=None, ip=None) -> Float32:
        """
        silu(a) = a * sigmoid(a) = a * (1 + tanh(a / 2)) / 2 = (0.5 * a) * tanh(0.5 * a) + (0.5 * a)
        This compiles down to 3 SASS instructions: FMUL to get 0.5 * a, MUFU.TANH, and FFMA.
        """
        # return a / (1.0 + cute.arch.exp2(-a * math.log2(math.e)))
        a_half = 0.5 * a
        # return a_half * self.tanh(a_half) + a_half
        return self.fma(a_half, self.tanh(a_half), a_half)

    @dsl_user_op
    def relu(self, a: float | Float32, *, loc=None, ip=None) -> Float32:
        return cute.arch.fmax(a, 0.0)

    @dsl_user_op
    def relu_sq(self, a: float | Float32, *, loc=None, ip=None) -> Float32:
        return a * cute.arch.fmax(a, 0.0)

    @dsl_user_op
    def gelu(self, a: Float32, *, loc=None, ip=None) -> Float32:
        # gelu(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x^3)))
        c0 = const_expr(math.sqrt(2 / math.pi))  # √(2/π)
        c1 = 0.044715
        a2 = a * a
        # inner = √(2/π) * (x + 0.044715*x^3)
        inner = c0 * self.fma(c1, a2 * a, a)
        return 0.5 * a * self.fma(1.0, self.tanh(inner), 1.0)

    @dsl_user_op
    def elem_pointer(
        self, x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
    ) -> cute.Pointer:
        return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

    @dsl_user_op
    def min_i32(self, a: int | Int32, b: int | Int32, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),  # return type
                [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
                "min.s32 $0, $1, $2;",
                "=r,r,r",  # output, input constraints
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def prefetch_gather_idx_for_A_when_vary_M(
        self,
        mAIdx: cute.Tensor,
        M_offset: int,
        M_boundary: int,
        copy_elems_per_thr_load: int,  # m n k l
    ) -> cute.Tensor:
        assert const_expr(not self.compute_weight_gradient)
        M, K = self.tile_M, self.tile_K

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        stride_1_tile, other_tile = K, M

        threads_per_stride_1_dim = const_expr(stride_1_tile // copy_elems_per_thr_load)
        num_other_dim_per_load = const_expr(
            self.num_load_A_threads // threads_per_stride_1_dim
        )

        tmAIdx = cute.make_fragment((num_other_dim_per_load,), dtype=mAIdx.element_type)

        for i in cutlass.range_constexpr(num_other_dim_per_load):
            other_dim_offset = (
                const_expr(i * num_other_dim_per_load)
                + tidx // threads_per_stride_1_dim
            )

            if other_dim_offset < M_boundary:
                M_i = M_offset + other_dim_offset
                tmAIdx[i] = mAIdx[M_i]

        return tmAIdx

    @cute.jit
    def prefetch_scatter_idx_for_D_when_vary_M(
        self,
        mD: cute.Tensor,  # unused, kept for symmetry
        mDIdx: cute.Tensor,
        D_r2g_thr_copy,
        tcDgcD_flat_partition: cute.Tensor,
        epi_tile_layout: cute.Layout,
        epi_tile_num: int,
        copy_elems_per_thr_load: int,  # unused here, but fine to keep
        tile_coord_mnkl: Tuple[int, int, None, int],  # (block_M, block_N, _, batch)
        MIdx_cur_group: int,
        MIdx_next_group: int,
    ) -> cute.Tensor:
        # Same base M offset as store_D_scatter
        block_M, block_N = tile_coord_mnkl[0], tile_coord_mnkl[1]
        M_offset = block_M * const_expr(self.tile_M) + MIdx_cur_group

        tDcD0 = D_r2g_thr_copy.partition_D(
            tcDgcD_flat_partition[None, None, *epi_tile_layout.get_hier_coord(0)]
        )
        num_load_per_thread = const_expr(cute.size(tDcD0, mode=[1]))

        tmDIdx = cute.make_fragment(
            (epi_tile_num * num_load_per_thread,), dtype=mDIdx.element_type
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            tDcD_slice = D_r2g_thr_copy.partition_D(
                tcDgcD_flat_partition[
                    None, None, *epi_tile_layout.get_hier_coord(epi_idx)
                ]
            )

            for i in cutlass.range_constexpr(num_load_per_thread):
                # Same coordinate source as in store_D_scatter
                MIdx_in_tile, _ = tDcD_slice[0, i, 0]
                MIdx = M_offset + MIdx_in_tile

                if MIdx < MIdx_next_group:
                    tmDIdx[epi_idx * num_load_per_thread + i] = mDIdx[MIdx]

        return tmDIdx

    @cute.jit
    def prefetch_gather_idx_for_A_when_vary_K(
        self,
        mAIdx: cute.Tensor,
        sAIdx: cute.Tensor,
        token_group_size: int,
        K_offset: int,
    ) -> cute.Tensor:
        assert const_expr(self.compute_weight_gradient and self.is_A_gather)

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        # !!! cannot be removed for correctness !!!
        cute.arch.barrier(
            barrier_id=NamedBarrierGemm.Prolog,
            number_of_threads=self.num_load_A_threads,
        )

        for i in cutlass.range_constexpr(
            cute.ceil_div(self.prefetch_token_idx_size, self.num_load_A_threads)
        ):
            offset = const_expr(i * self.num_load_A_threads) + tidx
            kidx = K_offset + offset

            if kidx < token_group_size:
                sAIdx[offset] = mAIdx[kidx]

        # !!! cannot be removed for correctness !!!
        cute.arch.barrier(
            barrier_id=NamedBarrierGemm.Prolog,
            number_of_threads=self.num_load_A_threads,
        )

    @cute.jit
    def load_A_gather(
        self,
        mA: cute.Tensor,
        tmAIdx: Optional[cute.Tensor],
        sAIdx_prefetch: cute.Tensor,
        M_offset: cutlass.Int32,
        tAsA: cute.Tensor,
        tApA: cute.Tensor,
        A_g2s_thr_copy,
        K_offset: cutlass.Int32,
        token_group_size: cutlass.Int32,
        copy_elems_per_thr_load: cutlass.Int32,
    ):
        M, K = self.tile_M, self.tile_K

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx - self.tma_warp_id * cute.arch.WARP_SIZE

        if const_expr(self.compute_weight_gradient):
            stride_1_tile, other_tile = M, K
        else:
            stride_1_tile, other_tile = K, M

        threads_per_stride_1_dim = const_expr(stride_1_tile // copy_elems_per_thr_load)
        num_other_dim_per_load = const_expr(
            self.num_load_A_threads // threads_per_stride_1_dim
        )

        K_offset_mod_smem_load = K_offset % const_expr(self.prefetch_token_idx_size)
        for i in cutlass.range_constexpr(
            cute.ceil_div(other_tile, num_other_dim_per_load)
        ):
            stride_1_dim_offset = (
                tidx % threads_per_stride_1_dim
            ) * copy_elems_per_thr_load
            other_dim_offset = (
                const_expr(i * num_other_dim_per_load)
                + tidx // threads_per_stride_1_dim
            )

            if const_expr(self.compute_weight_gradient):
                MIdx = M_offset + stride_1_dim_offset
                KIdx_global = K_offset + other_dim_offset

                if KIdx_global < token_group_size and MIdx < mA.shape[0]:
                    KIdx = sAIdx_prefetch[K_offset_mod_smem_load + other_dim_offset]
                    # KIdx = mAIdx_mk[K_offset + other_dim_offset]
                    tPrAptr = self.elem_pointer(mA, (MIdx, KIdx)).align(
                        self.universal_copy_bits // copy_elems_per_thr_load
                    )
                    mA_cur_copy = cute.make_tensor(
                        tPrAptr, ((copy_elems_per_thr_load, 1), 1)
                    )

                    cute.copy(A_g2s_thr_copy, mA_cur_copy, tAsA[None, None, i])
                else:
                    tAsA[None, None, i].fill(0.0)

            else:
                MIdx = tmAIdx[i]
                KIdx = K_offset + stride_1_dim_offset

                tPrAptr = self.elem_pointer(mA, (MIdx, KIdx)).align(
                    self.universal_copy_bits // copy_elems_per_thr_load
                )
                mA_cur_copy = cute.make_tensor(
                    tPrAptr, ((copy_elems_per_thr_load, 1), 1)
                )
                cute.copy(
                    A_g2s_thr_copy,
                    mA_cur_copy,
                    tAsA[None, i, None],
                    pred=tApA[None, i, None],
                )

    @cute.jit
    def store_D_scatter(
        self,
        mD: cute.Tensor,  # m, n, k, l
        mDIdx: cute.Tensor,
        tmDIdx: cute.Tensor,  # assume to have same size as mD
        tDrD: cute.Tensor,
        tDcD_slice: cute.Tensor,  # ((8, 1), 16, 1)
        D_r2g_thr_copy,
        epi_idx: cutlass.Int32,
        copy_elems_per_thr_load: cutlass.Int32,
        tile_coord_mnkl: Tuple[int, int, None, int],  # m n k l
        MIdx_cur_group: int,
        MIdx_next_group: int,
    ):
        block_M, block_N = tile_coord_mnkl[0], tile_coord_mnkl[1]

        M_offset = block_M * const_expr(self.tile_M) + MIdx_cur_group
        N_offset = block_N * const_expr(self.tile_N)

        num_load_per_thread = const_expr(cute.size(tDcD_slice, mode=[1]))
        for i in cutlass.range_constexpr(num_load_per_thread):
            MIdx_in_epi_tile, NIdx_in_epi_tile = tDcD_slice[0, i, 0]

            MIdx = M_offset + MIdx_in_epi_tile
            NIdx = N_offset + NIdx_in_epi_tile

            if MIdx < MIdx_next_group and NIdx < mD.shape[1]:
                if const_expr(self.is_scatter_idx_prefetched):
                    SIdx = tmDIdx[i + epi_idx * num_load_per_thread]
                else:
                    SIdx = mDIdx[MIdx]  # equivalent
                tPDptr = self.elem_pointer(mD, (SIdx, NIdx)).align(
                    self.universal_copy_bits // copy_elems_per_thr_load
                )

                mD_cur_copy = cute.make_tensor(
                    tPDptr, ((copy_elems_per_thr_load, 1), 1)
                )

                cute.copy(
                    D_r2g_thr_copy,
                    tDrD[None, i, None],
                    mD_cur_copy,
                )

    @cute.jit
    def fetch_scattered_S(
        self,
        tidx: int,
        mS: cute.Tensor,
        mS_scatter_idx: cute.Tensor,
        sS_staged: cute.Tensor,
        tile_coord_mnkl: Tuple[int, int, None, int],  # m n k l
        MIdx_cur_group: int,
        MIdx_next_group: int,
    ):
        block_M = tile_coord_mnkl[0]
        M = self.tile_M

        M_s = block_M * M + MIdx_cur_group

        for i in cutlass.range_constexpr(cute.ceil_div(M, self.num_epi_threads)):
            sS_offset = const_expr(i * self.num_epi_threads) + tidx
            M_i = M_s + sS_offset

            if M_i < MIdx_next_group and sS_offset < M:
                sIdx = mS_scatter_idx[M_i]
                sS_staged[sS_offset] = self.s_dtype(mS[sIdx])

    @dsl_user_op
    def prmt(
        self, a: int | Int32, b: int | Int32, c: int | Int32, *, loc=None, ip=None
    ) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [
                    Int32(a).ir_value(loc=loc, ip=ip),
                    Int32(b).ir_value(loc=loc, ip=ip),
                    Int32(c).ir_value(loc=loc, ip=ip),
                ],
                "prmt.b32 $0, $1, $2, $3;",
                "=r,r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def pack2x16_as_f32(
        self,
        a: Union[cutlass.BFloat16, cutlass.Float16],
        b: Union[cutlass.BFloat16, cutlass.Float16],
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Float32:
        vec_src_type = T.bf16() if a.dtype == cutlass.BFloat16 else T.f16()

        vec_f16x2 = vector.from_elements(
            T.vector(2, vec_src_type), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
        )
        vec_f32x1 = vector.bitcast(T.vector(1, T.f32()), vec_f16x2)
        return cutlass.Float32(
            vector.extract(
                vec_f32x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip
            )
        )

    @dsl_user_op
    def unpack2x16_as_2xf32(
        self, a: Float32, dtype: cutlass.Numeric, *, loc=None, ip=None
    ) -> Tuple[cutlass.Float32, cutlass.Float32]:

        vec_dst_type = T.bf16() if dtype == cutlass.BFloat16 else T.f16()

        vec_f32x1 = vector.from_elements(
            T.vector(1, T.f32()), (a.ir_value(),), loc=loc, ip=ip
        )
        vec_f16x2 = vector.bitcast(T.vector(2, vec_dst_type), vec_f32x1)
        res0 = Float32(
            vector.extract(
                vec_f16x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip
            )
        )
        res1 = Float32(
            vector.extract(
                vec_f16x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip
            )
        )
        return res0, res1

    @cute.jit
    def permute_gated_Cregs_b16(self, t: cute.Tensor) -> None:
        assert t.element_type.width == 16
        assert (
            cute.size(t.shape) % 4 == 0
        ), "Tensor size must be a multiple of 4 for b16 permutation"
        t_u32 = cute.recast_tensor(t, Int32)

        quad_idx = cute.arch.lane_idx() % 4
        lane_03 = quad_idx == 0 or quad_idx == 3
        selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
        selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
        # upper_map = [0, 3, 1, 2]
        # lower_map = [1, 2, 0, 3]
        # upper_idx = upper_map[quad_idx]
        # indexing isn't supported so we have to do arithmetic
        upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
        lower_idx = upper_idx ^ 1

        # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
        width = 4
        mask = cute.arch.WARP_SIZE - width
        clamp = cute.arch.WARP_SIZE - 1
        mask_and_clamp = const_expr(mask << 8 | clamp)

        for i in cutlass.range_constexpr(cute.size(t_u32.shape) // 2):
            upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
            upper0 = upper if lane_03 else lower
            lower0 = lower if lane_03 else upper
            upper0 = cute.arch.shuffle_sync(
                upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp
            )
            lower0 = cute.arch.shuffle_sync(
                lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp
            )
            t_u32[i * 2 + 0] = self.prmt(upper0, lower0, selector_upper)
            t_u32[i * 2 + 1] = self.prmt(upper0, lower0, selector_lower)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        mD: cute.Tensor,
        mY: Optional[cute.Tensor],
        mS: Optional[cute.Tensor],
        mDS_partial: Optional[cute.Tensor],
        mMoffset: cute.Tensor,
        mAIdx: Optional[cute.Tensor],
        mDIdx: Optional[cute.Tensor],
        mS_scatter_idx: Optional[cute.Tensor],
        mA_tensormap: Optional[cute.Tensor],
        mB_tensormap: Optional[cute.Tensor],
        mC_tensormap: Optional[cute.Tensor],
        mD_tensormap: Optional[cute.Tensor],
        mY_tensormap: Optional[cute.Tensor],
        mTileCount_semaphore: Optional[cute.Pointer],
        mBatchIdx_schedule_order: Optional[cute.Tensor],
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type if mC is not None else None
        self.d_dtype = mD.element_type
        self.s_dtype = cutlass.Float32

        self.a_layout = utils.LayoutEnum.from_tensor(mA)
        self.b_layout = utils.LayoutEnum.from_tensor(mB)
        self.c_layout = (
            cutlass.utils.LayoutEnum.from_tensor(mC) if mC is not None else None
        )
        self.d_layout = utils.LayoutEnum.from_tensor(mD)

        self.use_bias = const_expr(mBias is not None)
        if const_expr(self.use_bias):
            assert (
                not self.compute_weight_gradient
            ), "Bias addition is not supported when computing weight gradients"
            self.bias_dtype = mBias.element_type
            self.bias_layout = utils.LayoutEnum.from_tensor(mBias)
        else:
            self.bias_dtype = None
            self.bias_layout = None

        if const_expr(self.need_adhoc_epilogue_store):
            self.y_dtype = mY.element_type
            self.y_layout = utils.LayoutEnum.from_tensor(mY)
        else:
            self.y_layout = self.y_dtype = None

        if const_expr(mC is not None):
            assert self.acc_dtype == cutlass.Float32
            assert (
                self.need_epilogue_load
            ), "Set need_epilogue_load = True or set mC = None"

        if const_expr(mS is not None):
            assert (
                self.compute_dz_and_partial_ds_and_y1s
            ), "Set compute_dz_and_partial_ds = True or set mS = None"
            assert mDS_partial is not None
            assert mY is not None

        if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        if const_expr(mBatchIdx_schedule_order is not None):
            assert (
                mTileCount_semaphore is None
            ), "we only define a static scheduling order for static persistent tile scheduler"

        self.tensormap_management_bytes = (
            self.tensormap_bytes_total
            if const_expr(
                self.tensormap_update_mode == cutlass.utils.TensorMapUpdateMode.SMEM
            )
            else 0
        ) + self.tensor_memory_management_bytes

        self._setup_attributes()

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, self.tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )

        if const_expr(self.is_A_gather):
            A_tiled_copy = self._make_tiled_copy_2D(
                mA,
                self.tile_M,
                self.tile_K,
                self.a_layout == cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.num_load_A_threads,
                self.universal_copy_bits,
                is_g2s=True,
            )
            tma_atom_a = tma_tensor_a = None
        else:
            A_tiled_copy = None
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                mA,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                self.cluster_shape_mnk[1],
            )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        if const_expr(self.need_epilogue_load):
            tma_atom_c, tma_tensor_c = self._make_tma_epi_atoms_and_tensors(
                mC, self.c_epi_smem_layout_staged, self.c_epi_tile, store_or_load="load"
            )
        else:
            tma_atom_c, tma_tensor_c = None, None

        atom_bias = None
        if const_expr(self.use_bias):
            atom_bias = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cute.nvgpu.cpasync.LoadCacheMode.ALWAYS
                ),
                mBias.element_type,
                num_bits_per_copy=self.universal_copy_bits,
            )

            thread_per_row = self.tile_shape_mnk[1] // (
                self.universal_copy_bits // mBias.element_type.width
            )
            thread_layout = cute.make_ordered_layout((1, thread_per_row), order=(1, 0))
            value_layout = cute.make_layout(
                (1, self.universal_copy_bits // mBias.element_type.width)
            )
            atom_bias = cute.make_tiled_copy_tv(atom_bias, thread_layout, value_layout)

        if const_expr(self.d_epi_smem_layout_staged is not None):
            tma_atom_d, tma_tensor_d = self._make_tma_epi_atoms_and_tensors(
                mD,
                self.d_epi_smem_layout_staged,
                self.d_epi_tile,
                store_or_load="store",
            )
        else:
            tma_atom_d, tma_tensor_d = None, None

        if const_expr(mDIdx is not None):
            copy_elems = self.universal_copy_bits // mD.element_type.width
            assert self.num_epi_threads % (self.d_epi_tile[1] // copy_elems) == 0

            D_tiled_copy = self._make_tiled_copy_2D(
                mD,
                self.d_epi_tile[0],
                self.d_epi_tile[1],
                self.d_layout.is_n_major_c(),
                self.num_epi_threads,
                self.universal_copy_bits,
                is_g2s=False,
            )
        else:
            D_tiled_copy = None

        if const_expr(self.need_adhoc_epilogue_store):
            tma_atom_y, tma_tensor_y = self._make_tma_epi_atoms_and_tensors(
                mY,
                self.y_epi_smem_layout_staged,
                self.y_epi_tile,
                store_or_load="store",
            )
        else:
            tma_atom_y, tma_tensor_y = None, None

        if const_expr(self.compute_weight_gradient):
            assert const_expr(
                not self.compute_dz_and_partial_ds_and_y1s
            ), "weight grad computation conflicts with activation grad computation"

            problem_shape_ntile_mnl = cute.ceil_div(
                mD.shape[:2], self.tile_shape_mnk[:2]
            ) + (mD.shape[2],)
            TileScheduler = SonicMoETileScheduler
            tile_sched_args = TileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                raster_order=self.raster_order,
                group_size=self.L2_group_size,
                cluster_shape_mnk=self.cluster_shape_mnk,
                is_persistent=self.is_persistent,
                tile_count_semaphore=mTileCount_semaphore,
                batch_idx_permute=mBatchIdx_schedule_order,
            )
        else:
            problem_shape_ntile_mnl = (
                None,
                cute.ceil_div(mD.shape[1], self.tile_shape_mnk[1]),
                mMoffset.shape[0] - 1,
            )
            TileScheduler = SonicMoEVarlenMTileScheduler
            tile_sched_args = VarlenMTileSchedulerArguments(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                total_m=mD.shape[0],
                cu_seqlens_m=mMoffset,
                raster_order=self.raster_order,
                group_size=self.L2_group_size,
                tile_shape_mn=self.tile_shape_mnk[:2],
                cluster_shape_mnk=self.cluster_shape_mnk,
                is_persistent=self.is_persistent,
                tile_count_semaphore=mTileCount_semaphore,
            )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid = TileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        c_epi_smem_size = (
            cute.cosize(self.c_epi_smem_layout_staged)
            if const_expr(self.need_epilogue_load)
            else 0
        )
        bias_epi_smem_size = (
            cute.cosize(self.bias_epi_smem_layout_staged)
            if const_expr(self.use_bias)
            else 0
        )
        d_epi_smem_size = (
            cute.cosize(self.d_epi_smem_layout_staged)
            if const_expr(self.is_persistent and (self.d_epi_stage > 0))
            else 0
        )
        y_epi_smem_size = (
            cute.cosize(self.y_epi_smem_layout_staged)
            if const_expr(self.need_adhoc_epilogue_store) and self.is_persistent
            else 0
        )
        s_epi_smem_size = (
            cute.cosize(self.s_epi_smem_layout_staged)
            if const_expr(self.compute_dz_and_partial_ds_and_y1s)
            else 0
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_tensormaps], 64
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, d_epi_smem_size],
                self.buffer_align_bytes,
            ]
            sched_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.sched_stage * 2
            ]
            tile_count: cute.struct.MemRange[cutlass.Int32, self.sched_stage]
            if const_expr(self.need_epilogue_load):
                sC: cute.struct.Align[
                    cute.struct.MemRange[self.c_dtype, c_epi_smem_size],
                    self.buffer_align_bytes,
                ]
                epi_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.c_epi_stage * 2
                ]

            if const_expr(self.use_bias):
                sBias: cute.struct.Align[
                    cute.struct.MemRange[self.bias_dtype, bias_epi_smem_size],
                    self.buffer_align_bytes,
                ]

            if const_expr(self.compute_dz_and_partial_ds_and_y1s):
                sS: cute.struct.Align[
                    cute.struct.MemRange[self.s_dtype, s_epi_smem_size],
                    self.buffer_align_bytes,
                ]

            if const_expr(self.need_adhoc_epilogue_store):
                sY: cute.struct.Align[
                    cute.struct.MemRange[self.y_dtype, y_epi_smem_size],
                    self.buffer_align_bytes,
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
            if const_expr(self.compute_weight_gradient and self.is_A_gather):
                sAIdx_prefetch: cute.struct.Align[
                    cute.struct.MemRange[
                        self.index_dtype, self.prefetch_token_idx_size
                    ],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage
        allocated_smem_size = (
            self.shared_storage.size_in_bytes() + self.tensormap_management_bytes
        )
        # Launch the kernel synchronously
        self.kernel(
            A_tiled_copy,
            mA,
            tma_atom_a,
            tma_tensor_a,
            mB,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            mC,
            mBias,
            atom_bias,
            D_tiled_copy,
            tma_atom_d,
            tma_tensor_d,
            mD,
            tma_atom_y,
            tma_tensor_y,
            mY,
            mS,
            mDS_partial,
            mMoffset,
            mAIdx,
            mDIdx,
            mS_scatter_idx,
            mA_tensormap,
            mB_tensormap,
            mC_tensormap,
            mD_tensormap,
            mY_tensormap,
            tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.prefetch_AIdx_smem_layout_staged,
            self.c_epi_smem_layout_staged,
            self.bias_epi_smem_layout_staged,
            self.d_epi_smem_layout_staged,
            self.y_epi_smem_layout_staged,
            self.s_epi_smem_layout_staged,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=allocated_smem_size,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def update_tma_desc_ptr(
        self,
        mTensor: cute.Tensor,
        tma_atom: cute.CopyAtom,
        tensormap_manager: TensorMapManagerSm90,
        tensormap_ptr: cute.Pointer,
        token_start: Int32,
        token_group_size: Int32,
        is_tma_warp: bool,
        tensormap_smem_ptr: Optional[cute.Pointer] = None,
        address_space: cute.AddressSpace = cute.AddressSpace.generic,
    ) -> cute.Pointer:
        if const_expr(self.compute_weight_gradient):
            tensor_shape = (mTensor.shape[0], token_group_size)
            start_ptr = (mTensor.iterator + token_start * mTensor.stride[1]).toint()
        else:
            tensor_shape = (token_group_size, mTensor.shape[1])
            start_ptr = (mTensor.iterator + token_start * mTensor.stride[0]).toint()

        tensor_gmem_ptr = cute.make_ptr(
            mTensor.element_type,
            start_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        real_tensor = cute.make_tensor(
            tensor_gmem_ptr,
            cute.make_layout(tensor_shape, stride=mTensor.stride),
        )
        if const_expr(
            self.tensormap_update_mode == cutlass.utils.TensorMapUpdateMode.GMEM
        ):
            tensormap_manager.update_tensormap(
                (real_tensor,),
                (tma_atom,),
                tensormap_gmem_ptr=(tensormap_ptr,),
                is_manager_warp=is_tma_warp,
                tensormap_smem_ptr=None,
            )
        else:
            assert tensormap_smem_ptr is not None
            tensormap_manager.update_tensormap(
                (real_tensor,),
                (tma_atom,),
                tensormap_gmem_ptr=(tensormap_ptr,),
                is_manager_warp=is_tma_warp,
                tensormap_smem_ptr=(tensormap_smem_ptr,),
            )

        tensormap_manager.fence_tensormap_update(tensormap_ptr)

    @cute.jit
    def align_tensormap_smem_ptr(self, base_ptr: cute.Pointer):
        return cute.make_ptr(
            cutlass.Int64,
            base_ptr.toint(),
            cute.AddressSpace.smem,
            assumed_align=128,
        )

    @cute.jit
    def allocate_new_tensormap_smem_ptr(self, tensormap_smem_ptr: cute.Pointer):
        return self.align_tensormap_smem_ptr(
            tensormap_smem_ptr + self.bytes_per_tensormap // 8
        )

    @cute.jit
    def swiglu_derivative(
        self, g: Float32, u: Float32, dy1: Float32
    ) -> Tuple[Float32, Float32, Float32]:
        half_g = 0.5 * g
        tanh_half_g = self.tanh(half_g)

        sig_g = self.fma(0.5, tanh_half_g, 0.5)
        sig_n_g = 1 - sig_g

        silu_g = self.fma(half_g, tanh_half_g, half_g)

        dg = dy1 * (u * self.fma(silu_g, sig_n_g, sig_g))
        du = dy1 * silu_g

        swiglu_output = silu_g * u
        return dg, du, swiglu_output

    @cute.jit
    def reglu_derivative(
        self, g: Float32, u: Float32, dy1: Float32
    ) -> Tuple[Float32, Float32, Float32]:
        relu_g = cute.arch.fmax(0.0, g)

        relu_prime_g = 1.0
        if g < Float32(0.0):
            relu_prime_g = 0.0  # derivative of ReLU

        dg = dy1 * u * relu_prime_g
        du = dy1 * relu_g

        reglu_output = u * relu_g
        return dg, du, reglu_output

    @cute.jit
    def geglu_derivative(
        self, g: Float32, u: Float32, dy1: Float32
    ) -> Tuple[Float32, Float32, Float32]:
        # gelu(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715*g^3)))
        sqrt_2_over_pi = const_expr(math.sqrt(2 / math.pi))
        c = 0.044715

        g2 = g * g
        g3 = g2 * g

        # t = sqrt(2/pi) * (g + c*g^3)
        t = sqrt_2_over_pi * self.fma(c, g3, g)

        tanh_t = self.tanh(t)
        one_plus_th = 1.0 + tanh_t  # 1 + tanh(t)
        gelu_g = 0.5 * g * one_plus_th  # gelu(g)

        # d th / d g = (1 - tanh(t)^2) * sqrt(2/pi) * (1 + 3c g^2)
        sech2 = self.fma(-tanh_t, tanh_t, 1.0)
        dt_dg = sech2 * sqrt_2_over_pi * self.fma(3.0 * c, g2, 1.0)

        # d gelu / d g = 0.5*(1 + tanh(t)) + 0.5*g*dt_dg
        gelu_prime = 0.5 * self.fma(g, dt_dg, one_plus_th)

        # Chain rule for y = gelu(g) * u
        dg = dy1 * u * gelu_prime
        du = dy1 * gelu_g

        geglu_output = u * gelu_g
        return dg, du, geglu_output

    @cute.jit
    def silu_derivative(self, x: Float32, dy1: Float32) -> Tuple[Float32, Float32]:
        half_x = 0.5 * x
        tanh_half_x = self.tanh(half_x)

        sig_x = self.fma(0.5, tanh_half_x, 0.5)
        sig_n_x = 1 - sig_x

        silu_x = self.fma(half_x, tanh_half_x, half_x)
        dx = dy1 * self.fma(silu_x, sig_n_x, sig_x)

        return dx, silu_x

    @cute.jit
    def relu_derivative(self, x: Float32, dy1: Float32) -> Tuple[Float32, Float32]:
        relu_x = cute.arch.fmax(0.0, x)

        relu_prime_x = 1.0
        if x < Float32(0.0):
            relu_prime_x = 0.0  # derivative of ReLU

        dx = dy1 * relu_prime_x
        return dx, relu_x

    @cute.jit
    def relu_sq_derivative(self, x: Float32, dy1: Float32) -> Tuple[Float32, Float32]:
        relu_x = cute.arch.fmax(x, 0.0)
        relu_sq_output = relu_x * x
        dx = dy1 * (2.0 * relu_x)
        return dx, relu_sq_output

    @cute.jit
    def gelu_derivative(self, x: Float32, dy1: Float32) -> Tuple[Float32, Float32]:
        # gelu(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715*g^3)))
        sqrt_2_over_pi = const_expr(math.sqrt(2 / math.pi))
        c = 0.044715

        x2 = x * x
        x3 = x2 * x

        # t = sqrt(2/pi) * (g + c*g^3)
        t = sqrt_2_over_pi * self.fma(c, x3, x)

        tanh_t = self.tanh(t)
        one_plus_tanh_t = 1.0 + tanh_t  # 1 + tanh(t)
        gelu_x = 0.5 * x * one_plus_tanh_t

        # d th / d g = (1 - tanh(t)^2) * sqrt(2/pi) * (1 + 3c g^2)
        sech2 = self.fma(-tanh_t, tanh_t, 1.0)
        dt_dg = sech2 * sqrt_2_over_pi * self.fma(3.0 * c, x2, 1.0)

        # d gelu / d g = 0.5*(1 + tanh(t)) + 0.5*g*dt_dg
        gelu_prime = 0.5 * self.fma(x, dt_dg, one_plus_tanh_t)

        # Chain rule for y = gelu(g) * u
        dx = dy1 * gelu_prime

        return dx, gelu_x

    @cute.jit
    def compute_activation(self, tRS_rD, tRS_rY):
        if const_expr(self.is_glu):
            # tRS_sY: (((2, 4), 1), 1, 1, (1, 4))
            # (((2, 4), 1), 1, 1)
            if const_expr(self.compute_swiglu):
                act_func = self.silu
            elif const_expr(self.compute_reglu):
                act_func = self.relu
            elif const_expr(self.compute_geglu):
                act_func = self.gelu
            else:
                raise NotImplementedError()

            for i in cutlass.range_constexpr(cute.size(tRS_rD) // 2):
                tRS_rY[i] = (
                    act_func(tRS_rD[const_expr(2 * i)]) * tRS_rD[const_expr(2 * i + 1)]
                ).to(self.y_dtype)

            self.permute_gated_Cregs_b16(tRS_rY)

        elif const_expr(self.is_normal_act):
            assert cute.size(tRS_rD) == cute.size(tRS_rY)
            if const_expr(self.compute_relu_sq):
                act_func = self.relu_sq
            elif const_expr(self.compute_relu):
                act_func = self.relu
            elif const_expr(self.compute_silu):
                act_func = self.silu
            elif const_expr(self.compute_gelu):
                act_func = self.gelu
            else:
                raise NotImplementedError()

            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                tRS_rY[i] = act_func(tRS_rD[i]).to(self.y_dtype)

        else:
            raise NotImplementedError()

    @cute.jit
    def compute_backward_activation(
        self, tRS_rAcc, sS, tRS_rcD, tRS_rC, tRS_rD, tRS_rD_out, tRS_rY, epi_idx: Int32
    ):
        if const_expr(self.is_glu):
            # if we compute glu activation,
            #   we will assume the incoming C dtype as FP32, and we will output final result in FP32 (decompress to BF16 in caller side)

            if const_expr(self.compute_swiglu):
                bwd_act_func = self.swiglu_derivative
            elif const_expr(self.compute_reglu):
                bwd_act_func = self.reglu_derivative
            elif const_expr(self.compute_geglu):
                bwd_act_func = self.geglu_derivative
            else:
                raise NotImplementedError()

            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                g, u = self.unpack2x16_as_2xf32(tRS_rC[i], self.a_dtype)
                dy = tRS_rD[i]
                dg, du, fwd_output = bwd_act_func(g, u, dy)
                tRS_rAcc[const_expr(epi_idx * cute.size(tRS_rD) + i)] = dy * fwd_output
                s = sS[tRS_rcD[i]]
                tRS_rD_out[i] = self.pack2x16_as_f32(
                    self.a_dtype(dg * s), self.a_dtype(du * s)
                )
                tRS_rY[i] = self.y_dtype(fwd_output * s)

        elif const_expr(self.is_normal_act):
            if const_expr(self.compute_relu_sq):
                bwd_act_func = self.relu_sq_derivative
            elif const_expr(self.compute_relu):
                bwd_act_func = self.relu_derivative
            elif const_expr(self.compute_gelu):
                bwd_act_func = self.gelu_derivative
            elif const_expr(self.compute_silu):
                bwd_act_func = self.silu_derivative
            else:
                raise NotImplementedError()

            for i in cutlass.range_constexpr(cute.size(tRS_rD)):
                z = tRS_rC[i]
                dy = tRS_rD[i]
                dz, fwd_output = bwd_act_func(z, dy)
                tRS_rAcc[const_expr(epi_idx * cute.size(tRS_rD) + i)] = dy * fwd_output
                s = sS[tRS_rcD[i]]
                tRS_rD_out[i] = self.a_dtype(dz * s)
                tRS_rY[i] = self.y_dtype(fwd_output * s)

        else:
            raise NotImplementedError()

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        A_tiled_copy: Optional[cute.TiledCopy],
        mA_mkl: cute.Tensor,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl_tma: Optional[cute.Tensor],
        mB_nkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl_tma: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl_tma: Optional[cute.Tensor],
        mC_mnl: cute.Tensor,
        mBias_nl: Optional[cute.Tensor],
        cpasync_atom_bias: Optional[cute.CopyAtom],
        D_tiled_copy: Optional[cute.TiledCopy],
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl_tma: cute.Tensor,
        mD_mnl: cute.Tensor,
        tma_atom_y: Optional[cute.CopyAtom],
        mY_mnl_tma: Optional[cute.Tensor],
        mY_mnl: Optional[cute.Tensor],
        mS_ml: Optional[cute.Tensor],
        mDS_partial: Optional[cute.Tensor],
        mTokenoffset: cute.Tensor,
        mAIdx_mkl: cute.Tensor,
        mDIdx_mnl: Optional[cute.Tensor],
        mS_scatter_idx: Optional[cute.Tensor],
        mA_tensormap: Optional[cute.Tensor],
        mB_tensormap: Optional[cute.Tensor],
        mC_tensormap: Optional[cute.Tensor],
        mD_tensormap: cute.Tensor,
        mY_tensormap: Optional[cute.Tensor],
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        prefetch_AIdx_smem_layout_staged: Optional[cute.Layout],
        c_epi_smem_layout_staged: Optional[cute.ComposedLayout],
        bias_epi_smem_layout_staged: Optional[cute.Layout],
        d_epi_smem_layout_staged: cute.ComposedLayout,
        y_epi_smem_layout_staged: Optional[cute.ComposedLayout],
        s_epi_smem_layout_staged: Optional[cute.Layout],
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # Assume: M: 2048, N: 512, K: 1024, L: 4
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            if const_expr(not self.is_A_gather):
                cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if const_expr(not self.inference_mode):
                cpasync.prefetch_descriptor(tma_atom_d)
            if const_expr(self.need_adhoc_epilogue_store):
                cpasync.prefetch_descriptor(tma_atom_y)
            if const_expr(tma_atom_c is not None):
                cpasync.prefetch_descriptor(tma_atom_c)

        A_thr_copy_elems = self.universal_copy_bits // mA_mkl.element_type.width

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        if const_expr(self.is_A_gather):
            tma_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        else:
            tma_copy_bytes = cute.size_in_bytes(
                self.a_dtype, a_smem_layout
            ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        shared_storage = smem.allocate(self.shared_storage)

        # Threads/warps participating in this pipeline
        if const_expr(self.is_A_gather):
            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 1 + self.num_load_A_threads
            )
            # Each warp will contribute to the arrive count with the number of mcast size
            mcast_size = self.num_mcast_ctas_b
            pipeline_class = PipelineTmaCpAsync
        else:
            # Threads/warps participating in this pipeline
            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            # Each warp will contribute to the arrive count with the number of mcast size
            mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            pipeline_class = pipeline.PipelineTmaAsync

        consumer_arrive_cnt = mcast_size * (self.num_mma_threads // cute.arch.WARP_SIZE)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline_class.create(
            barrier_storage=shared_storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if const_expr(self.need_epilogue_load):
            # Threads/warps participating in this pipeline
            epi_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            # Each warp will contribute 1 to the arrive count
            consumer_arrive_cnt = self.num_epi_threads // cute.arch.WARP_SIZE
            epi_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            c_smem_layout = cute.slice_(c_epi_smem_layout_staged, (None, None, 0))
            tma_copy_c_bytes = cute.size_in_bytes(self.c_dtype, c_smem_layout)
            epi_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=shared_storage.epi_pipeline_array_ptr.data_ptr(),
                num_stages=self.c_epi_stage,
                producer_group=epi_pipeline_producer_group,
                consumer_group=epi_pipeline_consumer_group,
                tx_count=tma_copy_c_bytes,
            )
        else:
            epi_pipeline = None

        sA = shared_storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = shared_storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        if const_expr(not self.is_persistent):
            sD_ptr = cute.recast_ptr(
                sA.iterator, d_epi_smem_layout_staged.inner, dtype=self.d_dtype
            )
            sD = cute.make_tensor(sD_ptr, d_epi_smem_layout_staged.outer)

            if const_expr(self.inference_mode and self.need_adhoc_epilogue_store):
                next_ptr = sD_ptr
            else:
                next_ptr = sD_ptr + cute.cosize(d_epi_smem_layout_staged)

            if const_expr(self.need_adhoc_epilogue_store):
                sY_ptr = cute.recast_ptr(
                    next_ptr, y_epi_smem_layout_staged.inner, dtype=self.y_dtype
                )
                sY = cute.make_tensor(sY_ptr, y_epi_smem_layout_staged.outer)
                next_ptr = sY_ptr + cute.cosize(y_epi_smem_layout_staged)
            else:
                sY = None

        else:
            if const_expr(self.need_adhoc_epilogue_store):
                sY = shared_storage.sY.get_tensor(
                    y_epi_smem_layout_staged.outer,
                    swizzle=y_epi_smem_layout_staged.inner,
                )
                if const_expr(self.inference_mode):
                    sD = cute.make_tensor(
                        cute.recast_ptr(
                            sY.iterator,
                            d_epi_smem_layout_staged.inner,
                            dtype=self.d_dtype,
                        ),
                        d_epi_smem_layout_staged.outer,
                    )
                else:
                    sD = shared_storage.sD.get_tensor(
                        d_epi_smem_layout_staged.outer,
                        swizzle=d_epi_smem_layout_staged.inner,
                    )
            else:
                sY = None
                sD = shared_storage.sD.get_tensor(
                    d_epi_smem_layout_staged.outer,
                    swizzle=d_epi_smem_layout_staged.inner,
                )

        if const_expr(self.compute_weight_gradient and self.is_A_gather):
            sAIdx_prefetch = shared_storage.sAIdx_prefetch.get_tensor(
                prefetch_AIdx_smem_layout_staged
            )
        else:
            sAIdx_prefetch = None

        if const_expr(self.compute_dz_and_partial_ds_and_y1s):
            sS = shared_storage.sS.get_tensor(
                s_epi_smem_layout_staged, dtype=self.s_dtype
            )
        else:
            sS = None

        if const_expr(self.need_epilogue_load):
            sC = shared_storage.sC.get_tensor(
                c_epi_smem_layout_staged.outer, swizzle=c_epi_smem_layout_staged.inner
            )
        else:
            sC = None

        if const_expr(self.use_bias):
            sBias = shared_storage.sBias.get_tensor(bias_epi_smem_layout_staged)
        else:
            sBias = None

        sched_pipeline = None
        tile_count = None
        if const_expr(tile_sched_params.tile_count_semaphore is not None):
            sched_pipeline = self.make_sched_pipeline(
                cta_layout_mnk,
                sched_pipeline_mbar_ptr=shared_storage.sched_pipeline_array_ptr.data_ptr(),
            )
            tile_count = shared_storage.tile_count.get_tensor((self.sched_stage,))

        a_tensormap_smem_ptr = b_tensormap_smem_ptr = c_tensormap_smem_ptr = (
            d_tensormap_smem_ptr
        ) = y_tensormap_smem_ptr = None
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_smem_ptr = shared_storage.tensormap_buffer.data_ptr()
            tensormap_smem_ptr = self.align_tensormap_smem_ptr(tensormap_smem_ptr)

            if const_expr(self.compute_weight_gradient):
                if const_expr(not self.is_A_gather):
                    tensormap_smem_ptr = a_tensormap_smem_ptr = (
                        self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                    )
                tensormap_smem_ptr = b_tensormap_smem_ptr = (
                    self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                )
            else:
                if const_expr(self.pingpong):
                    if const_expr(
                        not (self.inference_mode and self.need_adhoc_epilogue_store)
                    ):
                        tensormap_smem_ptr = d0_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                        tensormap_smem_ptr = d1_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                        d_tensormap_smem_ptr = (
                            d0_tensormap_smem_ptr
                            if warp_idx // 4 == 0
                            else d1_tensormap_smem_ptr
                        )

                    if const_expr(self.need_adhoc_epilogue_store):
                        tensormap_smem_ptr = y0_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                        tensormap_smem_ptr = y1_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                        y_tensormap_smem_ptr = (
                            y0_tensormap_smem_ptr
                            if warp_idx // 4 == 0
                            else y1_tensormap_smem_ptr
                        )

                    if const_expr(self.need_epilogue_load):
                        tensormap_smem_ptr = c0_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                        tensormap_smem_ptr = c1_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                        c_tensormap_smem_ptr = (
                            c0_tensormap_smem_ptr
                            if warp_idx // 4 == 0
                            else c1_tensormap_smem_ptr
                        )

                else:
                    if const_expr(
                        not (self.inference_mode and self.need_adhoc_epilogue_store)
                    ):
                        tensormap_smem_ptr = d_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )
                    if const_expr(self.need_adhoc_epilogue_store):
                        tensormap_smem_ptr = y_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )

                    if const_expr(self.need_epilogue_load):
                        tensormap_smem_ptr = c_tensormap_smem_ptr = (
                            self.allocate_new_tensormap_smem_ptr(tensormap_smem_ptr)
                        )

        grid_dim = cute.arch.grid_dim()
        bid = cute.arch.block_idx()
        tensormap_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )
        tensormap_manager = TensorMapManagerSm90(
            self.tensormap_update_mode, self.bytes_per_tensormap
        )

        if const_expr(self.compute_weight_gradient):
            if const_expr(not self.is_A_gather and (mA_tensormap is not None)):
                a_tensormap_ptr = tensormap_manager.get_tensormap_ptr(
                    mA_tensormap[tensormap_workspace_idx, None].iterator
                )
            else:
                a_tensormap_ptr = None

            if const_expr(mB_tensormap is not None):
                b_tensormap_ptr = tensormap_manager.get_tensormap_ptr(
                    mB_tensormap[tensormap_workspace_idx, None].iterator
                )
            else:
                b_tensormap_ptr = None

            if cutlass.const_expr(
                self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
            ):
                if const_expr(not self.is_A_gather):
                    tensormap_a_init_ptr = a_tensormap_smem_ptr
                tensormap_b_init_ptr = b_tensormap_smem_ptr
            else:
                if const_expr(not self.is_A_gather):
                    tensormap_a_init_ptr = b_tensormap_ptr
                tensormap_b_init_ptr = b_tensormap_ptr

        else:
            if const_expr(self.pingpong):
                tensormap_workspace_idx = tensormap_workspace_idx * 2 + warp_idx // 4

            if const_expr(
                (mD_tensormap is not None)
                and (not (self.inference_mode and self.need_adhoc_epilogue_store))
            ):
                d_tensormap_ptr = tensormap_manager.get_tensormap_ptr(
                    mD_tensormap[tensormap_workspace_idx, None].iterator
                )
            else:
                d_tensormap_ptr = None

            if const_expr(self.need_adhoc_epilogue_store):
                assert mY_tensormap is not None
                y_tensormap_ptr = tensormap_manager.get_tensormap_ptr(
                    mY_tensormap[tensormap_workspace_idx, None].iterator
                )
            else:
                y_tensormap_ptr = None

            if const_expr(self.need_epilogue_load):
                assert mC_tensormap is not None
                c_tensormap_ptr = tensormap_manager.get_tensormap_ptr(
                    mC_tensormap[tensormap_workspace_idx, None].iterator
                )
            else:
                c_tensormap_ptr = None

            if cutlass.const_expr(
                self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
            ):
                tensormap_d_init_ptr = d_tensormap_smem_ptr
                tensormap_y_init_ptr = y_tensormap_smem_ptr
                tensormap_c_init_ptr = c_tensormap_smem_ptr
            else:
                tensormap_d_init_ptr = d_tensormap_ptr
                tensormap_y_init_ptr = y_tensormap_ptr
                tensormap_c_init_ptr = c_tensormap_ptr

        TileSchedulerCls = partial(
            TileScheduler.create, tile_sched_params, tile_count, sched_pipeline
        )

        k_tile_cnt = cute.ceil_div(cute.size(mA_mkl.shape[1]), self.tile_shape_mnk[2])
        c_tile_cnt = (
            cute.size(cute.ceil_div(self.tile_shape_mnk[:2], self.c_epi_tile))
            if const_expr(self.need_epilogue_load)
            else Int32(0)
        )

        if warp_idx >= self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            prolog_loading_warp_ids = (
                [
                    const_expr(self.tma_warp_id + i)
                    for i in range(self.num_load_A_threads // cute.arch.WARP_SIZE)
                ]
                if const_expr(self.is_A_gather)
                else [const_expr(self.tma_warp_id)]
            )

            if warp_idx in prolog_loading_warp_ids:
                is_tma_warp = cutlass.Boolean(warp_idx == self.tma_warp_id)
                cta_rank_in_cluster = cute.arch.make_warp_uniform(
                    cute.arch.block_idx_in_cluster()
                )
                cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

                a_mcast_mask = cute.make_layout_image_mask(
                    cta_layout_mnk, cluster_coord_mnk, mode=1
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = cute.make_layout_image_mask(
                    cta_layout_mnk, cluster_coord_mnk, mode=0
                )
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                mainloop_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                is_scheduler_warp = warp_idx == self.tma_warp_id
                if const_expr(cute.size(cta_layout_mnk) > 1):
                    is_scheduler_warp = (
                        is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                    )

                tile_scheduler = TileSchedulerCls(is_scheduler_warp=is_scheduler_warp)
                work_tile = tile_scheduler.initial_work_tile_info()

                last_batch_idx = cutlass.Int32(-1)
                token_group_size = cutlass.Int32(0)

                mcA_mkl = cute.make_identity_tensor((mA_mkl.shape[0], mA_mkl.shape[1]))

                TIdx_cur_group = TIdx_next_group = cutlass.Int32(0)
                mAIdx_mk = cute.domain_offset((0,), mAIdx_mkl)

                gA_mk = None
                A_g2s_thr_copy = None
                if const_expr(self.is_A_gather):
                    A_g2s_thr_copy = A_tiled_copy.get_slice(
                        tidx - self.tma_warp_id * cute.arch.WARP_SIZE
                    )
                    gA_mk = cute.local_tile(
                        mA_mkl, (self.tile_M, self.tile_K), (0, None)
                    )
                    tAgA = A_g2s_thr_copy.partition_S(gA_mk)

                if const_expr(self.compute_weight_gradient):
                    if const_expr(not self.is_A_gather):
                        tensormap_manager.init_tensormap_from_atom(
                            tma_atom_a,
                            tensormap_a_init_ptr,
                            is_manager_warp=is_tma_warp,
                        )
                    tensormap_manager.init_tensormap_from_atom(
                        tma_atom_b,
                        tensormap_b_init_ptr,
                        is_manager_warp=is_tma_warp,
                    )
                    tensormap_manager.fence_tensormap_initialization()

                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # (bM, bK, RestK)
                    if batch_idx != last_batch_idx:
                        TIdx_cur_group, TIdx_next_group = cute.arch.make_warp_uniform(
                            mTokenoffset[batch_idx]
                        ), cute.arch.make_warp_uniform(mTokenoffset[batch_idx + 1])
                        token_group_size = TIdx_next_group - TIdx_cur_group

                        if const_expr(self.is_A_gather):
                            if const_expr(self.compute_weight_gradient):
                                mcA_mkl = cute.make_identity_tensor(
                                    (mA_mkl.shape[0], token_group_size)
                                )
                            else:
                                mcA_mkl = cute.make_identity_tensor(
                                    (token_group_size, mA_mkl.shape[1])
                                )

                            mAIdx_mk = cute.domain_offset((TIdx_cur_group,), mAIdx_mkl)

                        if const_expr(self.compute_weight_gradient):
                            if const_expr(not self.is_A_gather):
                                assert a_tensormap_ptr is not None
                                self.update_tma_desc_ptr(
                                    mA_mkl,
                                    tma_atom_a,
                                    tensormap_manager,
                                    a_tensormap_ptr,
                                    TIdx_cur_group,
                                    token_group_size,
                                    is_tma_warp,
                                    tensormap_smem_ptr=a_tensormap_smem_ptr,
                                )
                            if const_expr(b_tensormap_ptr is not None):
                                self.update_tma_desc_ptr(
                                    mB_nkl,
                                    tma_atom_b,
                                    tensormap_manager,
                                    b_tensormap_ptr,
                                    TIdx_cur_group,
                                    token_group_size,
                                    is_tma_warp,
                                    tensormap_smem_ptr=b_tensormap_smem_ptr,
                                    # cute.AddressSpace.generic
                                )
                            k_tile_cnt = cute.ceil_div(
                                token_group_size, self.tile_shape_mnk[2]
                            )

                        last_batch_idx = batch_idx

                    if const_expr(self.is_A_gather):
                        cA = cute.local_tile(
                            mcA_mkl,
                            (self.tile_M, self.tile_K),
                            (tile_coord_mnkl[0], None),
                        )

                        tAsA = A_g2s_thr_copy.partition_D(sA)
                        tAcA = A_g2s_thr_copy.partition_D(cA)

                        tApA = cute.make_fragment(
                            cute.make_layout(
                                (
                                    tAgA.shape[0][1],
                                    cute.size(tAgA, mode=[1]),
                                    cute.size(tAgA, mode=[2]),
                                ),
                                stride=(cute.size(tAgA, mode=[1]), 1, 0),
                            ),
                            cutlass.Boolean,
                        )

                        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
                            for m in cutlass.range_constexpr(tApA.shape[1]):
                                if const_expr(self.compute_weight_gradient):
                                    tApA[rest_v, m, 0] = cute.elem_less(
                                        tAcA[(0, rest_v), m, 0, 0][0], mA_mkl.shape[0]
                                    )
                                else:
                                    tApA[rest_v, m, 0] = cute.elem_less(
                                        tAcA[(0, rest_v), m, 0, 0][0], token_group_size
                                    )
                    else:
                        if const_expr(self.compute_weight_gradient):
                            # update TMA map instead
                            mA_mk = cute.domain_offset((0, 0), mA_mkl_tma)
                        else:
                            mA_mk = cute.domain_offset((TIdx_cur_group, 0), mA_mkl_tma)

                        gA_mk_cur = cute.local_tile(
                            mA_mk,
                            (self.tile_M, self.tile_K),
                            (tile_coord_mnkl[0], None),
                        )

                        a_cta_layout = cute.make_layout(
                            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
                        )
                        a_cta_crd = cluster_coord_mnk[1]

                        tAsA, tAgA_mkl = cpasync.tma_partition(
                            tma_atom_a,
                            a_cta_crd,
                            a_cta_layout,
                            cute.group_modes(sA, 0, 2),
                            cute.group_modes(gA_mk_cur, 0, 2),
                        )

                    if const_expr(self.compute_weight_gradient):
                        gB_nk = cute.local_tile(
                            mB_nkl_tma,
                            (self.tile_N, self.tile_K),
                            (tile_coord_mnkl[1], None),
                        )
                    else:
                        gB_nk = cute.local_tile(
                            mB_nkl_tma,
                            self.tile_shape_mnk,
                            tile_coord_mnkl,
                            proj=(None, 1, 1),
                        )

                    b_cta_layout = cute.make_layout(
                        cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
                    )
                    b_cta_crd = cluster_coord_mnk[0]
                    tBsB, tBgB_nkl = cpasync.tma_partition(
                        tma_atom_b,
                        b_cta_crd,
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_nk, 0, 2),
                    )

                    peek_ab_empty_status = cutlass.Boolean(True)
                    if 0 < k_tile_cnt:
                        peek_ab_empty_status = mainloop_pipeline.producer_try_acquire(
                            mainloop_producer_state
                        )

                    if const_expr(self.is_A_gather):
                        M_offset = cute.arch.make_warp_uniform(
                            tile_coord_mnkl[0] * const_expr(self.tile_M)
                        )
                        if const_expr(self.compute_weight_gradient):
                            tmAIdx = None
                            M_boundary = mA_mkl.shape[0]
                        else:
                            M_boundary = cute.arch.make_warp_uniform(
                                self.min_i32(
                                    const_expr(self.tile_M), token_group_size - M_offset
                                )
                            )
                            tmAIdx = self.prefetch_gather_idx_for_A_when_vary_M(
                                mAIdx_mk, M_offset, M_boundary, A_thr_copy_elems
                            )

                    if const_expr(self.compute_weight_gradient):
                        if const_expr(self.is_A_gather):
                            a_tma_desc_ptr = None
                        else:
                            a_tma_desc_ptr = tensormap_manager.get_tensormap_ptr(
                                a_tensormap_ptr, cute.AddressSpace.generic
                            )

                        b_tma_desc_ptr = tensormap_manager.get_tensormap_ptr(
                            b_tensormap_ptr, cute.AddressSpace.generic
                        )
                    else:
                        a_tma_desc_ptr = None
                        b_tma_desc_ptr = None

                    for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                        if const_expr(self.is_A_gather):
                            mainloop_pipeline.producer_acquire(
                                mainloop_producer_state,
                                peek_ab_empty_status,
                                is_tma_warp=is_tma_warp,
                            )
                        else:
                            mainloop_pipeline.producer_acquire(
                                mainloop_producer_state, peek_ab_empty_status
                            )

                        if is_tma_warp:
                            cute.copy(
                                tma_atom_b,
                                tBgB_nkl[None, k_tile],
                                tBsB[None, mainloop_producer_state.index],
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=b_mcast_mask,
                                tma_desc_ptr=b_tma_desc_ptr,
                            )

                        K_offset = k_tile * const_expr(self.tile_K)

                        if const_expr(
                            self.compute_weight_gradient and self.is_A_gather
                        ):
                            if K_offset % const_expr(self.prefetch_token_idx_size) == 0:
                                self.prefetch_gather_idx_for_A_when_vary_K(
                                    mAIdx_mk, sAIdx_prefetch, token_group_size, K_offset
                                )

                        if const_expr(self.is_A_gather):
                            self.load_A_gather(
                                mA_mkl,
                                tmAIdx,
                                sAIdx_prefetch,
                                M_offset,
                                tAsA[None, None, None, mainloop_producer_state.index],
                                tApA,
                                A_g2s_thr_copy,
                                K_offset,
                                token_group_size,
                                A_thr_copy_elems,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_mkl[None, k_tile],
                                tAsA[None, mainloop_producer_state.index],
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                                tma_desc_ptr=a_tma_desc_ptr,
                            )

                        if const_expr(not self.is_A_gather):
                            # Mainloop pipeline's producer commit is a NOP
                            mainloop_pipeline.producer_commit(mainloop_producer_state)
                        else:
                            mainloop_pipeline.producer_cpasync_commit(
                                mainloop_producer_state
                            )
                        mainloop_producer_state.advance()

                        peek_ab_empty_status = cutlass.Boolean(True)
                        if k_tile + 1 < k_tile_cnt:
                            peek_ab_empty_status = (
                                mainloop_pipeline.producer_try_acquire(
                                    mainloop_producer_state
                                )
                            )

                    tile_scheduler.fetch_next_work(is_scheduler_warp=is_scheduler_warp)
                    tile_scheduler.advance_to_next_work(
                        is_scheduler_warp=is_scheduler_warp
                    )
                    work_tile = tile_scheduler.get_current_work()

                if const_expr(self.pingpong):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    tile_scheduler.advance_to_next_work(
                        is_scheduler_warp=is_scheduler_warp
                    )
                    # End of persistent scheduler loop
                mainloop_pipeline.producer_tail(mainloop_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.tma_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            is_tma_warp = cutlass.Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            if const_expr(not self.compute_weight_gradient):
                if const_expr(
                    not (self.inference_mode and self.need_adhoc_epilogue_store)
                ):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_atom_d,
                        tensormap_d_init_ptr,
                        is_manager_warp=is_tma_warp,
                    )
                if const_expr(self.need_adhoc_epilogue_store):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_atom_y,
                        tensormap_y_init_ptr,
                        is_manager_warp=is_tma_warp,
                    )
                if const_expr(self.need_epilogue_load):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_atom_c,
                        tensormap_c_init_ptr,
                        is_manager_warp=is_tma_warp,
                    )

            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(
                tidx // self.num_threads_per_warp_group
            )
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if not self.pingpong else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

            acc_shape = tiled_mma.partition_shape_C(
                cute.select(self.tile_shape_mnk, mode=[0, 1])
            )
            acc = cute.make_fragment(acc_shape, self.acc_dtype)

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            mainloop_consumer_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_epi_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.c_epi_stage
            )

            if const_expr(not self.compute_weight_gradient):
                tensormap_manager.fence_tensormap_initialization()

            tile_scheduler = TileSchedulerCls()
            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    if const_expr(self.compute_weight_gradient):
                        wg0_batch_idx = (
                            tile_scheduler.initial_work_tile_info().tile_idx[-1]
                        )
                        wg0_token_group_size = cute.arch.make_warp_uniform(
                            mTokenoffset[wg0_batch_idx + 1]
                        ) - cute.arch.make_warp_uniform(mTokenoffset[wg0_batch_idx])
                        tile_scheduler.advance_to_next_work()
                        mainloop_consumer_read_state.advance_iters(
                            cute.ceil_div(wg0_token_group_size, self.tile_shape_mnk[2])
                        )
                    else:
                        tile_scheduler.advance_to_next_work()
                        mainloop_consumer_read_state.advance_iters(k_tile_cnt)

                    # mainloop_consumer_read_state.advance_iters(k_tile_cnt)
                    if const_expr(self.need_epilogue_load):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)

            work_tile = tile_scheduler.initial_work_tile_info()
            last_batch_idx = cutlass.Int32(-1)
            token_group_size = cutlass.Int32(0)

            TIdx_cur_group = TIdx_next_group = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx = tile_coord_mnkl[3]
                is_group_changed = batch_idx != last_batch_idx
                if is_group_changed:
                    # construct tensor D based on real address, shape and stride information
                    TIdx_cur_group, TIdx_next_group = cute.arch.make_warp_uniform(
                        mTokenoffset[batch_idx]
                    ), cute.arch.make_warp_uniform(mTokenoffset[batch_idx + 1])
                    token_group_size = cute.arch.make_warp_uniform(
                        TIdx_next_group - TIdx_cur_group
                    )
                    if const_expr(self.compute_weight_gradient):
                        k_tile_cnt = cute.arch.make_warp_uniform(
                            cute.ceil_div(token_group_size, self.tile_shape_mnk[2])
                        )
                    else:
                        if const_expr(
                            (not self.inference_mode)
                            or (not self.need_adhoc_epilogue_store)
                        ):
                            assert (
                                d_tensormap_smem_ptr is not None
                                and d_tensormap_ptr is not None
                            )
                            self.update_tma_desc_ptr(
                                mD_mnl,
                                tma_atom_d,
                                tensormap_manager,
                                d_tensormap_ptr,
                                TIdx_cur_group,
                                token_group_size,
                                is_tma_warp,
                                tensormap_smem_ptr=d_tensormap_smem_ptr,
                                # cute.AddressSpace.generic
                            )
                        if const_expr(self.need_adhoc_epilogue_store):
                            assert (
                                y_tensormap_smem_ptr is not None
                                and y_tensormap_ptr is not None
                            )
                            self.update_tma_desc_ptr(
                                mY_mnl,
                                tma_atom_y,
                                tensormap_manager,
                                y_tensormap_ptr,
                                TIdx_cur_group,
                                token_group_size,
                                is_tma_warp,
                                tensormap_smem_ptr=y_tensormap_smem_ptr,
                                # cute.AddressSpace.generic
                            )
                        if const_expr(self.need_epilogue_load):
                            assert (
                                c_tensormap_smem_ptr is not None
                                and c_tensormap_ptr is not None
                            )
                            self.update_tma_desc_ptr(
                                mC_mnl,
                                tma_atom_c,
                                tensormap_manager,
                                c_tensormap_ptr,
                                TIdx_cur_group,
                                token_group_size,
                                is_tma_warp,
                                tensormap_smem_ptr=c_tensormap_smem_ptr,
                                # cute.AddressSpace.generic
                            )
                        last_batch_idx = batch_idx

                k_pipe_mmas = 1
                mainloop_consumer_release_state = mainloop_consumer_read_state.clone()
                num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")

                peek_ab_full_status = cutlass.Boolean(True)

                if k_tile_cnt == 0:
                    acc.fill(0.0)

                if k_tile_cnt > 0:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_read_state
                    )

                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                num_k_blocks = cute.size(tCrA, mode=[2])

                for k_tile in cutlass.range(num_prologue_mma):
                    # Wait for A/B buffer to be ready
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_read_state, peek_ab_full_status
                    )
                    warpgroup.fence()
                    for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_blk_coord = (
                            None,
                            None,
                            k_blk_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc
                        )
                        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile + 1 < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )

                for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_read_state, peek_ab_full_status
                    )
                    # WGMMA
                    warpgroup.fence()
                    for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_blk_coord = (
                            None,
                            None,
                            k_blk_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc
                        )
                    warpgroup.commit_group()
                    # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
                    warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_read_state.advance()
                    mainloop_consumer_release_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile + 1 < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                warpgroup.wait_group(0)
                for k_tile in cutlass.range(num_prologue_mma, unroll=1):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                if const_expr(self.pingpong):
                    if const_expr(self.compute_weight_gradient):
                        other_batch_idx = tile_scheduler.prefetch_next_work().tile_idx[
                            -1
                        ]
                        other_token_group_size = cute.arch.make_warp_uniform(
                            mTokenoffset[other_batch_idx + 1]
                        ) - cute.arch.make_warp_uniform(mTokenoffset[other_batch_idx])
                        mainloop_consumer_read_state.advance_iters(
                            cute.ceil_div(
                                other_token_group_size, self.tile_shape_mnk[2]
                            )
                        )
                    else:
                        mainloop_consumer_read_state.advance_iters(k_tile_cnt)

                    # Update starting mainloop pipeline state for the next tile

                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")

                epilogue_barrier = pipeline.NamedBarrier(
                    barrier_id=int(NamedBarrierGemm.Epilogue),
                    num_threads=self.num_epi_threads,
                )

                # Wait for all warp groups in the thread block to finish, because smem for tensor
                # A in the mainloop is reused in the epilogue if not persistent.
                if const_expr(not self.is_persistent):
                    epilogue_barrier.arrive_and_wait()

                copy_atom_D_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.d_layout,
                    elem_ty_d=self.d_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_D = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(self.d_layout.is_m_major_c(), 4),
                    self.d_dtype,
                )
                tiled_copy_D_atom = cute.make_tiled_copy_C_atom(copy_atom_D, tiled_mma)
                tiled_copy_D_r2s = cute.make_tiled_copy_S(
                    copy_atom_D_r2s, tiled_copy_D_atom
                )
                # (R2S, R2S_M, R2S_N, PIPE_D)
                tRS_sD = tiled_copy_D_r2s.get_slice(tidx).partition_D(sD)
                tRS_rD_layout = cute.make_layout(
                    tiled_copy_D_r2s.get_slice(tidx).partition_S(sD).shape[:3]
                )
                tRS_rD = cute.make_fragment(tRS_rD_layout, self.acc_dtype)

                if const_expr(self.need_epilogue_load):
                    copy_atom_C = cute.make_copy_atom(
                        warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(),
                            num_matrices=(4 if self.c_epi_tile[1] % 16 == 0 else 2),
                        ),
                        cutlass.Float16,  # this is just to get the right source layout
                    )
                    tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma
                    )
                    copy_atom_C_s2r = sm90_get_smem_load_op(self.c_layout, self.c_dtype)
                    tiled_copy_C_s2r = cute.make_tiled_copy_S(
                        copy_atom_C_s2r, tiled_copy_C_atom
                    )
                    thr_copy_C_s2r = tiled_copy_C_s2r.get_slice(tidx)
                    tSR_sC = thr_copy_C_s2r.partition_S(sC)
                    tRS_rC = cute.make_fragment(tRS_rD_layout, self.c_dtype)
                    tSR_rC = thr_copy_C_s2r.retile(tRS_rC)
                else:
                    thr_copy_C_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                if const_expr(self.need_adhoc_epilogue_store):
                    copy_atom_Y_r2s = sm90_utils.sm90_get_smem_store_op(
                        self.y_layout,
                        elem_ty_d=self.y_dtype,
                        elem_ty_acc=self.acc_dtype,
                    )
                    copy_atom_Y = cute.make_copy_atom(
                        warp.StMatrix8x8x16bOp(self.y_layout.is_m_major_c(), 4),
                        self.y_dtype,
                    )
                    tiled_copy_Y_atom = cute.make_tiled_copy_C_atom(
                        copy_atom_Y, tiled_mma
                    )
                    tiled_copy_Y_r2s = cute.make_tiled_copy_S(
                        copy_atom_Y_r2s, tiled_copy_Y_atom
                    )
                    tRS_sY = tiled_copy_Y_r2s.get_slice(tidx).partition_D(sY)

                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_D_r2s.retile(acc)
                # tRS_rAcc: tensor<ptr<f32, rmem, align<32>> o ((8,8),3,1):((1,8),64,0)>

                # (bM, bN)
                batch_idx = tile_coord_mnkl[3]
                if const_expr(self.compute_weight_gradient):
                    gD_mn = cute.local_tile(
                        mD_mnl_tma[None, None, batch_idx],
                        (self.tile_M, self.tile_N),
                        tile_coord_mnkl[:2],
                    )
                else:
                    gD_mn = cute.local_tile(
                        mD_mnl_tma, (self.tile_M, self.tile_N), tile_coord_mnkl[:2]
                    )

                copy_elems_D = self.universal_copy_bits // mD_mnl.element_type.width
                tdgd_for_tma_partition = cute.zipped_divide(gD_mn, self.d_epi_tile)

                if const_expr(self.need_adhoc_epilogue_store):
                    y_tile_size = (self.tile_M, self.tile_N)
                    if const_expr(
                        self.is_glu and not self.compute_dz_and_partial_ds_and_y1s
                    ):
                        y_tile_size = (self.tile_M, self.tile_N // 2)

                    gY_mn = cute.local_tile(
                        mY_mnl_tma, y_tile_size, tile_coord_mnkl[:2]
                    )

                    tygy_for_tma_partition = cute.zipped_divide(gY_mn, self.y_epi_tile)
                # bSG_sD: tensor<ptr<bf16, smem, align<1024>, S<2,4,3>> o ((2048,1),(1,4)):((1,0),(0,2048))>
                # bSG_gD: tensor<(?{div=128},?{div=192},?) o (((32,64),1),(3,4)):(((1@0,1@1),0),(64@1,32@0))>
                if const_expr(self.inference_mode and self.need_adhoc_epilogue_store):
                    bSG_sD = bSG_gD = None
                else:
                    bSG_sD, bSG_gD = cpasync.tma_partition(
                        tma_atom_d,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sD, 0, 2),
                        tdgd_for_tma_partition,
                    )

                if const_expr(self.need_adhoc_epilogue_store):
                    bSG_sY, bSG_gY = cpasync.tma_partition(
                        tma_atom_y,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sY, 0, 2),
                        tygy_for_tma_partition,
                    )
                    assert const_expr(
                        cute.size(tdgd_for_tma_partition, mode=[1])
                    ) == const_expr(cute.size(tygy_for_tma_partition, mode=[1]))

                if const_expr(self.use_bias):
                    expert_idx = tile_coord_mnkl[-1]
                    expert_elem_load = const_expr(
                        self.universal_copy_bits // mBias_nl.element_type.width
                    )
                    gBias = cute.local_tile(
                        mBias_nl,
                        (1, self.tile_shape_mnk[1]),
                        (expert_idx, tile_coord_mnkl[1]),
                    )
                    cBias = cute.local_tile(
                        cute.make_identity_tensor((1, mBias_nl.shape[1])),
                        (1, self.tile_shape_mnk[1]),
                        (0, tile_coord_mnkl[1]),
                    )

                    thr_copy_bias = cpasync_atom_bias.get_slice(tidx)
                    tBiasgBias = thr_copy_bias.partition_S(gBias)
                    tBiassBias = thr_copy_bias.partition_D(sBias)
                    tBiascBias = thr_copy_bias.partition_S(cBias)

                    thread_per_row = const_expr(
                        self.tile_shape_mnk[1] // expert_elem_load
                    )
                    if tidx < thread_per_row:
                        if tBiascBias[0][1] < mBias_nl.shape[1]:
                            cute.copy(thr_copy_bias, tBiasgBias, tBiassBias)
                        else:
                            tBiassBias.fill(0.0)

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    # cannot be removed for correctness!
                    epilogue_barrier.arrive_and_wait()

                    partition_for_epi_fn = partial(
                        partition_for_epilogue,
                        epi_tile=self.d_epi_tile,
                        tiled_copy=tiled_copy_D_r2s,
                        tidx=tidx,
                        reference_src=True,
                    )
                    sBias_retiled = partition_for_epi_fn(
                        cute.make_tensor(
                            sBias.iterator,
                            cute.make_layout((self.tile_M, self.tile_N), stride=(0, 1)),
                        )
                    )

                epi_tile_num = const_expr(cute.size(tdgd_for_tma_partition, mode=[1]))

                epi_tile_shape = tdgd_for_tma_partition.shape[1]
                num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                if const_expr(mDIdx_mnl is not None):
                    mcD = cute.make_identity_tensor((self.tile_M, self.tile_N))

                    tcDgcD_flat_partition = cute.flat_divide(mcD, self.d_epi_tile)
                    D_r2g_thr_copy = D_tiled_copy.get_slice(tidx)

                    TIdx_cur_group, TIdx_next_group = (
                        mTokenoffset[batch_idx],
                        mTokenoffset[batch_idx + 1],
                    )
                    if const_expr(self.is_scatter_idx_prefetched):
                        tmDIdx = self.prefetch_scatter_idx_for_D_when_vary_M(
                            mD_mnl,
                            mDIdx_mnl,
                            D_r2g_thr_copy,
                            tcDgcD_flat_partition,
                            epi_tile_layout,
                            epi_tile_num,
                            const_expr(
                                self.universal_copy_bits // mD_mnl.element_type.width
                            ),
                            tile_coord_mnkl,
                            TIdx_cur_group,
                            TIdx_next_group,
                        )
                    else:
                        tmDIdx = None
                else:
                    mcD = tcDgcD_flat_partition = D_r2g_thr_copy = tmDIdx = None

                if const_expr(not self.compute_weight_gradient):
                    if const_expr(
                        self.inference_mode and self.need_adhoc_epilogue_store
                    ):
                        d_tma_desc_ptr = None
                    else:
                        d_tma_desc_ptr = tensormap_manager.get_tensormap_ptr(
                            d_tensormap_ptr,
                            cute.AddressSpace.generic,
                        )
                    if const_expr(self.need_adhoc_epilogue_store):
                        y_tma_desc_ptr = tensormap_manager.get_tensormap_ptr(
                            y_tensormap_ptr,
                            cute.AddressSpace.generic,
                        )
                    if const_expr(self.need_epilogue_load):
                        c_tma_desc_ptr = tensormap_manager.get_tensormap_ptr(
                            c_tensormap_ptr,
                            cute.AddressSpace.generic,
                        )
                else:
                    d_tma_desc_ptr = y_tma_desc_ptr = c_tma_desc_ptr = None

                if const_expr(self.compute_dz_and_partial_ds_and_y1s):
                    TIdx_cur_group, TIdx_next_group = cute.arch.make_warp_uniform(
                        mTokenoffset[batch_idx]
                    ), cute.arch.make_warp_uniform(mTokenoffset[batch_idx + 1])
                    self.fetch_scattered_S(
                        tidx,
                        mS_ml,
                        mS_scatter_idx,
                        sS,
                        tile_coord_mnkl,
                        TIdx_cur_group,
                        TIdx_next_group,
                    )
                    epilogue_barrier.arrive_and_wait()

                    cD = cute.make_identity_tensor((self.tile_M, self.tile_N))
                    tDcD = tiled_mma.get_slice(tidx).partition_C(cD)
                    tRS_rcD_retiled = tiled_copy_D_r2s.retile(tDcD)
                    tRS_rcD = cute.make_fragment_like(
                        tRS_rD, dtype=mS_scatter_idx.element_type
                    )

                if const_expr(self.need_epilogue_load):
                    # mC_mn = cute.domain_offset((mTokenoffset[batch_idx], 0), mC_mnl_tma)
                    gC = cute.local_tile(
                        mC_mnl_tma, (self.tile_M, self.tile_N), tile_coord_mnkl[:2]
                    )
                    tCgC_for_tma_partition = cute.zipped_divide(gC, self.c_epi_tile)
                    bGS_sC, bGS_gC = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        tCgC_for_tma_partition,
                    )

                    for epi_idx in cutlass.range(
                        min(epi_tile_num, self.c_epi_stage), unroll=1
                    ):
                        if is_tma_warp:
                            epi_pipeline.producer_acquire(epi_producer_state)
                            # Get the global memory coordinate for the current epi tile
                            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                            cute.copy(
                                tma_atom_c,
                                bGS_gC[None, gmem_coord],
                                bGS_sC[None, epi_producer_state.index],
                                tma_bar_ptr=epi_pipeline.producer_get_barrier(
                                    epi_producer_state
                                ),
                                tma_desc_ptr=c_tma_desc_ptr,
                            )
                            # Epi pipeline's producer commit is a NOP
                            epi_pipeline.producer_commit(epi_producer_state)
                        epi_producer_state.advance()

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from acc to D registers
                    # tRS_sD: (((2, 4), 1), 1, 2, (1, 4))
                    # tRS_rD = cute.make_fragment_like(tRS_sD[None, None, None, 0], self.acc_dtype) # (((2, 4), 1), 1, 2)

                    # tRS_rD: tensor<ptr<f32, rmem, align<16>> o (((2,4),1),1,2):(((1,2),0),0,8)>
                    for epi_v in cutlass.range_constexpr(
                        cute.size(tRS_rD)
                    ):  # cute.size(tRS_rD): 16
                        tRS_rD[epi_v] = tRS_rAcc[
                            const_expr(epi_idx * cute.size(tRS_rD) + epi_v)
                        ]
                        if const_expr(self.compute_dz_and_partial_ds_and_y1s):
                            tRS_rcD[epi_v] = tRS_rcD_retiled[
                                const_expr(epi_idx * cute.size(tRS_rD) + epi_v)
                            ][0]

                    if const_expr(self.need_epilogue_load):
                        epi_pipeline.consumer_wait(epi_read_state)
                        cute.copy(
                            thr_copy_C_s2r,
                            tSR_sC[None, None, None, epi_read_state.index],
                            tSR_rC,
                        )
                        # Fence to make sure shared memory read is visible to TMA load
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            epi_pipeline.consumer_release(epi_read_state)
                        epi_read_state.advance()
                        if const_expr(epi_idx + self.c_epi_stage < epi_tile_num):
                            if is_tma_warp:
                                epi_pipeline.producer_acquire(epi_producer_state)
                                # Get the global memory coordinate for the current epi tile
                                gmem_coord = epi_tile_layout.get_hier_coord(
                                    epi_idx + self.c_epi_stage
                                )
                                cute.copy(
                                    tma_atom_c,
                                    bGS_gC[None, gmem_coord],
                                    bGS_sC[None, epi_producer_state.index],
                                    tma_bar_ptr=epi_pipeline.producer_get_barrier(
                                        epi_producer_state
                                    ),
                                    tma_desc_ptr=c_tma_desc_ptr,
                                )
                                # Epi pipeline's producer commit is a NOP
                                epi_pipeline.producer_commit(epi_producer_state)
                            epi_producer_state.advance()

                    if const_expr(self.use_bias):
                        sBias_retiled_and_grouped = cute.group_modes(
                            sBias_retiled, 3, cute.rank(sBias_retiled)
                        )
                        sBias_retiled_and_grouped_epi = sBias_retiled_and_grouped[
                            None, None, None, epi_tile_layout.get_hier_coord(epi_idx)
                        ]
                        rBias_retiled_epi_r = cute.make_fragment(
                            sBias_retiled_and_grouped_epi.layout,
                            dtype=mBias_nl.element_type,
                        )
                        cute.autovec_copy(
                            cute.filter_zeros(sBias_retiled_and_grouped_epi),
                            cute.filter_zeros(rBias_retiled_epi_r),
                        )

                        for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                            tRS_rD[epi_v] = tRS_rD[epi_v] + self.acc_dtype(
                                rBias_retiled_epi_r[epi_v]
                            )

                    if const_expr(self.compute_dz_and_partial_ds_and_y1s):
                        tRS_rD_out = cute.make_fragment_like(
                            tRS_rD,
                            (
                                cutlass.Float32
                                if const_expr(self.is_glu)
                                else self.d_dtype
                            ),
                        )
                        tRS_rY = cute.make_fragment_like(
                            tRS_sY[None, None, None, 0], self.y_dtype
                        )
                        self.compute_backward_activation(
                            tRS_rAcc,
                            sS,
                            tRS_rcD,
                            tRS_rC,
                            tRS_rD,
                            tRS_rD_out,
                            tRS_rY,
                            epi_idx,
                        )

                    elif const_expr(
                        not (self.inference_mode and self.need_adhoc_epilogue_store)
                    ):
                        tRS_rD_out = cute.make_fragment_like(tRS_rD, self.d_dtype)
                        tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))

                    if const_expr(
                        (self.is_glu or self.is_normal_act)
                        and not self.compute_dz_and_partial_ds_and_y1s
                    ):
                        tRS_rY = cute.make_fragment_like(
                            tRS_sY[None, None, None, 0], self.y_dtype
                        )
                        self.compute_activation(tRS_rD, tRS_rY)

                    # Copy from D registers to shared memory
                    if const_expr(
                        self.inference_mode and self.need_adhoc_epilogue_store
                    ):
                        epi_buffer = (num_prev_subtiles + epi_idx) % cute.size(
                            tRS_sY, mode=[3]
                        )
                    else:
                        epi_buffer = (num_prev_subtiles + epi_idx) % cute.size(
                            tRS_sD, mode=[3]
                        )

                    if const_expr(
                        not (self.inference_mode and self.need_adhoc_epilogue_store)
                    ):
                        cute.copy(
                            tiled_copy_D_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                    if const_expr(self.need_adhoc_epilogue_store):
                        cute.copy(
                            tiled_copy_Y_r2s,
                            tRS_rY,
                            tRS_sY[(None, None, None, epi_buffer)],
                        )

                    if const_expr(mDIdx_mnl is not None):
                        epilogue_barrier.arrive_and_wait()
                        tDsD = D_r2g_thr_copy.partition_S(sD[None, None, epi_buffer])
                        tDrD = cute.make_fragment_like(tDsD)
                        cute.autovec_copy(tDsD, tDrD)

                        tDcD_slice = D_r2g_thr_copy.partition_D(
                            tcDgcD_flat_partition[
                                None, None, *epi_tile_layout.get_hier_coord(epi_idx)
                            ]
                        )
                        self.store_D_scatter(
                            mD_mnl,
                            mDIdx_mnl,
                            tmDIdx,
                            tDrD,
                            tDcD_slice,
                            D_r2g_thr_copy,
                            epi_idx,
                            copy_elems_D,
                            tile_coord_mnkl,
                            TIdx_cur_group,
                            TIdx_next_group,
                        )
                        epilogue_barrier.arrive_and_wait()

                    else:
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        epilogue_barrier.arrive_and_wait()
                        # Get the global memory coordinate for the current epi tile.
                        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                        # Copy from shared memory to global memory
                        if is_tma_warp:
                            if const_expr(
                                not (
                                    self.inference_mode
                                    and self.need_adhoc_epilogue_store
                                )
                            ):
                                cute.copy(
                                    tma_atom_d,
                                    bSG_sD[None, epi_buffer],
                                    bSG_gD[None, gmem_coord],
                                    tma_desc_ptr=d_tma_desc_ptr,
                                )
                            if const_expr(self.need_adhoc_epilogue_store):
                                cute.copy(
                                    tma_atom_y,
                                    bSG_sY[None, epi_buffer],
                                    bSG_gY[None, gmem_coord],
                                    tma_desc_ptr=y_tma_desc_ptr,
                                )
                            cute.arch.cp_async_bulk_commit_group()
                            if const_expr(
                                self.inference_mode and self.need_adhoc_epilogue_store
                            ):
                                cute.arch.cp_async_bulk_wait_group(
                                    const_expr(self.y_epi_stage - 1), read=True
                                )
                            else:
                                cute.arch.cp_async_bulk_wait_group(
                                    const_expr(self.d_epi_stage - 1), read=True
                                )

                        epilogue_barrier.arrive_and_wait()

                if const_expr(self.compute_dz_and_partial_ds_and_y1s):
                    y1 = make_acc_tensor_mn_view(acc)
                    cD = cute.make_identity_tensor((self.tile_M, self.tile_N))
                    tDcD = tiled_mma.get_slice(tidx).partition_C(cD)
                    tDcD_mn = make_acc_tensor_mn_view(tDcD)

                    tile_M_offset = cute.arch.make_warp_uniform(
                        TIdx_cur_group + tile_coord_mnkl[0] * self.tile_M
                    )

                    mDS_partial_M, mDS_partial_N = mDS_partial.shape
                    mDS_partial_flatten_view = cute.make_tensor(
                        mDS_partial.iterator, (mDS_partial_M * mDS_partial_N,)
                    )
                    for r in cutlass.range_constexpr(cute.size(y1, mode=[0])):
                        col_sum = cutlass.Float32(0.0)

                        M_tile_idx = tDcD_mn[r, 0][0]
                        for c in cutlass.range_constexpr(cute.size(y1, mode=[1])):
                            col_sum = col_sum + y1[r, c]

                        col_sum = warp_reduce(col_sum, operator.add, width=4)

                        M_idx_raw = tile_M_offset + M_tile_idx
                        if tidx % 4 == 0 and M_idx_raw < TIdx_next_group:
                            M_idx = mS_scatter_idx[M_idx_raw]
                            N_idx = tile_coord_mnkl[1]
                            mDS_partial_flatten_view[M_idx * mDS_partial_N + N_idx] = (
                                col_sum.to(mDS_partial.element_type)
                            )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signalling
                    # the next WG's epilogue.
                    if const_expr(self.need_epilogue_load):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                    if warp_idx == 0 or warp_idx == 4:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                tile_scheduler.advance_to_next_work(
                    advance_count=1 if not self.pingpong else self.mma_warp_groups
                )
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            if const_expr(not self.pingpong):
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

    def generate_tensormap(self, m, n, l):
        if not self.is_persistent:
            total_m = m * l
            block_size_m = self.tile_M * self.cluster_shape_mnk[0]
            block_size_n = self.tile_N * self.cluster_shape_mnk[1]
            total_clusters_m_max = (total_m + l * (block_size_m - 1)) // block_size_m
            total_clusters_max = total_clusters_m_max * (
                (n + block_size_n - 1) // block_size_n
            )
            total_ctas = (
                total_clusters_max
                * self.cluster_shape_mnk[0]
                * self.cluster_shape_mnk[1]
            )
        else:
            total_ctas = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        if self.pingpong:
            total_ctas *= 2
        # 128 bytes per tensormap
        tensormaps_torch = torch.empty(
            total_ctas, 128 // 8, dtype=torch.int64, device="cuda"
        )
        tensormaps_tensor = from_dlpack(
            tensormaps_torch, assumed_align=128
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        return tensormaps_tensor

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierGemm.MmaWG0 if stage == "mma" else NamedBarrierGemm.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def make_sched_pipeline(
        self, cluster_layout_mnk: cute.Layout, sched_pipeline_mbar_ptr: cute.Pointer
    ):
        # Threads/warps participating in this pipeline
        sched_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_layout_mnk)
        # Each warp that are not the scheduler warp will contribute 1 to the arrive count
        consumer_arrive_cnt = (
            (self.mma_warp_groups if not self.pingpong else 1) * 4
            + max(self.num_load_A_threads // cute.arch.WARP_SIZE, 1)
        ) * cluster_size - 1
        sched_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=sched_pipeline_mbar_ptr,
            num_stages=self.sched_stage,
            producer_group=sched_pipeline_producer_group,
            consumer_group=sched_pipeline_consumer_group,
            # If there's cluster, the consumers must arrive at the mbar of CTA 0 in the cluster.
            consumer_mask=None if const_expr(cluster_size == 1) else 0,
        )

    def _compute_stages(
        self,
        tile_shape_mnk: Tuple[int, int, int],
        initial_d_epi_stage: int,
        d_epi_tile: Optional[Tuple[int, int]],
        c_epi_tile: Optional[Tuple[int, int]],
        y_epi_tile: Optional[Tuple[int, int]],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        c_dtype: Optional[Type[cutlass.Numeric]],
        y_dtype: Optional[Type[cutlass.Numeric]],
        smem_capacity: int,
        occupancy: int,
        overlap_sD_sA: bool,
    ) -> Tuple[int, int, int]:
        d_epi_stage = (
            initial_d_epi_stage
            if const_expr(not self.need_epilogue_load)
            else initial_d_epi_stage // 2
        )
        y_epi_stage = d_epi_stage

        if self.inference_mode and self.need_adhoc_epilogue_store:
            d_epi_stage = 0

        if overlap_sD_sA:
            epi_bytes = 0
        else:
            d_bytes_per_stage = cute.size(d_epi_tile) * d_dtype.width // 8
            epi_bytes = d_bytes_per_stage * d_epi_stage

            if y_dtype is not None or const_expr(self.need_adhoc_epilogue_store):
                y_bytes_per_stage = cute.size(y_epi_tile) * y_dtype.width // 8
                epi_bytes += y_bytes_per_stage * y_epi_stage
            else:
                y_bytes_per_stage = 0

        c_epi_stage = (
            0
            if (c_dtype is None or const_expr(not self.need_epilogue_load))
            else d_epi_stage
        )
        if c_dtype is not None and const_expr(self.need_epilogue_load):
            c_bytes_per_stage = cute.size(c_epi_tile) * c_dtype.width // 8 * c_epi_stage
            epi_bytes += c_bytes_per_stage * c_epi_stage
            d_epi_stage = c_epi_stage
        else:
            c_bytes_per_stage = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        remaining_bytes = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
            - self.prefetch_token_idx_size * 4
            - (
                self.tile_shape_mnk[1] * (self.bias_dtype.width // 8)
                if self.use_bias
                else 0
            )
            - 1024  # aligned  self.tensormap_management_bytes
        )
        ab_stage = remaining_bytes // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        if not overlap_sD_sA:
            if self.inference_mode and self.need_adhoc_epilogue_store:
                epi_stage_delta = (remaining_bytes - ab_bytes_per_stage * ab_stage) // (
                    y_bytes_per_stage + c_bytes_per_stage
                )
                y_epi_stage += epi_stage_delta
            else:
                epi_stage_delta = (remaining_bytes - ab_bytes_per_stage * ab_stage) // (
                    d_bytes_per_stage + y_bytes_per_stage + c_bytes_per_stage
                )
                d_epi_stage += epi_stage_delta
                y_epi_stage += epi_stage_delta

            if c_epi_stage > 0:
                c_epi_stage += epi_stage_delta

        if not self.need_adhoc_epilogue_store:
            y_epi_stage = 0

        return ab_stage, c_epi_stage, d_epi_stage, y_epi_stage

    def _sm90_compute_tile_shape_or_override(
        self,
        tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Type[cutlass.Numeric],
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(self.epi_tile_size, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        elif tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(self.epi_tile_size, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = (
                64
                if element_type.width == 8
                else min(self.epi_tile_size, tile_shape_mnk[1])
            )
            tile_m = math.gcd(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: Tuple[int, int, int],
        c_epi_tile: Tuple[int, int],
        bias_epi_tile: Tuple[int, int],
        d_epi_tile: Tuple[int, int],
        y_epi_tile: Optional[Tuple[int, int]],
        a_dtype: Type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        prefetch_idx_size: Optional[int],
        ab_stage: int,
        c_dtype: Optional[Type[cutlass.Numeric]],
        c_layout: Optional[cutlass.utils.LayoutEnum],
        bias_dtype: Optional[Type[cutlass.Numeric]],
        bias_layout: Optional[cutlass.utils.LayoutEnum],
        d_dtype: Type[cutlass.Numeric],
        d_layout: utils.LayoutEnum,
        y_dtype: Optional[Type[cutlass.Numeric]],
        y_layout: Optional[utils.LayoutEnum],
        s_dtype: Optional[Type[cutlass.Numeric]],
        c_epi_stage: int,
        d_epi_stage: int,
        y_epi_stage: int,
    ) -> Tuple[
        cute.ComposedLayout,
        cute.ComposedLayout,
        Optional[cute.ComposedLayout],
        cute.ComposedLayout,
        Optional[cute.ComposedLayout],
    ]:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K

        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        d_smem_shape = d_epi_tile
        d_major_mode_size = d_epi_tile[1] if d_layout.is_n_major_c() else d_epi_tile[0]
        d_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                d_layout,
                d_dtype,
                d_major_mode_size,
            ),
            d_dtype,
        )
        if d_epi_stage > 0:
            d_epi_smem_layout_staged = cute.tile_to_shape(
                d_smem_layout_atom,
                cute.append(d_smem_shape, d_epi_stage),
                order=(1, 0, 2) if d_layout.is_m_major_c() else (0, 1, 2),
            )
        else:
            # calculating the layout
            d_epi_smem_layout_staged = cute.tile_to_shape(
                d_smem_layout_atom,
                cute.append(d_smem_shape, 1),
                order=(1, 0, 2) if d_layout.is_m_major_c() else (0, 1, 2),
            )

        if y_epi_tile is not None:
            y_smem_shape = y_epi_tile
            # we force `y` to have same major mode as `z`. Otherwise the epilogue write is tricky
            y_major_mode_size = (
                y_epi_tile[1] if y_layout.is_n_major_c() else y_epi_tile[0]
            )
            y_smem_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    y_layout,
                    y_dtype,
                    y_major_mode_size,
                ),
                y_dtype,
            )
            y_epi_smem_layout_staged = cute.tile_to_shape(
                y_smem_layout_atom,
                cute.append(y_smem_shape, y_epi_stage),
                order=(1, 0, 2) if y_layout.is_m_major_c() else (0, 1, 2),
            )
        else:
            y_epi_smem_layout_staged = None

        if c_dtype is not None:
            assert c_layout is not None
            c_smem_shape = c_epi_tile
            c_major_mode_size = (
                c_epi_tile[1] if c_layout.is_n_major_c() else c_epi_tile[0]
            )
            c_smem_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size),
                c_dtype,
            )
            c_epi_smem_layout_staged = cute.tile_to_shape(
                c_smem_layout_atom,
                cute.append(c_smem_shape, c_epi_stage),
                order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
            )
        else:
            c_epi_smem_layout_staged = None

        if (
            bias_dtype is not None
            and bias_layout is not None
            and bias_epi_tile is not None
        ):
            bias_epi_smem_layout_staged = cute.make_layout((1, tile_shape_mnk[1]))
        else:
            bias_epi_smem_layout_staged = None

        if s_dtype is not None:
            s_epi_smem_layout_staged = cute.make_layout((tile_shape_mnk[0],))
        else:
            s_epi_smem_layout_staged = None

        if prefetch_idx_size > 0:
            prefetched_token_idx_smem_layout = cute.make_layout((prefetch_idx_size,))
        else:
            prefetched_token_idx_smem_layout = None

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            c_epi_smem_layout_staged,
            bias_epi_smem_layout_staged,
            d_epi_smem_layout_staged,
            y_epi_smem_layout_staged,
            s_epi_smem_layout_staged,
            prefetched_token_idx_smem_layout,
        )

    @staticmethod
    def _make_tma_epi_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
        store_or_load: str,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for storing D or loading C.

        :param tensor_d: Output tensor D
        :type tensor_d: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        assert store_or_load in ["load", "store"]
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_d.shape), epi_tile
        )
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if store_or_load == "load"
            else cpasync.CopyBulkTensorTileS2GOp()
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            op, tensor_d, epi_smem_layout, d_cta_v_layout
        )
        return tma_atom_d, tma_tensor_d

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    def _make_tiled_copy_2D(
        self,
        tensor: cute.Tensor,
        tile_shape_0: cute.Int32,
        tile_shape_1: cute.Int32,
        is_row_major: bool,
        threads_for_copy: Union[cutlass.Int32, int],
        universal_copy_bits: cutlass.Int32,
        is_g2s: Optional[bool] = True,
    ) -> cute.TiledCopy:
        copy_atom = cute.make_copy_atom(
            (
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
                )
                if const_expr(is_g2s)
                else cute.nvgpu.CopyUniversalOp()
            ),
            tensor.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        copy_elems = universal_copy_bits // tensor.element_type.width
        shape_dim_1 = cute.size(tile_shape_1) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout(
            (threads_for_copy // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if not is_row_major:
            shape_dim_0 = cute.size(tile_shape_0) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, threads_for_copy // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if is_row_major
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # tested a_dtype
        if a_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
        }:
            is_valid = False
        # tested acc_dtype
        if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            is_valid = False
        # tested d_dtype
        if out_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False
        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False
        return is_valid
