# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.
# ruff: noqa: E741 -- preserve names from the vendored CUTLASS example.

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

# This file is ported from the CUTLASS dense block-scaled GEMM example
# and adapted for the current Blackwell GeForce target.

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo

from .cute_utils import (
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)


# Vendored from b12x.cute.fp4 (so this kernel does not depend on the external
# b12x package). Used only by the kernel's opt-in split-K atomic-reduction path.
@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + offset
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def scatter_add_bf16(addr: Int64, val_f32, *, loc=None, ip=None):
    """BF16 atomic reduction add to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 packed; cvt.rn.satfinite.bf16.f32 packed, $1; red.relaxed.gpu.global.add.noftz.bf16 [$0], packed; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def scatter_add_bf16x2(addr: Int64, val0_f32, val1_f32, *, loc=None, ip=None):
    """BF16x2 atomic reduction add to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val0_f32.ir_value(loc=loc, ip=ip),
            val1_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 packed; cvt.rn.satfinite.bf16x2.f32 packed, $2, $1; red.relaxed.gpu.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


logger = logging.getLogger(__name__)
_DENSE_LOAD_PATHS = ("tma", "cpasync")


@dataclass(frozen=True)
class _DenseGemmPlan:
    mma_tiler_mn: Tuple[int, int]
    load_path: Literal["tma", "cpasync"]
    swap_ab: bool


# @dsl_user_op on PersistentTileSchedulerParams.__init__ can rename attributes
# (e.g. raster_along_m -> _raster_along_m, cluster_shape_major_fdd ->
# cluster_shape_m_fdd) but __extract_mlir_values__ (used by TVM-FFI)
# still references the original names.
_orig_extract = utils.PersistentTileSchedulerParams.__extract_mlir_values__

# Map of source-code attr name -> runtime attr name set by @dsl_user_op
_ATTR_RENAMES = {
    "raster_along_m": "_raster_along_m",
    "cluster_shape_major_fdd": "cluster_shape_m_fdd",
    "cluster_shape_minor_fdd": "cluster_shape_n_fdd",
}


def _patched_extract(self):
    for src_name, dst_name in _ATTR_RENAMES.items():
        if not hasattr(self, src_name) and hasattr(self, dst_name):
            setattr(self, src_name, getattr(self, dst_name))
    return _orig_extract(self)


utils.PersistentTileSchedulerParams.__extract_mlir_values__ = _patched_extract


def _convert_layout_acc_mn(
    acc_layout: cute.Layout, transpose: bool = False
) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if cutlass.const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _convert_layout_acc_mn(acc.layout, transpose=transpose)
    )


class DenseGemmKernel:
    """Implements batched matrix multiplication (C = A x SFA x B x SFB) for
    Blackwell GeForce architecture using warp-level MMA.

    Key architectural differences from the tcgen05 donor path:
    - No TMEM, no tcgen05, no 2-CTA instructions, no multi-cluster
    - Warp-level MMA: MmaMXF4NVF4Op atom m16n8k64, atom_layout=(4,2,1)
    - 256 MMA threads + 32 DMA = 288 total threads
    - PipelineTmaAsync (not PipelineTmaUmma)
    - Manual atom unroll workaround for CuTe DSL compiler SF address space bug
    - Cluster shape always (1,1,1)

    Notes:
        - Supported combinations:
            * NVF4: A/B: Float4E2M1FN, SF: Float8E4M3FN, sf_vec_size: 16
            * MXF4: A/B: Float4E2M1FN, SF: Float8E8M0FNU, sf_vec_size: 32
        - Tile shape constraints:
            * tile_m must be divisible by 128
            * tile_n must be divisible by 128
            * tile_k must be divisible by 64 (sf_vec_size=16) or 128 (sf_vec_size=32)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        mma_k: int = 64,
        tile_k: Optional[int] = None,
        single_work_tile_per_cta: bool = False,
        use_prefetch: bool = False,
        enable_pdl: bool = True,
        direct_one_m_tile_scheduler: bool = False,
        split_k_slices: int = 1,
        split_k_atomic_bf16: bool = False,
        use_m1_non_tma_a: bool = False,
        use_m1_non_tma_c: bool = False,
        use_m1_non_tma_sfa: bool = False,
        load_path: Literal["tma", "cpasync"] = "tma",
        swap_ab: bool = False,
        k_loop_unroll: int = 2,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.mma_k = mma_k
        if tile_k is None:
            tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.mma_tile_shape_mnk = (
            (mma_tiler_mn[1], mma_tiler_mn[0], tile_k)
            if swap_ab
            else self.tile_shape_mnk
        )
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.cluster_shape_mnk = (1, 1, 1)  # Always (1,1,1) on the current target
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.single_work_tile_per_cta = single_work_tile_per_cta
        self.use_prefetch = use_prefetch
        self.enable_pdl = enable_pdl
        self.direct_one_m_tile_scheduler = direct_one_m_tile_scheduler
        self.split_k_slices = split_k_slices
        self.split_k_atomic_bf16 = split_k_atomic_bf16
        self.use_m1_non_tma_a = use_m1_non_tma_a
        self.use_m1_non_tma_c = use_m1_non_tma_c
        self.use_m1_non_tma_sfa = use_m1_non_tma_sfa
        self.load_path = load_path
        self.swap_ab = swap_ab
        self.k_loop_unroll = k_loop_unroll
        mma_atom_mn = (self.mma_tile_shape_mnk[0], self.mma_tile_shape_mnk[1])
        if mma_atom_mn in ((16, 64), (16, 128)):
            self.atom_shape = (1, 2, 1)
        elif mma_atom_mn in ((32, 64), (32, 128)):
            self.atom_shape = (2, 2, 1)
        else:
            self.atom_shape = (4, 2, 1)

        self.tiled_mma = None
        self.occupancy = 1
        if mma_atom_mn in ((16, 64), (16, 128)):
            self.num_mma_warps = 2
        elif mma_atom_mn in ((32, 64), (32, 128)):
            self.num_mma_warps = 4
        else:
            self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        # FP4-only target (NVF4 sf_vec_size=16 / MXF4 sf_vec_size=32). The MXFP8
        # warp-MMA path was dropped: FlashInfer only drives this kernel for FP4,
        # and cute.nvgpu.warp.MmaMXF8Op is absent in the public cutlass-dsl build.
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_shape = self.atom_shape
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.mma_tile_shape_mnk,
            self.sf_vec_size,
            False,  # is_mxfp8: FP4-only
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        # Bare atom for manual unroll workaround (avoids hasAuxTensor address space bug)
        self.mma_atom = cute.make_mma_atom(mma_op)
        # Compute atom loop bounds from tile shape and atom/layout shape
        # MMA atom: m16n8k64 for FP4.
        mma_m, mma_n, mma_k = 16, 8, self.mma_k
        self.num_m_tiles = self.mma_tile_shape_mnk[0] // (mma_m * atom_shape[0])
        self.num_n_tiles = self.mma_tile_shape_mnk[1] // (mma_n * atom_shape[1])
        self.num_k_blocks = self.mma_tile_shape_mnk[2] // mma_k

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        assert (
            self.epi_stage > 0
        ), "epi_stage <= 0, not enough shared memory. This configuration will be skipped."

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
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
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A
            b: Input tensor B
            sfa: Scale factor tensor for A
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

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
        if cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_atom_sfa = tma_atom_b
            tma_tensor_sfa = sfa_tensor
        else:
            tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
                sfa_tensor,
                self.sfa_smem_layout_staged,
                self.sfa_tile_shape_mk,
                1,
                internal_type=cutlass.Int16,
            )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
            self.direct_one_m_tile_scheduler,
            self.split_k_slices,
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
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
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
            a,
            tma_atom_b,
            tma_tensor_b,
            b,
            tma_atom_sfa,
            tma_tensor_sfa,
            sfa_tensor,
            tma_atom_sfb,
            tma_tensor_sfb,
            sfb_tensor,
            tma_atom_c,
            tma_tensor_c,
            c,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    def _partition_fragment_SFA(
        self,
        sfa_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        thrfrg_sfa_layout = self._thrfrg_SFA(sfa_tensor.layout, thr_mma)
        thr_tensor = cute.make_tensor(sfa_tensor.iterator, thrfrg_sfa_layout)
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
        partitioned_sfa = thr_tensor[thr_vmk, (None, None)]
        partitioned_sfa = cute.group_modes(cute.flatten(partitioned_sfa), 0, 2)
        return cute.make_fragment_like(partitioned_sfa)

    def _partition_fragment_SFB(
        self,
        sfb_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        thrfrg_sfb_layout = self._thrfrg_SFB(sfb_tensor.layout, thr_mma)
        thr_tensor = cute.make_tensor(sfb_tensor.iterator, thrfrg_sfb_layout)
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
        partitioned_sfb = thr_tensor[thr_vnk, (None, None)]
        partitioned_sfb = cute.group_modes(cute.flatten(partitioned_sfb), 0, 2)
        partitioned_sfb = cute.group_modes(partitioned_sfb, 1, 3)
        return cute.make_fragment_like(partitioned_sfb)

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma: cute.TiledMma):
        assert cute.rank(sfa_tensor) >= 2

        atom_shape_mnk = tiled_mma.shape_mnk
        atom_sfa_layout = cute.make_layout(
            shape=((2, 2, 8), 64), stride=((8, 0, 1), 16)
        )
        permutation_mnk = tiled_mma.permutation_mnk
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        # Reorder the tensor for TiledAtom
        t_tile = (permutation_mnk[0], permutation_mnk[2])
        t_tensor = cute.logical_divide(sfa_tensor, t_tile)

        # Tile the tensor for the Atom
        a_tile = (
            cute.make_layout((atom_shape_mnk[0])),
            cute.make_layout((atom_shape_mnk[2])),
        )
        a_tensor = cute.zipped_divide(t_tensor, a_tile)

        # Transform the Atom mode from (M,K) to (Thr,Val)
        tv_tensor = cute.composition(a_tensor, (atom_sfa_layout, None))

        # Tile the tensor for the Thread
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_layout_vmnk[1])),
                cute.make_layout(cute.size(thr_layout_vmnk[3])),
            ),
        )
        thr_tensor = cute.zipped_divide(tv_tensor, thr_tile)
        return thr_tensor

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma: cute.TiledMma):
        assert cute.rank(sfb_tensor) >= 2

        atom_shape_mnk = tiled_mma.shape_mnk
        atom_sfb_layout = cute.make_layout(shape=((4, 8), 64), stride=((0, 1), 8))
        permutation_mnk = tiled_mma.permutation_mnk
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        # Reorder the tensor for TiledAtom
        t_tile = (permutation_mnk[1], permutation_mnk[2])
        t_tensor = cute.logical_divide(sfb_tensor, t_tile)

        # Tile the tensor for the Atom
        a_tile = (
            cute.make_layout((atom_shape_mnk[1])),
            cute.make_layout((atom_shape_mnk[2])),
        )
        a_tensor = cute.zipped_divide(t_tensor, a_tile)

        # Transform the Atom mode from (N,K) to (Thr,Val)
        tv_tensor = cute.composition(a_tensor, (atom_sfb_layout, None))

        # Tile the tensor for the Thread
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_layout_vmnk[2])),
                cute.make_layout(cute.size(thr_layout_vmnk[3])),
            ),
        )
        thr_tensor = cute.zipped_divide(tv_tensor, thr_tile)
        return thr_tensor

    def _get_layoutSFA_TV(self, tiled_mma: cute.TiledMma):
        if tiled_mma.permutation_mnk is not None:
            perm_m = tiled_mma.permutation_mnk[0]
            perm_k = tiled_mma.permutation_mnk[2]
            tile_m = cute.size(perm_m)
            tile_k = cute.size(perm_k)
        else:
            tile_shape_mnk = tiled_mma.shape_mnk * tiled_mma.thr_layout_vmnk
            tile_m = cute.size(tile_shape_mnk[0])
            tile_k = cute.size(tile_shape_mnk[2])

        ref_A = cute.make_layout((tile_m, tile_k))
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        atile = (
            None,
            (
                cute.make_layout(
                    shape=(
                        cute.size(thr_layout_vmnk[1]),
                        cute.size(thr_layout_vmnk[2]),
                    ),
                    stride=(1, 0),
                ),
                None,
            ),
        )

        thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
        thrfrg_sfa = self._thrfrg_SFA(ref_A, tiled_mma)
        layout_tv_1 = cute.composition(thrfrg_sfa, (atile, None))
        layout_tv = cute.composition(layout_tv_1, (thridx_2_thrid, None))
        return layout_tv

    def _get_layoutSFB_TV(self, tiled_mma: cute.TiledMma):
        if tiled_mma.permutation_mnk is not None:
            perm_n_layout = tiled_mma.permutation_mnk[1]
            perm_k = tiled_mma.permutation_mnk[2]
            tile_n = cute.size(perm_n_layout)
            tile_k = cute.size(perm_k)
        else:
            tile_shape_mnk = tiled_mma.shape_mnk * tiled_mma.thr_layout_vmnk
            tile_n = cute.size(tile_shape_mnk[1])
            tile_k = cute.size(tile_shape_mnk[2])

        ref_B = cute.make_layout((tile_n, tile_k))
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        atile = (
            None,
            (
                cute.make_layout(
                    shape=(
                        cute.size(thr_layout_vmnk[1]),
                        cute.size(thr_layout_vmnk[2]),
                    ),
                    stride=(0, 1),
                ),
                None,
            ),
        )

        thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
        thrfrg_sfb = self._thrfrg_SFB(ref_B, tiled_mma)
        layout_tv = cute.composition(thrfrg_sfb, (atile, None))
        layout_tv = cute.composition(layout_tv, (thridx_2_thrid, None))
        return layout_tv

    @cute.jit
    def _make_cpasync_tiled_copy(
        self,
        dtype: cutlass.Constexpr,
        tile_cols: cutlass.Constexpr[int],
    ) -> cute.TiledCopy:
        copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        async_copy_elems = copy_bits // dtype.width
        t_shape_dim_1 = tile_cols // async_copy_elems
        assert self.num_threads_per_warp % t_shape_dim_1 == 0
        t_layout = cute.make_ordered_layout(
            (self.num_threads_per_warp // t_shape_dim_1, t_shape_dim_1),
            order=(1, 0),
        )
        v_layout = cute.make_layout((1, async_copy_elems))
        return cute.make_tiled_copy_tv(atom_async_copy, t_layout, v_layout)

    @cute.jit
    def _make_scale_tiled_copy(
        self,
        dtype: cutlass.Constexpr,
    ) -> cute.TiledCopy:
        copy_bits = dtype.width
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        return cute.make_tiled_copy_tv(
            atom_async_copy,
            cute.make_layout((self.num_threads_per_warp,)),
            cute.make_layout((copy_bits // dtype.width,)),
        )

    @cute.jit
    def _predicate_cpasync_rows(
        self,
        tCc: cute.Tensor,
        row_limit: Int32,
    ) -> cute.Tensor:
        tPred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    cute.size(tCc, mode=[0, 1]),
                    cute.size(tCc, mode=[1]),
                    cute.size(tCc, mode=[2]),
                ),
                stride=(cute.size(tCc, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tPred.shape[0]):
            for rest_k in cutlass.range_constexpr(tPred.shape[2]):
                tPred[rest_v, 0, rest_k] = tCc[(0, rest_v), 0, rest_k][0] < row_limit
        return tPred

    @cute.jit
    def _cpasync_copy_2d(
        self,
        tiled_copy: cute.TiledCopy,
        tG: cute.Tensor,
        tS: cute.Tensor,
        tC: cute.Tensor,
        row_limit: Int32,
        predicate_rows: cutlass.Constexpr[bool],
    ) -> None:
        if cutlass.const_expr(predicate_rows):
            tP = self._predicate_cpasync_rows(tC, row_limit)
        for rest_m in cutlass.range_constexpr(cute.size(tS.shape[1])):
            if cutlass.const_expr(predicate_rows):
                cute.copy(
                    tiled_copy,
                    tG[None, rest_m, None],
                    tS[None, rest_m, None],
                    pred=tP[None, rest_m, None],
                )
            else:
                cute.copy(
                    tiled_copy,
                    tG[None, rest_m, None],
                    tS[None, rest_m, None],
                )

    @cute.jit
    def _scale_copy_2d(
        self,
        tiled_copy: cute.TiledCopy,
        tG: cute.Tensor,
        tS: cute.Tensor,
        tC: cute.Tensor,
        row_limit: Int32,
    ) -> None:
        tP = cute.make_rmem_tensor(cute.make_layout(tS.shape), cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(tP)):
            tP[i] = cute.elem_less(tC[i][0][0][0], row_limit)
        for rest_m in cutlass.range_constexpr(cute.size(tS.shape[1])):
            cute.copy(
                tiled_copy,
                tG[None, rest_m, None],
                tS[None, rest_m, None],
                pred=tP[None, rest_m, None],
            )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        directA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        directB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        directSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        directSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        directC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        # Keep alpha in FP32 for precision
        alpha_value = alpha[0].to(cutlass.Float32)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            if cutlass.const_expr(
                self.load_path == "tma" and not self.use_m1_non_tma_a
            ):
                cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.load_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(
                self.load_path == "tma" and not self.use_m1_non_tma_sfa
            ):
                cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.load_path == "tma"):
                cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.use_m1_non_tma_c):
                cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        if cutlass.const_expr(self.use_m1_non_tma_sfa):
            tma_copy_bytes = cute.size_in_bytes(
                self.b_dtype, b_smem_layout
            ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            if cutlass.const_expr(not self.use_m1_non_tma_a):
                tma_copy_bytes += cute.size_in_bytes(self.a_dtype, a_smem_layout)
        else:
            tma_copy_bytes = (
                cute.size_in_bytes(self.a_dtype, a_smem_layout)
                + cute.size_in_bytes(self.b_dtype, b_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
                + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
            )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        if cutlass.const_expr(self.load_path == "cpasync"):
            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_threads_per_warp,
            )
            mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps * self.num_threads_per_warp,
            )
            mainloop_pipeline = pipeline.PipelineAsync.create(
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                barrier_storage=mainloop_pipeline_array_ptr,
            )
        else:
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

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
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
        if cutlass.const_expr(not self.use_m1_non_tma_sfa):
            gSFA_mkl = cute.local_tile(
                mSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            self.sfb_tile_shape_nk,
            (None, None, None),
        )
        if cutlass.const_expr(self.load_path == "cpasync"):
            gA_cpasync_mkl = cute.local_tile(
                directA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            gB_cpasync_nkl = cute.local_tile(
                directB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            gSFA_cpasync_mkl = cute.local_tile(
                directSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
            gSFB_cpasync_nkl = cute.local_tile(
                directSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        if cutlass.const_expr(self.load_path == "tma" and not self.use_m1_non_tma_a):
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        if cutlass.const_expr(self.load_path == "tma"):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )

        # TMA partitions for SFA
        if cutlass.const_expr(self.load_path == "tma" and not self.use_m1_non_tma_sfa):
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sSFA, 0, 2),
                cute.group_modes(gSFA_mkl, 0, 2),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        if cutlass.const_expr(self.load_path == "tma"):
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sSFB, 0, 2),
                cute.group_modes(gSFB_nkl, 0, 2),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)

        if cutlass.const_expr(self.load_path == "cpasync"):
            cpasync_tiled_copy_A = self._make_cpasync_tiled_copy(
                self.a_dtype,
                self.tile_shape_mnk[2],
            )
            cpasync_tiled_copy_B = self._make_cpasync_tiled_copy(
                self.b_dtype,
                self.tile_shape_mnk[2],
            )
            cpasync_tiled_copy_SF = self._make_scale_tiled_copy(self.sf_dtype)
            cA_mkl = cute.make_identity_tensor(cute.shape(directA_mkl))
            cA_cpasync_mkl = cute.local_tile(
                cA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            cB_nkl = cute.make_identity_tensor(cute.shape(directB_nkl))
            cB_cpasync_nkl = cute.local_tile(
                cB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            cSFA_mkl = cute.make_identity_tensor(cute.shape(directSFA_mkl))
            cSFA_cpasync_mkl = cute.local_tile(
                cSFA_mkl,
                self.sfa_tile_shape_mk,
                (None, None, None),
            )
            cSFB_nkl = cute.make_identity_tensor(cute.shape(directSFB_nkl))
            cSFB_cpasync_nkl = cute.local_tile(
                cSFB_nkl,
                self.sfb_tile_shape_nk,
                (None, None, None),
            )

            cpasync_lane = tidx % self.num_threads_per_warp
            thr_cpasync_A = cpasync_tiled_copy_A.get_slice(cpasync_lane)
            thr_cpasync_B = cpasync_tiled_copy_B.get_slice(cpasync_lane)
            thr_cpasync_SF = cpasync_tiled_copy_SF.get_slice(cpasync_lane)
            tAgA_cpasync_mkl = thr_cpasync_A.partition_S(gA_cpasync_mkl)
            tAsA_cpasync = thr_cpasync_A.partition_D(sA)
            tAcA_cpasync_mkl = thr_cpasync_A.partition_S(cA_cpasync_mkl)
            tBgB_cpasync_nkl = thr_cpasync_B.partition_S(gB_cpasync_nkl)
            tBsB_cpasync = thr_cpasync_B.partition_D(sB)
            tBcB_cpasync_nkl = thr_cpasync_B.partition_S(cB_cpasync_nkl)
            tAgSFA_cpasync_mkl = thr_cpasync_SF.partition_S(gSFA_cpasync_mkl)
            tAsSFA_cpasync = thr_cpasync_SF.partition_D(sSFA)
            tAcSFA_cpasync_mkl = thr_cpasync_SF.partition_S(cSFA_cpasync_mkl)
            tBgSFB_cpasync_nkl = thr_cpasync_SF.partition_S(gSFB_cpasync_nkl)
            tBsSFB_cpasync = thr_cpasync_SF.partition_D(sSFB)
            tBcSFB_cpasync_nkl = thr_cpasync_SF.partition_S(cSFB_cpasync_nkl)

        # Make fragments. swap_ab keeps public C[M,N] unchanged but presents
        # B as MMA-A and A as MMA-B.
        if cutlass.const_expr(self.swap_ab):
            tCsA = thr_mma.partition_A(sB)
            tCsB = thr_mma.partition_B(sA)
        else:
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        if cutlass.const_expr(self.swap_ab):
            tCrSFA_full = self._partition_fragment_SFA(
                sSFB[None, None, 0], thr_mma, tidx
            )
            tCrSFB_full = self._partition_fragment_SFB(
                sSFA[None, None, 0], thr_mma, tidx
            )
            c_mma = cute.make_identity_tensor(
                (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
            )
            tCgC = thr_mma.partition_C(c_mma)
        else:
            tCrSFA_full = self._partition_fragment_SFA(
                sSFA[None, None, 0], thr_mma, tidx
            )
            tCrSFB_full = self._partition_fragment_SFB(
                sSFB[None, None, 0], thr_mma, tidx
            )
            tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_wait()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        block_idx = cute.arch.block_idx()
        k_tile_start = Int32(0)
        k_tile_iter_cnt = k_tile_cnt
        if cutlass.const_expr(self.split_k_slices > 1):
            k_tiles_per_split = k_tile_cnt // self.split_k_slices
            k_tile_start = Int32(block_idx[1]) * Int32(k_tiles_per_split)
            k_tile_iter_cnt = k_tiles_per_split

        # Tile scheduler
        if cutlass.const_expr(self.direct_one_m_tile_scheduler):
            direct_tile_valid = Int32(block_idx[2]) < Int32(
                tile_sched_params.problem_shape_ntile_mnl[1]
            )
            work_tile = WorkTileInfo(
                (Int32(0), Int32(block_idx[2]), Int32(0)),
                direct_tile_valid,
            )
        else:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, block_idx, cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            if cutlass.const_expr(self.swap_ab):
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                    self.b_dtype,
                )
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                    self.a_dtype,
                )
            else:
                atom_copy_ldmatrix_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                    self.a_dtype,
                )
                atom_copy_ldmatrix_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                    self.b_dtype,
                )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(
                sB if cutlass.const_expr(self.swap_ab) else sA
            )
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(
                sA if cutlass.const_expr(self.swap_ab) else sB
            )
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(
                sSFB if cutlass.const_expr(self.swap_ab) else sSFA
            )
            tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(tCrSFA_full)
            tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(
                sSFA if cutlass.const_expr(self.swap_ab) else sSFB
            )
            tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
                sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
                if cutlass.const_expr(self.swap_ab):
                    if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                        sSFB_tile = cute.local_tile(
                            sSFB,
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (sfb_tile_offset, 0, None),
                        )
                        tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(
                            sSFB_tile
                        )
                        tCrSFA_tile = self._partition_fragment_SFA(
                            sSFB_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(
                            tCrSFA_tile
                        )
                    else:
                        tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                        tCrSFA_tile = tCrSFA_full
                        tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                    if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                        sSFA_tile = cute.local_tile(
                            sSFA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (sfa_tile_offset, 0, None),
                        )
                        tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(
                            sSFA_tile
                        )
                        tCrSFB_tile = self._partition_fragment_SFB(
                            sSFA_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(
                            tCrSFB_tile
                        )
                    else:
                        tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                        tCrSFB_tile = tCrSFB_full
                        tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                else:
                    if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                        sSFA_tile = cute.local_tile(
                            sSFA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (sfa_tile_offset, 0, None),
                        )
                        tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(
                            sSFA_tile
                        )
                        tCrSFA_tile = self._partition_fragment_SFA(
                            sSFA_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(
                            tCrSFA_tile
                        )
                    else:
                        tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                        tCrSFA_tile = tCrSFA_full
                        tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                    if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                        sSFB_tile = cute.local_tile(
                            sSFB,
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (sfb_tile_offset, 0, None),
                        )
                        tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(
                            sSFB_tile
                        )
                        tCrSFB_tile = self._partition_fragment_SFB(
                            sSFB_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(
                            tCrSFB_tile
                        )
                    else:
                        tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                        tCrSFB_tile = tCrSFB_full
                        tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_iter_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
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

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                # The fragments hold a full stage of scale factors, so copy
                # them once per stage.
                cute.copy(
                    smem_tiled_copy_SFA,
                    tCsSFA_p_filtered,
                    tCrSFA_copy_view_filtered,
                )
                cute.copy(
                    smem_tiled_copy_SFB,
                    tCsSFB_p_filtered,
                    tCrSFB_copy_view_filtered,
                )

                for _k_tile in range(
                    0,
                    k_tile_iter_cnt - 1,
                    1,
                    unroll=self.k_loop_unroll,
                ):  # type: ignore[call-overload]
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Manual atom unroll: avoids hasAuxTensor address space bug
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    accumulators[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    accumulators[None, _mt, _nt],
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

                        if k_block_idx == num_k_blocks - 1:
                            # The MMAs have already read the old scale
                            # factors, so the new stage can overwrite them.
                            tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                            tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                            tCrSFA_copy_view_filtered = cute.filter_zeros(
                                tCrSFA_tile_copy_view
                            )
                            tCrSFB_copy_view_filtered = cute.filter_zeros(
                                tCrSFB_tile_copy_view
                            )
                            cute.copy(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered,
                                tCrSFA_copy_view_filtered,
                            )
                            cute.copy(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered,
                                tCrSFB_copy_view_filtered,
                            )

                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
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
                    # Manual atom unroll: avoids hasAuxTensor address space bug
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_tile[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_tile[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                accumulators[None, _mt, _nt],
                                tCrA[None, _mt, k_block_idx],
                                tCrB[None, _nt, k_block_idx],
                                accumulators[None, _mt, _nt],
                            )

                if cutlass.const_expr(self.swap_ab):
                    acc_mn = _reshape_acc_to_mn(accumulators, transpose=True)
                    c_identity = cute.make_identity_tensor(
                        (self.tile_shape_mnk[1], self.tile_shape_mnk[0])
                    )
                    coord_mn = _reshape_acc_to_mn(
                        thr_mma.partition_C(c_identity),
                        transpose=True,
                    )
                    for acc_m in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
                        for acc_n in cutlass.range_constexpr(
                            cute.size(acc_mn.shape[1])
                        ):
                            coord = coord_mn[acc_m, acc_n]
                            m_coord = (
                                tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0])
                                + coord[1]
                            )
                            n_coord = (
                                tile_coord_mnl[1] * Int32(self.tile_shape_mnk[1])
                                + coord[0]
                            )
                            if m_coord < Int32(
                                directC_mnl.shape[0]
                            ) and n_coord < Int32(directC_mnl.shape[1]):
                                directC_mnl[
                                    (
                                        m_coord,
                                        n_coord,
                                        tile_coord_mnl[2],
                                    )
                                ] = epilogue_op(
                                    (alpha_value * acc_mn[acc_m, acc_n]).to(
                                        self.c_dtype
                                    )
                                )
                    if cutlass.const_expr(self.single_work_tile_per_cta):
                        work_tile = WorkTileInfo(
                            work_tile.tile_idx,
                            cutlass.Boolean(0),
                        )
                    else:
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()

                if cutlass.const_expr(not self.swap_ab):
                    # EPILOGUE
                    _is_m_major = self.c_layout.is_m_major_c()
                    if cutlass.const_expr(self.c_dtype.width == 16):
                        copy_atom_r2s = cute.make_copy_atom(
                            cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                            self.c_dtype,
                        )
                    else:
                        copy_atom_r2s = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.c_dtype,
                        )

                    if cutlass.const_expr(self.c_dtype.width == 16):
                        copy_atom_C = cute.make_copy_atom(
                            cute.nvgpu.warp.StMatrix8x8x16bOp(
                                self.c_layout.is_m_major_c(),
                                2,
                            ),
                            self.c_dtype,
                        )
                    else:
                        copy_atom_C = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), self.c_dtype
                        )

                    tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(
                        copy_atom_C, tiled_mma
                    )

                    tiled_copy_r2s = cute.make_tiled_copy_S(
                        copy_atom_r2s,
                        tiled_copy_C_Atom,
                    )

                    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                    tRS_sD = thr_copy_r2s.partition_D(sC)
                    tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                    tRS_rD_layout = cute.make_layout(rD_shape[:3])
                    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                    sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                    tcgc_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )

                    bSG_sD, bSG_gD = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sepi_for_tma_partition,
                        tcgc_for_tma_partition,
                    )

                    epi_rest_m = bSG_gD.shape[1][0]
                    epi_rest_n = bSG_gD.shape[1][1]
                    epi_tile_m = self.epi_tile[0]
                    epi_tile_n = self.epi_tile[1]
                    mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                    mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                    has_multi_epi_store = cutlass.const_expr(
                        not (
                            self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1
                        )
                    )
                    tma_store_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    )
                    tma_store_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.epi_stage,
                        producer_group=tma_store_producer_group,
                    )

                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        for epi_n in cutlass.range_constexpr(epi_rest_n):
                            MmaMPerEpiM = epi_tile_m // mma_tile_m
                            MmaNPerEpiN = epi_tile_n // mma_tile_n
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
                                    mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                    mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                    tRS_rD_slice = tRS_rD[
                                        (None, mma_m_in_epi, mma_n_in_epi)
                                    ]
                                    tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        tRS_rD_slice[elem_idx] = tRS_rAcc_slice[
                                            elem_idx
                                        ]

                            gmem_coord = (epi_m, epi_n)
                            if cutlass.const_expr(self.split_k_slices > 1):
                                acc_mn = _reshape_acc_to_mn(accumulators)
                                c_identity = cute.make_identity_tensor(
                                    cute.slice_(self.tile_shape_mnk, (None, None, 0))
                                )
                                coord_mn = _reshape_acc_to_mn(
                                    thr_mma.partition_C(c_identity)
                                )
                                if cutlass.const_expr(self.split_k_atomic_bf16):
                                    for acc_m in cutlass.range_constexpr(
                                        cute.size(acc_mn.shape[0])
                                    ):
                                        for acc_n_pair in cutlass.range_constexpr(
                                            cute.size(acc_mn.shape[1]) // 2
                                        ):
                                            acc_n0 = acc_n_pair * 2
                                            acc_n1 = acc_n0 + 1
                                            coord0 = coord_mn[acc_m, acc_n0]
                                            coord1 = coord_mn[acc_m, acc_n1]
                                            m_coord0 = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord0[0]
                                            )
                                            n_coord0 = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord0[1]
                                            )
                                            m_coord1 = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord1[0]
                                            )
                                            n_coord1 = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord1[1]
                                            )
                                            if (
                                                m_coord0 < Int32(directC_mnl.shape[0])
                                                and m_coord1
                                                < Int32(directC_mnl.shape[0])
                                                and n_coord0
                                                < Int32(directC_mnl.shape[1])
                                                and n_coord1
                                                < Int32(directC_mnl.shape[1])
                                            ):
                                                c_offset = cute.crd2idx(
                                                    (
                                                        m_coord0,
                                                        n_coord0,
                                                        Int32(0),
                                                    ),
                                                    directC_mnl.layout,
                                                )
                                                scatter_add_bf16x2(
                                                    get_ptr_as_int64(
                                                        directC_mnl,
                                                        c_offset,
                                                    ),
                                                    alpha_value * acc_mn[acc_m, acc_n0],
                                                    alpha_value * acc_mn[acc_m, acc_n1],
                                                )
                                        if cutlass.const_expr(
                                            cute.size(acc_mn.shape[1]) % 2 == 1
                                        ):
                                            acc_n = cute.size(acc_mn.shape[1]) - 1
                                            coord = coord_mn[acc_m, acc_n]
                                            m_coord = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord[0]
                                            )
                                            n_coord = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord[1]
                                            )
                                            if m_coord < Int32(
                                                directC_mnl.shape[0]
                                            ) and n_coord < Int32(directC_mnl.shape[1]):
                                                c_offset = cute.crd2idx(
                                                    (
                                                        m_coord,
                                                        n_coord,
                                                        Int32(0),
                                                    ),
                                                    directC_mnl.layout,
                                                )
                                                scatter_add_bf16(
                                                    get_ptr_as_int64(
                                                        directC_mnl,
                                                        c_offset,
                                                    ),
                                                    alpha_value * acc_mn[acc_m, acc_n],
                                                )
                                else:
                                    split_idx = Int32(block_idx[1])
                                    for acc_m in cutlass.range_constexpr(
                                        cute.size(acc_mn.shape[0])
                                    ):
                                        for acc_n in cutlass.range_constexpr(
                                            cute.size(acc_mn.shape[1])
                                        ):
                                            coord = coord_mn[acc_m, acc_n]
                                            m_coord = (
                                                tile_coord_mnl[0]
                                                * Int32(self.tile_shape_mnk[0])
                                                + coord[0]
                                            )
                                            n_coord = (
                                                tile_coord_mnl[1]
                                                * Int32(self.tile_shape_mnk[1])
                                                + coord[1]
                                            )
                                            if m_coord < Int32(
                                                directC_mnl.shape[0]
                                            ) and n_coord < Int32(directC_mnl.shape[1]):
                                                directC_mnl[
                                                    (m_coord, n_coord, split_idx)
                                                ] = (alpha_value * acc_mn[acc_m, acc_n])
                            else:
                                # Type conversion with alpha scaling
                                tRS_rD_out = cute.make_rmem_tensor(
                                    tRS_rD_layout.shape, self.c_dtype
                                )
                                acc_vec = tRS_rD.load()
                                # Multiply alpha in FP32 before converting to c_dtype
                                # to avoid overflow when c_dtype is FP16
                                acc_vec = epilogue_op(
                                    (alpha_value * acc_vec).to(self.c_dtype)
                                )
                                tRS_rD_out.store(acc_vec)

                                # Register to shared memory
                                epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                                    tRS_sD, mode=[3]
                                )
                                if has_multi_epi_store:
                                    self.epilog_sync_barrier.arrive_and_wait()
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                self.epilog_sync_barrier.arrive_and_wait()

                                # Copy from shared memory to global memory
                                if cutlass.const_expr(self.use_m1_non_tma_c):
                                    for n_iter in cutlass.range_constexpr(
                                        (
                                            self.epi_tile[1]
                                            + self.num_mma_warps
                                            * self.num_threads_per_warp
                                            - 1
                                        )
                                        // (
                                            self.num_mma_warps
                                            * self.num_threads_per_warp
                                        )
                                    ):
                                        n_local = Int32(tidx) + Int32(
                                            n_iter
                                            * self.num_mma_warps
                                            * self.num_threads_per_warp
                                        )
                                        n_coord = (
                                            tile_coord_mnl[1]
                                            * Int32(self.tile_shape_mnk[1])
                                            + Int32(epi_n * self.epi_tile[1])
                                            + n_local
                                        )
                                        if n_local < Int32(
                                            self.epi_tile[1]
                                        ) and n_coord < Int32(directC_mnl.shape[1]):
                                            directC_mnl[
                                                (
                                                    Int32(0),
                                                    n_coord,
                                                    tile_coord_mnl[2],
                                                )
                                            ] = sC[(Int32(0), n_local, epi_buffer)]
                                else:
                                    if warp_idx == 0:
                                        cute.copy(
                                            tma_atom_c,
                                            bSG_sD[(None, epi_buffer)],
                                            bSG_gD[(None, gmem_coord)],
                                        )
                                        if has_multi_epi_store:
                                            tma_store_pipeline.producer_commit()
                                            tma_store_pipeline.producer_acquire()

                    # Advance to the next work tile
                    if cutlass.const_expr(self.single_work_tile_per_cta):
                        work_tile = WorkTileInfo(
                            work_tile.tile_idx,
                            cutlass.Boolean(0),
                        )
                    else:
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()
                    if has_multi_epi_store and cutlass.const_expr(
                        self.split_k_slices == 1
                    ):
                        tma_store_pipeline.producer_tail()

        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(
                    self.load_path == "tma" and not self.use_m1_non_tma_a
                ):
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                if cutlass.const_expr(self.load_path == "tma"):
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                if cutlass.const_expr(
                    self.load_path == "tma" and not self.use_m1_non_tma_sfa
                ):
                    sfa_tile_coord_m = tile_coord_mnl[0] // self.sfa_tiles_per_block
                    tAgSFA_mkl = tAgSFA[
                        (None, sfa_tile_coord_m, None, tile_coord_mnl[2])
                    ]
                if cutlass.const_expr(self.load_path == "tma"):
                    sfb_tile_coord_n = tile_coord_mnl[1] // self.sfb_tiles_per_block
                    tBgSFB_nkl = tBgSFB[
                        (None, sfb_tile_coord_n, None, tile_coord_mnl[2])
                    ]
                if cutlass.const_expr(self.load_path == "cpasync"):
                    cpasync_sfa_tile_coord_m = (
                        tile_coord_mnl[0] // self.sfa_tiles_per_block
                    )
                    cpasync_sfb_tile_coord_n = (
                        tile_coord_mnl[1] // self.sfb_tiles_per_block
                    )

                mainloop_producer_state.reset_count()

                for _k_tile in range(
                    0,
                    k_tile_iter_cnt,
                    1,
                    unroll=self.k_loop_unroll,
                ):  # type: ignore[call-overload]
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    k_tile_global = k_tile_start + mainloop_producer_state.count
                    if cutlass.const_expr(self.load_path == "tma"):
                        tBgB_k = tBgB_nkl[(None, k_tile_global)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                        if cutlass.const_expr(not self.use_m1_non_tma_a):
                            tAgA_k = tAgA_mkl[(None, k_tile_global)]
                            tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                            tAgSFA_k = tAgSFA_mkl[(None, k_tile_global)]
                            tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                        tBgSFB_k = tBgSFB_nkl[(None, k_tile_global)]
                        tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    if cutlass.const_expr(self.load_path == "cpasync"):
                        tAgA_cpasync_k = tAgA_cpasync_mkl[
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[0],
                                k_tile_global,
                                tile_coord_mnl[2],
                            )
                        ]
                        tAsA_cpasync_pipe = tAsA_cpasync[
                            (None, None, None, mainloop_producer_state.index)
                        ]
                        tAcA_cpasync_k = cute.slice_(
                            tAcA_cpasync_mkl,
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[0],
                                k_tile_global,
                                tile_coord_mnl[2],
                            ),
                        )
                        tBgB_cpasync_k = tBgB_cpasync_nkl[
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[1],
                                k_tile_global,
                                tile_coord_mnl[2],
                            )
                        ]
                        tBsB_cpasync_pipe = tBsB_cpasync[
                            (None, None, None, mainloop_producer_state.index)
                        ]
                        tBcB_cpasync_k = cute.slice_(
                            tBcB_cpasync_nkl,
                            (
                                None,
                                None,
                                None,
                                tile_coord_mnl[1],
                                k_tile_global,
                                tile_coord_mnl[2],
                            ),
                        )
                        tAgSFA_cpasync_k = cute.filter_zeros(
                            tAgSFA_cpasync_mkl[
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfa_tile_coord_m,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                )
                            ]
                        )
                        tAsSFA_cpasync_pipe = cute.filter_zeros(
                            tAsSFA_cpasync[
                                (None, None, None, mainloop_producer_state.index)
                            ]
                        )
                        tAcSFA_cpasync_k = cute.filter_zeros(
                            cute.slice_(
                                tAcSFA_cpasync_mkl,
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfa_tile_coord_m,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                ),
                            )
                        )
                        tBgSFB_cpasync_k = cute.filter_zeros(
                            tBgSFB_cpasync_nkl[
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfb_tile_coord_n,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                )
                            ]
                        )
                        tBsSFB_cpasync_pipe = cute.filter_zeros(
                            tBsSFB_cpasync[
                                (None, None, None, mainloop_producer_state.index)
                            ]
                        )
                        tBcSFB_cpasync_k = cute.filter_zeros(
                            cute.slice_(
                                tBcSFB_cpasync_nkl,
                                (
                                    None,
                                    None,
                                    None,
                                    cpasync_sfb_tile_coord_n,
                                    k_tile_global,
                                    tile_coord_mnl[2],
                                ),
                            )
                        )
                        self._cpasync_copy_2d(
                            cpasync_tiled_copy_A,
                            tAgA_cpasync_k,
                            tAsA_cpasync_pipe,
                            tAcA_cpasync_k,
                            Int32(directA_mkl.shape[0]),
                            True,
                        )
                        self._cpasync_copy_2d(
                            cpasync_tiled_copy_B,
                            tBgB_cpasync_k,
                            tBsB_cpasync_pipe,
                            tBcB_cpasync_k,
                            Int32(directC_mnl.shape[1]),
                            True,
                        )
                        self._scale_copy_2d(
                            cpasync_tiled_copy_SF,
                            tAgSFA_cpasync_k,
                            tAsSFA_cpasync_pipe,
                            tAcSFA_cpasync_k,
                            Int32(directA_mkl.shape[0]),
                        )
                        self._scale_copy_2d(
                            cpasync_tiled_copy_SF,
                            tBgSFB_cpasync_k,
                            tBsSFB_cpasync_pipe,
                            tBcSFB_cpasync_k,
                            Int32(directC_mnl.shape[1]),
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                    elif cutlass.const_expr(self.use_m1_non_tma_a):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        for a_iter in cutlass.range_constexpr(
                            (self.tile_shape_mnk[2] + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            k_local = lane + Int32(a_iter * self.num_threads_per_warp)
                            if k_local < Int32(self.tile_shape_mnk[2]):
                                k_coord = (
                                    k_tile_global * Int32(self.tile_shape_mnk[2])
                                    + k_local
                                )
                                sA[
                                    (
                                        Int32(0),
                                        k_local,
                                        mainloop_producer_state.index,
                                    )
                                ] = directA_mkl[
                                    (
                                        Int32(0),
                                        k_coord,
                                        tile_coord_mnl[2],
                                    )
                                ]
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )

                    if cutlass.const_expr(self.load_path == "cpasync"):
                        pass
                    elif cutlass.const_expr(self.use_m1_non_tma_sfa):
                        lane = Int32(tidx % self.num_threads_per_warp)
                        scale_groups_per_k_tile = (
                            self.tile_shape_mnk[2] // self.sf_vec_size
                        )
                        sfa_slots = self.sfa_tile_shape_mk[0] * scale_groups_per_k_tile
                        for sfa_iter in cutlass.range_constexpr(
                            (sfa_slots + self.num_threads_per_warp - 1)
                            // self.num_threads_per_warp
                        ):
                            linear = lane + Int32(sfa_iter * self.num_threads_per_warp)
                            m_local = linear // Int32(scale_groups_per_k_tile)
                            scale_group = linear - m_local * Int32(
                                scale_groups_per_k_tile
                            )
                            k_local_sfa = scale_group * Int32(self.sf_vec_size)
                            k_coord_sfa = (
                                k_tile_global * Int32(self.tile_shape_mnk[2])
                                + k_local_sfa
                            )
                            if linear < Int32(sfa_slots):
                                sSFA[
                                    (
                                        m_local,
                                        k_local_sfa,
                                        mainloop_producer_state.index,
                                    )
                                ] = directSFA_mkl[
                                    (
                                        Int32(0),
                                        k_coord_sfa,
                                        tile_coord_mnl[2],
                                    )
                                ]
                        cute.arch.fence_proxy("async.shared", space="cta")
                    else:
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_k,
                            tAsSFA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                    if cutlass.const_expr(self.load_path == "tma"):
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_k,
                            tBsSFB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                    if cutlass.const_expr(self.load_path == "cpasync"):
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                if cutlass.const_expr(self.single_work_tile_per_cta):
                    work_tile = WorkTileInfo(
                        work_tile.tile_idx,
                        cutlass.Boolean(0),
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if cutlass.const_expr(self.enable_pdl):
            griddepcontrol_launch_dependents()
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        sf_dtype,
        sfa_smem_layout,
        sfb_smem_layout,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )
        epi_stage = min(epi_stage_max, 4)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        raw_ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        ab_stage = max(1, min(raw_ab_stage, 4))
        if tile_shape_mnk[0] in (16, 64) and tile_shape_mnk[1] == 128:
            ab_stage = max(1, min(raw_ab_stage, 5))
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
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
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
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

        sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )
        sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
        direct_one_m_tile_scheduler: bool,
        split_k_slices: int,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        if cutlass.const_expr(split_k_slices > 1):
            grid = (1, split_k_slices, num_ctas_mnl[1])
        else:
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c,
        epi_smem_layout_staged,
        epi_tile: tuple,
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        smem_tile: tuple,
        mcast_dim: int,
        internal_type=None,
    ) -> tuple:
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
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size: int,
        c_dtype,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        *,
        load_path: str = "tma",
        swap_ab: bool = False,
    ) -> bool:
        # The current target only supports cluster (1,1)
        if cluster_shape_mn != (1, 1):
            return False
        if load_path not in _DENSE_LOAD_PATHS:
            return False
        # FP4-only target (NVF4/MXF4). The MXFP8 warp-MMA path was dropped.
        if ab_dtype != cutlass.Float4E2M1FN:
            return False
        if swap_ab:
            if l != 1:
                return False
            if sf_vec_size != 16:
                return False
        if load_path == "cpasync" and (sf_vec_size != 16 or l != 1):
            return False
        # FP4 experiments allow narrow N tiles. The scale-factor smem paths
        # still allocate full 128-element SF blocks, but the live MMA tile may
        # consume only 16/32 columns.
        if (
            mma_tiler_mn[0] % 64 != 0
            or mma_tiler_mn[1] % 16 != 0
            or mma_tiler_mn[1] > 128
            or (mma_tiler_mn[1] < 64 and not swap_ab)
        ):
            return False
        # Current target MMA constraints:
        #   sf_vec_size=16 requires sf_dtype=Float8E4M3FN
        #   sf_vec_size=32 requires sf_dtype=Float8E8M0FNU
        if sf_vec_size == 16 and sf_dtype != cutlass.Float8E4M3FN:
            return False
        if sf_vec_size == 32 and sf_dtype != cutlass.Float8E8M0FNU:
            return False
        # Public output is 16-bit; split-K internally uses FP32 partial output.
        if c_dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        # A must be K-major, B must be K-major
        if a_major != "k" or b_major != "k":
            return False
        # K floor is 32 (TMA assumed_align=16 on K-major packed FP4), not tile_k: the
        # mainloop predicates the partial tile, so ragged K works. Mirror gemm_base.py.
        if k % 32 != 0:
            return False
        return True

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
    ):
        """Wrapper matching the SM100 compile interface."""
        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(
            mA.element_type == cutlass.Uint8 and mB.element_type == cutlass.Uint8
        ):
            k = k_raw * 2
            a_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
            b_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        elif cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError("Unsupported mixed input dtypes for block-scaled GEMM.")
        else:
            k = k_raw
            a_ptr = mA.iterator
            b_ptr = mB.iterator

        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        # C is always row-major (m, n): b12x swap_ab is device-internal (the
        # epilogue writes logical [m, n]), so this `swap_ab` arg is unused here.
        c_tensor = cute.make_tensor(
            mC.iterator,
            layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
        )
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


_ALPHA_ONE_CACHE: dict = {}


def _select_default_mma_tiler_mn(
    m: int,
    n: int,
    sm_count: int,
    *,
    expected_m: Optional[int] = None,
    k: Optional[int] = None,
) -> Tuple[int, int]:
    # FP4-only tile selector. The MXFP8 regimes (16x64/16x128/32x128 decode
    # tiles, narrow-N 64x64) were dropped along with the MXFP8 warp-MMA path.
    coarse_tile = (128, 128)
    plan_m = expected_m if expected_m is not None else m
    if plan_m == 1 and k is not None:
        # Flushed M=1 FP4 probe (benchmarks/probe_dense_fp4_tile_load_sweep.py)
        # across the repo's common shapes:
        #   * wide/medium N: (64,128)/TMA has the best geomean and wins nearly all
        #     shapes.
        #   * N=1024,K=5376: (64,64)/TMA wins the boundary by a small margin.
        #   * N<=512 with long K: (64,32)/TMA+swap_ab is the only clear tiny-N win.
        # Keep the tile selector tile-only; the launch planner below attaches
        # swap_ab to the narrow tile.
        if n <= 512 and k >= 4096:
            return (64, 32)
        if n <= 1024:
            return (64, 64)
        return (64, 128)

    coarse_tiles = ((m + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # The coarse CTA-count heuristic misses exact-small-M, wide-N cases: a wide
    # N dimension can generate plenty of CTAs even while each 128-row M tile is
    # mostly empty. Keep using the narrower 64x128 tile while the 128x128 plan
    # still leaves the GPU below the existing half-SM occupancy proxy.
    if n > 1536:
        if m <= 64:
            return (64, 128)
        if m <= 256 and coarse_tiles < max(1, sm_count // 2):
            return (64, 128)
    if m <= 128 and coarse_tiles < max(1, sm_count // 2):
        if n > 1536:
            return (64, 128)
        medium_tile = (128, 64)
        medium_tiles = ((m + medium_tile[0] - 1) // medium_tile[0]) * (
            (n + medium_tile[1] - 1) // medium_tile[1]
        )
        if medium_tiles < max(1, sm_count // 2):
            return (64, 64)
        return (128, 64)
    return coarse_tile


def _select_default_dense_gemm_plan(
    m: int,
    n: int,
    k: int,
    sm_count: int,
    *,
    expected_m: Optional[int] = None,
) -> _DenseGemmPlan:
    tile = _select_default_mma_tiler_mn(
        m,
        n,
        sm_count,
        expected_m=expected_m,
        k=k,
    )
    return _DenseGemmPlan(
        mma_tiler_mn=tile,
        load_path="tma",
        swap_ab=(tile[1] < 64),
    )


# Alias for FlashInfer integration
Sm120B12xBlockScaledDenseGemmKernel = DenseGemmKernel
