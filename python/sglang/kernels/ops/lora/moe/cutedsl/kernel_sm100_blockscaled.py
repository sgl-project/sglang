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

"""Persistent block-scaled GEMM (MXFP8, MXFP4, NVFP4) for masked slabs or
aligned flat expert segments.

Physical tensors are A [E, M, K], B [E, N, K], and C [E, M, N]; FP4 operands
pack two elements per byte along K. SFA and SFB carry one scale per
``sf_vec_size`` elements along K in the tcgen05 atom order
``(32, 4, M/128, 4, K/(4*sf_vec_size))`` of 512-byte blocks, so every row tile
of 128 owns whole atoms. Masking and flat segments follow kernel_sm100_bf16.

NVFP4 checkpoints carry per-expert global scales outside the block scales;
``alpha`` [E, halves] multiplies each output tile in the epilogue, selecting
the half by the weight row (gate or up rows of a fused W13). Per-token NVFP4
activations add ``tok_scale`` [rows, E | 1]: each accumulator element is
multiplied by alpha and then by its token's stored scale, two products in
that order (alpha times the scale is never formed, so a subnormal scale
cannot erase an output, and the accumulator times a scale near the BF16
maximum cannot overflow before alpha is applied).

The epilogue LoRA (``lora_k``) adds one BF16 rank-``lora_k`` MMA per adapter
slot after the K loop; the planes carry the raw BF16 bridge. With
``lora_acc="shared"`` it accumulates into the base accumulator, which is the
sum only where nothing scales the accumulator afterwards (the MX builds).
With ``lora_acc="separate"`` (the NVFP4 builds, whose epilogue applies alpha)
it accumulates into a second fp32 TMEM accumulator D that the epilogue adds
last: ``C * alpha + D`` with static activation scales and
``(C * alpha) * scale + D`` with per-token ones. No reciprocal of alpha is
ever formed, so an expert whose alpha is zero or subnormal keeps a finite
bridge and a finite sum.

Mainloop and scale-factor plumbing derive from CUTLASS v4.8.0
blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py.
"""

from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from sglang.kernels.ops.lora.moe.cutedsl.scheduler import (
    MoESchedulerParams,
    create_moe_tile_scheduler,
    resolve_scheduler_params_and_grid,
)


@cute.jit
def _epilogue_store_live(
    gemm_kernel,
    epi_tidx: cutlass.Int32,
    warp_idx: cutlass.Int32,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    stores_issued: cutlass.Int32,
    live_cols: cutlass.Int32,
    epilogue_op: cutlass.Constexpr,
    mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
    sColScale: cute.Tensor,
    col_alpha: cutlass.Float32,
    use_col_scale: cutlass.Constexpr = False,
    separate_acc=None,
) -> Tuple[pipeline.PipelineState, cutlass.Int32]:
    """cutlass.utils.gemm.sm100.epilogue_tma_store, storing only the subtiles
    whose first column lies below ``live_cols``; ``stores_issued`` counts the
    stores so the C staging buffers rotate over stored subtiles only.
    ``epilogue_op`` runs on the FP32 accumulator, before the C conversion.
    ``separate_acc`` = (the LoRA accumulator D, its pipeline and consumer
    state, the tile's has-LoRA flag) or ``use_col_scale`` scale the base
    accumulator here, element by element: by ``col_alpha`` (the tile's
    alpha) and then, with ``use_col_scale``, by ``sColScale`` at its column
    (the token's stored per-token scale); the scaled elements of a tile with
    LoRA get their D added last. Each product and the sum round to fp32
    separately."""
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    if cutlass.const_expr(use_col_scale or separate_acc is not None):
        tTR_rAcc_flat = cute.group_modes(tTR_rAcc, 0, cute.rank(tTR_rAcc))
        frag_size = cute.size(tTR_rAcc.shape)
    if cutlass.const_expr(use_col_scale):
        # (element, subtile) -> (weight row, token) within the CTA tile.
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        cC = cute.make_identity_tensor(
            (gemm_kernel.cta_tile_shape_mnk[0], gemm_kernel.cta_tile_shape_mnk[1])
        )
        tTR_cC_epi = thr_copy_t2r.partition_D(cute.flat_divide(cC, epi_tile))
        tTR_cC = cute.group_modes(
            cute.group_modes(tTR_cC_epi, 3, cute.rank(tTR_cC_epi)), 0, 3
        )
    if cutlass.const_expr(separate_acc is not None):
        # D is partitioned as the base accumulator: element i of each
        # fragment is the same (weight row, token).
        tCtAccD_base, d_pipeline, d_consumer_state, has_lora = separate_acc
        tCtAccD = transform_partitioned_tensor_layout(tCtAccD_base)
        tiled_copy_t2r_d, tTR_tAccD_base, tTR_rD = epilogue_tmem_copy_and_partition(
            gemm_kernel, epi_tidx, tCtAccD, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
        )
        tTR_rD_flat = cute.group_modes(tTR_rD, 0, cute.rank(tTR_rD))
        frag_size_d = cute.size(tTR_rD.shape)
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )
    # ((ATOM_V, REST_V), EPI_M, EPI_N)
    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
    acc_pipeline.consumer_wait(acc_consumer_state)
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
    if cutlass.const_expr(separate_acc is not None):
        tTR_tAccD = tTR_tAccD_base[
            (None, None, None, None, None, d_consumer_state.index)
        ]
        d_pipeline.consumer_wait(d_consumer_state)
        tTR_tAccD = cute.group_modes(tTR_tAccD, 3, cute.rank(tTR_tAccD))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    for subtile_idx in cutlass.range_constexpr(subtile_cnt):
        # Subtiles enumerate M first; the column start decides whether it is live.
        col0 = (subtile_idx // gemm_kernel.epi_m_cnt) * gemm_kernel.epi_tile_n
        if col0 < live_cols:
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            if cutlass.const_expr(use_col_scale):
                for i in cutlass.range_constexpr(frag_size):
                    coord = tTR_cC[(i, subtile_idx)]
                    tTR_rAcc_flat[i] = (tTR_rAcc_flat[i] * col_alpha) * sColScale[
                        coord[1]
                    ]
            if cutlass.const_expr(separate_acc is not None and not use_col_scale):
                # Static scales: alpha reaches the base accumulator alone.
                for i in cutlass.range_constexpr(frag_size):
                    tTR_rAcc_flat[i] = tTR_rAcc_flat[i] * col_alpha
            if cutlass.const_expr(separate_acc is not None):
                # D is added after the base scaling; a tile without LoRA MMAs
                # leaves D stale and skips it.
                if has_lora:
                    cute.copy(
                        tiled_copy_t2r_d,
                        tTR_tAccD[(None, None, None, subtile_idx)],
                        tTR_rD,
                    )
                    for i in cutlass.range_constexpr(frag_size_d):
                        tTR_rAcc_flat[i] = tTR_rAcc_flat[i] + tTR_rD_flat[i]
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(epilogue_op(acc_vec).to(gemm_kernel.c_dtype))
            c_buffer = stores_issued % gemm_kernel.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()
            if warp_idx == gemm_kernel.epilogue_warp_id[0]:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()
            stores_issued += 1

    epilog_sync_barrier.arrive_and_wait()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
        if cutlass.const_expr(separate_acc is not None):
            d_pipeline.consumer_release(d_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state, stores_issued


class GroupedGemmKernelSm100BlockScaled:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        mma_inst_tile_k: int = 4,
        persistent_clusters: int | None = None,
        swap_ab: bool = False,
        contiguous_segments: bool = False,
        sf_vec_size: int = 32,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
        lora_k: int = 0,
        lora_stages: int = 4,
        lora_acc: str = "shared",
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # The K extent here is a placeholder. _setup_attributes computes it.
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.mma_inst_tile_k = mma_inst_tile_k
        self.persistent_clusters = persistent_clusters
        self.swap_ab = swap_ab
        self.contiguous_segments = contiguous_segments
        self.sf_vec_size = sf_vec_size
        self.sf_dtype: Type[cutlass.Numeric] = sf_dtype
        # Epilogue LoRA: one BF16 rank-``lora_k`` MMA per adapter slot after
        # the K loop, operands through ``lora_stages`` TMA stages; 0 compiles
        # none of it. ``lora_acc`` picks the accumulator: "shared" = the base
        # accumulator, "separate" = a second TMEM accumulator D added in the
        # epilogue after the base accumulator's scaling (builds with alpha).
        if lora_k % 16:
            raise ValueError(
                f"the epilogue LoRA rank must be a multiple of 16, got {lora_k}"
            )
        if lora_acc not in ("shared", "separate"):
            raise ValueError(f"unknown epilogue LoRA accumulator mode {lora_acc!r}")
        self.lora_k = lora_k
        self.lora_stages = lora_stages
        self.lora_acc = lora_acc if lora_k else "shared"
        if contiguous_segments and not swap_ab:
            raise ValueError("contiguous_segments requires swap_ab")
        if cluster_shape_mn != (1, 1) or use_2cta_instrs:
            raise ValueError(
                "the block-scaled kernel runs 1-CTA MMAs in (1, 1) clusters"
            )
        self.arch = "sm_100"

        self.cta_group = tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2

    LIVE_EPI_TOKENS = 16

    def _create_tiled_mma(self, mma_tiler_mn):
        return utils.sm100.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            mma_tiler_mn,
        )

    def _setup_attributes(self):
        tiled_mma = self._create_tiled_mma(self.mma_tiler[:2])
        # SFB is loaded in whole 128-row atoms even for narrower token tiles.
        self.mma_tiler_mn_sfb = (
            self.mma_tiler[0],
            cute.round_up(self.mma_tiler[1], 128),
        )
        tiled_mma_sfb = self._create_tiled_mma(self.mma_tiler_mn_sfb)

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (*self.mma_tiler_mn_sfb, self.mma_tiler[2])
        # Token tiles narrower than a scale atom load the whole atom and shift
        # the MMA's TMEM view of it by the tile's row offset.
        self.sfb_tiles_per_atom = max(1, 128 // self.mma_tiler[1])
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        epi_m, epi_n = cute.size(epi_tile[0]), cute.size(epi_tile[1])
        # Decode tiles hold a few live tokens: narrow subtiles let the
        # epilogue skip the padded ones (the store is M-major, so 8 is legal).
        if self.cta_tile_shape_mnk[1] <= 64 and epi_n > self.LIVE_EPI_TOKENS:
            epi_n = self.LIVE_EPI_TOKENS
            epi_tile = (epi_m, epi_n)
        self.epi_tile = epi_tile
        self.epi_tile_n = epi_n
        self.epi_m_cnt = self.cta_tile_shape_mnk[0] // epi_m

        c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        # The LoRA operands are K-major BF16 tiles of the same M x N as the
        # block-scaled MMA, so their MMA shares the TMEM accumulator layout.
        lora_bytes = 0
        self.tiled_mma_lora = tiled_mma
        self.lora_mma_tiler = self.mma_tiler
        if self.lora_k:
            self.tiled_mma_lora = utils.sm100.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            self.lora_mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], self.lora_k)
            lw_one = utils.sm100.make_smem_layout_a(
                self.tiled_mma_lora, self.lora_mma_tiler, cutlass.BFloat16, 1
            )
            lp_one = utils.sm100.make_smem_layout_b(
                self.tiled_mma_lora, self.lora_mma_tiler, cutlass.BFloat16, 1
            )
            self.lora_stage_bytes = cute.size_in_bytes(
                cutlass.BFloat16, lw_one
            ) + cute.size_in_bytes(cutlass.BFloat16, lp_one)
            # Plus alignment slack for the two staged buffers.
            lora_bytes = self.lora_stages * self.lora_stage_bytes + 2048
        # The epilogue's per-token scale vector: one fp32 per token column.
        tok_scale_bytes = 4 * self.mma_tiler[1]

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            c_smem_layout,
            lora_bytes + tok_scale_bytes,
        )
        # The separate LoRA accumulator takes N TMEM columns per stage after
        # the base stages and the two scale-factor copies (at most 32 columns)
        # of the 512: two stages beside 64-wide token tiles, one beside
        # 128-wide ones, none beside 256-wide ones. The other builds keep one
        # unused stage's barriers.
        self.lora_d_stages = 1
        if self.lora_acc == "separate":
            used = self.num_acc_stage * self.mma_tiler[1] + 32
            if used + self.mma_tiler[1] > 512:
                raise ValueError(
                    "the separate epilogue LoRA accumulator does not fit in TMEM "
                    f"beside the base accumulator for N = {self.mma_tiler[1]}"
                )
            self.lora_d_stages = 2 if used + 2 * self.mma_tiler[1] <= 512 else 1

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )
        self.lw_smem_layout_staged = self.a_smem_layout_staged
        self.lp_smem_layout_staged = self.b_smem_layout_staged
        if self.lora_k:
            self.lw_smem_layout_staged = utils.sm100.make_smem_layout_a(
                self.tiled_mma_lora,
                self.lora_mma_tiler,
                cutlass.BFloat16,
                self.lora_stages,
            )
            self.lp_smem_layout_staged = utils.sm100.make_smem_layout_b(
                self.tiled_mma_lora,
                self.lora_mma_tiler,
                cutlass.BFloat16,
                self.lora_stages,
            )

        # The accumulator stages plus both scale-factor copies fit in TMEM.
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        tok_scale: cute.Tensor,
        lora_w: cute.Tensor,
        lora_p: cute.Tensor,
        lora_slots: cute.Tensor,
        group_m: cute.Tensor,
        direct_schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        use_alpha: cutlass.Constexpr = False,
        alpha_rows_per_half: cutlass.Constexpr = 1,
        use_tok_scale: cutlass.Constexpr = False,
        lora_experts: cutlass.Constexpr = 1,
    ):
        """``tok_scale`` (rows, expert | 1) fp32 per-token scales multiply each
        output column before alpha when ``use_tok_scale``; unread otherwise.
        ``lora_w`` (hidden, rank, slot * lora_experts + expert) BF16 weights
        and ``lora_p`` (rows, rank, plane) BF16 bridge planes, one plane per
        slot (flat rows) or per (slot, expert) slab, add
        ``lora_w @ lora_p^T`` to each tile for the first ``lora_slots[0]``
        slots when the kernel was built with ``lora_k``; unread otherwise."""
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(sfa.element_type != self.sf_dtype):
            raise TypeError(f"scale type must be {self.sf_dtype}: {sfa.element_type}")
        if cutlass.const_expr(self.lora_acc == "separate" and not use_alpha):
            raise ValueError(
                "the separate epilogue LoRA accumulator is a build with alpha: "
                "its epilogue adds D after scaling the base accumulator"
            )

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma(self.mma_tiler[:2])
        tiled_mma_sfb = self._create_tiled_mma(self.mma_tiler_mn_sfb)

        # Scale factors fill the tcgen05 atom layout over the operand shapes.
        # ((Atom_M, Rest_M), (Atom_K, Rest_K), RestL)
        sfa = cute.make_tensor(
            sfa.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size),
        )
        sfb = cute.make_tensor(
            sfb.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size),
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

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

        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_op = utils.sm100.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Without lora_k the LoRA atoms alias the A/B atoms and stay unread.
        tma_atom_lw, tma_tensor_lw = tma_atom_a, tma_tensor_a
        tma_atom_lp, tma_tensor_lp = tma_atom_b, tma_tensor_b
        if cutlass.const_expr(self.lora_k > 0):
            lw_smem_layout = cute.slice_(
                self.lw_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_lw, tma_tensor_lw = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                lora_w,
                lw_smem_layout,
                self.lora_mma_tiler,
                self.tiled_mma_lora,
                self.cluster_layout_vmnk.shape,
            )
            lp_smem_layout = cute.slice_(
                self.lp_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_lp, tma_tensor_lp = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                lora_p,
                lp_smem_layout,
                self.lora_mma_tiler,
                self.tiled_mma_lora,
                self.cluster_layout_vmnk.shape,
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = resolve_scheduler_params_and_grid(
            a=a,
            cta_tile_k=self.cta_tile_shape_mnk[2],
            cluster_shape_mn=self.cluster_shape_mn,
            swap_ab=self.swap_ab,
            persistent_clusters=self.persistent_clusters,
            max_active_clusters=max_active_clusters,
        )

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            self.tiled_mma_lora,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_lw,
            tma_tensor_lw,
            tma_atom_lp,
            tma_tensor_lp,
            alpha,
            tok_scale,
            lora_slots,
            group_m,
            direct_schedule,
            schedule_tiles,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.lw_smem_layout_staged,
            self.lp_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            use_alpha,
            alpha_rows_per_half,
            use_tok_scale,
            lora_experts,
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
        tiled_mma_sfb: cute.TiledMma,
        tiled_mma_lora: cute.TiledMma,
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
        tma_atom_lw: cute.CopyAtom,
        mLW_mkl: cute.Tensor,
        tma_atom_lp: cute.CopyAtom,
        mLP_nkl: cute.Tensor,
        alpha: cute.Tensor,
        tok_scale: cute.Tensor,
        lora_slots: cute.Tensor,
        group_m: cute.Tensor,
        direct_schedule: cute.Tensor,
        schedule_tiles: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        lw_smem_layout_staged: cute.ComposedLayout,
        lp_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: MoESchedulerParams,
        epilogue_op: cutlass.Constexpr,
        use_alpha: cutlass.Constexpr,
        alpha_rows_per_half: cutlass.Constexpr,
        use_tok_scale: cutlass.Constexpr,
        lora_experts: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.lora_k > 0):
                cpasync.prefetch_descriptor(tma_atom_lw)
                cpasync.prefetch_descriptor(tma_atom_lp)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

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
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            lora_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.lora_stages * 2
            ]
            lora_d_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.lora_d_stages * 2]
            col_scale: cute.struct.MemRange[cutlass.Float32, self.cta_tile_shape_mnk[1]]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # The epilogue's per-token scales for the tile in flight.
        sColScale = cute.make_tensor(
            storage.col_scale.data_ptr(),
            cute.make_layout(self.cta_tile_shape_mnk[1]),
        )

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
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

        if cutlass.const_expr(self.lora_k > 0):
            # One stage per (tile, slot): the TMA warp fills, the MMA warp frees.
            lora_producer, lora_consumer = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.lora_full_mbar_ptr.data_ptr(),
                num_stages=self.lora_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, num_tma_producer
                ),
                tx_count=self.lora_stage_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
        if cutlass.const_expr(self.lora_acc == "separate"):
            # The D accumulator: the MMA warp fills a stage per tile, the
            # epilogue warps free it once they have read it.
            d_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.lora_d_mbar_ptr.data_ptr(),
                num_stages=self.lora_d_stages,
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
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )
        if cutlass.const_expr(self.lora_k > 0):
            # (MMA, MMA_M, MMA_K, STAGE) weights and (MMA, MMA_N, MMA_K, STAGE) planes
            sLW = smem.allocate_tensor(
                element_type=cutlass.BFloat16,
                layout=lw_smem_layout_staged.outer,
                byte_alignment=1024,
                swizzle=lw_smem_layout_staged.inner,
            )
            sLP = smem.allocate_tensor(
                element_type=cutlass.BFloat16,
                layout=lp_smem_layout_staged.outer,
                byte_alignment=1024,
                swizzle=lp_smem_layout_staged.inner,
            )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL): whole 128-row scale atoms per tile.
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        if cutlass.const_expr(self.lora_k > 0):
            # (bM, bK, RestM, RestK, RestL) weights; (bN, bK, RestN, RestK, RestL) planes
            gLW_mkl = cute.local_tile(
                mLW_mkl,
                cute.slice_(self.lora_mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gLP_nkl = cute.local_tile(
                mLP_nkl,
                cute.slice_(self.lora_mma_tiler, (0, None, None)),
                (None, None, None),
            )
            thr_mma_lora = tiled_mma_lora.get_slice(mma_tile_coord_v)
            tCgLW = thr_mma_lora.partition_A(gLW_mkl)
            tCgLP = thr_mma_lora.partition_B(gLP_nkl)
            tLWsLW, tLWgLW = cpasync.tma_partition(
                tma_atom_lw,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sLW, 0, 3),
                cute.group_modes(tCgLW, 0, 3),
            )
            tLPsLP, tLPgLP = cpasync.tma_partition(
                tma_atom_lp,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sLP, 0, 3),
                cute.group_modes(tCgLP, 0, 3),
            )
            # (MMA, MMA_M, MMA_K, STAGE) and (MMA, MMA_N, MMA_K, STAGE)
            tCrLW = tiled_mma_lora.make_fragment_A(sLW)
            tCrLP = tiled_mma_lora.make_fragment_B(sLP)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        # (MMA, MMA_M, MMA_N, D_STAGE): the separate LoRA accumulator.
        tCtAccD_fake = tiled_mma_lora.make_fragment_C(
            cute.append(acc_shape, self.lora_d_stages)
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
                    work_tile.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    work_tile.tile_n_idx,
                    work_tile.expert_idx,
                )
                if cutlass.const_expr(self.swap_ab):
                    mma_tile_coord_mnl = (
                        work_tile.tile_n_idx // cute.size(tiled_mma.thr_id.shape),
                        work_tile.tile_m_idx,
                        work_tile.expert_idx,
                    )
                a_rest_l_coord = mma_tile_coord_mnl[2]
                if cutlass.const_expr(self.contiguous_segments):
                    # Flat rows use slot 0 plus seg_offsets; weights keep the expert.
                    seg_base_tile = (
                        group_m[work_tile.expert_idx] // self.cta_tile_shape_mnk[1]
                    )
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        seg_base_tile + mma_tile_coord_mnl[1],
                        cutlass.Int32(0),
                    )
                    a_rest_l_coord = work_tile.expert_idx

                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, a_rest_l_coord)]
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, a_rest_l_coord)
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                tBgSFB_slice = tBgSFB[
                    (
                        None,
                        mma_tile_coord_mnl[1] // self.sfb_tiles_per_atom,
                        None,
                        mma_tile_coord_mnl[2],
                    )
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
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, handle.count)],
                        tAsSFA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, handle.count)],
                        tBsSFB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )

                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < work_tile.k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                if cutlass.const_expr(self.lora_k > 0):
                    # The tile's LoRA operands follow its K loads, one stage per
                    # slot: weights of (slot, expert) and the slot's bridge
                    # plane. The MMA warp frees a stage after the tile's K loop
                    # and the slot's MMA, so a slot count past the stage count
                    # only waits on loads already in flight; issued ahead of the
                    # K loads, that wait would need this tile's K loop first and
                    # the two warps would deadlock.
                    num_slots = lora_slots[0]
                    for slot in cutlass.range(0, num_slots, 1, unroll=1):
                        lora_handle = lora_producer.acquire_and_advance()
                        weight_l = slot * lora_experts + work_tile.expert_idx
                        plane_l = slot
                        if cutlass.const_expr(not self.contiguous_segments):
                            plane_l = weight_l
                        cute.copy(
                            tma_atom_lw,
                            tLWgLW[(None, mma_tile_coord_mnl[0], 0, weight_l)],
                            tLWsLW[(None, lora_handle.index)],
                            tma_bar_ptr=lora_handle.barrier,
                        )
                        cute.copy(
                            tma_atom_lp,
                            tLPgLP[(None, mma_tile_coord_mnl[1], 0, plane_l)],
                            tLPsLP[(None, lora_handle.index)],
                            tma_bar_ptr=lora_handle.barrier,
                        )

                work_tile = tile_sched.advance_to_next_work()

            ab_producer.tail()
            if cutlass.const_expr(self.lora_k > 0):
                lora_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base, tCtSFA, tCtSFB, sfb_tmem_cols, tCtAccD_base = (
                self._tmem_regions(
                    tmem_ptr,
                    tiled_mma,
                    tCtAcc_fake,
                    tCtAccD_fake,
                    sfa_smem_layout_staged,
                    sfb_smem_layout_staged,
                )
            )
            tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_copy_and_partition(
                sSFA, tCtSFA
            )
            tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_copy_and_partition(
                sSFB, tCtSFB
            )

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            if cutlass.const_expr(self.lora_k > 0 and self.lora_acc != "separate"):
                tiled_mma_lora.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(self.lora_acc == "separate"):
                d_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.lora_d_stages
                )

            while work_tile.is_valid_tile:
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.sfb_tiles_per_atom > 1):
                    # Rows r, r + 32, r + 64, r + 96 of an atom share TMEM lane
                    # r as four words, so a tile's rows start N // 32 words in.
                    token_tile = work_tile.tile_n_idx
                    if cutlass.const_expr(self.swap_ab):
                        token_tile = work_tile.tile_m_idx
                    if cutlass.const_expr(self.contiguous_segments):
                        token_tile = (
                            token_tile
                            + group_m[work_tile.expert_idx]
                            // self.cta_tile_shape_mnk[1]
                        )
                    sfb_shift = (token_tile % self.sfb_tiles_per_atom) * (
                        self.mma_tiler[1] // 32
                    )
                    tCtSFB_mma = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + sfb_tmem_cols + sfb_shift, dtype=self.sf_dtype
                        ),
                        tCtSFB.layout,
                    )

                for k_tile in range(work_tile.k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        stage_coord = (None, None, None, None, handle.index)
                        cute.copy(
                            tiled_copy_s2t_sfa, tCsSFA_s2t[stage_coord], tCtSFA_s2t
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb, tCsSFB_s2t[stage_coord], tCtSFB_s2t
                        )

                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB_mma],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        handle.release()

                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < work_tile.k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if cutlass.const_expr(self.lora_k > 0):
                    num_slots = lora_slots[0]
                    if cutlass.const_expr(self.lora_acc == "separate"):
                        # One rank-K BF16 MMA per slot into D (the first slot
                        # overwrites it); the stage is free once the epilogue
                        # warps have read the tile that last used it. A tile
                        # without slots leaves D stale and the epilogue skips it.
                        if is_leader_cta:
                            d_pipeline.producer_acquire(d_producer_state)
                        tCtAccD = tCtAccD_base[
                            (None, None, None, d_producer_state.index)
                        ]
                        tiled_mma_lora.set(tcgen05.Field.ACCUMULATE, False)
                        for slot in cutlass.range(0, num_slots, 1, unroll=1):
                            if is_leader_cta:
                                lora_handle = lora_consumer.wait_and_advance()
                                lora_crd = (None, None, None, lora_handle.index)
                                cute.gemm(
                                    tiled_mma_lora,
                                    tCtAccD,
                                    tCrLW[lora_crd],
                                    tCrLP[lora_crd],
                                    tCtAccD,
                                )
                                tiled_mma_lora.set(tcgen05.Field.ACCUMULATE, True)
                                lora_handle.release()
                        if is_leader_cta:
                            d_pipeline.producer_commit(d_producer_state)
                        d_producer_state.advance()
                    else:
                        # One rank-K BF16 MMA per slot into the same accumulator.
                        for slot in cutlass.range(0, num_slots, 1, unroll=1):
                            if is_leader_cta:
                                lora_handle = lora_consumer.wait_and_advance()
                                lora_crd = (None, None, None, lora_handle.index)
                                cute.gemm(
                                    tiled_mma_lora,
                                    tCtAcc,
                                    tCrLW[lora_crd],
                                    tCrLP[lora_crd],
                                    tCtAcc,
                                )
                                lora_handle.release()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                work_tile = tile_sched.advance_to_next_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # (EPI_TILE_M, EPI_TILE_N, STAGE)
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
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAccD_base = self._tmem_regions(
                tmem_ptr,
                tiled_mma,
                tCtAcc_fake,
                tCtAccD_fake,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
            )[4]

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            if cutlass.const_expr(self.lora_acc == "separate"):
                d_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.lora_d_stages
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
            stores_issued = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                mma_tile_coord_mnl = (
                    work_tile.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    work_tile.tile_n_idx,
                    work_tile.expert_idx,
                )
                # Masked slabs store only the live token subtiles of a tile.
                live_cols = cutlass.Int32(self.cta_tile_shape_mnk[1])
                if cutlass.const_expr(self.swap_ab):
                    mma_tile_coord_mnl = (
                        work_tile.tile_n_idx // cute.size(tiled_mma.thr_id.shape),
                        work_tile.tile_m_idx,
                        work_tile.expert_idx,
                    )
                    if cutlass.const_expr(not self.contiguous_segments):
                        live_cols = (
                            group_m[work_tile.expert_idx]
                            - work_tile.tile_m_idx * self.cta_tile_shape_mnk[1]
                        )
                if cutlass.const_expr(self.contiguous_segments):
                    # Same segment base as the TMA warp.
                    seg_base_tile = (
                        group_m[work_tile.expert_idx] // self.cta_tile_shape_mnk[1]
                    )
                    mma_tile_coord_mnl = (
                        mma_tile_coord_mnl[0],
                        seg_base_tile + mma_tile_coord_mnl[1],
                        cutlass.Int32(0),
                    )
                tile_op = epilogue_op
                col_alpha = cutlass.Float32(1.0)
                if cutlass.const_expr(use_alpha):
                    # Gate rows and up rows of a fused W13 carry separate scales.
                    half = (
                        mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0]
                    ) // alpha_rows_per_half
                    alpha_value = alpha[(work_tile.expert_idx, half)]
                    if cutlass.const_expr(use_tok_scale or self.lora_acc == "separate"):
                        # Applied per element: before the token's scale, and
                        # to the base accumulator only when D is added.
                        col_alpha = alpha_value
                    else:

                        def tile_op(x, alpha_value=alpha_value):
                            return epilogue_op(x * alpha_value)

                if cutlass.const_expr(use_tok_scale):
                    # Stage the tile's token scales; the previous tile's
                    # epilogue ended on this barrier, so the slots are free.
                    token_base = mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                    epi_threads = 32 * len(self.epilogue_warp_id)
                    for chunk in cutlass.range_constexpr(
                        (self.cta_tile_shape_mnk[1] + epi_threads - 1) // epi_threads
                    ):
                        col = tidx + chunk * epi_threads
                        if col < self.cta_tile_shape_mnk[1]:
                            sColScale[col] = tok_scale[
                                (token_base + col, mma_tile_coord_mnl[2])
                            ]
                    epilog_sync_barrier.arrive_and_wait()
                # Whether this tile's LoRA MMAs ran (the batch's slot bound).
                separate_acc = None
                if cutlass.const_expr(self.lora_acc == "separate"):
                    has_lora = lora_slots[0] > 0
                    separate_acc = (
                        tCtAccD_base,
                        d_pipeline,
                        d_consumer_state,
                        has_lora,
                    )

                work_tile = tile_sched.advance_to_next_work()

                acc_consumer_state, stores_issued = _epilogue_store_live(
                    self,
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    stores_issued,
                    live_cols,
                    tile_op,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                    sColScale,
                    col_alpha,
                    use_tok_scale,
                    separate_acc,
                )
                if cutlass.const_expr(self.lora_acc == "separate"):
                    d_consumer_state.advance()

            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    def _tmem_regions(
        self,
        tmem_ptr,
        tiled_mma: cute.TiledMma,
        tCtAcc_fake: cute.Tensor,
        tCtAccD_fake: cute.Tensor,
        sfa_smem_layout_staged: cute.ComposedLayout,
        sfb_smem_layout_staged: cute.ComposedLayout,
    ):
        """The accumulator stages, the two scale-factor tensors after them
        and, after those, the separate LoRA accumulator at a column aligned
        to the tile width (the accumulator stage stride)."""
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(
            tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
            dtype=self.sf_dtype,
        )
        # (MMA, MMA_M, MMA_K)
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            ),
        )
        sfb_tmem_cols = tcgen05.find_tmem_tensor_col_offset(
            tCtAcc_base
        ) + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
        sfb_tmem_ptr = cute.recast_ptr(tmem_ptr + sfb_tmem_cols, dtype=self.sf_dtype)
        # (MMA, MMA_N, MMA_K)
        tCtSFB = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            ),
        )
        width = self.mma_tiler[1]
        d_col = (
            (sfb_tmem_cols + tcgen05.find_tmem_tensor_col_offset(tCtSFB) + width - 1)
            // width
            * width
        )
        if self.lora_acc == "separate" and isinstance(d_col, int):
            if d_col + self.lora_d_stages * width > self.num_tmem_alloc_cols:
                raise ValueError(
                    f"the separate epilogue LoRA accumulator ({self.lora_d_stages} "
                    f"stages of {width} columns at column {d_col}) exceeds the "
                    f"{self.num_tmem_alloc_cols} TMEM columns"
                )
        # (MMA, MMA_M, MMA_N, D_STAGE)
        tCtAccD_base = cute.make_tensor(tmem_ptr + d_col, tCtAccD_fake.layout)
        return tCtAcc_base, tCtSFA, tCtSFB, sfb_tmem_cols, tCtAccD_base

    def _s2t_copy_and_partition(
        self, sSF: cute.Tensor, tSF: cute.Tensor
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition one scale-factor copy from SMEM into TMEM."""
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, thr_copy_s2t.partition_S(tCsSF_compact)
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
        c_smem_layout: cute.Layout,
        reserved_bytes: int = 0,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 2

        a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024 + reserved_bytes

        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage

    def check_supported_dtypes(
        self,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ):
        sf_dtype = self.sf_dtype
        if a_dtype is not b_dtype:
            raise testing.CantImplementError(
                f"the block-scaled GEMM takes matching A/B; got {a_dtype}, {b_dtype}"
            )
        valid = {
            (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
        }
        if (a_dtype, sf_dtype, self.sf_vec_size) not in valid:
            raise testing.CantImplementError(
                f"unsupported block-scaled combination {a_dtype}, {sf_dtype}, "
                f"sf_vec_size={self.sf_vec_size}"
            )
        if c_dtype is not cutlass.BFloat16:
            raise testing.CantImplementError(f"C must be BF16; got {c_dtype}")
        if self.acc_dtype is not cutlass.Float32:
            raise testing.CantImplementError(
                f"the accumulator must be FP32; got {self.acc_dtype}"
            )

    def check_mma_tiler_and_cluster_shape(self):
        if self.mma_tiler_mn[0] != 128:
            raise testing.CantImplementError(
                f"Invalid mma tiler M for 1-CTA block-scaled MMA: {self.mma_tiler_mn}"
            )
        # 32-token tiles fault (misaligned address) with a one-word SFB shift.
        if self.mma_tiler_mn[1] not in (64, 128, 256):
            raise testing.CantImplementError(
                f"Invalid mma tiler N: {self.mma_tiler_mn[1]}; token tiles are "
                "half, one or two 128-row scale atoms"
            )
        if self.cluster_shape_mn != (1, 1):
            raise testing.CantImplementError(
                f"Invalid cluster shape: {self.cluster_shape_mn}"
            )

    def check_tensor_alignment(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ):
        def check_contiguous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contiguous_16B_alignment(a_dtype, a_major == "m", (m, k, l))
            or not check_contiguous_16B_alignment(b_dtype, b_major == "n", (n, k, l))
            or not check_contiguous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            raise testing.CantImplementError(
                f"Invalid tensor alignment: {m}, {n}, {k}, {l}, {a_dtype}, {b_dtype}, {c_dtype}, {a_major}, {b_major}, {c_major}"
            )

    def can_implement(
        self,
        mnkl: Tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        try:
            self.check_supported_dtypes(a_dtype, b_dtype, c_dtype)
            self.check_mma_tiler_and_cluster_shape()

            m, n, k, l = mnkl
            self.check_tensor_alignment(
                m, n, k, l, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
            )
        except testing.CantImplementError:
            return False
        return True
