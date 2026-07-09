# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 SGLang Team
"""LL (low-latency) FP8 block-scaled GEMM, CuTeDSL, SM100/SM120.

out[M, N] = (scale_A * A_fp8[M, K]) @ (scale_B * B_fp8[N, K]).T

Warp-specialized (1 DMA warp-group via cp.async + 4 MMA warps), fixed
tile_m=16 -- purpose-built for tiny-M decode (M<=16, one CTA-worth of tokens
per M-tile), not a general-shape GEMM. FP8 operands are passed to the kernel
viewed as bf16 (2 fp8 bytes per bf16 element) so the SM90-style
`LdMatrix8x8x16bOp` ldmatrix atom (built for 16-bit types) can load them
directly; the actual multiply-accumulate recasts the loaded bf16 registers
back to fp8 and issues a real `MmaFP8Op` (mma.sync.m16n8k32.e4m3) instruction.
Scale factors are ue8m0 (power-of-two, DeepGEMM-compatible) packed 4-per-int32.

Kernel body (`LLFp8BlockGemm`, `fused_ue8m0_scale`, `_make_pred`) is an
unmodified copy of vLLM's
`vllm/model_executor/kernels/linear/cute_dsl/_ll_fp8_block_warpspecialized.py`
(2026-07-08 diff) -- verified correct and faster than both sglang's own
CUTLASS-C++ JIT `fp8_blockwise_scaled_mm` and DeepGEMM's native SM120
`fp8_gemm_nt` at every tested M in [1,16] (see the bench_ll_fp8_block_gemm.py
benchmark: ~1.1-1.6x over the JIT kernel, ~2-3.3x over DeepGEMM). Only the
scale-layout adapters below (`_pack_ue8m0_last_axis`,
`_ue8m0_weight_scale_for_ll`) and the sglang op-registration wrapper are new.
"""

from __future__ import annotations

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline

# ///////////////////////////////////////////////////////////////////////////
# Split-K DSMEM helpers (verbatim from cutedsl_bf16_gemm.py's split-K diff --
# see that file for the full docstrings on why st.async+mbarrier-tx-completion
# avoids the release/acquire fences a plain PTX cluster arrive would need)
# ///////////////////////////////////////////////////////////////////////////


@dsl_user_op
def _st_async_shared_cluster_v4_f32(addr, mbar, v0, v1, v2, v3, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
        " [$0], {$2, $3, $4, $5}, [$1];",
        "r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_async_shared_cluster_f32(addr, mbar, val, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
            cutlass.Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32"
        " [$0], $2, [$1];",
        "r,r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _mbarrier_wait_parity_spin(addr, phase, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(phase).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n"
        "@!P1 bra LAB_WAIT;\n"
        "}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )


from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import direct_register_custom_op

# ///////////////////////////////////////////////////////////////////////////
# Kernel (verbatim from the vLLM diff -- see module docstring)
# ///////////////////////////////////////////////////////////////////////////


def _make_pred(tXgX, tXcX, dim_size):
    num_vec = tXgX.shape[0][1]
    num_mn = cute.size(tXgX, mode=[1])
    num_k = cute.size(tXgX, mode=[2])
    pred = cute.make_rmem_tensor(
        cute.make_layout((num_vec, num_mn, num_k), stride=(num_mn, 1, 0)),
        cutlass.Boolean,
    )
    for rv in range(num_vec):
        for mn in range(num_mn):
            pred[rv, mn, 0] = cute.elem_less(tXcX[(0, rv), mn, 0, 0][0], dim_size)
    return pred


@dsl_user_op
def fused_ue8m0_scale(sa_packed, sb_packed, byte_idx, *, loc=None, ip=None):
    """Fused scale: 2^(ea+eb-254) from packed ue8m0 A and B scales."""
    f32 = cutlass.Float32.mlir_type
    val_a = sa_packed.ir_value(loc=loc, ip=ip)
    val_b = sb_packed.ir_value(loc=loc, ip=ip)
    idx = byte_idx.ir_value(loc=loc, ip=ip)
    res = _llvm.inline_asm(
        f32,
        [val_a, val_b, idx],
        "{"
        ".reg .u32 ea, eb, combined;"
        "prmt.b32 ea, $1, 0, $3;"
        "and.b32 ea, ea, 0xFF;"
        "prmt.b32 eb, $2, 0, $3;"
        "and.b32 eb, eb, 0xFF;"
        "add.u32 combined, ea, eb;"
        "sub.u32 combined, combined, 127;"
        "shl.b32 combined, combined, 23;"
        "mov.b32 $0, combined;"
        "}",
        "=f,r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(res)


@dsl_user_op
class LLFp8BlockGemm:
    def __init__(
        self,
        tile_n: int = 16,
        tile_k: int = 256,
        num_stages: int = 2,
        num_dma_warps: int = 4,
        split_k: int = 1,
        fp32_scale: bool = False,
        *,
        loc=None,
        ip=None,
    ):
        if not 1 <= split_k <= 8:
            raise ValueError(f"split_k={split_k} unsupported (1..8, portable cluster)")
        self.ab_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.out_dtype = cutlass.BFloat16
        self.tile_m = 16
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.tile_k_fp8 = tile_k * 2
        self.num_stages = num_stages
        # fp32_scale: use plain fp32 block scales (one value per 128-wide
        # K-block, no packing) instead of the default ue8m0 (power-of-two)
        # scales packed 4-per-int32. The MMA itself is always a plain,
        # unscaled fp8xfp8 mma.sync -- scaling is a separate fp32
        # multiply-accumulate in the epilogue regardless of this flag (see
        # `fused_ue8m0_scale`'s docstring), so this only changes how the
        # per-block scale value is stored/loaded/decoded, not the compute
        # path. Trades away `fused_ue8m0_scale`'s bit-trick (~7 PTX
        # instructions, but requires lossy power-of-two-rounded scales) for a
        # single native fp32 multiply against the weight's actual
        # (non-power-of-two) block scale -- removes the accuracy tax of
        # `requant_weight_ue8m0` entirely. 4x more scale smem/DMA traffic
        # (one fp32 per sub-block instead of 4 packed into one int32), but
        # scale data is tiny relative to A/B operands so this is negligible.
        self.fp32_scale = fp32_scale
        self.mma_shape = (16, 8, 16)
        self.atom_layout = (1, 1, 1)
        self.num_mma_warps = 4
        self.num_dma_threads = num_dma_warps * 32
        self.num_mma_threads = self.num_mma_warps * 32
        self.num_threads = self.num_dma_threads + self.num_mma_threads
        # split_k CTAs of a (split_k,1,1) cluster each run the FULL mainloop
        # pipeline over an interleaved subset of K-tiles (rank r takes tiles
        # r, r+split_k, r+2*split_k, ...), then combine their partial fp32
        # accumulators cross-CTA via DSMEM in the epilogue (see `kernel`
        # below). cluster_shape is (1,1,1) -- a no-op single-CTA "cluster" --
        # when split_k==1.
        self.split_k = split_k
        self.cluster_shape = (split_k, 1, 1)

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        major_size = min(smem_tiler[1], 64)
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        copy_elems = copy_bits // dtype.width
        k_threads = cute.size(self.tile_k) // copy_elems
        thread_layout = cute.make_layout(
            (num_threads // k_threads, k_threads), stride=(k_threads, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
        stream: CUstream,
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, copy_bits, (bM, bK, self.num_stages)
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, copy_bits, (bN, bK, self.num_stages)
        )
        atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy(
            atom_g2s, mA.element_type, copy_bits, self.num_dma_threads
        )
        tiled_copy_B = self._make_gmem_tiled_copy(
            atom_g2s, mB.element_type, copy_bits, self.num_dma_threads
        )
        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_shape)
        perm_mnk = (
            self.atom_layout[0] * self.mma_shape[0],
            self.atom_layout[1] * self.mma_shape[1] * (self.tile_n // 8),
            self.atom_layout[2] * self.mma_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk
        )
        op_fp8 = cute.nvgpu.warp.MmaFP8Op(
            cutlass.Float8E4M3FN, self.acc_dtype, (16, 8, 32)
        )
        perm_mnk_fp8 = (
            self.atom_layout[0] * 16,
            self.atom_layout[1] * 8 * (self.tile_n // 8),
            self.atom_layout[2] * 32,
        )
        tiled_mma_fp8 = cute.make_tiled_mma(
            op_fp8, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk_fp8
        )
        # split_k>1: split_k CTAs per output M-tile, grouped into one cluster
        # each (grid is already an exact multiple of split_k -- no round_up
        # needed, unlike a 2-CTA-cluster scheme that pairs *adjacent* M-tiles).
        grid_m = cute.ceil_div(cute.size(mC, mode=[0]), bM) * self.split_k
        grid_n = cute.ceil_div(cute.size(mC, mode=[1]), bN)
        self.kernel(
            mA,
            mB,
            mC,
            mSA,
            mSB,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            tiled_mma_fp8,
        ).launch(
            grid=[cute.size(grid_m), cute.size(grid_n), 1],
            block=[self.num_threads, 1, 1],
            cluster=self.cluster_shape,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mA,
        mB,
        mC,
        mSA,
        mSB,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tiled_mma_fp8: cute.TiledMma,
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_idx = tidx // 32
        is_dma = warp_idx < (self.num_dma_threads // 32)
        dma_tidx = tidx
        mma_tidx = tidx - self.num_dma_threads
        N_out = cute.size(mC, mode=[1])
        M_out = cute.size(mC, mode=[0])

        # split_k>1: `split_k` consecutive CTAs along the grid's M axis form
        # one cluster and all work the SAME output M-tile (`real_bid_m`);
        # `split_rank` (the cluster-local rank, a hardware register read --
        # not `bid_m % split_k` -- matching cutedsl_bf16_gemm.py's pattern)
        # picks this CTA's interleaved K-tile subset.
        if cutlass.const_expr(self.split_k > 1):
            real_bid_m = bid_m // self.split_k
            split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        else:
            real_bid_m = bid_m
            split_rank = cutlass.Int32(0)

        cta_tiler = (bM, bN, bK)
        coord = (real_bid_m, bid_n, None)
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=coord, proj=(1, 1, None))
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        cA = cute.local_tile(mcA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))

        # split_k>1: leader's landing zone for (split_k-1) peers' partial
        # fp32 accumulators, one (bM*bN) slot per peer -- elided (1 dummy
        # element) when split_k==1. `splitk_bar`'s single mbarrier tracks
        # cross-CTA DSMEM tx-completion (see the epilogue below).
        RED_BUF_LEN: cutlass.Constexpr = (
            (self.split_k - 1) * bM * bN if self.split_k > 1 else 1
        )

        # fp32_scale: one scale value per 128-wide K-block, SCALE_SUBBLOCKS
        # of them per pipeline stage (one per MMA warp's sub-block, see the
        # `sg`-loop below) instead of ue8m0's single packed-int32-per-stage.
        SCALE_SUBBLOCKS: cutlass.Constexpr = self.tile_k_fp8 // 128
        if cutlass.const_expr(self.fp32_scale):
            ScaleDtype = cutlass.Float32
            SA_SCALE_ELEMS: cutlass.Constexpr = bM * num_stages * SCALE_SUBBLOCKS
            SB_SCALE_ELEMS: cutlass.Constexpr = num_stages * SCALE_SUBBLOCKS
        else:
            ScaleDtype = cutlass.Int32
            SA_SCALE_ELEMS: cutlass.Constexpr = bM * num_stages
            SB_SCALE_ELEMS: cutlass.Constexpr = num_stages

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16
            ]
            sa_scale: cute.struct.Align[
                cute.struct.MemRange[ScaleDtype, SA_SCALE_ELEMS], 4
            ]
            sb_scale: cute.struct.Align[
                cute.struct.MemRange[ScaleDtype, SB_SCALE_ELEMS], 4
            ]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, num_stages * 2], 8
            ]
            splitk_bar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]
            red_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, RED_BUF_LEN], 16
            ]

        smem = cutlass.utils.SmemAllocator()
        storage_ptr = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
        storage = SharedStorage(storage_ptr)
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)
        bar_splitk = storage.splitk_bar.data_ptr()
        red_buf_ptr = storage.red_buf.data_ptr()

        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_dma_threads
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_threads
        )
        mainloop_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
        )

        if cutlass.const_expr(self.split_k > 1):
            # bar_splitk: single elected arrive_and_expect_tx (count=1) --
            # peers signal via st.async tx-completion, not a normal arrive.
            if tidx == 0:
                cute.arch.mbarrier_init(bar_splitk, 1)
            cute.arch.mbarrier_init_fence()
            # Cluster-wide sync so every peer's bar_splitk/red_buf address is
            # initialized before any CTA issues a remote DSMEM write to it.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        total_k_tiles = cute.size(gA, mode=[2])
        sSA_ptr = storage.sa_scale.data_ptr()
        sSB_ptr = storage.sb_scale.data_ptr()
        n_block_idx = bid_n * bN // 128
        TILE_K_FP8_C: cutlass.Constexpr = self.tile_k_fp8
        # fp32_scale: one gmem scale element per 128-wide K-block
        # (`num_scale_k`). ue8m0: one packed int32 per 4 K-blocks
        # (`num_packed_k`) -- kept as a distinct name since it indexes a
        # different (4x coarser) gmem array, not just a different dtype.
        if cutlass.const_expr(self.fp32_scale):
            num_scale_k = total_k_tiles * TILE_K_FP8_C // 128
        else:
            num_packed_k = (total_k_tiles * TILE_K_FP8_C // 128) // 4

        # split_k>1: this rank's interleaved K-tile subset (rank r takes
        # global tiles r, r+split_k, r+2*split_k, ...); split_k==1 keeps the
        # original single-rank behavior (tiles_this_rank == total_k_tiles).
        if cutlass.const_expr(self.split_k > 1):
            tiles_this_rank = (
                total_k_tiles - split_rank + self.split_k - 1
            ) // self.split_k
        else:
            tiles_this_rank = total_k_tiles

        if is_dma:
            cute.arch.setmaxregister_decrease(40)
            thr_A = tiled_copy_A.get_slice(dma_tidx)
            thr_B = tiled_copy_B.get_slice(dma_tidx)
            tAgA = thr_A.partition_S(gA)
            tAsA = thr_A.partition_D(sA)
            tBgB = thr_B.partition_S(gB)
            tBsB = thr_B.partition_D(sB)
            tAcA = thr_A.partition_S(cA)
            tBcB = thr_B.partition_S(cB)

            tApA = _make_pred(tAgA, tAcA, mA.shape[0])

            tBpB = _make_pred(tBgB, tBcB, mB.shape[0])

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_stages
            )

            # Pre-create scale tensors (like A/B tile tensors)
            n_repr = n_block_idx * 128
            m_slot = dma_tidx % bM
            global_m = real_bid_m * bM + m_slot
            safe_m = global_m if global_m < M_out else M_out - 1
            if cutlass.const_expr(self.fp32_scale):
                gSA = cute.make_tensor(
                    (mSA.iterator + safe_m * num_scale_k).align(4),
                    cute.make_layout((num_scale_k,)),
                )
                gSB = cute.make_tensor(
                    (mSB.iterator + n_repr).align(4),
                    cute.make_layout((num_scale_k,), stride=(N_out,)),
                )
                sSA = cute.make_tensor(
                    sSA_ptr,
                    cute.make_layout(
                        (bM, num_stages, SCALE_SUBBLOCKS),
                        stride=(SCALE_SUBBLOCKS, bM * SCALE_SUBBLOCKS, 1),
                    ),
                )
                sSB = cute.make_tensor(
                    sSB_ptr,
                    cute.make_layout(
                        (num_stages, SCALE_SUBBLOCKS), stride=(SCALE_SUBBLOCKS, 1)
                    ),
                )
            else:
                gSA = cute.make_tensor(
                    (mSA.iterator + safe_m * num_packed_k).align(4),
                    cute.make_layout((num_packed_k,)),
                )
                gSB = cute.make_tensor(
                    (mSB.iterator + n_repr).align(4),
                    cute.make_layout((num_packed_k,), stride=(N_out,)),
                )
                sSA = cute.make_tensor(
                    sSA_ptr,
                    cute.make_layout((bM, num_stages), stride=(1, bM)),
                )
                sSB = cute.make_tensor(
                    sSB_ptr,
                    cute.make_layout((num_stages,)),
                )

            # Peeled first (local) iteration: B data + B scale before wait,
            # A data after wait, A scale overlaps with cp.async A. This
            # rank's first global K-tile is `split_rank` (== 0 when
            # split_k==1, recovering the original behavior exactly).
            g_k_first = split_rank
            mainloop_pipeline.producer_acquire(producer_state)
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, g_k_first],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )
            if cutlass.const_expr(self.fp32_scale):
                scale_k_base = g_k_first * TILE_K_FP8_C // 128
                for sb4 in cutlass.range_constexpr(SCALE_SUBBLOCKS):
                    sSB[producer_state.index, sb4] = gSB[scale_k_base + sb4]
            else:
                packed_k = (g_k_first * TILE_K_FP8_C // 128) // 4
                sSB[producer_state.index] = gSB[packed_k]

            cute.arch.griddepcontrol_wait()

            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, g_k_first],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            if cutlass.const_expr(self.fp32_scale):
                for sb4 in cutlass.range_constexpr(SCALE_SUBBLOCKS):
                    sSA[m_slot, producer_state.index, sb4] = gSA[scale_k_base + sb4]
            else:
                sSA[m_slot, producer_state.index] = gSA[packed_k]
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

            for local_k_tile in range(1, tiles_this_rank):
                mainloop_pipeline.producer_acquire(producer_state)
                if cutlass.const_expr(self.split_k > 1):
                    g_k_idx = split_rank + self.split_k * local_k_tile
                else:
                    g_k_idx = local_k_tile
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, g_k_idx],
                    tAsA[None, None, None, producer_state.index],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, g_k_idx],
                    tBsB[None, None, None, producer_state.index],
                    pred=tBpB,
                )
                if cutlass.const_expr(self.fp32_scale):
                    scale_k_base = g_k_idx * TILE_K_FP8_C // 128
                    for sb4 in cutlass.range_constexpr(SCALE_SUBBLOCKS):
                        sSA[m_slot, producer_state.index, sb4] = gSA[scale_k_base + sb4]
                        sSB[producer_state.index, sb4] = gSB[scale_k_base + sb4]
                else:
                    packed_k = (g_k_idx * TILE_K_FP8_C // 128) // 4
                    sSA[m_slot, producer_state.index] = gSA[packed_k]
                    sSB[producer_state.index] = gSB[packed_k]
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            mainloop_pipeline.producer_tail(producer_state)

        else:
            cute.arch.setmaxregister_increase(232)
            lane_id = mma_tidx % 32
            mma_warp_idx = mma_tidx // 32
            NUM_MMA_WARPS: cutlass.Constexpr = self.num_mma_warps

            thr_mma = tiled_mma.get_slice(lane_id)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type
            )
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mB.element_type
            )
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane_id)
            thr_s2r_B = tiled_s2r_B.get_slice(lane_id)
            tCsA_v = thr_s2r_A.partition_S(sA)
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCsB_v = thr_s2r_B.partition_S(sB)
            tCrB_v = thr_s2r_B.retile(tCrB)

            num_k_block = cute.size(tCrA, mode=[2])
            K_PER_WARP: cutlass.Constexpr = num_k_block // NUM_MMA_WARPS
            KB_PER_SCALE: cutlass.Constexpr = 4
            SCALE_GROUPS_PER_WARP: cutlass.Constexpr = K_PER_WARP // KB_PER_SCALE

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages
            )

            m_row_0 = lane_id // 4
            m_row_1 = m_row_0 + 8

            # Pre-create smem scale tensors for reading
            if cutlass.const_expr(self.fp32_scale):
                sSA_mma = cute.make_tensor(
                    sSA_ptr,
                    cute.make_layout(
                        (bM, num_stages, SCALE_SUBBLOCKS),
                        stride=(SCALE_SUBBLOCKS, bM * SCALE_SUBBLOCKS, 1),
                    ),
                )
                sSB_mma = cute.make_tensor(
                    sSB_ptr,
                    cute.make_layout(
                        (num_stages, SCALE_SUBBLOCKS), stride=(SCALE_SUBBLOCKS, 1)
                    ),
                )
            else:
                sSA_mma = cute.make_tensor(
                    sSA_ptr,
                    cute.make_layout((bM, num_stages), stride=(1, bM)),
                )
                sSB_mma = cute.make_tensor(
                    sSB_ptr,
                    cute.make_layout((num_stages,)),
                )

            for local_k_tile in range(tiles_this_rank):
                mainloop_pipeline.consumer_wait(consumer_state)

                tCsA_p = tCsA_v[None, None, None, consumer_state.index]
                tCsB_p = tCsB_v[None, None, None, consumer_state.index]

                # A+B scales from smem (loaded by DMA warps this iteration)
                stage = consumer_state.index
                if cutlass.const_expr(not self.fp32_scale):
                    sa0_packed = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sa0_packed[0] = sSA_mma[m_row_0, stage]
                    sa1_packed = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sa1_packed[0] = sSA_mma[m_row_1, stage]
                    sb_packed = cute.make_rmem_tensor((1,), cutlass.Int32)
                    sb_packed[0] = sSB_mma[stage]

                # This rank's global K-tile for `local_k_tile` (see the DMA
                # warp's identical g_k_idx computation above) -- the scale
                # array is indexed by absolute position in the full K range,
                # not by the per-rank loop counter.
                if cutlass.const_expr(self.split_k > 1):
                    g_k_tile = split_rank + self.split_k * local_k_tile
                else:
                    g_k_tile = local_k_tile

                for sg in cutlass.range(SCALE_GROUPS_PER_WARP, unroll_full=True):
                    sg_global = mma_warp_idx * SCALE_GROUPS_PER_WARP + sg

                    if cutlass.const_expr(self.fp32_scale):
                        # sg_global already indexes directly into this
                        # stage's SCALE_SUBBLOCKS-wide sub-block dimension --
                        # no packed-word decode needed, just a native fp32
                        # multiply against the weight's actual block scale.
                        scale_m0 = (
                            sSA_mma[m_row_0, stage, sg_global]
                            * sSB_mma[stage, sg_global]
                        )
                        scale_m1 = (
                            sSA_mma[m_row_1, stage, sg_global]
                            * sSB_mma[stage, sg_global]
                        )
                    else:
                        k_fp8_base = g_k_tile * TILE_K_FP8_C + sg_global * 128
                        scale_k_idx = k_fp8_base // 128
                        packed_k = scale_k_idx // 4
                        byte_k = scale_k_idx - packed_k * 4

                        scale_m0 = fused_ue8m0_scale(
                            sa0_packed[0], sb_packed[0], byte_k
                        )
                        scale_m1 = fused_ue8m0_scale(
                            sa1_packed[0], sb_packed[0], byte_k
                        )

                    tCrP = tiled_mma_fp8.make_fragment_C(tCgC)
                    tCrP.fill(0.0)
                    for kb in cutlass.range(KB_PER_SCALE, unroll_full=True):
                        k_block = mma_warp_idx * K_PER_WARP + sg * KB_PER_SCALE + kb
                        cute.copy(
                            tiled_s2r_A,
                            tCsA_p[None, None, k_block],
                            tCrA_v[None, None, 0],
                        )
                        cute.copy(
                            tiled_s2r_B,
                            tCsB_p[None, None, k_block],
                            tCrB_v[None, None, 0],
                        )
                        cute.gemm(
                            tiled_mma_fp8,
                            tCrP,
                            cute.recast_tensor(
                                tCrA[None, None, 0], cutlass.Float8E4M3FN
                            ),
                            cute.recast_tensor(
                                tCrB[None, None, 0], cutlass.Float8E4M3FN
                            ),
                            tCrP,
                        )

                    p0, p1, p2, p3 = tCrP[0], tCrP[1], tCrP[2], tCrP[3]
                    p4, p5, p6, p7 = tCrP[4], tCrP[5], tCrP[6], tCrP[7]
                    tCrC[0] = tCrC[0] + p0 * scale_m0
                    tCrC[1] = tCrC[1] + p1 * scale_m0
                    tCrC[2] = tCrC[2] + p2 * scale_m1
                    tCrC[3] = tCrC[3] + p3 * scale_m1
                    tCrC[4] = tCrC[4] + p4 * scale_m0
                    tCrC[5] = tCrC[5] + p5 * scale_m0
                    tCrC[6] = tCrC[6] + p6 * scale_m1
                    tCrC[7] = tCrC[7] + p7 * scale_m1

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # Signal dependent kernels after GEMM is done
            if mma_tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()
                # pass
            cute.arch.sync_threads()

            # Epilogue: warp reduction + global store
            smem_red_ptr = cute.arch.alloc_smem(
                cutlass.Float32, bM * bN * NUM_MMA_WARPS, alignment=16
            )
            smem_warp = cute.make_tensor(
                smem_red_ptr + mma_warp_idx * bM * bN,
                cute.make_layout((bM, bN), stride=(bN, 1)),
            )
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)
            cute.arch.sync_threads()

            num_elems: cutlass.Constexpr = bM * bN
            elems_per_thread: cutlass.Constexpr = num_elems // self.num_mma_threads

            # This thread's intra-CTA-reduced totals (this rank's own partial
            # K-sum for split_k>1, or the full sum for split_k==1) and their
            # validity (OOB M/N tiles are masked out and never stored).
            totals = cute.make_rmem_tensor((elems_per_thread,), cutlass.Float32)
            valid = cute.make_rmem_tensor((elems_per_thread,), cutlass.Boolean)
            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * self.num_mma_threads + mma_tidx
                m = idx // bN
                n = idx % bN
                global_m = real_bid_m * bM + m
                global_n = bid_n * bN + n
                is_valid = cute.elem_less(global_m, M_out) and cute.elem_less(
                    global_n, N_out
                )
                valid[ei] = is_valid
                total = cutlass.Float32(0.0)
                if is_valid:
                    for w in cutlass.range_constexpr(NUM_MMA_WARPS):
                        p = smem_red_ptr + w * bM * bN + idx
                        t = cute.make_tensor(p, cute.make_layout((1,)))
                        r = cute.make_rmem_tensor((1,), cutlass.Float32)
                        cute.autovec_copy(t, r)
                        total = total + r[0]
                totals[ei] = total

            # split_k>1: merge the (split_k-1) peers' partial K-sums (fp32)
            # into the leader's `totals` via DSMEM st.async + mbarrier
            # tx-completion -- no producer-side release fence, no
            # consumer-side acquire fence needed (see the helper docstrings
            # at the top of this file / cutedsl_bf16_gemm.py's split-K diff).
            do_store = cutlass.Boolean(True)
            if cutlass.const_expr(self.split_k > 1):
                do_store = split_rank == 0
                frag_sz: cutlass.Constexpr = elems_per_thread
                if split_rank != 0:
                    remote_ptr = cute.arch.map_dsmem_ptr(red_buf_ptr, 0)
                    base_addr = remote_ptr.toint() + (
                        (
                            (split_rank - 1) * (self.num_mma_threads * frag_sz)
                            + mma_tidx * frag_sz
                        )
                        * 4
                    )
                    remote_bar = cute.arch.map_dsmem_ptr(bar_splitk, 0).toint()
                    if cutlass.const_expr(frag_sz % 4 == 0):
                        for i in cutlass.range_constexpr(frag_sz // 4):
                            _st_async_shared_cluster_v4_f32(
                                base_addr + i * 16,
                                remote_bar,
                                totals[i * 4],
                                totals[i * 4 + 1],
                                totals[i * 4 + 2],
                                totals[i * 4 + 3],
                            )
                    else:
                        for i in cutlass.range_constexpr(frag_sz):
                            _st_async_shared_cluster_f32(
                                base_addr + i * 4, remote_bar, totals[i]
                            )
                else:
                    # Single elected arrive declares the expected tx bytes
                    # (all (split_k-1) peers' contributions); the phase
                    # completes once every peer's st.async has landed. NOT
                    # elect_one -- that elects one lane PER WARP and this
                    # epilogue spans 4 warps (4 arrives would corrupt the
                    # count-1 barrier).
                    if mma_tidx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_splitk,
                            (self.split_k - 1) * self.num_mma_threads * frag_sz * 4,
                        )
                    red_layout = cute.make_layout(
                        (self.split_k - 1, self.num_mma_threads, frag_sz),
                        stride=(self.num_mma_threads * frag_sz, frag_sz, 1),
                    )
                    rLocal = cute.make_tensor(red_buf_ptr, red_layout)
                    _mbarrier_wait_parity_spin(bar_splitk.toint(), 0)
                    for s in cutlass.range_constexpr(self.split_k - 1):
                        for i in cutlass.range_constexpr(frag_sz):
                            totals[i] = totals[i] + rLocal[s, mma_tidx, i]

            if do_store:
                for ei in cutlass.range_constexpr(elems_per_thread):
                    if valid[ei]:
                        idx = ei * self.num_mma_threads + mma_tidx
                        m = idx // bN
                        n = idx % bN
                        global_m = real_bid_m * bM + m
                        global_n = bid_n * bN + n
                        out_p = (mC.iterator + global_m * N_out + global_n).align(2)
                        out_t = cute.make_tensor(out_p, cute.make_layout((1,)))
                        out_r = cute.make_rmem_tensor((1,), self.out_dtype)
                        out_r[0] = totals[ei].to(self.out_dtype)
                        out_t[0] = out_r[0]

        cute.arch.sync_threads()


# ///////////////////////////////////////////////////////////////////////////
# Scale-layout adapters (new -- the kernel's ue8m0 packing does NOT match
# either sglang's or DeepGEMM's native ue8m0 layouts, see below)
# ///////////////////////////////////////////////////////////////////////////
#
# Weight: sglang's native weight is fp8 quantized against an *arbitrary*
# (non-power-of-two) fp32 per-block scale. This kernel needs a power-of-two
# (ue8m0) scale -- naively rounding the existing scale up to the nearest
# power of two WITHOUT re-deriving the fp8 weight values against that new
# scale corrupts every element by up to ~2x per 128x128 block (independently
# per block, so it doesn't just look like "the output is off by a constant
# factor" -- it decorrelates entirely once summed over many K-blocks).
# `requant_weight_ue8m0` (sglang's own DeepGEMM-integration helper: dequant
# with the old scale, re-quantize fp8 against the new power-of-two scale) is
# the correct fix and is reused directly rather than hand-rolled.
#
# `requant_weight_ue8m0`'s returned scale is *also* the DeepGEMM-native
# packed ue8m0 int32 layout, i.e. `(N, K//128//4)` int32 row-major (one
# packed word per weight row, broadcast-redundant across each 128-row block)
# -- but physically it's a *transposed view* over `(K//128//4, N)` storage.
# This kernel's raw pointer math (`mSB.iterator + n_block_idx*128`,
# stride=N) needs that physical `(K//128//4, N)` layout directly, so
# `.t().contiguous()` is required before handing it to the op (a bare
# `.reshape(-1)` on the transposed view would silently force a copy in the
# wrong, *logical* row-major order instead of the physical one).
#
# Both the requantized weight and its packed scale are computed once per
# weight and cached on the `weight_scale` parameter, mirroring sglang's
# existing `format_ue8m0` flag pattern in
# `requant_block_scale_ue8m0_for_deepgemm`.
#
# Activation scale: sglang's `sglang_per_token_group_quant_fp8_ue8m0` already
# produces packed ue8m0 int32 on GPU, quantized directly against the
# power-of-two scale (no mismatch issue -- it's a single fused quantize, not
# a post-hoc scale conversion). Its returned tensor is a *transposed view*
# just like the weight scale: physical storage is `(K//128//4, M)`, exposed
# as a `(M, K//128//4)` logical shape via strides. This kernel wants the
# actual *physical* layout to be `(M, K//128//4)`
# (`mSA.iterator + m*num_packed_k`, contiguous per row) -- a `.contiguous()`
# call on sglang's output forces exactly that (cheap: M is tiny, <=64 rows).


_REQUANT_CHUNK_ROWS = 128  # multiple of the 128-row block size


def _requant_weight_for_ll(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

    cached = getattr(weight_scale, "_ll_requant", None)
    if cached is not None:
        return cached

    # Requantize in row-chunks aligned to the 128-row block granularity
    # instead of dequantizing the WHOLE weight into one fp32/bf16 buffer at
    # once. For a large MoE model, each expert's gate_up_proj/down_proj
    # weight hits this on first use (lazily, during decode-graph warmup) --
    # a single-shot dequant of one big (N,K) weight can itself be hundreds
    # of MB, which is exactly the allocation that OOM'd an already
    # near-full-memory server mid warmup (see conversation: crash inside
    # block_quant_dequant during bs=30 decode-graph capture). Chunking
    # bounds the transient dequant buffer to one chunk's size regardless of
    # the full weight's size; each 128-row block is quantized independently
    # so chunking on 128-row boundaries is numerically exact, not an
    # approximation.
    n = weight.shape[0]
    if n <= _REQUANT_CHUNK_ROWS:
        new_weight, new_scale = requant_weight_ue8m0(weight, weight_scale, [128, 128])
    else:
        weight_chunks = []
        scale_chunks = []
        for start in range(0, n, _REQUANT_CHUNK_ROWS):
            end = min(start + _REQUANT_CHUNK_ROWS, n)
            w_chunk, s_chunk = requant_weight_ue8m0(
                weight[start:end], weight_scale[start // 128 : end // 128], [128, 128]
            )
            weight_chunks.append(w_chunk)
            scale_chunks.append(s_chunk)
        new_weight = torch.cat(weight_chunks, dim=0)
        new_scale = torch.cat(scale_chunks, dim=0)

    new_scale = new_scale.t().contiguous()  # (K//128//4, N) physical
    cached = (new_weight, new_scale)
    try:
        weight_scale._ll_requant = cached
    except AttributeError:
        pass  # not a Parameter/Tensor subclass that allows extra attrs
    return cached


# ///////////////////////////////////////////////////////////////////////////
# sglang op registration
# ///////////////////////////////////////////////////////////////////////////

_compiled_cache: dict = {}


def pick_ll_tile_n(n: int, k: int) -> int:
    """tile_n autotune (was a TODO in the original vLLM kernel: "add tile_n,
    tile_k, num_stages, num_dma_warps to autotuning space"). CUPTI cold-L2
    sweep over tile_n in {8,16,32} at m in {1,4,16} across 5 real decode-time
    (n,k) shapes found the winner tracks total per-row work (n*k), not m or
    either dimension alone -- e.g. (n=6144,k=5120) [n*k=31M] wants tile_n=8,
    (n=17408,k=5120) [n*k=89M] wants tile_n=16 despite the *same* k.

    tile_n=32 is DISABLED: independently verified to produce wrong results
    (cos_sim~0.71 vs the plain-matmul reference, reproduces at every M tried)
    regardless of split_k -- pre-existing in the epilogue's accumulator
    unpacking, unrelated to split_k, not yet root-caused. This previously
    meant `pick_ll_tile_n` silently routed real shapes like gate_up_proj
    (n=5120,k=34816, work=178M) through a numerically-wrong kernel; capping
    at 16 here is the safe stopgap until the tile_n=32 bug is fixed. See
    vllm-ll-fp8-block-gemm-test.md memory for the full sweep."""
    work = n * k
    if work < 40_000_000:
        return 8
    return 16


def pick_ll_split_k(n: int, k: int, tile_n: int) -> int:
    """split_k autotune. CUPTI cold-L2 sweep (tile_n in {8,16} x split_k in
    {1,2,3,4}) across the 5 real decode-time shapes from `pick_ll_tile_n`'s
    own sweep, m in {1,4,16}: split_k only won for ONE of the five --
    out_proj (n=6144,k=5120, tile_n=8, k_tiles=20) -- by a consistent but
    modest ~3-4% (split_k=3). Every other shape got *worse* with any
    split_k>1, INCLUDING down_proj, which has the exact same k_tiles=20 as
    out_proj but a wider tile_n=16 -- so this is NOT just a "short K" effect:
    at tile_n=8 the per-K-tile MMA work is small enough that mainloop
    pipeline latency dominates, and splitting K creates more overlapping
    short pipelines to hide it; at tile_n=16+ the per-tile work is already
    enough to hide that latency, so split_k only adds DSMEM/cluster overhead
    for no benefit. Gated on both conditions actually observed (tile_n==8
    and short-ish K) rather than generalizing past them -- this is a narrow,
    conservative win, not a general "split_k helps large K" heuristic (it
    does not: gate_up_proj/qkv_gate_proj/in_proj_qkvz all have much larger K
    and were all worse with any split_k). See vllm-ll-fp8-block-gemm-test.md
    memory for the full sweep."""
    if tile_n != 8:
        return 1
    k_tiles = k // 256
    if 10 <= k_tiles <= 25:
        return 3
    return 1


def _compile_ll_kernel(
    mA, mB, mC, mSA, mSB, tile_n: int, split_k: int, fp32_scale: bool = False
) -> object:
    cache_key = (tile_n, split_k, fp32_scale)
    compiled = _compiled_cache.get(cache_key)
    if compiled is not None:
        return compiled
    gemm = LLFp8BlockGemm(
        tile_n=tile_n,
        tile_k=256,
        num_stages=2,
        num_dma_warps=4,
        split_k=split_k,
        fp32_scale=fp32_scale,
    )
    stream0 = CUstream(torch.cuda.default_stream().cuda_stream)
    # `--enable-tvm-ffi` (paired with `enable_tvm_ffi=True` on every
    # `from_dlpack` below) switches the compiled callable to a lighter-weight
    # TVM-FFI calling convention with less Python/ctypes marshalling overhead
    # per invocation -- dropped when this was first ported from vLLM's
    # `ll_fp8_block.py`, which matters more here than for a typical kernel
    # since this one is specifically meant to minimize per-call latency at
    # tens-of-microseconds durations, where host-side dispatch overhead is a
    # non-trivial fraction of the total.
    compiled = cute.compile(
        gemm, mA, mB, mC, mSA, mSB, stream0, options="--enable-tvm-ffi"
    )
    _compiled_cache[cache_key] = compiled
    return compiled


def use_ll_fp8_block_gemm_shape(n: int, k: int) -> bool:
    """M-independent half of `use_ll_fp8_block_gemm`'s check: N must be a
    multiple of 32 (the widest tile_n candidate, so `pick_ll_tile_n` never
    has to fall back) and K must be a multiple of 512 (tile_k_fp8) since the
    mainloop has no K-remainder handling. Split out so callers that need to
    know before M is known (e.g. eager load-time requant, below) don't have
    to fake an M value."""
    return n % 32 == 0 and k % 512 == 0


def use_ll_fp8_block_gemm(m: int, n: int, k: int) -> bool:
    """Shape/tile constraints across all tile_n candidates {8,16,32}:
    tile_m=16 (fixed in-kernel) with no swap option, so this only helps for
    small M."""
    return m <= 64 and use_ll_fp8_block_gemm_shape(n, k)


def requant_weight_for_ll_fp8_block_gemm(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> None:
    """Eagerly populate `_requant_weight_for_ll`'s cache on `weight_scale`.

    Must run during `process_weights_after_loading` (mirrors DeepGEMM's
    `requant_block_scale_ue8m0_for_deepgemm`, called from the same site) so
    the extra full-size fp8 weight copy this kernel needs is materialized --
    and counted against `avail_mem` -- before KV-cache/mamba-cache sizing and
    CUDA graph capture. Previously this only happened lazily on the first
    forward call, which lands mid decode-graph warmup after those pools
    already claimed most of the GPU, with no room left for it -- see the
    `_requant_weight_for_ll` OOM in the module's git history / conversation
    log for the crash this fixes."""
    _requant_weight_for_ll(weight, weight_scale)


def _ll_fp8_block_gemm_run(
    q_input: torch.Tensor,
    input_scale_ue8m0: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    out: torch.Tensor,
    tile_n: int,
    split_k: int,
    fp32_scale: bool,
) -> None:
    # Mark M/N/K as dynamic (with a divisibility hint) so the SAME compiled
    # kernel handles every shape -- without this, `cute.compile` bakes in
    # whatever M/N/K happened to be passed on the FIRST call, and the
    # constant-keyed `_compiled_cache` below would silently reuse that
    # compiled kernel (with the wrong dimensions baked in) for every
    # subsequent, differently-shaped call. Mirrors vLLM's original
    # `_get_compiled` in `ll_fp8_block.py`.
    div = 8
    mA = (
        from_dlpack(q_input.view(torch.bfloat16), assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mB = (
        from_dlpack(weight.view(torch.bfloat16), assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    mC = (
        from_dlpack(out, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=div)
    )
    # 16 (not the int32-minimum 4) matches flashinfer's own scale-factor
    # tensors (see dense_blockscaled_gemm's grouped-gemm sibling,
    # `assumed_align=16` on sfa_ptr/sfb_ptr) -- PyTorch's CUDA caching
    # allocator aligns every fresh allocation to well beyond 16 bytes, and
    # both of these are always fresh (`.contiguous()`/`torch.empty(...)`
    # output from `sglang_per_token_group_quant_fp8_ue8m0` and
    # `requant_weight_ue8m0`), never a raw pass-through of an arbitrary
    # caller-provided slice, so the stronger hint is safe here.
    mSA = from_dlpack(
        input_scale_ue8m0.reshape(-1), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic()
    mSB = from_dlpack(
        weight_scale_ue8m0.reshape(-1), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic()
    compiled = _compile_ll_kernel(mA, mB, mC, mSA, mSB, tile_n, split_k, fp32_scale)
    current_stream = CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(mA, mB, mC, mSA, mSB, current_stream)


def _ll_fp8_block_gemm_fake(
    q_input: torch.Tensor,
    input_scale_ue8m0: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_ue8m0: torch.Tensor,
    out: torch.Tensor,
    tile_n: int,
    split_k: int,
    fp32_scale: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="ll_fp8_block_gemm",
    op_func=_ll_fp8_block_gemm_run,
    mutates_args=["out"],
    fake_impl=_ll_fp8_block_gemm_fake,
)


@debug_kernel_api
def ll_fp8_block_scaled_mm(
    q_input: torch.Tensor,
    input_scale_ue8m0: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_compact: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """out[M, N] = (ue8m0-scale * q_input[M, K]) @ (block-scale * weight[N, K]).T

    q_input: fp8 e4m3, (M, K), M <= 64.
    input_scale_ue8m0: packed ue8m0 int32, logical shape (M, K//128//4) --
        e.g. straight from `sglang_per_token_group_quant_fp8_ue8m0(...).
        contiguous()` (see module docstring for why `.contiguous()` matters).
    weight: fp8 e4m3, (N, K), quantized against `weight_scale_compact` (an
        arbitrary, non-power-of-two scale -- as sglang normally produces).
    weight_scale_compact: sglang's *native* compact fp32 block scale,
        (N // 128, K // 128). Requantized (weight *and* scale, cached on this
        tensor) to this kernel's ue8m0 layout internally, so callers don't
        need to know about the format mismatch.
    """
    assert out_dtype == torch.bfloat16, "LL FP8 block GEMM only supports bf16 output"
    m, k = q_input.shape
    n = weight.shape[0]
    tile_n = pick_ll_tile_n(n, k)
    split_k = pick_ll_split_k(n, k, tile_n)
    weight_ue8m0, weight_scale_ue8m0 = _requant_weight_for_ll(
        weight, weight_scale_compact
    )
    out = torch.empty(m, n, dtype=out_dtype, device=q_input.device)
    torch.ops.sglang.ll_fp8_block_gemm(
        q_input,
        input_scale_ue8m0,
        weight_ue8m0,
        weight_scale_ue8m0,
        out,
        tile_n,
        split_k,
        False,
    )
    return out


def _broadcast_weight_scale_for_ll(weight_scale_compact: torch.Tensor) -> torch.Tensor:
    """(N//128, K//128) fp32 -> physical (K//128, N) fp32, one scale value
    per weight row (broadcast across each 128-row block), matching the
    layout `_requant_weight_for_ll` produces for the ue8m0 path -- but here
    it's a pure reshape/broadcast (no dequant/requant of the weight or its
    scale values), since the kernel's fp32-scale mode consumes the scale
    as-is instead of demanding a power-of-two value.

    Cached on `weight_scale_compact` (mirrors `_ll_requant` on the ue8m0
    path) since it's the same tensor across every forward call for a given
    weight.
    """
    cached = getattr(weight_scale_compact, "_ll_fp32_broadcast", None)
    if cached is not None:
        return cached
    n_blocks, k_blocks = weight_scale_compact.shape
    broadcast = weight_scale_compact.repeat_interleave(128, dim=0)  # (N, K//128)
    result = broadcast.t().contiguous()  # (K//128, N) physical
    try:
        weight_scale_compact._ll_fp32_broadcast = result
    except AttributeError:
        pass
    return result


@debug_kernel_api
def ll_fp8_block_scaled_mm_fp32(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_compact: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Same GEMM as `ll_fp8_block_scaled_mm`, but with plain fp32 block
    scales instead of ue8m0 (power-of-two) -- no weight/scale requantization,
    so no accuracy loss from rounding the scale to the nearest power of two.

    q_input: fp8 e4m3, (M, K), M <= 64.
    input_scale: plain fp32, (M, K // 128), row-major contiguous.
    weight: fp8 e4m3, (N, K), quantized against `weight_scale_compact` --
        used as-is, never re-quantized (unlike the ue8m0 path).
    weight_scale_compact: sglang's native compact fp32 block scale,
        (N // 128, K // 128).
    """
    assert out_dtype == torch.bfloat16, "LL FP8 block GEMM only supports bf16 output"
    m, k = q_input.shape
    n = weight.shape[0]
    tile_n = pick_ll_tile_n(n, k)
    split_k = pick_ll_split_k(n, k, tile_n)
    input_scale = input_scale.contiguous()
    weight_scale = _broadcast_weight_scale_for_ll(weight_scale_compact)
    out = torch.empty(m, n, dtype=out_dtype, device=q_input.device)
    torch.ops.sglang.ll_fp8_block_gemm(
        q_input, input_scale, weight, weight_scale, out, tile_n, split_k, True
    )
    return out
