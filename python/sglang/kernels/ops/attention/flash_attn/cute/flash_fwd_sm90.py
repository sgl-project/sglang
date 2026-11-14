# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

from functools import partial
from types import SimpleNamespace
from typing import Callable, Literal, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import BFloat16, Float16, Float32, Int32, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from sglang.kernels.ops.attention.flash_attn.cute import pipeline as pipeline_custom
from sglang.kernels.ops.attention.flash_attn.cute import utils
from sglang.kernels.ops.attention.flash_attn.cute.block_info import BlockInfo
from sglang.kernels.ops.attention.flash_attn.cute.block_sparse_utils import (
    consume_block_sparse_loads,
    produce_block_sparse_loads,
)
from sglang.kernels.ops.attention.flash_attn.cute.block_sparsity import (
    BlockSparseTensors,
)
from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned,
)
from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd import (
    FlashAttentionForwardBase,
)
from sglang.kernels.ops.attention.flash_attn.cute.mask import AttentionMask
from sglang.kernels.ops.attention.flash_attn.cute.named_barrier import NamedBarrierFwd
from sglang.kernels.ops.attention.flash_attn.cute.pack_gqa import (
    PackGQA,
    make_packgqa_tiled_tma_atom,
    pack_gqa_layout,
)
from sglang.kernels.ops.attention.flash_attn.cute.paged_kv import PagedKVManager
from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sglang.kernels.ops.attention.flash_attn.cute.softmax import (
    Softmax,
    apply_score_mod_inner,
)
from sglang.kernels.ops.attention.flash_attn.cute.tile_scheduler import (
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        has_bias: bool = False,
        bias_block_size: int = 128,
        rel_extent_padded: int = 128,
        fp8_kv_dequant: bool = False,
        fp8_kv_mma: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        # FP8 e4m3 paged K/V is staged in smem, cast to fp16 by the MMA warpgroups,
        # and its descales are folded into the softmax scale and the final O scale.
        self.fp8_kv_dequant = fp8_kv_dequant
        # hdim 512 alternative to the cast: QK and PV run as FP8 WGMMA. Q and P are
        # each split into an e4m3 high term and an e4m3 residual (compensated pair),
        # and PV is added to the fp32 O once per KV block because FP8 WGMMA
        # accumulates below fp32 precision.
        self.fp8_kv_mma = fp8_kv_mma
        if fp8_kv_mma:
            assert fp8_kv_dequant and self.tile_hdim == self.tile_hdimv == 512
            # K and V are TMA'd by page; the PV ordering below needs the overlap loop.
            assert intra_wg_overlap and not paged_kv_non_tma
            # P is scaled before its e4m3 cast so small probabilities keep precision
            # over long contexts; the inverse joins the final O scale.
            self.fp8_p_scale = 256.0
            # K needs no conversion, so it is TMA'd into one of two sK slots and the
            # next K load overlaps the current QK.
            self.num_k_slots_mma = 2
            # Causal/local builds split the QK sum across two accumulators; one
            # accumulator chain there makes ptxas serialize the WGMMAs (C7511).
            self.qk_split_mma = self.is_causal or self.is_local
        if fp8_kv_dequant:
            assert self.dtype == Float16, "FP8-KV dequant computes in fp16"
            # hdim 512: two MMA warpgroups share 64 Q rows and each owns one
            # 256-wide half of O and V. hdim 256: each owns 64 Q rows and reads
            # all of V.
            assert self.tile_hdim == self.tile_hdimv and self.tile_hdim in (256, 512)
            # The cast writes only slot 0 of sK/sV; one staging slot feeds both.
            assert self.num_stages == 1
            # RS PV leaves no sP, which keeps sQ + sK + sV + sStage within 227 KB.
            assert mma_pv_is_rs and not has_bias and not self.Q_in_regs
        self.buffer_align_bytes = 1024
        self.use_tma_KV = not paged_kv_non_tma
        self.has_bias = has_bias
        self.rel_extent_padded = rel_extent_padded
        if has_bias:
            assert rel_extent_padded % self.tile_n == 0
        self.bias_n_max = rel_extent_padded // self.tile_n if has_bias else 0
        self.bias_block_size = bias_block_size
        assert self.use_tma_KV or not (
            self.check_hdim_oob or self.check_hdim_v_oob
        ), "Paged KV does not support irregular head dim"
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        atom_layout_n = 2 if max(self.tile_hdim, self.tile_hdimv) > 256 else 1
        mma_dtype = self.kv_dtype if self.fp8_kv_mma else self.dtype
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, atom_layout_n, 1),
            tiler_mn=(64, self.tile_n),
        )
        # FP8 WGMMA takes K-major operands only, so the FP8 path stores V transposed.
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            warpgroup.OperandMajorMode.K,
            (
                warpgroup.OperandMajorMode.K
                if self.fp8_kv_mma
                else warpgroup.OperandMajorMode.MN
            ),
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                atom_layout_n,
                1,
            ),
            tiler_mn=(64, min(256, self.tile_hdimv)),
            a_source=(
                warpgroup.OperandSource.RMEM
                if self.mma_pv_is_rs
                else warpgroup.OperandSource.SMEM
            ),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                self.buffer_align_bytes,
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_Bias_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        sBias_struct = cute.struct.Align[
            cute.struct.MemRange[
                self.bias_dtype if const_expr(self.has_bias) else cutlass.BFloat16,
                cute.cosize(self.sBias_layout) if const_expr(self.has_bias) else 0,
            ],
            1024,
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageQKVBias:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_Bias: mbar_ptr_Bias_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sBias: sBias_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQVBias:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_Bias: mbar_ptr_Bias_struct
            sQ: sQV_struct
            sK: sK_struct
            sBias: sBias_struct
            sP: sP_struct

        if const_expr(self.fp8_kv_dequant):
            sStage_struct = cute.struct.Align[
                cute.struct.MemRange[
                    self.kv_dtype, cute.cosize(self.sStage_layout)
                ],
                self.buffer_align_bytes,
            ]
            if const_expr(self.fp8_kv_mma):
                sQ_mma_struct, sK_mma_struct, sVt_mma_struct = [
                    cute.struct.Align[
                        cute.struct.MemRange[self.kv_dtype, cute.cosize(layout)],
                        self.buffer_align_bytes,
                    ]
                    for layout in (self.sQ_mma_layout, self.sK_layout, self.sVt_mma_layout)
                ]
                mbar_ptr_K_mma_struct = cute.struct.MemRange[
                    cutlass.Int64, self.num_k_slots_mma * 2
                ]

                # sQ holds the fp16/bf16 Q as TMA'd and, once it is quantized, the
                # e4m3 residual term; sQ_mma the high term. K lands in sK straight
                # from GMEM; V goes through sStage and is transposed into sVt.
                @cute.struct
                class SharedStorageQKVMma:
                    mbar_ptr_Q: mbar_ptr_Q_struct
                    mbar_ptr_Stage: mbar_ptr_Q_struct
                    mbar_ptr_K: mbar_ptr_K_mma_struct
                    sQ: sQ_struct
                    sQ_mma: sQ_mma_struct
                    sK: sK_mma_struct
                    sVt: sVt_mma_struct
                    sStage: sStage_struct

                return SharedStorageQKVMma

            @cute.struct
            class SharedStorageQKVStage:
                mbar_ptr_Q: mbar_ptr_Q_struct
                mbar_ptr_Stage: mbar_ptr_Q_struct
                sV: sV_struct
                sQ: sQ_struct
                sK: sK_struct
                sStage: sStage_struct

            return SharedStorageQKVStage
        if const_expr(self.has_bias):
            return (
                SharedStorageQKVBias
                if const_expr(not self.Q_in_regs)
                else SharedStorageSharedQVBias
            )
        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    def _check_fp8_kv_types(self, mQ, mK, mV, mO, blocksparse_tensors):
        if const_expr(not (mK.element_type == mV.element_type == cutlass.Float8E4M3FN)):
            raise TypeError("FP8-KV dequant expects fp8 e4m3 K/V")
        if const_expr(mQ.element_type not in [Float16, BFloat16]):
            raise TypeError("FP8-KV dequant expects fp16 or bf16 Q")
        expected_o = [Float32] if self.is_split_kv else [Float16, BFloat16]
        if const_expr(mO.element_type not in expected_o):
            raise TypeError(f"FP8-KV dequant expects O in {expected_o}")
        if const_expr(blocksparse_tensors is not None):
            raise NotImplementedError("FP8-KV dequant does not support block sparsity")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        # DescaleTensors of (b, h_k) fp32 q/k/v descales, any may be None; FP8-KV only.
        descale_tensors=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        mBias: Optional[
            cute.Tensor
        ] = None,  # (b, s_q, h, rel_extent_padded) or (total_q, h, rel_extent_padded)
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        The FP8-KV dequant path instead takes fp16/bf16 Q and O with fp8 e4m3 K/V.
        """

        if const_expr(self.fp8_kv_dequant):
            self._check_fp8_kv_types(mQ, mK, mV, mO, blocksparse_tensors)
            self.kv_dtype = mK.element_type
            self.q_dtype = mQ.element_type
            # Split partials are fp32 accumulators written straight from acc_O.
            if const_expr(not self.is_split_kv):
                self.o_dtype = mO.element_type
        else:
            self._check_type(
                *(
                    t.element_type if t is not None else None
                    for t in (
                        mQ,
                        mK,
                        mV,
                        mO,
                        mLSE,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mSeqUsedQ,
                        mSeqUsedK,
                    )
                )
            )

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        self.bias_dtype = (
            mBias.element_type if const_expr(mBias is not None) else self.dtype
        )
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        if const_expr(mBias is not None):
            mBias = assume_tensor_aligned(mBias)
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ = layout_utils.select(mQ, Q_layout_transpose)
        if const_expr(mBias is not None):
            mBias = layout_utils.select(mBias, Q_layout_transpose)
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        if const_expr(self.is_split_kv):
            num_splits = mO.shape[0]
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = (
                [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            )
        else:
            num_splits = Int32(1)
            O_layout_transpose = (
                [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            )
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
        mO = layout_utils.select(mO, O_layout_transpose)
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {
            1: (256, 56),
            2: (240, 24),
            3: (160, 32),
        }[self.num_wg_mma]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and self.has_bias):
            raise NotImplementedError(
                "Block sparsity + sheared bias is not supported on SM90"
            )

        # FP8-KV: at hdim 512 the two MMA warpgroups split O/V along N
        # (atom_layout_n == 2), so each reads only its half of sV.
        self.v_split_across_wgs = max(self.tile_hdim, self.tile_hdimv) > 256
        # FP8-KV syncs both warpgroups on every K tile, which rules out ping-pong.
        self.use_scheduler_barrier = not self.fp8_kv_dequant and (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        # Split partials are float32; store them straight from registers (no TMA O).
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(
            self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)
        ):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        # FP8-KV: sK/sV hold the fp16 cast of the fp8 tile staged in sStage; with
        # FP8 MMA sK holds the e4m3 K itself.
        sKV_dtype = self.dtype if const_expr(self.fp8_kv_dequant) else mK.element_type
        if const_expr(self.fp8_kv_mma):
            sKV_dtype = self.kv_dtype
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(elem_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for elem_type, shape, stage in [
                (self.q_dtype, (self.tile_m, self.tile_hdim), None),
                (
                    sKV_dtype,
                    (self.tile_n, self.tile_hdim),
                    (
                        self.num_k_slots_mma
                        if const_expr(self.fp8_kv_mma)
                        else self.num_stages
                    ),
                ),
                (sKV_dtype, (self.tile_n, self.tile_hdimv), self.num_stages),
                # sO is the fp16/bf16 store buffer; unused for split: mO is fp32.
                (self.o_dtype, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        # One fp8 staging tile serves both K and V (tile_hdim == tile_hdimv); with
        # FP8 MMA it holds only V.
        self.sStage_layout = None
        self.sQ_mma_layout = None
        self.sVt_mma_layout = None
        if const_expr(self.fp8_kv_dequant):
            self.sStage_layout = sm90_utils.make_smem_layout(
                self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), 1
            )
        if const_expr(self.fp8_kv_mma):
            self.sQ_mma_layout = sm90_utils.make_smem_layout(
                self.kv_dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
            )
            # V^T as (head_dim_v, tile_n), K-major: the PV B operand.
            self.sVt_mma_layout = sm90_utils.make_smem_layout(
                self.kv_dtype,
                LayoutEnum.ROW_MAJOR,
                (self.tile_hdimv, self.tile_n),
                self.num_stages,
            )
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )
        if const_expr(mBias is not None):
            self.sBias_layout = sm90_utils.make_smem_layout(
                self.bias_dtype,
                LayoutEnum.ROW_MAJOR,
                (self.bias_block_size, self.tile_n),
                self.num_stages,
            )
        else:
            self.sBias_layout = None

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mBias is not None):
                mBias = pack_gqa_layout(
                    mBias, self.qhead_per_kvhead, nheads_kv, head_idx=2
                )
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        gmem_tiled_copy_Bias = cpasync.CopyBulkTensorTileG2SOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        if const_expr(mBias is not None):
            self.tma_copy_bytes["Bias"] = cute.size_in_bytes(
                self.bias_dtype, cute.select(self.sBias_layout, mode=[0, 1])
            )
        # FP8-KV: TMA writes the fp8 K/V tiles to sStage, not to sK/sV.
        sK_tma_layout, sV_tma_layout = (
            (self.sStage_layout, self.sStage_layout)
            if const_expr(self.fp8_kv_dequant)
            else (self.sK_layout, self.sV_layout)
        )
        make_tiled_tma_atom_fn = (
            partial(
                make_packgqa_tiled_tma_atom,
                qhead_per_kvhead=self.qhead_per_kvhead,
                head_idx=2,
            )
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(sK_tma_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(sV_tma_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
        tma_atom_Bias, tma_tensor_Bias = None, None
        if const_expr(mBias is not None):
            tma_atom_Bias, tma_tensor_Bias = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_Bias,
                mBias,
                cute.select(self.sBias_layout, mode=[0, 1]),
                (self.bias_block_size, self.tile_n),
                1,  # No mcast
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            (
                cute.size(mQ.shape[3])
                if const_expr(mCuSeqlensQ is None)
                else cute.size(mCuSeqlensQ.shape[0] - 1)
            ),
            num_splits,
            (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mQ.shape[1],
            mV.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        if const_expr(self.has_bias):
            base_softmax_scale = softmax_scale
            softmax_scale_log2, softmax_scale = utils.LOG2_E, None
        else:
            base_softmax_scale = None
            softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
                softmax_scale, self.score_mod
            )
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_tensor_Bias,
            tma_atom_Bias,
            softmax_scale_log2,
            softmax_scale,
            base_softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.sBias_layout,
            self.sStage_layout,
            self.sQ_mma_layout,
            self.sVt_mma_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            num_splits,
            aux_data,
            fastdiv_mods,
            descale_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        mBias: Optional[cute.Tensor],
        tma_atom_Bias: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        base_softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        sBias_layout: cute.ComposedLayout | None,
        sStage_layout: cute.ComposedLayout | None,
        sQ_mma_layout: cute.ComposedLayout | None,
        sVt_mma_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        num_splits: Int32 = 1,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        descale_tensors=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                tma_atom_Bias,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        pipeline_stage = None
        if const_expr(self.fp8_kv_dequant):
            pipeline_k = pipeline_v = None
            pipeline_stage = self._make_pipeline_stage(
                storage.mbar_ptr_Stage.data_ptr(), tma_warp, load_threads, mma_warps
            )
            if const_expr(self.fp8_kv_mma):
                pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                    barrier_storage=storage.mbar_ptr_K.data_ptr(),
                    num_stages=self.num_k_slots_mma,
                    producer_group=tma_warp,
                    consumer_group=mma_warps,
                    tx_count=self.tma_copy_bytes["K"],
                    defer_sync=True,
                )
        elif const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
        if const_expr(self.has_bias):
            pipeline_bias = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Bias.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Bias"],
                defer_sync=True,
            )
        else:
            pipeline_bias = None

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(
            sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype
        )
        # FP8-KV with bf16 Q: the QK MMA reads an fp16 view of the same bytes,
        # which the MMA warpgroups narrow in place once per tile.
        sQ_mma = sQ
        sQ_res = None
        if const_expr(self.fp8_kv_mma):
            sQ_mma = storage.sQ_mma.get_tensor(
                sQ_mma_layout.outer, swizzle=sQ_mma_layout.inner
            )
            # The fp16/bf16 Q bytes are dead once quantized; reuse them for the
            # residual term.
            sQ_res = storage.sQ.get_tensor(
                sQ_mma_layout.outer, swizzle=sQ_mma_layout.inner, dtype=self.kv_dtype
            )
        elif const_expr(self.q_dtype != self.dtype):
            sQ_mma = storage.sQ.get_tensor(
                sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.dtype
            )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(self.fp8_kv_mma):
            sVt = storage.sVt.get_tensor(
                sVt_mma_layout.outer, swizzle=sVt_mma_layout.inner
            )
            sV = layout_utils.transpose_view(sVt)
        else:
            if const_expr(not self.Q_in_regs):
                sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            else:
                sV = storage.sQ.get_tensor(
                    sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
                )
            # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
            sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        if const_expr(self.has_bias):
            sBias = storage.sBias.get_tensor(
                sBias_layout.outer, swizzle=sBias_layout.inner
            )
        else:
            sBias = None
        sStage = None
        if const_expr(self.fp8_kv_dequant):
            sStage = storage.sStage.get_tensor(
                sStage_layout.outer, swizzle=sStage_layout.inner
            )
        # reuse sQ's data iterator
        sO = storage.sQ.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=self.o_dtype
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mK.shape[0]
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks
                if blocksparse_tensors is not None
                else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if const_expr(self.fp8_kv_dequant):
                self.load_fp8_kv(
                    mQ,
                    mK,
                    mV,
                    sQ,
                    sStage,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    pipeline_q,
                    pipeline_stage,
                    gmem_tiled_copy_Q,
                    mPageTable,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    num_splits,
                    sK=sK,
                    pipeline_k=pipeline_k,
                )
            else:
                self.load(
                    mQ,
                    mK,
                    mV,
                    sQ,
                    sK,
                    sV,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    mBias,
                    sBias,
                    tma_atom_Bias,
                    pipeline_k,
                    pipeline_v,
                    pipeline_bias,
                    pipeline_q,
                    gmem_tiled_copy_Q,
                    mPageTable,
                    blocksparse_tensors,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    num_splits,
                )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                base_softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                num_splits,
                aux_data,
                fastdiv_mods,
                sBias,
                pipeline_bias,
                sQ_mma,
                sV,
                sStage,
                pipeline_stage,
                descale_tensors,
                sQ_res,
            )

    def _make_pipeline_stage(self, barrier_storage, tma_warp, load_threads, mma_warps):
        # One fp8 staging slot: K and V tiles alternate through it in consumer order.
        if const_expr(self.use_tma_KV):
            return pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=barrier_storage,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
        return pipeline_custom.PipelineCpAsync.create(
            barrier_storage=barrier_storage,
            num_stages=1,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        mBias: Optional[cute.Tensor],
        sBias: Optional[cute.Tensor],
        tma_atom_Bias: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_bias: Optional[pipeline.PipelineAsync],
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = 1,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(
            not self.use_tma_KV or not self.use_tma_Q
        )
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)
        is_bias_load_warp = warp_idx_in_wg == 0

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(
                        mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0)
                    )
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size == n_block_size) ===
                    if const_expr(mPageTable is not None):
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None)
                        )
                    else:
                        # Non-paged TMA
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.tile_hdim), (None, 0)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.tile_hdimv), (None, 0)
                        )
                    # TODO: mcast
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_K_fn, pipeline_k
                    )
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_V_fn, pipeline_v
                    )
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                        self.num_threads_per_warp_group,
                        mK.element_type,
                        arch=self.arch.major * 10 + self.arch.minor,
                    )
                if const_expr(self.has_bias):
                    mBias_cur = seqlen.offset_batch_Q(mBias, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    gBias = cute.local_tile(
                        mBias_cur, (self.bias_block_size, self.tile_n), (None, None)
                    )
                    tBsBias, tBgBias = cpasync.tma_partition(
                        tma_atom_Bias,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sBias, 0, 2),
                        cute.group_modes(gBias, 0, 2),
                    )
                    load_Bias = partial(
                        self.load_Bias,
                        tma_atom_Bias,
                        tBgBias,
                        tBsBias,
                        pipeline_bias=pipeline_bias,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )
                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m,
                        self.tile_hdim,
                        self.check_hdim_oob,
                        self.qhead_per_kvhead,
                    )

                if const_expr(not self.use_block_sparsity):
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(
                                0, q_producer_phase
                            )
                            load_Q(
                                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0)
                            )
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen, m_block, split_idx, num_splits
                    )
                    bias_max_idx = Int32(0)
                    num_bias_loads = Int32(0)
                    if const_expr(self.has_bias):
                        _, n_block_max_abs = block_info.get_n_block_min_max(
                            seqlen, m_block, split_idx, num_splits, absolute=True
                        )
                        bias_idx_offset = n_block_max_abs - n_block_max
                        bias_max_idx = self.bias_n_max - 1 - bias_idx_offset
                        num_bias_loads = min(
                            self.bias_n_max - bias_idx_offset, n_block_max - n_block_min
                        )
                    # if cute.arch.thread_idx()[0] == 0:
                    #     cute.printf("m_block = %d, n_block_min: %d, n_block_max: %d", m_block, n_block_min, n_block_max)
                    # Empty split (n_block_min == n_block_max): no K/V to load. The
                    # consumer skips this tile symmetrically, so the K/V pipelines stay
                    # balanced.
                    load_kv = True
                    if const_expr(self.is_split_kv):
                        load_kv = n_block_min < n_block_max
                    if load_kv:
                        # Clamp n_block to 0 when n_block_max == 0 (can happen with causal
                        # + pack_gqa when seqlen_k < tile_n). TMA handles n_block=-1
                        # gracefully (fills zeros), but cp.async would crash on
                        # out-of-bounds page table access.
                        n_block = (
                            n_block_max - 1
                            if const_expr(self.use_tma_KV)
                            else cutlass.max(n_block_max - 1, 0)
                        )
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )

                        # First iteration: load K on pipeline_k
                        if is_kv_load_warp:
                            pipeline_k.producer_acquire(kv_producer_state)
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                        if const_expr(self.has_bias):
                            if is_bias_load_warp and num_bias_loads > 0:
                                load_Bias(
                                    src_idx=(m_block, bias_max_idx),
                                    producer_state=kv_producer_state,
                                )

                        if is_kv_load_warp:
                            if const_expr(
                                not self.intra_wg_overlap or not self.use_tma_KV
                            ):
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                                for i in cutlass.range(
                                    n_block_max - 1 - n_block_min, unroll=1
                                ):
                                    n_block = n_block_max - 1 - i - 1
                                    page_idx = (
                                        mPageTable[batch_idx, n_block]
                                        if const_expr(
                                            mPageTable is not None and self.use_tma_KV
                                        )
                                        else None
                                    )
                                    if const_expr(not self.use_tma_KV):
                                        paged_kv_manager.load_page_table(n_block)
                                    pipeline_k.producer_acquire(kv_producer_state)
                                    load_K(
                                        block=n_block,
                                        producer_state=kv_producer_state,
                                        page_idx=page_idx,
                                    )
                                    if const_expr(self.has_bias):
                                        if is_bias_load_warp and i + 1 < num_bias_loads:
                                            load_Bias(
                                                src_idx=(m_block, bias_max_idx - i - 1),
                                                producer_state=kv_producer_state,
                                            )
                                    pipeline_v.producer_acquire(kv_producer_state)
                                    load_V(
                                        block=n_block,
                                        producer_state=kv_producer_state,
                                        page_idx=page_idx,
                                    )
                                    kv_producer_state.advance()
                            else:
                                for i in cutlass.range(
                                    n_block_max - 1 - n_block_min, unroll=1
                                ):
                                    n_block_prev = n_block_max - i - 1
                                    n_block = n_block_prev - 1
                                    page_idx = (
                                        mPageTable[batch_idx, n_block]
                                        if const_expr(mPageTable is not None)
                                        else None
                                    )
                                    page_idx_prev = (
                                        mPageTable[batch_idx, n_block_prev]
                                        if const_expr(mPageTable is not None)
                                        else None
                                    )
                                    kv_producer_state_prev = kv_producer_state.clone()
                                    kv_producer_state.advance()
                                    pipeline_k.producer_acquire(kv_producer_state)
                                    load_K(
                                        block=n_block,
                                        producer_state=kv_producer_state,
                                        page_idx=page_idx,
                                    )
                                    if const_expr(self.has_bias):
                                        if is_bias_load_warp and i + 1 < num_bias_loads:
                                            load_Bias(
                                                src_idx=(m_block, bias_max_idx - i - 1),
                                                producer_state=kv_producer_state,
                                            )
                                    pipeline_v.producer_acquire(kv_producer_state_prev)
                                    load_V(
                                        block=n_block_prev,
                                        producer_state=kv_producer_state_prev,
                                        page_idx=page_idx_prev,
                                    )
                                n_block = n_block_min
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                else:
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(
                                0, q_producer_phase
                            )
                            load_Q(
                                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0)
                            )
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            (
                                self.q_subtile_factor
                                if self.q_subtile_factor is not None
                                else 1
                            ),
                            split_idx,
                            num_splits,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(
                block, sX[None, None, producer_state.index], K_or_V
            )
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def load_fp8_kv(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sStage: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_stage: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: cute.Tensor,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = 1,
        sK: Optional[cute.Tensor] = None,
        pipeline_k: Optional[pipeline.PipelineAsync] = None,
    ):
        """FP8-KV producer: streams fp8 K/V tiles through the single sStage slot in
        the order the MMA warpgroups cast them (see the FP8-KV consumer section).
        With FP8 MMA, K goes straight to its sK slot and only V uses sStage."""
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)
        q_producer_phase = Int32(1)
        stage_phase = Int32(1)
        k_producer_state = None
        if const_expr(self.fp8_kv_mma):
            k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_k_slots_mma
            )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            q_producer_phase = self._load_q_tile(
                mQ_cur,
                sQ,
                tma_atom_Q,
                pipeline_q,
                gmem_tiled_copy_Q,
                m_block,
                seqlen.seqlen_q,
                q_producer_phase,
            )
            stage = partial(
                self._stage_kv_tile,
                self._make_stage_copy(
                    mK, mV, sStage, tma_atom_K, tma_atom_V, mPageTable, batch_idx,
                    head_idx_kv, tidx, seqlen.seqlen_k,
                ),
                pipeline_stage=pipeline_stage,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            # An empty split loads nothing; its consumer skips the tile as well.
            has_kv = True
            if const_expr(self.is_split_kv):
                has_kv = n_block_min < n_block_max
            if is_kv_load_warp and has_kv:
                if const_expr(self.fp8_kv_mma):
                    # Same order as the cast path below, but K has its own pipeline,
                    # so only the V loads advance the staging phase.
                    load_k = partial(
                        self._load_k_slot,
                        mK,
                        sK,
                        tma_atom_K,
                        mPageTable,
                        batch_idx,
                        head_idx_kv,
                        pipeline_k,
                    )
                    load_k(n_block_max - 1, k_producer_state)
                    k_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        load_k(n_block, k_producer_state)
                        k_producer_state.advance()
                        stage(n_block=n_block + 1, K_or_V="V", phase=stage_phase)
                        stage_phase ^= 1
                    stage(n_block=n_block_min, K_or_V="V", phase=stage_phase)
                    stage_phase ^= 1
                elif const_expr(self.intra_wg_overlap):
                    stage(n_block=n_block_max - 1, K_or_V="K", phase=stage_phase)
                    stage_phase ^= 1
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        stage(n_block=n_block, K_or_V="K", phase=stage_phase)
                        stage_phase ^= 1
                        stage(n_block=n_block + 1, K_or_V="V", phase=stage_phase)
                        stage_phase ^= 1
                    stage(n_block=n_block_min, K_or_V="V", phase=stage_phase)
                    stage_phase ^= 1
                else:
                    # The consumer always runs the first block, even on an empty
                    # tile (n_block_max <= n_block_min), so always stage its pair.
                    for i in cutlass.range(
                        cutlass.max(n_block_max - n_block_min, 1), unroll=1
                    ):
                        n_block = n_block_max - 1 - i
                        stage(n_block=n_block, K_or_V="K", phase=stage_phase)
                        stage_phase ^= 1
                        stage(n_block=n_block, K_or_V="V", phase=stage_phase)
                        stage_phase ^= 1
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _load_q_tile(
        self,
        mQ_cur: cute.Tensor,
        sQ: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        m_block: Int32,
        seqlen_q: Int32,
        q_producer_phase: Int32,
    ) -> Int32:
        # TMA: warp 0 issues the copy. Packed GQA: all 128 producer threads cp.async.
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if const_expr(self.use_tma_Q):
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )
            if warp_idx_in_wg == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
        else:
            tidx, _, _ = cute.arch.thread_idx()
            pack_gqa = PackGQA(
                self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
            )
            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
            pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen_q)
            cute.arch.cp_async_commit_group()
            pipeline_q.producer_commit_w_index(0)
        return q_producer_phase ^ 1

    @cute.jit
    def _make_stage_copy(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sStage: cute.Tensor,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        mPageTable: cute.Tensor,
        batch_idx: Int32,
        head_idx_kv: Int32,
        tidx: Int32,
        seqlen_k: Int32,
    ) -> SimpleNamespace:
        if const_expr(self.use_tma_KV):
            # page_size == tile_n: one TMA box per page, indexed through the page table.
            gK, gV = [
                cute.local_tile(
                    mX[None, None, head_idx_kv, None],
                    (self.tile_n, self.tile_hdim),
                    (0, 0, None),
                )
                for mX in (mK, mV)
            ]
            copy_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), gK, sStage
            )
            copy_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), gV, sStage
            )
            return SimpleNamespace(
                copy_K=copy_K, copy_V=copy_V, mPageTable=mPageTable, batch_idx=batch_idx
            )
        paged_kv_manager = PagedKVManager.create(
            mPageTable,
            mK,
            mV,
            FastDivmodDivisor(mK.shape[0]),
            batch_idx,
            head_idx_kv,
            tidx,
            seqlen_k,
            0,  # leftpad_k
            self.tile_n,
            self.tile_hdim,
            self.tile_hdimv,
            self.num_threads_per_warp_group,
            mK.element_type,
            arch=self.arch.major * 10 + self.arch.minor,
        )
        return SimpleNamespace(paged_kv_manager=paged_kv_manager, sStage=sStage)

    @cute.jit
    def _stage_kv_tile(
        self,
        stage_copy: SimpleNamespace,
        n_block: Int32,
        K_or_V: cutlass.Constexpr[str],
        phase: Int32,
        pipeline_stage: pipeline.PipelineAsync,
    ):
        # Clamped like the BF16 path: n_block_max can be 0 for an empty causal tile.
        n_block = cutlass.max(n_block, 0)
        if const_expr(self.use_tma_KV):
            page_idx = stage_copy.mPageTable[stage_copy.batch_idx, n_block]
            copy = stage_copy.copy_K if const_expr(K_or_V == "K") else stage_copy.copy_V
            pipeline_stage.producer_acquire_w_index_phase(0, phase)
            copy(
                src_idx=page_idx,
                dst_idx=0,
                tma_bar_ptr=pipeline_stage.sync_object_full.get_barrier(0),
            )
        else:
            stage_copy.paged_kv_manager.load_page_table(n_block)
            pipeline_stage.producer_acquire_w_index_phase(0, phase)
            stage_copy.paged_kv_manager.load_KV(
                n_block, stage_copy.sStage[None, None, 0], K_or_V
            )
            cute.arch.cp_async_commit_group()
            pipeline_stage.producer_commit_w_index(0)

    @cute.jit
    def _load_k_slot(
        self,
        mK: cute.Tensor,
        sK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mPageTable: cute.Tensor,
        batch_idx: Int32,
        head_idx_kv: Int32,
        pipeline_k: pipeline.PipelineAsync,
        n_block: Int32,
        k_state: pipeline.PipelineState,
    ):
        # FP8 MMA: TMA the e4m3 K page into its sK slot (page_size == tile_n).
        gK = cute.local_tile(
            mK[None, None, head_idx_kv, None], (self.tile_n, self.tile_hdim), (0, 0, None)
        )
        load_K, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K, 0, cute.make_layout(1), gK, sK
        )
        # Clamped like the cast path: n_block_max can be 0 for an empty causal tile.
        page_idx = mPageTable[batch_idx, cutlass.max(n_block, 0)]
        pipeline_k.producer_acquire(k_state)
        load_K(
            src_idx=page_idx,
            dst_idx=k_state.index,
            tma_bar_ptr=pipeline_k.sync_object_full.get_barrier(k_state.index),
        )

    @cute.jit
    def load_Bias(
        self,
        tma_atom_Bias: cute.CopyAtom,
        tBgBias: cute.Tensor,
        tBsBias: cute.Tensor,
        src_idx: cute.Coord,
        pipeline_bias: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
    ):
        pipeline_bias.producer_acquire(producer_state)
        cute.copy(
            tma_atom_Bias,
            tBgBias[None, src_idx[0], src_idx[1]],
            tBsBias[None, producer_state.index],
            tma_bar_ptr=pipeline_bias.producer_get_barrier(producer_state),
        )
        pipeline_bias.producer_commit(producer_state)

    @cute.jit
    def apply_bias(
        self,
        acc_S: cute.Tensor,
        thr_mma_qk: cute.ThrMma,
        sBias: cute.Tensor,
        pipeline_bias: pipeline.PipelineAsync,
        bias_softmax_scale: Float32,
        consumer_state: pipeline.PipelineState,
        apply_bias: bool,
    ):
        if apply_bias:
            pipeline_bias.consumer_wait(
                consumer_state, pipeline_bias.consumer_try_wait(consumer_state)
            )
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)
            for i in cutlass.range_constexpr(cute.size(acc_S.shape)):
                row = tScS[i][0]
                col = tScS[i][1]
                if (
                    const_expr(self.bias_block_size == self.tile_m)
                    or row < self.bias_block_size
                ):
                    acc_S[i] = acc_S[i] * bias_softmax_scale
                    acc_S[i] = acc_S[i] + sBias[row, col, consumer_state.index].to(
                        self.qk_acc_dtype
                    )
                else:
                    acc_S[i] = acc_S[i] * bias_softmax_scale
            pipeline_bias.consumer_release(consumer_state)
        elif const_expr(self.has_bias):
            for i in cutlass.range_constexpr(cute.size(acc_S.shape)):
                acc_S[i] = acc_S[i] * bias_softmax_scale

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        base_softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        num_splits: Int32 = 1,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        sBias: Optional[cute.Tensor] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        sQ_mma: Optional[cute.Tensor] = None,
        sV: Optional[cute.Tensor] = None,
        sStage: Optional[cute.Tensor] = None,
        pipeline_stage: Optional[pipeline.PipelineAsync] = None,
        descale_tensors=None,
        sQ_res: Optional[cute.Tensor] = None,
    ):
        aux_tensors = aux_data.tensors
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ_mma, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.num_k_slots_mma if const_expr(self.fp8_kv_mma) else self.num_stages,
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        dequant = None
        mma_fp8 = None
        if const_expr(self.fp8_kv_mma):
            mma_fp8 = self._make_fp8_mma(
                tiled_mma_qk,
                wg_mma_qk,
                tSrQ,
                tSrK,
                sQ,
                sQ_mma,
                sQ_res,
                sK,
                sVt,
                sStage,
                pipeline_k,
                pipeline_stage,
                tidx,
                warp_group_idx,
                warp_group_thread_layout(warp_group_idx),
                tOrP,
            )
            mma_one_n_block_all = partial(
                self.mma_one_n_block_intrawg_overlap_fp8_mma,
                acc_O=acc_O,
                tOrP=tOrP,
                check_inf=True,
                scores_scale=scores_scale,
                mma_fp8=mma_fp8,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap_fp8_mma,
                tOrP=tOrP,
                scores_scale=scores_scale,
                acc_O=acc_O,
                mma_fp8=mma_fp8,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap_fp8_mma,
                scores_scale=scores_scale,
                acc_O=acc_O,
                mma_fp8=mma_fp8,
            )
        elif const_expr(self.fp8_kv_dequant):
            dequant = self._make_fp8_kv_dequant(
                sQ, sQ_mma, sK, sV, sStage, pipeline_stage, tidx, warp_group_idx
            )
            mma_one_n_block_all = partial(
                (
                    self.mma_one_n_block_intrawg_overlap_fp8
                    if const_expr(self.intra_wg_overlap)
                    else self.mma_one_n_block_fp8
                ),
                mma_qk_fn=mma_qk_fn,
                acc_O=acc_O,
                tOrP=tOrP,
                check_inf=True,
                scores_scale=scores_scale,
                dequant=dequant,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap_fp8,
                mma_qk_fn=mma_qk_fn,
                tOrP=tOrP,
                scores_scale=scores_scale,
                acc_O=acc_O,
                dequant=dequant,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap_fp8,
                mma_pv_fn=mma_pv_fn,
                scores_scale=scores_scale,
                acc_O=acc_O,
                dequant=dequant,
            )
        else:
            mma_one_n_block_all = partial(
                (
                    self.mma_one_n_block_intrawg_overlap
                    if const_expr(self.intra_wg_overlap)
                    else self.mma_one_n_block
                ),
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                pipeline_v=pipeline_v,
                acc_O=acc_O,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                check_inf=True,
                scores_scale=scores_scale,
                thr_mma_qk=thr_mma_qk,
                sBias=sBias,
                pipeline_bias=pipeline_bias,
                bias_softmax_scale=base_softmax_scale,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                scores_scale=scores_scale,
                acc_O=acc_O,
                thr_mma_qk=thr_mma_qk,
                sBias=sBias,
                pipeline_bias=pipeline_bias,
                bias_softmax_scale=base_softmax_scale,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
                scores_scale=scores_scale,
                acc_O=acc_O,
            )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            # FP8-KV folds the q*k descale into the score scale and v into final O.
            softmax_tile, softmax_scale_tile, v_descale = (
                softmax,
                softmax_scale,
                Float32(1.0),
            )
            if const_expr(self.fp8_kv_dequant):
                softmax_tile, softmax_scale_tile, v_descale = (
                    self._fp8_kv_tile_softmax(
                        descale_tensors,
                        batch_idx,
                        head_idx,
                        softmax_scale_log2,
                        softmax_scale,
                        num_rows=cute.size(softmax.row_max),
                    )
                )
                if const_expr(self.fp8_kv_mma):
                    v_descale = v_descale * Float32(1.0 / self.fp8_p_scale)

            # Recompute fastdiv_mods if necessary for varlen with aux_tensors
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    (
                        seqlen_q_divmod
                        if not recompute_fastdiv_mods_q
                        else FastDivmodDivisor(seqlen.seqlen_q)
                    ),
                    (
                        seqlen_k_divmod
                        if not recompute_fastdiv_mods_k
                        else FastDivmodDivisor(seqlen.seqlen_k)
                    ),
                )

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=(
                        1.0 if const_expr(self.has_bias) else softmax_scale_tile
                    ),
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                )
            mma_one_n_block = partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax_tile,
                score_mod_fn=score_mod_fn,
            )
            process_first_half_block_tile = partial(
                process_first_half_block, softmax=softmax_tile
            )
            process_last_half_block_tile = partial(
                process_last_half_block, softmax=softmax_tile
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            num_bias_loads = Int32(0)
            if const_expr(self.has_bias):
                _, n_block_max_abs = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits, absolute=True
                )
                bias_idx_offset = n_block_max_abs - n_block_max
                num_bias_loads = min(
                    self.bias_n_max - bias_idx_offset, n_block_max - n_block_min
                )
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            if const_expr(self.fp8_kv_mma):
                self._quantize_q_fp8(mma_fp8)
            elif const_expr(self.q_dtype != self.dtype):
                self._narrow_q_fp16(dequant)
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = False
            has_work = True
            if const_expr(self.is_split_kv):
                has_work = n_block_min < n_block_max

            # ==========================================
            # MAINLOOP
            # ==========================================
            if const_expr(not self.use_block_sparsity):
                # ==========================================
                # No block-sparsity (original path)
                # ==========================================
                if has_work:
                    # First iteration with seqlen masking
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_first_half_block_tile(
                            n_block=n_block_max - 1,
                            seqlen=seqlen,
                            kv_consumer_state=kv_consumer_state,
                            mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                            score_mod_fn=score_mod_fn,
                            is_first_block=True,
                            apply_bias=const_expr(self.has_bias) and num_bias_loads > 0,
                        )
                    else:
                        self.warp_scheduler_barrier_sync()
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1,
                            seqlen=seqlen,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                            is_first_n_block=True,
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=True
                            ),
                            apply_bias=const_expr(self.has_bias) and num_bias_loads > 0,
                        )
                        O_should_accumulate = True
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
                    n_block_max -= 1
                    # Next couple of iterations with causal/local masking. With sheared bias,
                    # the bias tile carries the causal/local -inf padding for its right band.
                    if const_expr(self.is_causal or self.is_local or self.has_bias):
                        if const_expr(self.has_bias):
                            n_block_min_causal_local_mask = max(
                                n_block_max + 1 - num_bias_loads, n_block_min
                            )
                        else:
                            n_block_min_causal_local_mask = (
                                block_info.get_n_block_min_causal_local_mask(
                                    seqlen, m_block, n_block_min
                                )
                            )
                        # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                seqlen=seqlen,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                mask_fn=(
                                    None
                                    if const_expr(self.has_bias)
                                    else partial(
                                        mask_fn,
                                        mask_mod=self.mask_mod,
                                        mask_seqlen=False,
                                    )
                                ),
                                apply_bias=self.has_bias,
                            )
                            O_should_accumulate = True
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )
                    # The remaining iterations have no masking
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True
                    # Separate iterations with local masking on the left
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                seqlen=seqlen,
                                mma_pv_fn=partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                mask_fn=partial(
                                    mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                                ),
                            )
                            O_should_accumulate = True
                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)
                if has_work:
                    # Last "half" iteration
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_last_half_block_tile(
                            kv_consumer_state=kv_consumer_state,
                            zero_init=not O_should_accumulate,
                        )
                        O_should_accumulate = True
                    else:
                        self.warp_scheduler_barrier_arrive()
                else:
                    softmax_tile.reset()
                    acc_O.fill(0.0)

            else:
                # ==========================================
                # Block sparsity
                # ==========================================
                kv_consumer_state, O_should_accumulate, processed_any = (
                    consume_block_sparse_loads(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        m_block,
                        seqlen,
                        kv_consumer_state,
                        mma_pv_fn,
                        mma_one_n_block,
                        process_first_half_block_tile,
                        process_last_half_block_tile,
                        mask_fn,
                        score_mod_fn,
                        O_should_accumulate,
                        self.mask_mod,
                        fastdiv_mods,
                        self.intra_wg_overlap,
                        self.warp_scheduler_barrier_sync,
                        self.warp_scheduler_barrier_arrive,
                        self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                        (
                            self.q_subtile_factor
                            if self.q_subtile_factor is not None
                            else 1
                        ),
                        split_idx,
                        num_splits,
                    )
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax_tile.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_rmem_tensor_like(
                        softmax_tile.row_max, Float32
                    )
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = (
                            row % self.qhead_per_kvhead
                            + head_idx * self.qhead_per_kvhead
                        )
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
            if const_expr(self.is_split_kv and learnable_sink is not None):
                # Only split 0 folds the learnable sink into the denominator.
                if const_expr(not self.pack_gqa):
                    sink_val = sink_val if split_idx == 0 else -Float32.inf
                else:
                    if split_idx != 0:
                        sink_val.fill(-Float32.inf)

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax_tile.finalize(final_scale=v_descale, sink_val=sink_val)
            softmax_tile.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O,
                softmax_tile.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        thr_mma_qk: Optional[cute.ThrMma] = None,
        sBias: Optional[cute.Tensor] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        bias_softmax_scale: Optional[Float32] = None,
        apply_bias: bool = False,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        if const_expr(self.has_bias):
            self.apply_bias(
                acc_S,
                thr_mma_qk,
                sBias,
                pipeline_bias,
                bias_softmax_scale,
                kv_consumer_state,
                apply_bias,
            )

        # Apply score modification if present
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            if const_expr(max(self.tile_hdim, self.tile_hdimv) > 256):
                # Both warp groups consume the same P tile for hdim 512.
                # Wait for every reader before overwriting shared memory.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_mma_threads,
                )
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            if const_expr(max(self.tile_hdim, self.tile_hdimv) > 256):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_mma_threads,
                )
            else:
                cute.arch.sync_warp()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    # ///////////////////////////////////////////////////////////////////////////////
    # FP8-KV dequant consumer. Per K/V tile the MMA warpgroups take one sStage slot
    # in the producer's order (load_fp8_kv): with intra-WG overlap K[max-1], then
    # (K[n], V[n+1]) pairs, then V[min]; without it (K[n], V[n]) pairs. Every tile
    # uses an even number of slots, so the slot phases below are fixed.
    # ///////////////////////////////////////////////////////////////////////////////

    @cute.jit
    def _make_fp8_kv_dequant(
        self,
        sQ: cute.Tensor,
        sQ_mma: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sStage: cute.Tensor,
        pipeline_stage: pipeline.PipelineAsync,
        tidx: Int32,
        warp_group_idx: Int32,
    ) -> SimpleNamespace:
        # 8 fp16 is one 128-bit smem store; 8 fp8 one 64-bit load.
        vec = 8
        sStage0, sK0, sV0 = sStage[None, None, 0], sK[None, None, 0], sV[None, None, 0]
        # Both warpgroups read the whole K tile, so they cast it together.
        thr_copy_K = copy_utils.tiled_copy_2d(
            self.dtype, self.tile_hdim // vec, self.num_mma_threads, num_copy_elems=vec
        ).get_slice(tidx)
        if const_expr(self.v_split_across_wgs):
            # hdim 512: each warpgroup casts only the V half its own PV reads.
            half = self.tile_hdimv // self.num_wg_mma
            thr_copy_V = copy_utils.tiled_copy_2d(
                self.dtype, half // vec, self.num_threads_per_warp_group, num_copy_elems=vec
            ).get_slice(tidx % self.num_threads_per_warp_group)
            sStage_V = cute.local_tile(sStage0, (self.tile_n, half), (0, warp_group_idx))
            sV_half = cute.local_tile(sV0, (self.tile_n, half), (0, warp_group_idx))
        else:
            # hdim 256: both warpgroups read all of V, so they cast it together.
            thr_copy_V = copy_utils.tiled_copy_2d(
                self.dtype, self.tile_hdimv // vec, self.num_mma_threads, num_copy_elems=vec
            ).get_slice(tidx)
            sStage_V, sV_half = sStage0, sV0
        tQsrc = tQdst = None
        if const_expr(self.q_dtype != self.dtype):
            thr_copy_Q = copy_utils.tiled_copy_2d(
                self.dtype,
                self.tile_hdim // vec,
                self.num_mma_threads,
                num_copy_elems=vec,
            ).get_slice(tidx)
            tQsrc, tQdst = thr_copy_Q.partition_S(sQ), thr_copy_Q.partition_D(sQ_mma)
        return SimpleNamespace(
            pipeline_stage=pipeline_stage,
            warp_group_idx=warp_group_idx,
            tStage_K=thr_copy_K.partition_S(sStage0),
            tsK=thr_copy_K.partition_D(sK0),
            tStage_V=thr_copy_V.partition_S(sStage_V),
            tsV=thr_copy_V.partition_D(sV_half),
            tQsrc=tQsrc,
            tQdst=tQdst,
        )

    @cute.jit
    def _fp8_kv_tile_softmax(
        self,
        descale_tensors,
        batch_idx: Int32,
        head_idx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        num_rows: cutlass.Constexpr[int],
    ):
        head_idx_kv = (
            head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        )
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(descale_tensors is not None):
            if const_expr(descale_tensors.q_descale is not None):
                qk_descale *= Float32(descale_tensors.q_descale[batch_idx, head_idx_kv])
            if const_expr(descale_tensors.k_descale is not None):
                qk_descale *= Float32(descale_tensors.k_descale[batch_idx, head_idx_kv])
            if const_expr(descale_tensors.v_descale is not None):
                v_descale = Float32(descale_tensors.v_descale[batch_idx, head_idx_kv])
        # A score_mod sees scaled scores, so the descale joins softmax_scale there.
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale_log2 * qk_descale
        else:
            softmax_scale = softmax_scale * qk_descale
        softmax = Softmax.create(
            softmax_scale_log2, num_rows=num_rows, softmax_scale=softmax_scale
        )
        return softmax, softmax_scale, v_descale

    @cute.jit
    def _narrow_q_fp16(self, dequant: SimpleNamespace):
        # bf16 and fp16 share a size, so each thread rewrites only its own elements.
        rQ = cute.make_rmem_tensor_like(dequant.tQsrc)
        cute.autovec_copy(dequant.tQsrc, rQ)
        rQ_mma = cute.make_rmem_tensor_like(dequant.tQdst)
        rQ_mma.store(rQ.load().to(Float32).to(self.dtype))
        cute.autovec_copy(rQ_mma, dequant.tQdst)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ),
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def _cast_fp8_tile(
        self,
        pipeline_stage: pipeline.PipelineAsync,
        tStage: cute.Tensor,
        tDst: cute.Tensor,
        phase: Int32,
    ):
        rStage = cute.make_rmem_tensor_like(tStage)
        rDst = cute.make_rmem_tensor_like(tDst)
        pipeline_stage.consumer_wait_w_index_phase(0, phase)
        cute.autovec_copy(tStage, rStage)
        # The fp8 bytes are in registers; let the producer refill the slot.
        pipeline_stage.consumer_release_w_index(0)
        utils.cvt_tensor_e4m3_to_f16(rStage, rDst)
        cute.autovec_copy(rDst, tDst)
        # Generic-proxy stores must be visible to the WGMMA (async proxy) reads.
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _cast_k_fp8(self, dequant: SimpleNamespace, phase: Int32):
        self._cast_fp8_tile(dequant.pipeline_stage, dequant.tStage_K, dequant.tsK, phase)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.DequantKFull),
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def _cast_v_fp8(self, dequant: SimpleNamespace, phase: Int32):
        self._cast_fp8_tile(dequant.pipeline_stage, dequant.tStage_V, dequant.tsV, phase)
        if const_expr(self.v_split_across_wgs):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.DequantV0) + dequant.warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group,
            )
        else:
            # sV is free: DequantKEmpty (overlap) or DequantKFull (no overlap)
            # for this K tile already ordered the other warpgroup's last PV.
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.DequantVFull),
                number_of_threads=self.num_mma_threads,
            )

    # ///////////////////////////////////////////////////////////////////////////////
    # FP8-MMA consumer (fp8_kv_mma, hdim 512). The tile order is the cast path's
    # with intra-WG overlap, but K has its own two-slot pipeline and V alone goes
    # through sStage. kv_consumer_state has two stages and advances once per block,
    # so its (index, phase) addresses K's slot and its index is the one-slot V
    # staging phase; both stay valid across tiles.
    # ///////////////////////////////////////////////////////////////////////////////

    def _make_fp8_mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        wg_mma_qk,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        sQ: cute.Tensor,
        sQ_mma: cute.Tensor,
        sQ_res: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sStage: cute.Tensor,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_stage: pipeline.PipelineAsync,
        tidx: Int32,
        warp_group_idx: Int32,
        wg_thr_layout,
        tOrP: cute.Tensor,
    ) -> SimpleNamespace:
        # Trace-time setup (plain Python, called from the jit mma()).
        _, tSrQ_res, _ = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ_res, sK
        )
        pv = self._make_pv_scratch(sVt, wg_thr_layout, pv_cols=64)
        # Q: one 256-thread pass reads the fp16/bf16 tile and writes the e4m3 pair.
        vec = 8
        thr_copy_Q = copy_utils.tiled_copy_2d(
            self.kv_dtype, self.tile_hdim // vec, self.num_mma_threads, num_copy_elems=vec
        ).get_slice(tidx)
        # V: each warpgroup transposes the 256-wide half of V its own PV reads,
        # LDSM.T -> PRMT -> STSM (FA3's FP8 V transpose). Static half coordinates
        # keep the 128-bit alignment that STSM needs.
        half = self.tile_hdimv // 2
        sStageVt = layout_utils.transpose_view(sStage[None, None, 0])
        sVt0 = sVt[None, None, 0]
        ldsm_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), self.kv_dtype),
            cute.make_layout((32, 4, 1, 1), stride=(4, 1, 0, 0)),
            cute.make_layout((2, 2, 1, 4), stride=(1, 2, 16, 4)),
        )
        # FA3's bank-conflict-free STSM layout permutes V's output columns;
        # _permute_output_fp8 undoes it on O once per tile.
        stsm_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), self.kv_dtype),
            cute.make_layout((8, 4, 4, 1), stride=(4, 1, 32, 0)),
            cute.make_layout((1, 4, 2, 2), stride=(0, 1, 4, 8)),
        )
        wg_tidx = tidx % self.num_threads_per_warp_group
        ldsm_thr = ldsm_copy.get_slice(wg_tidx)
        stsm_thr = stsm_copy.get_slice(wg_tidx)
        tStage_V, tsVt = [], []
        for h in range(2):
            tS = ldsm_thr.partition_S(
                cute.flat_divide(cute.local_tile(sStageVt, (half, self.tile_n), (h, 0)), (64, 8))
            )
            tD = stsm_thr.partition_D(
                cute.flat_divide(cute.local_tile(sVt0, (half, self.tile_n), (h, 0)), (8, 16))
            )
            tStage_V.append(cute.group_modes(tS, 1, cute.rank(tS)))
            tsVt.append(cute.group_modes(tD, 1, cute.rank(tD)))
        qk_fn = partial(
            self._gemm_qk_fp8_compensated,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            # The residual term goes first, while the running sum is still small.
            tSrQ_res,
            tSrQ,
            tSrK,
        )
        return SimpleNamespace(
            qk_fn=qk_fn,
            pv=pv,
            tOrP=tOrP,
            tOrP_res=cute.make_rmem_tensor_like(tOrP),
            pipeline_k=pipeline_k,
            pipeline_stage=pipeline_stage,
            warp_group_idx=warp_group_idx,
            tQsrc=thr_copy_Q.partition_S(sQ),
            tQhi=thr_copy_Q.partition_D(sQ_mma),
            tQres=thr_copy_Q.partition_D(sQ_res),
            ldsm_copy=ldsm_copy,
            stsm_copy=stsm_copy,
            tStage_V0=tStage_V[0],
            tStage_V1=tStage_V[1],
            tsVt0=tsVt[0],
            tsVt1=tsVt[1],
        )

    @cute.jit
    def _quantize_q_fp8(self, m: SimpleNamespace):
        """Split the TMA'd fp16/bf16 Q tile into e4m3 high and residual terms."""
        rQ = cute.make_rmem_tensor_like(m.tQsrc)
        rHi = cute.make_rmem_tensor_like(m.tQhi)
        rRes = cute.make_rmem_tensor_like(m.tQres)
        cute.autovec_copy(m.tQsrc, rQ)
        q = rQ.load().to(Float32)
        q_hi = q.to(self.kv_dtype)
        rHi.store(q_hi)
        rRes.store((q - q_hi.to(Float32)).to(self.kv_dtype))
        cute.autovec_copy(rHi, m.tQhi)
        # Every thread holds its Q values in registers before sQ is overwritten.
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_mma_threads
        )
        cute.autovec_copy(rRes, m.tQres)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.NarrowQ), number_of_threads=self.num_mma_threads
        )

    @cute.jit
    def _gemm_two_fp8_a(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tA_first: cute.Tensor,
        tA_second: cute.Tensor,
        tB: cute.Tensor,
        zero_init: cutlass.Constexpr[bool],
        wg_wait: cutlass.Constexpr[int],
        mid: Optional[Callable] = None,
    ):
        """Two FP8 A terms against one B in one WGMMA commit group; mid() runs
        between them."""
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tA_first.shape[2])):
            cute.gemm(mma_atom, acc, tA_first[None, None, k], tB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        if const_expr(mid is not None):
            mid()
            warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tA_second.shape[2])):
            cute.gemm(mma_atom, acc, tA_second[None, None, k], tB[None, None, k], acc)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)

    @cute.jit
    def _gemm_qk_fp8_compensated(
        self,
        tiled_mma: cute.TiledMma,
        shape,
        tQ_first: cute.Tensor,
        tQ_second: cute.Tensor,
        tK: cute.Tensor,
        B_idx=None,
        wg_wait: cutlass.Constexpr[int] = -1,
        mid: Optional[Callable] = None,
        split_acc: Optional[cute.Tensor] = None,
    ):
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rK = tK if const_expr(B_idx is None) else tK[None, None, None, B_idx]
        if const_expr(not self.qk_split_mma):
            self._gemm_two_fp8_a(tiled_mma, acc, tQ_first, tQ_second, rK, True, wg_wait, mid)
        elif const_expr(split_acc is None):
            # First block: a private second accumulator, waited for and folded here.
            acc2 = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
            self._gemm_qk_fp8_split(tiled_mma, acc, acc2, tQ_first, tQ_second, rK, None)
            warpgroup.wait_group(0)
            self._fold_qk_split(acc, acc2)
        else:
            # The caller waits and folds split_acc before it reuses the scratch.
            self._gemm_qk_fp8_split(tiled_mma, acc, split_acc, tQ_first, tQ_second, rK, mid)
        return acc

    @cute.jit
    def _gemm_qk_fp8_split(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        acc2: cute.Tensor,
        tA_first: cute.Tensor,
        tA_second: cute.Tensor,
        rK: cute.Tensor,
        mid: Optional[Callable],
    ):
        """One commit group: the first half of the k-steps sums into acc, the rest
        into acc2. Causal/local builds otherwise serialize the WGMMAs (ptxas C7511)."""
        nk = const_expr(cute.size(tA_first.shape[2]))
        warpgroup.fence()
        a1 = cute.make_mma_atom(tiled_mma.op)
        a2 = cute.make_mma_atom(tiled_mma.op)
        a1.set(warpgroup.Field.ACCUMULATE, False)
        a2.set(warpgroup.Field.ACCUMULATE, False)
        for t in cutlass.range_constexpr(2):
            tA = tA_first if const_expr(t == 0) else tA_second
            if const_expr(t == 1 and mid is not None):
                mid()
                warpgroup.fence()
            for k in cutlass.range_constexpr(nk):
                if const_expr(k < nk // 2):
                    cute.gemm(a1, acc, tA[None, None, k], rK[None, None, k], acc)
                    a1.set(warpgroup.Field.ACCUMULATE, True)
                else:
                    cute.gemm(a2, acc2, tA[None, None, k], rK[None, None, k], acc2)
                    a2.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    @cute.jit
    def _fold_qk_split(self, acc: cute.Tensor, acc2: cute.Tensor):
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = acc[i] + acc2[i]

    def _make_pv_scratch(self, sVt: cute.Tensor, wg_thr_layout, pv_cols: int):
        """PV in pv_cols-wide column chunks: one 64 x pv_cols WGMMA per warpgroup into
        an fp32 scratch that is added to O after each chunk."""
        half = self.tile_hdimv // 2
        assert half % pv_cols == 0
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.kv_dtype,
            self.kv_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 2, 1),
            tiler_mn=(64, pv_cols),
            a_source=warpgroup.OperandSource.RMEM,
        )
        wg_mma = tiled_mma.get_slice(wg_thr_layout)
        # Rows [c * pv_cols, (c + 1) * pv_cols) of each warpgroup's half of sVt.
        chunk_layout = cute.make_layout(
            ((8, pv_cols // 8, 2), (self.tile_n, 1), (1, 1)),
            stride=((self.tile_n, 8 * self.tile_n, half * self.tile_n), (1, 0), (0, 0)),
        )
        tVs = []
        acc = None
        for c in range(half // pv_cols):
            sVt_c = cute.make_tensor(sVt.iterator + c * pv_cols * self.tile_n, chunk_layout)
            acc, _, tV_c = sm90_utils.partition_fragment_ABC(
                wg_mma, (self.tile_m, 2 * pv_cols, self.tile_n), None, sVt_c
            )
            tVs.append(tV_c)
        return SimpleNamespace(tiled_mma=tiled_mma, acc=acc, tVs=tVs, n8=pv_cols // 8)

    @cute.jit
    def _gemm_pv_fp8_promoted(self, m: SimpleNamespace, acc_O: cute.Tensor, B_idx):
        """O += P @ V for one KV block, each column chunk added to O in fp32."""
        pv = m.pv
        for c in cutlass.range_constexpr(len(pv.tVs)):
            self._gemm_two_fp8_a(
                pv.tiled_mma, pv.acc, m.tOrP, m.tOrP_res, pv.tVs[c][None, None, None, B_idx], True, 0
            )
            fo = cute.group_modes(acc_O, 1, cute.rank(acc_O))
            fc = cute.group_modes(pv.acc, 1, cute.rank(pv.acc))
            for mi in cutlass.range_constexpr(cute.size(fc.shape[1])):
                for e in cutlass.range_constexpr(2):
                    for r in cutlass.range_constexpr(2):
                        for j in cutlass.range_constexpr(pv.n8):
                            idx = ((e, r, c * pv.n8 + j), mi)
                            fo[idx] = fo[idx] + fc[((e, r, j), mi)]

    @cute.jit
    def _transpose_v_half(self, m: SimpleNamespace, tStage_V, tsVt):
        for i in cutlass.range_constexpr(cute.size(tStage_V.shape[1])):
            dst = tsVt[None, i]
            # Every lane's STSM address is 16-byte aligned; keep that through slicing.
            dst = cute.make_tensor(dst.iterator.align(16), dst.layout)
            rV = cute.make_rmem_tensor_like(dst)
            cute.copy(m.ldsm_copy, tStage_V[None, i], rV)
            rV_u32 = cute.recast_tensor(rV, cutlass.Uint32)
            for j in cutlass.range_constexpr(cute.size(rV_u32) // 2):
                upper = rV_u32[2 * j]
                lower = rV_u32[2 * j + 1]
                rV_u32[2 * j] = utils.prmt_b32(upper, lower, cutlass.Uint32(0x6420))
                rV_u32[2 * j + 1] = utils.prmt_b32(upper, lower, cutlass.Uint32(0x7531))
            cute.copy(m.stsm_copy, rV, dst)

    @cute.jit
    def _transpose_v_fp8(self, m: SimpleNamespace, phase: Int32):
        """Move this warpgroup's V half from sStage into sVt, transposed."""
        m.pipeline_stage.consumer_wait_w_index_phase(0, phase)
        if m.warp_group_idx == 0:
            self._transpose_v_half(m, m.tStage_V0, m.tsVt0)
        else:
            self._transpose_v_half(m, m.tStage_V1, m.tsVt1)
        # The LDSM reads complete before the slot is refilled; each warp releases.
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        m.pipeline_stage.consumer_release_w_index(0)
        # This warpgroup's PV reads only the half it wrote.
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.DequantV0) + m.warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group,
        )

    @cute.jit
    def _permute_Cregs_fp8(self, acc: cute.Tensor):
        """FA3's register permutation from the QK accumulator to the FP8 PV A operand."""
        acc_u64 = cute.recast_tensor(acc, cutlass.Int64)
        flat = cute.make_tensor(acc_u64.iterator, cute.make_layout(cute.size(acc_u64)))
        for i in cutlass.range_constexpr(cute.size(flat) // 4):
            tmp = flat[4 * i + 1]
            flat[4 * i + 1] = flat[4 * i + 2]
            flat[4 * i + 2] = tmp

    @cute.jit
    def _permute_output_fp8(self, acc: cute.Tensor):
        """Undo the O column permutation from the FP8 P layout and the STSM V store."""
        # The FP8 accumulator's local permutation first, then FA3's cross-lane one;
        # the order matters.
        frag32 = cute.group_modes(acc, 1, cute.rank(acc))
        for mi in cutlass.range_constexpr(cute.size(frag32.shape[1])):
            for j in cutlass.range_constexpr(cute.size(frag32.shape[0][1])):
                for i in cutlass.range_constexpr(frag32.shape[0][2] // 2):
                    tmp = frag32[((1, j, 2 * i), mi)]
                    frag32[((1, j, 2 * i), mi)] = frag32[((0, j, 2 * i + 1), mi)]
                    frag32[((0, j, 2 * i + 1), mi)] = tmp
        frag = cute.group_modes(cute.recast_tensor(acc, cutlass.Int64), 1, cute.rank(acc))
        n8 = frag.shape[0][2]
        assert n8 % 2 == 0
        quad_idx = cute.arch.lane_idx() % 4
        lane_03 = quad_idx == 0 or quad_idx == 3
        upper_src = Int32(0)
        if quad_idx == 1:
            upper_src = Int32(2)
        elif quad_idx == 2:
            upper_src = Int32(3)
        elif quad_idx == 3:
            upper_src = Int32(1)
        lower_src = upper_src ^ Int32(2)
        for mi in cutlass.range_constexpr(cute.size(frag.shape[1])):
            for j in cutlass.range_constexpr(cute.size(frag.shape[0][1])):
                for i in cutlass.range_constexpr(n8 // 2):
                    upper = frag[((0, j, 2 * i), mi)]
                    lower = frag[((0, j, 2 * i + 1), mi)]
                    upper0, lower0 = upper, lower
                    if not lane_03:
                        upper0, lower0 = lower, upper
                    upper0 = utils.shuffle_sync(upper0, upper_src, width=4)
                    lower0 = utils.shuffle_sync(lower0, lower_src, width=4)
                    if lane_03:
                        frag[((0, j, 2 * i), mi)] = upper0
                        frag[((0, j, 2 * i + 1), mi)] = lower0
                    else:
                        frag[((0, j, 2 * i), mi)] = lower0
                        frag[((0, j, 2 * i + 1), mi)] = upper0

    @cute.jit
    def _prepare_fp8_probabilities(self, acc_S: cute.Tensor, m: SimpleNamespace):
        """Softmax probabilities -> the compensated e4m3 pair of PV A fragments."""
        self._permute_Cregs_fp8(acc_S)
        layout = acc_S.layout
        n8 = layout.shape[0][2]
        assert n8 % 4 == 0
        # FP8 form of FA3's convert_layout_acc_Aregs.
        tP_acc = cute.make_tensor(
            acc_S.iterator,
            cute.make_layout(
                ((4, 2, 2), layout.shape[1], (n8 // 4, layout.shape[2])),
                stride=((1, 4, 8), layout.stride[1], (16, layout.stride[2])),
            ),
        )
        p = tP_acc.load() * Float32(self.fp8_p_scale)
        p_hi = p.to(self.kv_dtype)
        m.tOrP.store(p_hi)
        m.tOrP_res.store((p - p_hi.to(Float32)).to(self.kv_dtype))

    @cute.jit
    def first_half_block_overlap_fp8_mma(
        self,
        n_block: Int32,
        kv_consumer_state,
        tOrP: cute.Tensor,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        mma_fp8: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        apply_bias: bool = False,  # passed by the shared main loop; FP8-KV has no bias
    ):
        m = mma_fp8
        m.pipeline_k.consumer_wait_w_index_phase(
            kv_consumer_state.index, kv_consumer_state.phase
        )
        acc_S = m.qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        m.pipeline_k.consumer_release_w_index(kv_consumer_state.index)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
        self._prepare_fp8_probabilities(acc_S, m)
        acc_O.fill(0.0)
        scores_scale.store(row_scale.load())
        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap_fp8_mma(
        self,
        kv_consumer_state,
        zero_init: bool,
        mma_fp8: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        softmax.rescale_O(acc_O, scores_scale)
        self._transpose_v_fp8(mma_fp8, phase=kv_consumer_state.index)
        self._gemm_pv_fp8_promoted(mma_fp8, acc_O, B_idx=0)
        # O has been accumulated in permuted column order; restore it once per tile.
        self._permute_output_fp8(acc_O)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block_intrawg_overlap_fp8_mma(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_pv_fn: Callable,  # passed by the shared main loop; unused
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        mma_fp8: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        apply_bias: bool = False,  # passed by the shared main loop; FP8-KV has no bias
    ):
        # QK[n] overlaps the transpose of V[n + 1]; PV[n + 1] follows.
        m = mma_fp8
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        k_slot = smem_pipe_read.index
        m.pipeline_k.consumer_wait_w_index_phase(k_slot, smem_pipe_read.phase)

        def transpose_v():
            self._transpose_v_fp8(m, phase=smem_pipe_read_v.index)

        # The V transpose runs between the residual and high Q terms of QK.
        if const_expr(self.qk_split_mma):
            acc_S = m.qk_fn(B_idx=k_slot, wg_wait=-1, mid=transpose_v, split_acc=m.pv.acc)
        else:
            acc_S = m.qk_fn(B_idx=k_slot, wg_wait=-1, mid=transpose_v)
        softmax.rescale_O(acc_O, scores_scale)
        if const_expr(self.qk_split_mma):
            # PV reuses the scratch: QK lands and folds first.
            warpgroup.wait_group(0)
            self._fold_qk_split(acc_S, m.pv.acc)
        # Each PV chunk waits for all prior WGMMAs, so QK[n] is complete after it.
        self._gemm_pv_fp8_promoted(m, acc_O, B_idx=0)
        m.pipeline_k.consumer_release_w_index(k_slot)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        self._prepare_fp8_probabilities(acc_S, m)
        scores_scale.store(row_scale.load())
        return smem_pipe_read

    @cute.jit
    def first_half_block_overlap_fp8(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        tOrP: cute.Tensor,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        dequant: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        apply_bias: bool = False,  # passed by the shared main loop; FP8-KV has no bias
    ):
        self._cast_k_fp8(dequant, phase=Int32(0))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())
        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap_fp8(
        self,
        kv_consumer_state,
        mma_pv_fn: Callable,
        zero_init: bool,
        dequant: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        self._cast_v_fp8(dequant, phase=Int32(1))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block_intrawg_overlap_fp8(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        dequant: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        apply_bias: bool = False,  # passed by the shared main loop; FP8-KV has no bias
    ):
        # QK[n] overlaps the cast of V[n + 1] and the PV that consumes it.
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        # The other warpgroup's QK[n + 1] may still be reading sK.
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.DequantKEmpty),
            number_of_threads=self.num_mma_threads,
        )
        self._cast_k_fp8(dequant, phase=Int32(1))
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        self._cast_v_fp8(dequant, phase=Int32(0))
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        warpgroup.wait_group(1)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        # tOrP is the in-flight PV's A operand; overwrite it only after the wait.
        warpgroup.wait_group(0)
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        else:
            softmax.rescale_O(acc_O, row_scale)
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_fp8(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        dequant: SimpleNamespace,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        apply_bias: bool = False,  # passed by the shared main loop; FP8-KV has no bias
    ):
        # Each warpgroup finishes QK[n] before releasing V[n]'s slot, so K[n - 1]
        # cannot land in sK while the other warpgroup still reads it.
        self._cast_k_fp8(dequant, phase=Int32(0))
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), tOrP)
        softmax.rescale_O(acc_O, row_scale)
        self._cast_v_fp8(dequant, phase=Int32(1))
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        thr_mma_qk: Optional[cute.ThrMma] = None,
        sBias: Optional[cute.Tensor] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        bias_softmax_scale: Optional[Float32] = None,
        apply_bias: bool = False,
    ):
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(self.has_bias):
            self.apply_bias(
                acc_S,
                thr_mma_qk,
                sBias,
                pipeline_bias,
                bias_softmax_scale,
                smem_pipe_read,
                apply_bias,
            )

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            if const_expr(max(self.tile_hdim, self.tile_hdimv) > 256):
                # Both warp groups consume the same P tile for hdim 512.
                # Wait for every reader before overwriting shared memory.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_mma_threads,
                )
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            if const_expr(max(self.tile_hdim, self.tile_hdimv) > 256):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_mma_threads,
                )
            else:
                cute.arch.sync_warp()
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        thr_mma_qk: Optional[cute.ThrMma] = None,
        sBias: Optional[cute.Tensor] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        bias_softmax_scale: Optional[Float32] = None,
        apply_bias: bool = False,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(self.has_bias):
            self.apply_bias(
                acc_S,
                thr_mma_qk,
                sBias,
                pipeline_bias,
                bias_softmax_scale,
                smem_pipe_read,
                apply_bias,
            )

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            if const_expr(max(self.tile_hdim, self.tile_hdimv) > 256):
                # Both warp groups consume the same P tile for hdim 512.
                # Wait for every reader before overwriting shared memory.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PEmpty),
                    number_of_threads=self.num_mma_threads,
                )
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            if const_expr(max(self.tile_hdim, self.tile_hdimv) > 256):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.PFull),
                    number_of_threads=self.num_mma_threads,
                )
            else:
                cute.arch.sync_warp()
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
