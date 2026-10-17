# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# - paged KV
# Unsupported features that will be added later:
# - optimized dedicated kernels for all hdim-192/256 modes
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import enum
import math
import operator
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync

import sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.utils as utils
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn import (
    barrier,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn import (
    blackwell_helpers as sm100_utils,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn import (
    copy_utils,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.block_info import (
    BlockInfo,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.block_sparse_utils import (
    get_total_block_count,
    handle_block_sparse_empty_tile_correction_sm100,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.block_sparsity import (
    BlockSparseTensors,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.fast_math import (
    FastDivmod,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.mask import (
    AttentionMask,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.pack_gqa import (
    PackGQA,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.paged_kv import (
    PagedKVManager,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.seqlen_info import (
    SeqlenInfoQK,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.softmax import (
    SoftmaxSm100,
    apply_score_mod_inner,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.tile_scheduler import (
    ParamsBase,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
)


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemAlloc = enum.auto()
    WorkTile = enum.auto()


#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class FlashAttentionForwardSm100:
    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        split_kv_len: Optional[int] = None,
        force_persistent: bool = False,
        dynamic_splitkv_scheduler: bool = False,
        run_splitkv_scheduler: bool = True,
        dynamic_splitkv_balance: bool = False,
        gpu_partial_order: bool = False,
        fused_splitkv_combine: bool = False,
        fused_splitkv_combine_mode: str = "none",
        programmatic_launch_completion: bool = True,
        swa_right_only: bool = False,
        force_non_tma_o: bool = False,
        attention_mask_cls: Callable = AttentionMask,
    ):
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = (
            1 if self.head_dim_padded >= 256 or self.head_dim_v_padded >= 192 else 2
        )
        assert self.q_stage in [1, 2]

        # 2 Q tile per CTA
        self.cta_tiler = (
            self.q_stage * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.split_kv_len = split_kv_len
        self.dynamic_splitkv_scheduler = dynamic_splitkv_scheduler
        self.run_splitkv_scheduler = run_splitkv_scheduler
        self.dynamic_splitkv_balance = dynamic_splitkv_balance
        if gpu_partial_order and not dynamic_splitkv_scheduler:
            raise ValueError("gpu_partial_order requires dynamic_splitkv_scheduler")
        self.gpu_partial_order = gpu_partial_order
        self.fused_splitkv_combine = fused_splitkv_combine
        if fused_splitkv_combine_mode not in ("none", "producer", "queue", "pipeline"):
            raise ValueError(
                f"Unsupported fused_splitkv_combine_mode={fused_splitkv_combine_mode}"
            )
        self.fused_splitkv_combine_mode = fused_splitkv_combine_mode
        self.programmatic_launch_completion = programmatic_launch_completion
        self.swa_right_only = swa_right_only
        self.force_non_tma_o = force_non_tma_o
        self.prefetch_attention_mask = False
        self.attention_mask_cls = attention_mask_cls
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 2
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False
        overlap_sO_sQ = (
            self.head_dim_padded == 192 and self.head_dim_v_padded >= 64
        ) or (self.head_dim_v_padded >= 128 and self.is_split_kv)
        self.overlap_sO_sQ = overlap_sO_sQ
        # Note(wangbojun/codex): HD192 needs the Q/O smem alias to stay within
        # the launchable SM100 smem limit. Explicit persistent splitKV keeps the
        # alias and lets correctness/perf validation decide whether it is viable.
        if self.overlap_sO_sQ and not force_persistent:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if self.q_stage == 1:
            if not self.use_tma_KV:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q:  # fallback
            self.epilogue_warp_ids = (13, 14)

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(2 if self.q_stage == 2 else 1)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        elif self.q_stage == 1:
            self.num_regs_softmax = 224
            self.num_regs_correction = 80
            self.num_regs_other = 48
        else:
            # self.num_regs_softmax = 192 if self.is_causal or self.is_local else 184
            self.num_regs_softmax = 200
            # self.num_regs_softmax = 176
            # self.num_regs_correction = 96
            # self.num_regs_correction = 80
            # self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
            self.num_regs_correction = 64
            # self.num_regs_other = 32
            # self.num_regs_other = 64
            # self.num_regs_other = 80
            self.num_regs_other = 48
            # self.num_regs_other = 96 if self.is_causal or self.is_local else 80
            # self.num_regs_other = 64 if self.is_causal or self.is_local else 80
        self.num_regs_empty = 24
        self.work_tile_threads = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
            )
        )

        self.buffer_align_bytes = 1024
        self.tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwd.TmemAlloc),
            num_threads=cute.arch.WARP_SIZE,
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = (
            max(smem_size_q, smem_size_o)
            if self.overlap_sO_sQ
            else smem_size_q + smem_size_o
        )
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        smem_size_kv_per_stage = max(smem_size_k_per_stage, smem_size_v_per_stage)
        self.kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        if (
            self.q_stage == 1
            and self.head_dim_padded == 192
            and self.head_dim_v_padded == 192
        ):
            # The layout/alignment overhead pushes kv_stage=3 past the launchable SM100 SMEM limit.
            self.kv_stage = min(self.kv_stage, 2)
        if self.q_dtype.width == 8:
            self.kv_stage = min(self.kv_stage, 4)
        assert self.kv_stage >= 1, (
            f"SM100 flash attention cannot fit shared memory for "
            f"head_dim={self.head_dim_padded}, head_dim_v={self.head_dim_v_padded}, "
            f"q_stage={self.q_stage}"
        )
        self.acc_stage = 1
        self.epi_stage = 2
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        mNumSplitsDynamic: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
        mTotalSplitUnits: Optional[cute.Tensor],
        mOFinal: Optional[cute.Tensor],
        mLSEFinal: Optional[cute.Tensor],
        mNumSplitsActual: Optional[cute.Tensor],
        mSplitDoneCounter: Optional[cute.Tensor],
        mReductionQueue: Optional[cute.Tensor],
        mReductionQueueValid: Optional[cute.Tensor],
        mReductionQueueState: Optional[cute.Tensor],
        mFusedTimeline: Optional[cute.Tensor],
        mPartialOrder: Optional[cute.Tensor],
        mPartialOrderState: Optional[cute.Tensor],
        splitkv_scheduler_target_tasks: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mSigmoidGate: Optional[cute.Tensor] = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            for t in (mQ, mK, mV, mO)
        ]
        mOFinal = (
            cute.make_tensor(
                mOFinal.iterator,
                cute.make_layout(mOFinal.shape, stride=new_stride(mOFinal)),
            )
            if const_expr(mOFinal is not None)
            else None
        )
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = (
                [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            )
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = (
                [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            )
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            num_splits = Int32(1)
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )
        mOFinal = (
            cute.make_tensor(
                mOFinal.iterator,
                cute.select(
                    mOFinal.layout,
                    mode=[1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1],
                ),
            )
            if const_expr(mOFinal is not None)
            else None
        )
        mLSEFinal = (
            cute.make_tensor(
                mLSEFinal.iterator,
                cute.select(
                    mLSEFinal.layout,
                    mode=[2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0],
                ),
            )
            if const_expr(mLSEFinal is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = (
            [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        )
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.use_tma_O = (
            self.arch >= 90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not self.force_non_tma_o
        )
        # This can be tuned
        self.e2e_freq = 16
        if const_expr(
            self.head_dim_padded > 64
            and not self.is_causal
            and not self.is_local
            and self.pack_gqa
        ):
            self.e2e_freq = (
                32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10
            )

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.q_dtype,
            self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mOFinal is not None):
                shape_O_final_packed = (
                    (self.qhead_per_kvhead, mOFinal.shape[0]),
                    mOFinal.shape[1],
                    mK.shape[2],
                    *mOFinal.shape[3:],
                )
                stride_O_final_packed = (
                    (mOFinal.stride[2], mOFinal.stride[0]),
                    mOFinal.stride[1],
                    mOFinal.stride[2] * self.qhead_per_kvhead,
                    *mOFinal.stride[3:],
                )
                mOFinal = cute.make_tensor(
                    mOFinal.iterator,
                    cute.make_layout(
                        shape_O_final_packed, stride=stride_O_final_packed
                    ),
                )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator,
                    cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed),
                )
            if const_expr(mLSEFinal is not None):
                shape_LSE_final_packed = (
                    (self.qhead_per_kvhead, mLSEFinal.shape[0]),
                    mK.shape[2],
                    *mLSEFinal.shape[2:],
                )
                stride_LSE_final_packed = (
                    (mLSEFinal.stride[1], mLSEFinal.stride[0]),
                    mLSEFinal.stride[1] * self.qhead_per_kvhead,
                    *mLSEFinal.stride[2:],
                )
                mLSEFinal = cute.make_tensor(
                    mLSEFinal.iterator,
                    cute.make_layout(
                        shape_LSE_final_packed, stride=stride_LSE_final_packed
                    ),
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )
        else:
            tma_atom_K = None
            tma_atom_V = None

        o_cta_v_layout = cute.composition(
            cute.make_identity_layout(mO.shape), self.epi_tile
        )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                atom_universal_copy, tO_layout, vO_layout
            )

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        tile_sched_args = TileSchedulerArguments(
            cute.size(aux_tensors[3].shape[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[
                0
            ],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedK=mSeqUsedK,
            mNumSplitsDynamic=mNumSplitsDynamic,
            mSplitOffsets=mSplitOffsets,
            mTotalSplitUnits=mTotalSplitUnits,
            mNumSplitsActual=mNumSplitsActual,
            mPartialOrder=mPartialOrder,
            mPartialOrderState=mPartialOrderState,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            is_causal=self.is_causal,
            is_local=self.is_local,
            split_kv_len=self.split_kv_len,
            dynamic_splitkv_scheduler=self.dynamic_splitkv_scheduler,
            gpu_partial_order=self.gpu_partial_order,
            dynamic_splitkv_balance=self.dynamic_splitkv_balance,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = (
            self.mbar_load_kv_empty_offset + self.kv_stage
        )
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + 2
        self.mbar_O_full_offset = self.mbar_S_full_offset + 2
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + 2
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + 2
        self.mbar_corr_epi_full_offset = (
            self.mbar_softmax_corr_empty_offset + self.epi_stage
        )
        self.mbar_corr_epi_empty_offset = (
            self.mbar_corr_epi_full_offset + self.epi_stage
        )
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + 2
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        self.mbar_total = self.mbar_P_full_2_offset + 2

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout)
            if const_expr(not self.overlap_sO_sQ)
            else cutlass.max(
                cute.cosize(sQ_layout),
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
        )

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Stage paged-TMA page indices once per KV tile. The same page
            # index is consumed by both K and V TMA loads, so a shared ring
            # avoids redundant per-lane global page-table loads.
            sPageRing: cute.struct.MemRange[Int32, self.kv_stage]
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale = None
        else:
            # NB: If a users passes in a score mod, we want to apply the score-mod in the sm_scaled qk
            # But in the original base 10. We hijack softmax_scale_log2 to just be the change of base
            # and correctly apply the softmax_scale prior to score_mod in the softmax step
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmod.create(seqlen_q)
            seqlen_k_divmod = FastDivmod.create(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError(
                "Block sparsity + paged KV not supported on SM100"
            )

        # Launch the kernel synchronously
        if const_expr(self.dynamic_splitkv_scheduler and self.run_splitkv_scheduler):
            assert mNumSplitsDynamic is not None
            assert mSplitOffsets is not None
            assert mTotalSplitUnits is not None
            if const_expr(self.gpu_partial_order):
                assert mPartialOrder is not None
                assert mPartialOrderState is not None
            self.splitkv_scheduler_kernel(
                mPartialOrder,
                mPartialOrderState,
                mNumSplitsDynamic,
                mSplitOffsets,
                mTotalSplitUnits,
                mNumSplitsActual,
                mCuSeqlensQ,
                mCuSeqlensK,
                mSeqUsedK,
                mQ.shape[3]
                if const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ.shape[0] - 1,
                cute.size(mQ.shape[2]),
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1],
                num_splits,
                Int32(self.cta_tiler[0]),
                Int32(self.cta_tiler[1]),
                window_size_left,
                window_size_right,
                splitkv_scheduler_target_tasks,
            ).launch(
                grid=[1, 1, 1],
                block=[32, 1, 1],
                stream=stream,
            )
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mNumSplitsDynamic,
            mSplitOffsets,
            mTotalSplitUnits,
            mOFinal,
            mLSEFinal,
            mNumSplitsActual,
            mSplitDoneCounter,
            mReductionQueue,
            mReductionQueueValid,
            mReductionQueueState,
            mFusedTimeline,
            mPartialOrder,
            mPartialOrderState,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            mSigmoidGate,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.jit
    def _splitkv_scheduler_seqlen_q(
        self,
        batch_idx: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
    ) -> Int32:
        seqlen_q = Int32(0)
        if const_expr(mCuSeqlensQ is not None):
            seqlen_q = mCuSeqlensQ[batch_idx + Int32(1)] - mCuSeqlensQ[batch_idx]
        return seqlen_q

    @cute.jit
    def _splitkv_scheduler_effective_n_blocks(
        self,
        batch_idx: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        seqlen_k_static: Int32,
        tile_m: Int32,
        tile_n: Int32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
    ) -> Int32:
        seqlen_q = self._splitkv_scheduler_seqlen_q(batch_idx, mCuSeqlensQ)
        seqlen_k = seqlen_k_static
        if const_expr(mSeqUsedK is not None):
            seqlen_k = mSeqUsedK[batch_idx]
        elif const_expr(mCuSeqlensK is not None):
            seqlen_k = mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[batch_idx]

        n_block_max = cute.ceil_div(seqlen_k, tile_n)
        if const_expr(
            self.is_causal or (self.is_local and window_size_right is not None)
        ):
            logical_q = seqlen_q
            if const_expr(self.pack_gqa):
                logical_q *= self.qhead_per_kvhead
            m_idx_max = cute.ceil_div(logical_q, tile_m) * tile_m
            if const_expr(self.pack_gqa):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead)
            n_idx = m_idx_max + seqlen_k - seqlen_q
            n_idx_right = (
                n_idx if const_expr(self.is_causal) else n_idx + window_size_right
            )
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, tile_n))

        n_block_min = Int32(0)
        if const_expr(self.is_local and window_size_left is not None):
            n_idx_left = seqlen_k - seqlen_q - window_size_left
            n_block_min = cutlass.max(n_idx_left // tile_n, 0)

        effective_n_blocks = cutlass.max(Int32(0), n_block_max - n_block_min)
        if seqlen_q <= Int32(0) or seqlen_k <= Int32(0):
            effective_n_blocks = Int32(0)
        return effective_n_blocks

    @cute.kernel
    def splitkv_scheduler_kernel(
        self,
        mPartialOrder: Optional[cute.Tensor],
        mPartialOrderState: Optional[cute.Tensor],
        mActualSplits: cute.Tensor,
        mSplitOffsets: cute.Tensor,
        mTotalSplitUnits: cute.Tensor,
        mNumSplitsActual: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        num_batch: Int32,
        num_head: Int32,
        seqlen_k_static: Int32,
        num_splits: Int32,
        tile_m: Int32,
        tile_n: Int32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        target_partial_tasks: Int32,
    ):
        lane_idx = cute.arch.lane_idx()
        local_active_m_blocks = Int32(0)
        local_max_split_units = Int32(0)
        local_effective_n_blocks = Int32(0)
        local_max_effective_n_blocks = Int32(0)
        batch_idx = lane_idx
        while batch_idx < num_batch:
            seqlen_q = self._splitkv_scheduler_seqlen_q(batch_idx, mCuSeqlensQ)
            logical_q = seqlen_q
            if const_expr(self.pack_gqa):
                logical_q *= self.qhead_per_kvhead
            num_m_blocks = cute.ceil_div(logical_q, tile_m)
            effective_n_blocks = self._splitkv_scheduler_effective_n_blocks(
                batch_idx,
                mCuSeqlensQ,
                mCuSeqlensK,
                mSeqUsedK,
                seqlen_k_static,
                tile_m,
                tile_n,
                window_size_left,
                window_size_right,
            )
            if const_expr(
                self.split_kv_len is not None and not self.dynamic_splitkv_balance
            ):
                n_blocks_per_split_target = cutlass.max(
                    Int32(1),
                    Int32(
                        (self.split_kv_len + self.n_block_size - 1) // self.n_block_size
                    ),
                )
                num_kv_splits = cutlass.max(
                    Int32(1),
                    cutlass.min(
                        num_splits, effective_n_blocks // n_blocks_per_split_target
                    ),
                )
            else:
                num_kv_splits = cutlass.max(
                    Int32(1), cutlass.min(num_splits, effective_n_blocks)
                )
            if effective_n_blocks <= Int32(0):
                num_kv_splits = Int32(0)
            mActualSplits[batch_idx] = num_kv_splits
            mSplitOffsets[batch_idx + Int32(1)] = effective_n_blocks
            if num_kv_splits > Int32(0):
                local_active_m_blocks += num_m_blocks
                local_max_split_units += num_m_blocks * num_kv_splits
                local_effective_n_blocks += num_m_blocks * effective_n_blocks
                local_max_effective_n_blocks = cutlass.max(
                    local_max_effective_n_blocks, effective_n_blocks
                )
            batch_idx += Int32(cute.arch.WARP_SIZE)
        cute.arch.sync_warp()

        active_m_blocks = utils.warp_reduce(local_active_m_blocks, operator.add)
        max_split_units = utils.warp_reduce(local_max_split_units, operator.add)
        total_effective_n_blocks = utils.warp_reduce(
            local_effective_n_blocks, operator.add
        )
        max_effective_n_blocks = utils.warp_reduce(local_max_effective_n_blocks, max)

        # Note(wangbojun/codex): schedule by real q/k lengths. CUDA graph
        # padding rows have q_len=0 or kv_len=0 and therefore contribute no
        # partial work, while workspace shape remains graph-stable.
        split_overhead_blocks = Int32(128)
        plateau_split_blocks_value = (
            32768 + self.n_block_size - 1
        ) // self.n_block_size
        plateau_split_blocks = Int32(plateau_split_blocks_value)
        target_split_units = cute.ceil_div(target_partial_tasks, num_head)
        target_split_units = cutlass.max(active_m_blocks, target_split_units)
        if const_expr(self.dynamic_splitkv_balance):
            plateau_units = (
                total_effective_n_blocks + plateau_split_blocks - Int32(1)
            ) // plateau_split_blocks
            target_split_units = cutlass.max(
                target_split_units,
                plateau_units,
            )
        target_split_units = cutlass.min(max_split_units, target_split_units)
        balance_target_partial_tasks = target_partial_tasks
        if const_expr(self.dynamic_splitkv_balance):
            balance_target_partial_tasks = target_split_units * num_head
        blocks_per_split = Int32(1)
        if target_split_units > Int32(0):
            blocks_per_split = cutlass.max(
                Int32(1), cute.ceil_div(total_effective_n_blocks, target_split_units)
            )
        if const_expr(self.dynamic_splitkv_balance):
            candidate_makespan = cutlass.min(max_effective_n_blocks, blocks_per_split)
            makespan_reduction = max_effective_n_blocks - candidate_makespan
            if (
                active_m_blocks > Int32(0)
                and makespan_reduction < split_overhead_blocks
            ):
                blocks_per_split = max_effective_n_blocks
        if const_expr(not self.dynamic_splitkv_balance):
            blocks_per_split = Int32(1)
            if const_expr(self.split_kv_len is not None):
                blocks_per_split = cutlass.max(
                    Int32(1),
                    Int32(
                        (self.split_kv_len + self.n_block_size - 1) // self.n_block_size
                    ),
                )

        local_preliminary_partial_tasks = Int32(0)
        batch_idx = lane_idx
        while batch_idx < num_batch:
            seqlen_q = self._splitkv_scheduler_seqlen_q(batch_idx, mCuSeqlensQ)
            logical_q = seqlen_q
            if const_expr(self.pack_gqa):
                logical_q *= self.qhead_per_kvhead
            num_m_blocks = cute.ceil_div(logical_q, tile_m)
            effective_n_blocks = mSplitOffsets[batch_idx + Int32(1)]
            max_kv_splits = mActualSplits[batch_idx]
            if effective_n_blocks > Int32(0):
                prelim_splits = max_kv_splits
                if const_expr(self.dynamic_splitkv_balance):
                    split_candidate = cutlass.min(
                        max_kv_splits,
                        cute.ceil_div(effective_n_blocks, blocks_per_split),
                    )
                    prelim_splits = Int32(
                        cutlass.select_(
                            effective_n_blocks < split_overhead_blocks,
                            Int32(1),
                            split_candidate,
                        )
                    )
                local_preliminary_partial_tasks += (
                    num_m_blocks * prelim_splits * num_head
                )
            batch_idx += Int32(cute.arch.WARP_SIZE)

        preliminary_partial_tasks = utils.warp_reduce(
            local_preliminary_partial_tasks, operator.add
        )
        surplus_split_units = Int32(0)
        if (
            balance_target_partial_tasks > Int32(0)
            and preliminary_partial_tasks > balance_target_partial_tasks
        ):
            surplus_split_units = (
                preliminary_partial_tasks - balance_target_partial_tasks
            ) // num_head

        split_offset = Int32(0)
        eligible_offset = Int32(0)
        batch_base = Int32(0)
        if lane_idx == 0:
            mSplitOffsets[Int32(0)] = Int32(0)
        while batch_base < num_batch:
            batch_idx = batch_base + lane_idx
            actual_splits = Int32(0)
            eligible_for_surplus = Int32(0)
            num_m_blocks = Int32(0)
            if batch_idx < num_batch:
                seqlen_q = self._splitkv_scheduler_seqlen_q(batch_idx, mCuSeqlensQ)
                logical_q = seqlen_q
                if const_expr(self.pack_gqa):
                    logical_q *= self.qhead_per_kvhead
                num_m_blocks = cute.ceil_div(logical_q, tile_m)
                effective_n_blocks = mSplitOffsets[batch_idx + Int32(1)]
                max_kv_splits = mActualSplits[batch_idx]
                min_splits = Int32(1)
                if effective_n_blocks > Int32(0):
                    if const_expr(self.dynamic_splitkv_balance):
                        split_candidate = cute.ceil_div(
                            effective_n_blocks, blocks_per_split
                        )
                        split_candidate = cutlass.max(
                            min_splits, cutlass.min(max_kv_splits, split_candidate)
                        )
                        actual_splits = Int32(
                            cutlass.select_(
                                effective_n_blocks < split_overhead_blocks,
                                Int32(1),
                                split_candidate,
                            )
                        )
                    else:
                        actual_splits = max_kv_splits
                if effective_n_blocks <= Int32(0):
                    actual_splits = Int32(0)
                if const_expr(self.dynamic_splitkv_balance):
                    if actual_splits > min_splits:
                        eligible_for_surplus = num_m_blocks
            eligible_prefix = utils.warp_prefix_sum(eligible_for_surplus, lane_idx)
            if const_expr(self.dynamic_splitkv_balance):
                if (
                    eligible_for_surplus > Int32(0)
                    and eligible_offset + eligible_prefix <= surplus_split_units
                ):
                    actual_splits -= Int32(1)
            split_work_units = num_m_blocks * actual_splits
            split_prefix = utils.warp_prefix_sum(split_work_units, lane_idx)
            if batch_idx < num_batch:
                mActualSplits[batch_idx] = actual_splits
                if const_expr(mNumSplitsActual is not None):
                    mNumSplitsActual[batch_idx] = actual_splits
                mSplitOffsets[batch_idx + Int32(1)] = split_offset + split_prefix
            split_offset += utils.shuffle_sync(split_prefix, cute.arch.WARP_SIZE - 1)
            eligible_offset += utils.shuffle_sync(
                eligible_prefix, cute.arch.WARP_SIZE - 1
            )
            batch_base += Int32(cute.arch.WARP_SIZE)
        total_partial_tasks = split_offset * num_head
        total_active_tasks = active_m_blocks * num_head
        helper_cta_limit = Int32(
            (self.qhead_per_kvhead + len(self.correction_warp_ids) - 1)
            // len(self.correction_warp_ids)
            if const_expr(self.pack_gqa)
            else 1
        )
        if lane_idx == 0:
            barrier.st_release(
                utils.elem_pointer(mTotalSplitUnits, (Int32(0),)), split_offset
            )

            if const_expr(self.gpu_partial_order):
                assert mPartialOrder is not None
                assert mPartialOrderState is not None
                tail_room = total_partial_tasks % target_partial_tasks
                if tail_room == Int32(0):
                    tail_room = target_partial_tasks
                helper_cta_limit_tail = Int32(1)
                if total_active_tasks > Int32(0):
                    helper_cta_limit_tail = cutlass.max(
                        Int32(1), tail_room // total_active_tasks
                    )
                    helper_cta_limit_tail = cutlass.min(
                        helper_cta_limit_tail, helper_cta_limit
                    )
                mPartialOrderState[Int32(1)] = helper_cta_limit_tail

                count = Int32(0)
                batch_idx = Int32(0)
                while batch_idx < num_batch:
                    num_kv_splits = mActualSplits[batch_idx]
                    head_idx = Int32(0)
                    while head_idx < num_head:
                        helper_ctas = cutlass.min(num_kv_splits, helper_cta_limit_tail)
                        helper_start = num_kv_splits - helper_ctas
                        split_idx = Int32(0)
                        while split_idx < helper_start:
                            mPartialOrder[count] = (
                                batch_idx * num_head + head_idx
                            ) * num_splits + split_idx
                            count += Int32(1)
                            split_idx += Int32(1)
                        head_idx += Int32(1)
                    batch_idx += Int32(1)

                batch_idx = Int32(0)
                while batch_idx < num_batch:
                    num_kv_splits = mActualSplits[batch_idx]
                    head_idx = Int32(0)
                    while head_idx < num_head:
                        helper_ctas = cutlass.min(num_kv_splits, helper_cta_limit_tail)
                        split_idx = num_kv_splits - helper_ctas
                        while split_idx < num_kv_splits:
                            mPartialOrder[count] = (
                                batch_idx * num_head + head_idx
                            ) * num_splits + split_idx
                            count += Int32(1)
                            split_idx += Int32(1)
                        head_idx += Int32(1)
                    batch_idx += Int32(1)
                barrier.st_release(
                    utils.elem_pointer(mPartialOrderState, (Int32(0),)), count
                )
        barrier.launch_dependents()

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mNumSplitsDynamic: Optional[cute.Tensor],
        mSplitOffsets: Optional[cute.Tensor],
        mTotalSplitUnits: Optional[cute.Tensor],
        mOFinal: Optional[cute.Tensor],
        mLSEFinal: Optional[cute.Tensor],
        mNumSplitsActual: Optional[cute.Tensor],
        mSplitDoneCounter: Optional[cute.Tensor],
        mReductionQueue: Optional[cute.Tensor],
        mReductionQueueValid: Optional[cute.Tensor],
        mReductionQueueState: Optional[cute.Tensor],
        mFusedTimeline: Optional[cute.Tensor],
        mPartialOrder: Optional[cute.Tensor],
        mPartialOrderState: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mSigmoidGate: Optional[cute.Tensor] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Use the first N warps to initialize barriers
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_full_offset + i, 1)
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i,
                    len([self.mma_warp_id]),
                )
        if warp_idx == 2:
            for i in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i,
                    cute.arch.WARP_SIZE * 4,
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i,
                    cute.arch.WARP_SIZE * 4,
                )
        if warp_idx == 3:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_s0_s1_sequence_offset + i,
                        cute.arch.WARP_SIZE,
                    )
        if const_expr(not self.use_correction_warps_for_epi) and warp_idx == 4:
            for i in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
        if warp_idx == 5:
            for i in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                    cute.arch.WARP_SIZE
                    * (len(self.softmax0_warp_ids) + len(self.correction_warp_ids)),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id])
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 6:
            for i in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,
                    cute.arch.WARP_SIZE * len(self.softmax0_warp_ids),
                )
        if warp_idx == 7:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(
            mbar_ptr + self.mbar_load_kv_full_offset
        )

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype),
                sO_layout.outer,
            )

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )
        sPageRing = storage.sPageRing.get_tensor(cute.make_layout((self.kv_stage,)))

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(2 if self.q_stage == 2 else 1)
        )
        tOtOs = tuple(
            cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
            for stage in range(2 if self.q_stage == 2 else 1)
        )

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [
            cute.make_tensor(
                tOrP.iterator
                + self.qk_acc_dtype.width
                // self.q_dtype.width
                * self.tmem_p_offset[stage],
                tOrP.layout,
            )
            for stage in range(2 if self.q_stage == 2 else 1)
        ]

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            split_kv_len=None
            if const_expr(self.dynamic_splitkv_scheduler)
            else self.split_kv_len,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            self.attention_mask_cls,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if const_expr(len(self.empty_warp_ids) > 1):
            if warp_idx == self.empty_warp_ids[1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        assert len(self.empty_warp_ids) <= 2

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                mO,
                mReductionQueue,
                mFusedTimeline,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_kv,
                sPageRing,
                mbar_ptr,
                block_info,
                num_splits,
                mNumSplitsActual,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_tensors,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(
                    tmem_alloc_cols,
                    storage.tmem_holding_buf,
                    is_two_cta=False,
                )
                self.tmem_alloc_barrier.arrive_and_wait()

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStSs,
                tOtOs,
                tOrPs,
                pipeline_kv,
                mbar_ptr,
                mO,
                mReductionQueue,
                mFusedTimeline,
                block_info,
                num_splits,
                mNumSplitsActual,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=False)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if (
                warp_idx >= self.epilogue_warp_ids[0]
                and warp_idx <= self.epilogue_warp_ids[-1]
            ):
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    mOFinal,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    mNumSplitsActual,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    aux_tensors,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]
        ) or (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]):
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mO=mO,
                mLSE=mLSE,
                mReductionQueue=mReductionQueue,
                mFusedTimeline=mFusedTimeline,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                mNumSplitsActual=mNumSplitsActual,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                blocksparse_tensors=blocksparse_tensors,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(
                    0
                    if const_expr(self.q_stage == 1)
                    or warp_idx < self.softmax1_warp_ids[0]
                    else 1
                )
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(
                        tStS.iterator
                        + (
                            self.tmem_s_offset[0]
                            if stage == 0
                            else self.tmem_s_offset[1]
                        ),
                        tStS.layout,
                    ),
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(
                        tStS.iterator + self.tmem_s_offset[0], tStS.layout
                    )
                    softmax_loop(stage=0, tStSi=tStSi)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if (
                    warp_idx < self.correction_warp_ids[0]
                    and warp_idx >= self.softmax1_warp_ids[0]
                ):
                    tStSi = cute.make_tensor(
                        tStS.iterator + self.tmem_s_offset[1], tStS.layout
                    )
                    softmax_loop(stage=1, tStSi=tStSi)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtOs,
                sScale,
                mO,
                mLSE,
                mOFinal,
                mLSEFinal,
                mNumSplitsActual,
                mSplitDoneCounter,
                mReductionQueue,
                mReductionQueueValid,
                mReductionQueueState,
                mFusedTimeline,
                mPartialOrderState,
                mCuSeqlensQ,
                sO,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                mSigmoidGate,
                blocksparse_tensors,
                aux_tensors,
            )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        if const_expr(
            self.fused_splitkv_combine
            and (
                self.fused_splitkv_combine_mode == "queue"
                or self.fused_splitkv_combine_mode == "pipeline"
            )
        ):
            cute.arch.sync_threads()
            self.fused_splitkv_combine_loop(
                mO,
                mLSE,
                mOFinal,
                mLSEFinal,
                mNumSplitsActual,
                mReductionQueue,
                mReductionQueueValid,
                mReductionQueueState,
                mFusedTimeline,
                mCuSeqlensQ,
                mSigmoidGate,
            )

        return

    @cute.jit
    def _fused_timeline_mark(
        self,
        mFusedTimeline: Optional[cute.Tensor],
        slot: Int32,
        stage: cutlass.Constexpr[int],
        lane_idx: Int32,
    ) -> None:
        if const_expr(mFusedTimeline is not None):
            if lane_idx == 0:
                barrier.st_i64(
                    utils.elem_pointer(mFusedTimeline, (slot, stage)),
                    barrier.globaltimer_u64(),
                )

    @cute.jit
    def _fused_timeline_value(
        self,
        mFusedTimeline: Optional[cute.Tensor],
        slot: Int32,
        stage: cutlass.Constexpr[int],
        val: Int64,
        lane_idx: Int32,
    ) -> None:
        if const_expr(mFusedTimeline is not None):
            if lane_idx == 0:
                barrier.st_i64(utils.elem_pointer(mFusedTimeline, (slot, stage)), val)

    @cute.jit
    def _partial_timeline_slot(
        self,
        mO: cute.Tensor,
        mReductionQueue: Optional[cute.Tensor],
        batch_idx: Int32,
        head_idx: Int32,
        split_idx: Int32,
    ) -> Int32:
        offset = Int32(0)
        if const_expr(mReductionQueue is not None):
            offset = mReductionQueue.shape[0]
        return offset + (batch_idx * mO.shape[2] + head_idx) * mO.shape[3] + split_idx

    @cute.jit
    def _fused_splitkv_load_lse(
        self,
        mLSEPartial: cute.Tensor,
        q_row: Int32,
        q_offset: Int32,
        head_idx: Int32,
        split_idx: Int32,
    ) -> Float32:
        if const_expr(self.pack_gqa):
            ptr = utils.elem_pointer(
                mLSEPartial, ((q_row, q_offset), head_idx, split_idx)
            )
        else:
            ptr = utils.elem_pointer(mLSEPartial, (q_offset, head_idx, split_idx))
        return Float32(cute.make_tensor(ptr, (1,))[0])

    @cute.jit
    def _fused_splitkv_store_lse(
        self,
        mLSEFinal: cute.Tensor,
        q_row: Int32,
        q_offset: Int32,
        head_idx: Int32,
        val: Float32,
    ) -> None:
        if const_expr(self.pack_gqa):
            ptr = utils.elem_pointer(mLSEFinal, ((q_row, q_offset), head_idx))
        else:
            ptr = utils.elem_pointer(mLSEFinal, (q_offset, head_idx))
        cute.make_tensor(ptr, (1,))[0] = val

    @cute.jit
    def _fused_splitkv_load_o(
        self,
        mOPartial: cute.Tensor,
        q_row: Int32,
        q_offset: Int32,
        d_idx: Int32,
        head_idx: Int32,
        split_idx: Int32,
    ) -> Float32:
        if const_expr(self.pack_gqa):
            ptr = utils.elem_pointer(
                mOPartial, ((q_row, q_offset), d_idx, head_idx, split_idx)
            )
        else:
            ptr = utils.elem_pointer(mOPartial, (q_offset, d_idx, head_idx, split_idx))
        return Float32(cute.make_tensor(ptr, (1,))[0])

    @cute.jit
    def _fused_splitkv_store_o(
        self,
        mOFinal: cute.Tensor,
        q_row: Int32,
        q_offset: Int32,
        d_idx: Int32,
        head_idx: Int32,
        val: Float32,
    ) -> None:
        if const_expr(self.pack_gqa):
            ptr = utils.elem_pointer(mOFinal, ((q_row, q_offset), d_idx, head_idx))
        else:
            ptr = utils.elem_pointer(mOFinal, (q_offset, d_idx, head_idx))
        cute.make_tensor(ptr, (1,))[0] = val.to(mOFinal.element_type)

    @cute.jit
    def _fused_splitkv_lse(
        self,
        mLSEPartial: cute.Tensor,
        q_row: Int32,
        q_offset: Int32,
        head_idx: Int32,
        num_splits: Int32,
    ) -> Float32:
        lse_max = -Float32.inf
        for split_idx in cutlass.range(num_splits, unroll=1):
            lse_s = self._fused_splitkv_load_lse(
                mLSEPartial, q_row, q_offset, head_idx, split_idx
            )
            lse_max = utils.fmax(lse_max, lse_s)
        LOG2_E = math.log2(math.e)
        lse_sum = Float32(0.0)
        if lse_max != -Float32.inf:
            for split_idx in cutlass.range(num_splits, unroll=1):
                lse_s = self._fused_splitkv_load_lse(
                    mLSEPartial, q_row, q_offset, head_idx, split_idx
                )
                if lse_s != -Float32.inf:
                    lse_sum += utils.exp2f((lse_s - lse_max) * LOG2_E)
        return (
            utils.logf(lse_sum) + lse_max if lse_max != -Float32.inf else -Float32.inf
        )

    @cute.jit
    def fused_splitkv_combine_task(
        self,
        mOPartial: cute.Tensor,
        mLSEPartial: cute.Tensor,
        mOFinal: cute.Tensor,
        mLSEFinal: Optional[cute.Tensor],
        mNumSplitsActual: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mFusedTimeline: Optional[cute.Tensor],
        timeline_slot: Int32,
        task_id: Int32,
        lane_idx: Int32,
        mSigmoidGate: Optional[cute.Tensor] = None,
    ) -> None:
        self._fused_timeline_value(
            mFusedTimeline, timeline_slot, 6, Int64(task_id), lane_idx
        )
        qhead_per_task = const_expr(self.qhead_per_kvhead if self.pack_gqa else 1)
        base_task_id = task_id // qhead_per_task
        q_row = task_id - base_task_id * qhead_per_task
        num_sched_heads = mOPartial.shape[2]
        batch_idx = base_task_id // num_sched_heads
        head_idx = base_task_id - batch_idx * num_sched_heads
        q_offset = mCuSeqlensQ[batch_idx]
        num_splits = mNumSplitsActual[batch_idx]
        LOG2_E = math.log2(math.e)

        lse = Float32(0.0)
        self._fused_timeline_mark(mFusedTimeline, timeline_slot, 7, lane_idx)
        if lane_idx == 0:
            lse = self._fused_splitkv_lse(
                mLSEPartial, q_row, q_offset, head_idx, num_splits
            )
            if const_expr(mLSEFinal is not None):
                self._fused_splitkv_store_lse(mLSEFinal, q_row, q_offset, head_idx, lse)
        lse = utils.shuffle_sync(lse, 0)
        self._fused_timeline_mark(mFusedTimeline, timeline_slot, 8, lane_idx)

        elems_per_lane = const_expr(
            (self.head_dim_v_padded + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
        )
        out_acc = cute.make_rmem_tensor(elems_per_lane, Float32)
        out_acc.fill(0.0)

        self._fused_timeline_mark(mFusedTimeline, timeline_slot, 9, lane_idx)
        if lse != -Float32.inf:
            for split_idx in cutlass.range(num_splits, unroll=1):
                scale = Float32(0.0)
                if lane_idx == 0:
                    lse_s = self._fused_splitkv_load_lse(
                        mLSEPartial, q_row, q_offset, head_idx, split_idx
                    )
                    if lse_s != -Float32.inf:
                        scale = utils.exp2f((lse_s - lse) * LOG2_E)
                scale = utils.shuffle_sync(scale, 0)
                if scale != Float32(0.0):
                    for d_slot in cutlass.range_constexpr(elems_per_lane):
                        d_idx = lane_idx + d_slot * cute.arch.WARP_SIZE
                        if (
                            const_expr(not self.check_hdim_v_oob)
                            or d_idx < mOFinal.shape[1]
                        ):
                            out_acc[d_slot] += scale * self._fused_splitkv_load_o(
                                mOPartial,
                                q_row,
                                q_offset,
                                d_idx,
                                head_idx,
                                split_idx,
                            )

        # Head-wise sigmoid gate on the combined output (matches the in-kernel
        # non-split epilogue and the non-fused combine kernel). The gate is a
        # per-(token, query-head) scalar, so it distributes over the linear
        # LSE-weighted combine -- applying it once to out_acc here is exact.
        if const_expr(mSigmoidGate is not None):
            if const_expr(self.pack_gqa):
                q_head_idx = head_idx * self.qhead_per_kvhead + q_row
            else:
                q_head_idx = head_idx
            gate_scale = Float32(0.5) * cute.math.tanh(
                Float32(0.5) * Float32(mSigmoidGate[q_offset, q_head_idx])
            ) + Float32(0.5)
            for d_slot in cutlass.range_constexpr(elems_per_lane):
                out_acc[d_slot] = out_acc[d_slot] * gate_scale

        for d_slot in cutlass.range_constexpr(elems_per_lane):
            d_idx = lane_idx + d_slot * cute.arch.WARP_SIZE
            if const_expr(not self.check_hdim_v_oob) or d_idx < mOFinal.shape[1]:
                self._fused_splitkv_store_o(
                    mOFinal, q_row, q_offset, d_idx, head_idx, out_acc[d_slot]
                )
        self._fused_timeline_mark(mFusedTimeline, timeline_slot, 10, lane_idx)
        self._fused_timeline_mark(mFusedTimeline, timeline_slot, 11, lane_idx)

    @cute.jit
    def fused_splitkv_combine_loop(
        self,
        mOPartial: cute.Tensor,
        mLSEPartial: cute.Tensor,
        mOFinal: cute.Tensor,
        mLSEFinal: Optional[cute.Tensor],
        mNumSplitsActual: cute.Tensor,
        mReductionQueue: cute.Tensor,
        mReductionQueueValid: cute.Tensor,
        mReductionQueueState: cute.Tensor,
        mFusedTimeline: Optional[cute.Tensor],
        mCuSeqlensQ: cute.Tensor,
        mSigmoidGate: Optional[cute.Tensor] = None,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        num_tasks = mReductionQueue.shape[0]
        write_ptr = utils.elem_pointer(mReductionQueueState, (Int32(0),))
        read_ptr = utils.elem_pointer(mReductionQueueState, (Int32(1),))
        progress = barrier.ld_acquire(read_ptr)
        while progress < num_tasks:
            task_slot = num_tasks
            if lane_idx == 0:
                write_progress = barrier.ld_acquire(write_ptr)
                if progress < write_progress:
                    old_read = barrier.atom_cas_i32(
                        read_ptr, progress, progress + Int32(1)
                    )
                    if old_read == progress:
                        task_slot = progress
            task_slot = cute.arch.shuffle_sync(task_slot, 0)
            if task_slot < num_tasks:
                self._fused_timeline_mark(mFusedTimeline, task_slot, 3, lane_idx)
                self._fused_timeline_mark(mFusedTimeline, task_slot, 4, lane_idx)
                ready = Int32(0)
                ready_ptr = utils.elem_pointer(mReductionQueueValid, (task_slot,))
                while ready != Int32(1):
                    ready = barrier.ld_acquire(ready_ptr)
                self._fused_timeline_mark(mFusedTimeline, task_slot, 5, lane_idx)
                task_id = mReductionQueue[task_slot]
                self.fused_splitkv_combine_task(
                    mOPartial,
                    mLSEPartial,
                    mOFinal,
                    mLSEFinal,
                    mNumSplitsActual,
                    mCuSeqlensQ,
                    mFusedTimeline,
                    task_slot,
                    task_id,
                    lane_idx,
                    mSigmoidGate,
                )
            progress = barrier.ld_acquire(read_ptr)

    @cute.jit
    def fused_splitkv_try_combine_ready(
        self,
        mOPartial: cute.Tensor,
        mLSEPartial: cute.Tensor,
        mOFinal: cute.Tensor,
        mLSEFinal: Optional[cute.Tensor],
        mNumSplitsActual: cute.Tensor,
        mReductionQueue: cute.Tensor,
        mReductionQueueValid: cute.Tensor,
        mReductionQueueState: cute.Tensor,
        mFusedTimeline: Optional[cute.Tensor],
        mCuSeqlensQ: cute.Tensor,
        lane_idx: Int32,
        max_claims_per_warp: cutlass.Constexpr[int],
        mSigmoidGate: Optional[cute.Tensor] = None,
    ) -> None:
        num_tasks = mReductionQueue.shape[0]
        write_ptr = utils.elem_pointer(mReductionQueueState, (Int32(0),))
        read_ptr = utils.elem_pointer(mReductionQueueState, (Int32(1),))
        for _ in cutlass.range_constexpr(max_claims_per_warp):
            task_slot = num_tasks
            if lane_idx == 0:
                read_progress = barrier.ld_acquire(read_ptr)
                write_progress = barrier.ld_acquire(write_ptr)
                if read_progress < write_progress:
                    old_read = barrier.atom_cas_i32(
                        read_ptr, read_progress, read_progress + Int32(1)
                    )
                    if old_read == read_progress:
                        task_slot = read_progress
            task_slot = cute.arch.shuffle_sync(task_slot, 0)
            if task_slot < num_tasks:
                self._fused_timeline_mark(mFusedTimeline, task_slot, 3, lane_idx)
                self._fused_timeline_mark(mFusedTimeline, task_slot, 4, lane_idx)
                ready = Int32(0)
                ready_ptr = utils.elem_pointer(mReductionQueueValid, (task_slot,))
                while ready != Int32(1):
                    ready = barrier.ld_acquire(ready_ptr)
                self._fused_timeline_mark(mFusedTimeline, task_slot, 5, lane_idx)
                task_id = mReductionQueue[task_slot]
                self.fused_splitkv_combine_task(
                    mOPartial,
                    mLSEPartial,
                    mOFinal,
                    mLSEFinal,
                    mNumSplitsActual,
                    mCuSeqlensQ,
                    mFusedTimeline,
                    task_slot,
                    task_id,
                    lane_idx,
                    mSigmoidGate,
                )

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mReductionQueue: Optional[cute.Tensor],
        mFusedTimeline: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        sPageRing: cute.Tensor,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        mNumSplitsActual: Optional[cute.Tensor],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(mFusedTimeline is not None):
                partial_timeline_slot = self._partial_timeline_slot(
                    mO, mReductionQueue, batch_idx, head_idx, split_idx
                )
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 0, tidx
                )
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 12, tidx
                )
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
            )

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [
                        t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)
                    ]
                else:
                    mK_cur = cute.domain_offset(
                        (seqlen.offset_k, 0), mK[None, None, head_idx_kv]
                    )
                    mV_cur = cute.domain_offset(
                        (0, seqlen.offset_k), mV[None, None, head_idx_kv]
                    )
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
                )
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmod.create(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                    arch=self.arch,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
            )

            if const_expr(not self.use_block_sparsity):
                actual_num_splits = work_tile.actual_num_splits
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, actual_num_splits
                )
                process_tile = n_block_min < n_block_max
                if process_tile:
                    if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                        load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        self.load_page_idx_shared(
                            mPageTable,
                            sPageRing,
                            batch_idx,
                            n_block_first,
                            kv_producer_state.index,
                            tidx,
                        )
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    load_K(
                        block=n_block_max - 1,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )  # K0
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and (
                        const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE
                    ):
                        load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
                    q_producer_phase ^= 1
                    load_V(
                        block=n_block_max - 1,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )  # V0
                    kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            self.load_page_idx_shared(
                                mPageTable,
                                sPageRing,
                                batch_idx,
                                n_block,
                                kv_producer_state.index,
                                tidx,
                            )
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}, page_idx = {}", n_block, page_idx)
                        load_K(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )  # Ki
                        kv_producer_state.advance()
                        load_V(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )  # Vi
                        kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                )
            if const_expr(mFusedTimeline is not None):
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 13, tidx
                )

            # Only persistent dynamic SplitKV synchronizes the work-tile warps here.
            self.sync_dynamic_work_tile()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def load_page_idx_shared(
        self,
        mPageTable: cute.Tensor,
        sPageRing: cute.Tensor,
        batch_idx: Int32,
        n_block: Int32,
        ring_slot: Int32,
        tidx: Int32,
    ) -> Int32:
        if tidx == 0:
            sPageRing[ring_slot] = mPageTable[batch_idx, n_block]
        cute.arch.sync_warp()
        return sPageRing[ring_slot]

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        mO: cute.Tensor,
        mReductionQueue: Optional[cute.Tensor],
        mFusedTimeline: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        mNumSplitsActual: Optional[cute.Tensor],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op,
                self.tmem_s_offset[stage],
                tSrQs[stage],
                sA=sQ[None, None, None, stage],
                zero_init=True,
            )
            for stage in range(2 if self.q_stage == 2 else 1)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
            )
            for stage in range(2 if self.q_stage == 2 else 1)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(mFusedTimeline is not None):
                partial_timeline_slot = self._partial_timeline_slot(
                    mO, mReductionQueue, batch_idx, head_idx, split_idx
                )
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 14, cute.arch.lane_idx()
                )
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(
                    blocksparse_tensors, batch_idx, head_idx, m_block
                )
                process_tile = block_iter_count > Int32(0)
            else:
                actual_num_splits = work_tile.actual_num_splits
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, actual_num_splits
                )
                block_iter_count = n_block_max - n_block_min
                process_tile = n_block_min < n_block_max

            if process_tile:
                for stage in cutlass.range_constexpr(
                    2 if const_expr(self.q_stage == 2) else 1
                ):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_load_q_full_offset + stage,
                        mma_q_consumer_phase,
                    )
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(
                            sK_cur,
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                    gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(
                        2 if const_expr(self.q_stage == 2) else 1
                    ):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile has finished.
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        # 4. release accumulated O0_partial / O1_partial
                        # Don't need to signal O_full to the correction warps anymore since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # with cute.arch.elect_one():
                        #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # 5. release V(i-1)
                        if const_expr(
                            (self.q_stage == 2 and stage == 1) or self.q_stage == 1
                        ):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        # 3. release S0
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(
                        2 if const_expr(self.q_stage == 2) else 1
                    ):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(
                    2 if const_expr(self.q_stage == 2) else 1
                ):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                        P_full_O_rescaled_phase,
                    )
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=not O_should_accumulate,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            if const_expr(mFusedTimeline is not None):
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 15, cute.arch.lane_idx()
                )
            # Only persistent dynamic SplitKV synchronizes the work-tile warps here.
            self.sync_dynamic_work_tile()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mReductionQueue: Optional[cute.Tensor],
        mFusedTimeline: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        mNumSplitsActual: Optional[cute.Tensor],
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(mFusedTimeline is not None):
                partial_timeline_slot = self._partial_timeline_slot(
                    mO, mReductionQueue, batch_idx, head_idx, split_idx
                )
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 16, tidx
                )
            seqlen = SeqlenInfoCls(batch_idx)
            actual_num_splits = work_tile.actual_num_splits
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, actual_num_splits
            )

            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            shared_mask_kwargs = dict(
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )
            block_mask_mod = (
                self.mask_mod if const_expr(self.use_block_sparsity) else None
            )
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=block_mask_mod,
                fastdiv_mods=fastdiv_mods,
                **shared_mask_kwargs,
            )
            if const_expr(self.swa_right_only):
                right_mask_fn = partial(
                    mask.apply_mask_sm100_right_only,
                    fastdiv_mods=fastdiv_mods,
                    **shared_mask_kwargs,
                )
            else:
                right_mask_fn = mask_fn
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors, batch_idx, head_idx, m_block
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = tile_block_count > Int32(0)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=self.q_stage * m_block + stage,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage,
                    si_corr_producer_phase,
                )
                si_corr_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,
                    self.mbar_P_full_2_offset,
                    self.q_stage,
                    Int32(stage),
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # if tidx == 0:
                    #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage
                    )
                    # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])
            else:
                if tile_block_count > Int32(0):
                    (
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(right_mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(right_mask_fn, mask_seqlen=False),
                            )
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )
                    # The remaining iterations have no masking
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        n_block = n_block_max - n_tile - 1
                        (
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            si_corr_producer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                        )
                    # Separate iterations with local masking on the left
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            0, n_block_max - n_block_min, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                si_corr_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage
                    )

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + utils.log2f(softmax.row_sum[0])) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            if const_expr(mFusedTimeline is not None):
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 17, tidx
                )
            # Only persistent dynamic SplitKV synchronizes the work-tile warps here.
            self.sync_dynamic_work_tile()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScP = cute.composition(
            tScS, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )

        # Custom masks can issue metadata loads while the softmax warpgroup is
        # waiting for MMA to publish the score tile.
        if const_expr(self.prefetch_attention_mask):
            prefetched_mask = self.prefetch_attention_mask_sm100(
                m_block,
                n_block,
                thr_mma_qk,
                thr_tmem_load,
                aux_tensors,
            )

        # Wait for Si
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase
        )
        tSrS_t2r = cute.make_rmem_tensor(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                aux_tensors,
                fastdiv_mods,
            )

        if const_expr(mask_fn is not None):
            if const_expr(self.prefetch_attention_mask):
                self.apply_prefetched_attention_mask_sm100(
                    tSrS_t2r,
                    prefetched_mask,
                )
            else:
                mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        # print(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
            )
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(tScP).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            e2e=mask_fn is None and self.head_dim_padded <= 128,
            e2e_freq=self.e2e_freq,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(
                mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4
            )
        # print(tSrP_r2t_f32, tStP_r2t)
        # cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2]) // 4 * 3):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // 4 * 3, cute.size(tStP_r2t.shape[2])
        ):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage,
            si_corr_producer_phase,
        )
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        return (
            mma_si_consumer_phase ^ 1,
            si_corr_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def output_row_offset(self, m_block: Int32, aux_tensors: Optional[list]) -> Int32:
        return Int32(0)

    @cute.jit
    def output_seqlen_q(
        self,
        seqlen_q: Int32,
        m_block: Int32,
        aux_tensors: Optional[list],
    ) -> Int32:
        return seqlen_q

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mOFinal: Optional[cute.Tensor],
        mLSEFinal: Optional[cute.Tensor],
        mNumSplitsActual: Optional[cute.Tensor],
        mSplitDoneCounter: Optional[cute.Tensor],
        mReductionQueue: Optional[cute.Tensor],
        mReductionQueueValid: Optional[cute.Tensor],
        mReductionQueueState: Optional[cute.Tensor],
        mFusedTimeline: Optional[cute.Tensor],
        mPartialOrderState: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mSigmoidGate: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )

        # First iter: no correction is required
        for stage in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
            cute.arch.mbarrier_arrive(
                mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
            )

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            partial_timeline_slot = Int32(0)
            if const_expr(mFusedTimeline is not None):
                partial_timeline_slot = self._partial_timeline_slot(
                    mO, mReductionQueue, batch_idx, head_idx, split_idx
                )
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 18, tidx
                )
            seqlen = SeqlenInfoCls(batch_idx)
            output_seqlen_q = self.output_seqlen_q(
                seqlen.seqlen_q, m_block, aux_tensors
            )
            actual_num_splits = work_tile.actual_num_splits
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, actual_num_splits
            )

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                    None, None, head_idx, split_idx
                ]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                mO_cur = cute.domain_offset(
                    ((0, self.output_row_offset(m_block, aux_tensors)), 0), mO_cur
                )
            gO = None
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(
                    mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0)
                )
            if const_expr(
                self.is_split_kv
                and self.dynamic_splitkv_scheduler
                and mOFinal is not None
            ):
                mO_final_cur = seqlen.offset_batch_Q(mOFinal, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                gO_final = None
                if const_expr(not self.pack_gqa):
                    gO_final = cute.local_tile(
                        mO_final_cur,
                        (self.m_block_size, self.head_dim_v_padded),
                        (None, 0),
                    )

            # Default LSE to -inf for invalid split_idx tiles
            stats = [
                (
                    0.0,
                    -Float32.inf
                    if const_expr(mLSE is not None or learnable_sink is not None)
                    else None,
                    True,
                )
            ] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors, batch_idx, head_idx, m_block
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = total_block_count > Int32(0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                if const_expr(self.q_stage == 2):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 0,
                        softmax_corr_consumer_phase,
                    )
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + 0
                    )
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 1,
                        softmax_corr_consumer_phase,
                    )
                else:
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + 0,
                        softmax_corr_consumer_phase,
                    )
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + 0
                    )
                softmax_corr_consumer_phase ^= 1

                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(
                        2 if const_expr(self.q_stage == 2) else 1
                    ):
                        # wait for S0 / S1
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                            softmax_corr_consumer_phase,
                        )
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # should_rescale = True
                        # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv,
                                tOtOs[stage if self.q_stage == 2 else 0],
                                tidx,
                                scale,
                            )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
                        )
                        empty_stage = 1 - stage if const_expr(self.q_stage == 2) else 0
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_softmax_corr_empty_offset + empty_stage
                        )
                    softmax_corr_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + 1
                    )
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(
                            2 if const_expr(self.q_stage == 2) else 1
                        ):
                            q_head_idx = (
                                (self.q_stage * m_block + stage) * self.m_block_size
                                + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(
                                learnable_sink[q_head_idx]
                            )
                for stage in cutlass.range_constexpr(
                    2 if const_expr(self.q_stage == 2) else 1
                ):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        softmax_corr_consumer_phase,
                    )
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ]
                    else:
                        row_max = None
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_softmax_corr_empty_offset + stage
                    )
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += utils.exp2f(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_O_full_offset + stage,
                        o_corr_consumer_phase,
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage,
                            corr_epi_producer_phase,
                        )
                    sigmoid_gate_scale = None
                    if const_expr(mSigmoidGate is not None):
                        packed_row = (
                            self.q_stage * m_block + stage
                        ) * self.m_block_size + tidx
                        if const_expr(self.pack_gqa):
                            q_token_idx = packed_row // self.qhead_per_kvhead
                            q_head_offset = (
                                packed_row - q_token_idx * self.qhead_per_kvhead
                            )
                            q_head_idx = (
                                head_idx * self.qhead_per_kvhead + q_head_offset
                            )
                        else:
                            q_token_idx = packed_row
                            q_head_idx = head_idx
                        sigmoid_gate_scale = Float32(1.0)
                        if q_token_idx < seqlen.seqlen_q:
                            sigmoid_gate_scale = Float32(0.5) * cute.math.tanh(
                                Float32(0.5)
                                * Float32(
                                    mSigmoidGate[
                                        seqlen.offset_q + q_token_idx, q_head_idx
                                    ]
                                )
                            ) + Float32(0.5)
                    if const_expr(
                        self.is_split_kv
                        and self.dynamic_splitkv_scheduler
                        and mOFinal is not None
                    ):
                        if actual_num_splits == Int32(1):
                            self.correction_epilogue(
                                thr_mma_pv,
                                tOtOs[stage],
                                tidx,
                                stage,
                                m_block,
                                seqlen.seqlen_q,
                                scale,
                                sO[None, None, stage],
                                mO_final_cur,
                                gO_final,
                                gmem_tiled_copy_O,
                                sigmoid_gate_scale,
                            )
                        else:
                            self.correction_epilogue(
                                thr_mma_pv,
                                tOtOs[stage],
                                tidx,
                                stage,
                                m_block,
                                seqlen.seqlen_q,
                                scale,
                                sO[None, None, stage],
                                mO_cur,
                                gO,
                                gmem_tiled_copy_O,
                                None,
                            )
                    else:
                        self.correction_epilogue(
                            thr_mma_pv,
                            tOtOs[stage],
                            tidx,
                            stage,
                            m_block,
                            output_seqlen_q,
                            scale,
                            sO[None, None, stage],
                            mO_cur,
                            gO,
                            gmem_tiled_copy_O,
                            sigmoid_gate_scale
                            if const_expr(not self.is_split_kv)
                            else None,
                        )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage
                        )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    cute.arch.mbarrier_arrive(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage
                    )
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # WARNING: we need some code before the const_expr, see https://github.com/NVIDIA/cutlass/issues/2781
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                else:
                    gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset(
                            (offset,), mLSE[None, head_idx, split_idx]
                        )
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                mLSE_cur = cute.domain_offset(
                    ((0, self.output_row_offset(m_block, aux_tensors)),), mLSE_cur
                )
                if const_expr(
                    self.is_split_kv
                    and self.dynamic_splitkv_scheduler
                    and mLSEFinal is not None
                ):
                    if const_expr(not seqlen.has_cu_seqlens_q):
                        mLSE_final_cur = mLSEFinal[None, head_idx, batch_idx]
                    else:
                        offset = (
                            seqlen.offset_q
                            if const_expr(not self.pack_gqa)
                            else (0, seqlen.offset_q)
                        )
                        mLSE_final_cur = cute.domain_offset(
                            (offset,), mLSEFinal[None, head_idx]
                        )
                    if actual_num_splits == Int32(1):
                        self.correction_store_lse(
                            mLSE_final_cur,
                            stats,
                            tidx,
                            m_block,
                            output_seqlen_q,
                            softmax_scale_log2,
                        )
                    else:
                        self.correction_store_lse(
                            mLSE_cur,
                            stats,
                            tidx,
                            m_block,
                            output_seqlen_q,
                            softmax_scale_log2,
                        )
                else:
                    self.correction_store_lse(
                        mLSE_cur,
                        stats,
                        tidx,
                        m_block,
                        output_seqlen_q,
                        softmax_scale_log2,
                    )

            if const_expr(mFusedTimeline is not None):
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 19, tidx
                )
                self._fused_timeline_mark(
                    mFusedTimeline, partial_timeline_slot, 1, tidx
                )

            if const_expr(mSplitDoneCounter is not None):
                if actual_num_splits > Int32(1):
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierFwd.Epilogue),
                        number_of_threads=len(self.correction_warp_ids)
                        * cute.arch.WARP_SIZE,
                    )
                    num_sched_heads = mO.shape[2]
                    task_id = batch_idx * num_sched_heads + head_idx
                    done_ptr = utils.elem_pointer(mSplitDoneCounter, (task_id,))
                    expected_splits = mNumSplitsActual[batch_idx]
                    qhead_per_task = const_expr(
                        self.qhead_per_kvhead if self.pack_gqa else 1
                    )
                    helper_cta_limit = const_expr(
                        (qhead_per_task + len(self.correction_warp_ids) - 1)
                        // len(self.correction_warp_ids)
                    )
                    if const_expr(self.fused_splitkv_combine_mode == "producer"):
                        warp_idx = cute.arch.make_warp_uniform(
                            tidx // cute.arch.WARP_SIZE
                        )
                        lane_idx = cute.arch.lane_idx()
                        old_done = Int32(0)
                        if tidx == 0:
                            old_done = barrier.atom_add_i32(done_ptr, Int32(1))
                            sScale[0] = Float32(old_done)
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierFwd.Epilogue),
                            number_of_threads=len(self.correction_warp_ids)
                            * cute.arch.WARP_SIZE,
                        )
                        old_done = Int32(sScale[0])
                        helper_ctas = cutlass.min(
                            expected_splits, Int32(helper_cta_limit)
                        )
                        if const_expr(
                            self.gpu_partial_order and mPartialOrderState is not None
                        ):
                            helper_ctas = cutlass.min(
                                expected_splits, mPartialOrderState[Int32(1)]
                            )
                        helper_start = expected_splits - helper_ctas
                        if old_done >= helper_start:
                            done = old_done + Int32(1)
                            while done < expected_splits:
                                done = barrier.ld_acquire(done_ptr)
                            cute.arch.fence_acq_rel_gpu()
                            helper_rank = old_done - helper_start
                            for q_row_wave in cutlass.range_constexpr(helper_cta_limit):
                                q_row = (
                                    Int32(q_row_wave) * helper_ctas + helper_rank
                                ) * Int32(len(self.correction_warp_ids)) + warp_idx
                                if q_row < qhead_per_task:
                                    combine_task_id = task_id * qhead_per_task + q_row
                                    self._fused_timeline_mark(
                                        mFusedTimeline, combine_task_id, 2, lane_idx
                                    )
                                    self.fused_splitkv_combine_task(
                                        mO,
                                        mLSE,
                                        mOFinal,
                                        mLSEFinal,
                                        mNumSplitsActual,
                                        mCuSeqlensQ,
                                        mFusedTimeline,
                                        combine_task_id,
                                        combine_task_id,
                                        lane_idx,
                                        mSigmoidGate,
                                    )
                    else:
                        old_done = Int32(0)
                        if tidx == 0:
                            old_done = barrier.atom_add_i32(done_ptr, Int32(1))
                            if const_expr(
                                self.fused_splitkv_combine_mode == "pipeline"
                            ):
                                sScale[0] = Float32(old_done)
                            if old_done + Int32(1) == expected_splits:
                                cute.arch.fence_acq_rel_gpu()
                                for q_row in cutlass.range_constexpr(qhead_per_task):
                                    queue_idx = barrier.atom_add_i32(
                                        utils.elem_pointer(
                                            mReductionQueueState, (Int32(0),)
                                        ),
                                        Int32(1),
                                    )
                                    mReductionQueue[queue_idx] = (
                                        task_id * qhead_per_task + q_row
                                    )
                                    self._fused_timeline_mark(
                                        mFusedTimeline, queue_idx, 2, tidx
                                    )
                                    cute.arch.fence_acq_rel_gpu()
                                    barrier.st_release(
                                        utils.elem_pointer(
                                            mReductionQueueValid, (queue_idx,)
                                        ),
                                        Int32(1),
                                    )
                        if const_expr(self.fused_splitkv_combine_mode == "pipeline"):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierFwd.Epilogue),
                                number_of_threads=len(self.correction_warp_ids)
                                * cute.arch.WARP_SIZE,
                            )
                            old_done = Int32(sScale[0])
                            if old_done + Int32(1) == expected_splits:
                                self.fused_splitkv_try_combine_ready(
                                    mO,
                                    mLSE,
                                    mOFinal,
                                    mLSEFinal,
                                    mNumSplitsActual,
                                    mReductionQueue,
                                    mReductionQueueValid,
                                    mReductionQueueState,
                                    mFusedTimeline,
                                    mCuSeqlensQ,
                                    cute.arch.lane_idx(),
                                    qhead_per_task * 8,
                                    mSigmoidGate,
                                )

            # Advance to next tile
            # Only persistent dynamic SplitKV synchronizes the work-tile warps here.
            self.sync_dynamic_work_tile()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop
        barrier.launch_dependents()

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_store_lse(
        self,
        mLSE_cur: cute.Tensor,
        stats: list,
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        softmax_scale_log2: Float32,
    ):
        for stage in cutlass.range_constexpr(2 if const_expr(self.q_stage == 2) else 1):
            row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
            LN2 = math.log(2.0)
            lse = (
                (row_max * softmax_scale_log2 + utils.log2f(row_sum)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
            seqlen_q_store = (
                seqlen_q
                if const_expr(not self.pack_gqa)
                else seqlen_q * self.qhead_per_kvhead
            )
            row = (self.q_stage * m_block + stage) * self.m_block_size + tidx
            if row < seqlen_q_store:
                if const_expr(not self.pack_gqa):
                    gLSE = cute.local_tile(
                        mLSE_cur,
                        (self.m_block_size,),
                        (self.q_stage * m_block + stage,),
                    )
                    gLSE[tidx] = lse
                else:
                    q_token_idx = row // self.qhead_per_kvhead
                    q_head_idx = row - q_token_idx * self.qhead_per_kvhead
                    mLSE_first_dim = cute.make_tensor(
                        mLSE_cur.iterator,
                        cute.make_layout(
                            (mLSE_cur.shape[0],),
                            stride=(mLSE_cur.stride[0],),
                        ),
                    )
                    lse_ptr_i64 = utils.elem_pointer(
                        mLSE_first_dim, ((q_head_idx, q_token_idx),)
                    ).toint()
                    lse_gmem_ptr = cute.make_ptr(
                        mLSE_cur.element_type,
                        lse_ptr_i64,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
        sigmoid_gate_scale: Optional[Float32] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        ).get_slice(tidx)
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(
                tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOrO_frg_cvt = cute.make_rmem_tensor(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            if const_expr(sigmoid_gate_scale is not None):
                tOrO_frg_gated = cute.make_rmem_tensor(
                    tOrO_frg.shape, self.pv_acc_dtype
                )
                tOrO_frg_gated.store(tOrO_frg_cvt.load().to(self.pv_acc_dtype))
                for j in cutlass.range_constexpr(0, cute.size(tOrO_frg_gated), 2):
                    tOrO_frg_gated[j], tOrO_frg_gated[j + 1] = utils.mul_packed_f32x2(
                        (tOrO_frg_gated[j], tOrO_frg_gated[j + 1]),
                        (sigmoid_gate_scale, sigmoid_gate_scale),
                    )
                tOrO_frg_cvt.store(tOrO_frg_gated.load().to(self.o_dtype))
            cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
        # fence view async shared
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)

            # load acc O from smem to rmem for wider vectorization
            tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)
            # copy acc O from rmem to gmem
            if const_expr(not self.pack_gqa):
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOpO = utils.predicate_k(tOcO, limit=mO_cur.shape[1])
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    predicate_m = (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen_q
                        - (self.q_stage * m_block + stage) * self.m_block_size
                        - tOcO[0][0]
                    )
                    predicate = cute.make_fragment_like(tOpO[None, 0, None])
                    for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                        for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                            predicate[i, k] = (
                                tOpO[i, rest_m, k]
                                if const_expr(self.check_hdim_v_oob)
                                else True
                            ) and predicate_m
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                        pred=predicate,
                    )
            else:
                pack_gqa = PackGQA(
                    self.m_block_size,
                    self.head_dim_v_padded,
                    self.check_hdim_v_oob,
                    self.qhead_per_kvhead,
                )
                pack_gqa.store_O(
                    mO_cur,
                    tOrO,
                    gmem_tiled_copy_O,
                    tidx,
                    self.q_stage * m_block + stage,
                    seqlen_q,
                )

    @cute.jit
    def sync_dynamic_work_tile(self) -> None:
        if const_expr(self.dynamic_splitkv_scheduler and self.is_persistent):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WorkTile),
                number_of_threads=self.work_tile_threads,
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        mOFinal: Optional[cute.Tensor],
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        mNumSplitsActual: Optional[cute.Tensor],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            output_seqlen_q = self.output_seqlen_q(
                seqlen.seqlen_q, m_block, aux_tensors
            )
            actual_num_splits = work_tile.actual_num_splits
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, actual_num_splits
            )

            process_tile = n_block_min < n_block_max
            if process_tile:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx, split_idx
                    ]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    mO_cur = cute.domain_offset(
                        ((0, self.output_row_offset(m_block, aux_tensors)), 0), mO_cur
                    )
                if const_expr(
                    self.is_split_kv
                    and self.dynamic_splitkv_scheduler
                    and mOFinal is not None
                ):
                    mO_final_cur = seqlen.offset_batch_Q(mOFinal, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                if const_expr(self.use_tma_O):
                    gO = cute.local_tile(
                        mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0)
                    )
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range_constexpr(
                        2 if const_expr(self.q_stage == 2) else 1
                    ):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage,
                            epi_consumer_phase,
                        )
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=self.q_stage * m_block + stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(
                        2 if const_expr(self.q_stage == 2) else 1
                    ):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage
                        )
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor(
                        (self.m_block_size, self.head_dim_v_padded)
                    )
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    for stage in cutlass.range_constexpr(
                        2 if const_expr(self.q_stage == 2) else 1
                    ):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage,
                            epi_consumer_phase,
                        )
                        # 2. copy O0 / O1 to gmem
                        # load acc O from smem to rmem for wider vectorization
                        tOrO = cute.make_fragment_like(
                            tOsO[None, None, None, 0], self.o_dtype
                        )
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                        # copy acc O from rmem to gmem
                        if const_expr(not self.pack_gqa):
                            if const_expr(
                                self.is_split_kv
                                and self.dynamic_splitkv_scheduler
                                and mOFinal is not None
                            ):
                                if actual_num_splits == Int32(1):
                                    gO = cute.local_tile(
                                        mO_final_cur,
                                        (self.m_block_size, self.head_dim_v_padded),
                                        (None, 0),
                                    )
                                else:
                                    gO = cute.local_tile(
                                        mO_cur,
                                        (self.m_block_size, self.head_dim_v_padded),
                                        (None, 0),
                                    )
                            else:
                                gO = cute.local_tile(
                                    mO_cur,
                                    (self.m_block_size, self.head_dim_v_padded),
                                    (None, 0),
                                )
                            tOgO = gmem_thr_copy_O.partition_D(gO)
                            tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                            for rest_m in cutlass.range_constexpr(
                                cute.size(tOrO.shape[1])
                            ):
                                predicate_m = (
                                    t0OcO[0, rest_m, 0][0]
                                    < output_seqlen_q
                                    - (self.q_stage * m_block + stage)
                                    * self.m_block_size
                                    - tOcO[0][0]
                                )
                                predicate = cute.make_fragment_like(tOpO[None, 0, None])
                                for k in cutlass.range_constexpr(
                                    cute.size(predicate.shape[1])
                                ):
                                    for i in cutlass.range_constexpr(
                                        cute.size(predicate.shape[0])
                                    ):
                                        predicate[i, k] = (
                                            tOpO[i, rest_m, k]
                                            if const_expr(self.check_hdim_v_oob)
                                            else True
                                        ) and predicate_m
                                cute.copy(
                                    gmem_tiled_copy_O,
                                    tOrO[None, rest_m, None],
                                    tOgO[
                                        None,
                                        rest_m,
                                        None,
                                        self.q_stage * m_block + stage,
                                    ],
                                    pred=predicate,
                                )
                        else:
                            pack_gqa = PackGQA(
                                self.m_block_size,
                                self.head_dim_v_padded,
                                self.check_hdim_v_oob,
                                self.qhead_per_kvhead,
                            )
                            if const_expr(
                                self.is_split_kv
                                and self.dynamic_splitkv_scheduler
                                and mOFinal is not None
                            ):
                                if actual_num_splits == Int32(1):
                                    pack_gqa.store_O(
                                        mO_final_cur,
                                        tOrO,
                                        gmem_tiled_copy_O,
                                        tidx,
                                        self.q_stage * m_block + stage,
                                        output_seqlen_q,
                                    )
                                else:
                                    pack_gqa.store_O(
                                        mO_cur,
                                        tOrO,
                                        gmem_tiled_copy_O,
                                        tidx,
                                        self.q_stage * m_block + stage,
                                        output_seqlen_q,
                                    )
                            else:
                                pack_gqa.store_O(
                                    mO_cur,
                                    tOrO,
                                    gmem_tiled_copy_O,
                                    tidx,
                                    self.q_stage * m_block + stage,
                                    output_seqlen_q,
                                )
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage
                        )

                epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_full_ptr + stage, self.tma_copy_bytes["Q"]
            )
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage,
                    self.tma_copy_bytes[K_or_V],
                )
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            tXgX_cur = (
                tXgX[None, block]
                if const_expr(page_idx is None)
                else tXgX[None, 0, page_idx]
            )
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)
        else:
            assert paged_kv_manager is not None
            sX_cur = sX[None, None, None, stage]
            if const_expr(self.uneven_kv_smem):
                sX_cur = self.offset_kv_smem(sX_cur, stage, phase ^ 1)
            paged_kv_manager.load_KV(block, sX_cur, K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            offset = cute.assume(offset, divby=128 // self.k_dtype.width)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        if self.use_tma_KV:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids)
            )
            return cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=load_kv_mbar_ptr,
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
            )
        else:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                len(self.load_warp_ids) * cute.arch.WARP_SIZE,
            )
            return cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                barrier_storage=load_kv_mbar_ptr,
            )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset(
            (m_block * self.m_block_size, n_block * self.n_block_size), cS
        )
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical = q_physical // self.qhead_per_kvhead
            head_offset = q_physical - q_idx_logical * self.qhead_per_kvhead
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = seqlen_q_divmod.divmod(q_idx_logical)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead
            if cutlass.const_expr(self.pack_gqa)
            else 1,
        )
