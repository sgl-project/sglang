# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 GQA grouped-union sparse paged-KV prefill."""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.utils import LayoutEnum

from sglang.kernels.ops.attention.step5.cute.compile_cache import (
    cached_compile_function,
)
from sglang.kernels.ops.attention.step5.cute.sm90.kernel_utils import (
    cute_utils as utils,
)
from sglang.kernels.ops.attention.step5.cute.sm90.kernel_utils import (
    hopper_helpers as hop_helpers,
)
from sglang.kernels.ops.attention.step5.cute.sm90.kernel_utils.softmax import Softmax
from sglang.kernels.ops.attention.step5.cute.sm90.utils import elem_pointer


class _TokenWiseSparsePagedPrefillUnionGQA:
    """Grouped-query prefill consuming prebuilt union region metadata.

    The default sparse semantics are exact DSA-style: every query in the CTA
    attends only to its own selected region set, while K/V loading is still shared
    across the group's union set.
    """

    arch: int = 90
    SUPPORTED_LOGICAL_Q_HEADS = (4, 8, 16)
    SUPPORTED_HEAD_DIMS = (128, 192)
    PAGE_SIZE = 16
    KV_HEADS = 1
    NUM_STAGES = 2
    PRODUCER_REGS = 24
    MATH_REGS = 240
    PRODUCER_SUBGROUP_LANES = 8
    Q_LOAD_COPY_ELEMS = 8
    PRODUCER_ASYNC_COPY_ELEMS = 8
    REQUIRED_POINTER_ALIGN = 16
    SUPPORTED_Q_GROUPS = (8, 16, 32)
    WGMMA_ROWS = 64
    SUPPORTED_TOPOLOGIES = {
        (32, 64, 512),
        (8, 64, 512),
        (16, 64, 512),
    }

    @staticmethod
    def _expected_q_group(logical_q_heads: int) -> int:
        if int(logical_q_heads) == 4:
            return 32
        if int(logical_q_heads) == 8:
            return 16
        if int(logical_q_heads) == 16:
            return 8
        raise ValueError(
            "GQA grouped-union prefill supports only logical_q_heads in "
            f"{_TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_LOGICAL_Q_HEADS}, "
            f"got {int(logical_q_heads)}"
        )

    @staticmethod
    def _validate_q_group_contract(*, logical_q_heads: int, q_group: int) -> None:
        expected_q_group = _TokenWiseSparsePagedPrefillUnionGQA._expected_q_group(
            logical_q_heads
        )
        if int(q_group) != expected_q_group:
            raise ValueError(
                "GQA grouped-union prefill requires "
                f"q_group={expected_q_group} when logical_q_heads={int(logical_q_heads)}, "
                f"got q_group={int(q_group)}"
            )

    def __init__(
        self,
        *,
        dtype,
        logical_q_heads: int,
        head_dim: int,
        head_dim_v: int,
        block_n: int,
        num_threads: int,
        q_group: int,
        max_union_windows: int,
        per_query_positions: bool = False,
    ) -> None:
        self.per_query_positions = per_query_positions
        self.dtype = dtype
        self.logical_q_heads = int(logical_q_heads)
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.GROUP_SIZE = (
            4 if self.logical_q_heads <= 4 else 8 if self.logical_q_heads <= 8 else 16
        )
        self.GROUP_SIZE_PAD = self.GROUP_SIZE
        self.GROUP_SIZE_PAD_MMA = self.logical_q_heads
        self.slots_per_warpgroup = self.WGMMA_ROWS // self.GROUP_SIZE_PAD_MMA
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.q_group = int(q_group)
        self.max_union_windows = int(max_union_windows)
        self._validate_q_group_contract(
            logical_q_heads=self.logical_q_heads,
            q_group=self.q_group,
        )
        # b64/t512 spends the extra warps on producers. A cap of 232 keeps
        # compilation time bounded without slowing the kernel.
        self.math_regs = (
            232 if self.block_n == 64 and self.num_threads == 512 else self.MATH_REGS
        )
        # b64/t512 has enough shared-memory headroom for extra
        # K/V stages; this targets the remaining producer/consumer barrier waits.
        self.num_stages = (
            4
            if self.HEAD_DIM == 128 and self.block_n == 64 and self.num_threads == 512
            else int(self.NUM_STAGES)
        )
        self.num_warps = self.num_threads // 32
        self.num_math_warpgroups = (
            self.q_group + self.slots_per_warpgroup - 1
        ) // self.slots_per_warpgroup
        self.num_compute_warps = self.num_math_warpgroups * 4
        self.num_producer_warps = self.num_warps - self.num_compute_warps
        self.producer_threads = self.num_producer_warps * 32
        if self.num_producer_warps <= 0:
            raise ValueError("GQA grouped-union prefill requires producer warps")
        self.producer_subgroup_lanes = int(self.PRODUCER_SUBGROUP_LANES)
        self.producer_groups = self.producer_threads // self.producer_subgroup_lanes
        self.max_rows_per_group = (
            self.block_n + self.producer_groups - 1
        ) // self.producer_groups
        self.k_stage_arrive_threads = self.producer_threads
        self.v_stage_arrive_threads = self.producer_threads

        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("Only fp16/bf16 are supported")
        if self.logical_q_heads not in self.SUPPORTED_LOGICAL_Q_HEADS:
            raise ValueError(
                "GQA grouped-union prefill supports only logical_q_heads in "
                f"{self.SUPPORTED_LOGICAL_Q_HEADS}, got {self.logical_q_heads}"
            )
        if (
            self.HEAD_DIM not in self.SUPPORTED_HEAD_DIMS
            or self.HEAD_DIM_V not in self.SUPPORTED_HEAD_DIMS
        ):
            raise ValueError(
                "GQA grouped-union prefill supports only head_dim/head_dim_v in "
                f"{self.SUPPORTED_HEAD_DIMS}, got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "GQA grouped-union prefill keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if self.q_group not in self.SUPPORTED_Q_GROUPS:
            raise ValueError(
                f"q_group must be one of {self.SUPPORTED_Q_GROUPS}, got {self.q_group}"
            )
        if self.WGMMA_ROWS % self.GROUP_SIZE_PAD_MMA != 0:
            raise ValueError("logical_q_heads must tile the WGMMA M dimension")
        if self.num_producer_warps % 4 != 0:
            raise ValueError(
                "producer warps must keep compute warpgroups 4-warp aligned"
            )
        if (
            self.q_group,
            self.block_n,
            self.num_threads,
        ) not in self.SUPPORTED_TOPOLOGIES:
            raise ValueError(
                "GQA grouped-union prefill supports only "
                f"{sorted(self.SUPPORTED_TOPOLOGIES)}"
            )
        if self.block_n != 64:
            raise ValueError(
                f"GQA grouped-union prefill expects block_n=64, got {self.block_n}"
            )
        if self.max_union_windows <= 0:
            raise ValueError("GQA grouped-union prefill requires max_union_windows > 0")

    @staticmethod
    def _convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def _store_acc_into_pv_operand(self, acc, operand, Element) -> None:
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(Element))

    @cute.jit
    def _make_acc_into_pv_operand(self, acc, operand_layout_tv, Element):
        # WGMMA RS consumes P in operand-A TV layout,
        # while QK produces P in accumulator-C layout; FlashAttention does this
        # remap in registers to avoid an extra shared-memory P tile.
        operand = cute.make_rmem_tensor_like(
            self._convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        self._store_acc_into_pv_operand(acc, operand, Element)
        return operand

    @cute.jit
    def _apply_causal_token_mask(
        self,
        acc_S,
        tScS_mn,
        sTok,
        stage: Int32,
        q_base: Int32,
        positions: cute.Tensor,
        q_local_base: Int32,
        q_tile_count: Int32,
        wg_slot_base: Int32,
        logical_q_heads_i32: Int32,
    ) -> None:
        acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
        nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
        ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
        tile_tail_tok = sTok[Int32(self.block_n - 1), stage]
        # Union metadata is emitted in ascending logical
        # region order. Complete tiles before the warpgroup's first query position
        # cannot be touched by causal masking.
        tile_before_wg_min_q = (
            (q_tile_count >= (wg_slot_base + Int32(self.slots_per_warpgroup)))
            and (tile_tail_tok >= Int32(0))
            and (tile_tail_tok <= (q_base + wg_slot_base))
        )
        if const_expr(self.per_query_positions):
            tile_before_wg_min_q = False
        if not tile_before_wg_min_q:
            for r in cutlass.range_constexpr(nrow_s):
                row = tScS_mn[r, 0][0]
                slot_rel = row // Int32(self.GROUP_SIZE_PAD_MMA)
                h = row - slot_rel * Int32(self.GROUP_SIZE_PAD_MMA)
                q_slot = wg_slot_base + slot_rel
                q_pos = q_base + q_slot
                if const_expr(self.per_query_positions):
                    q_pos = Int32(0)
                    if q_slot < q_tile_count:
                        q_pos = positions[q_local_base + q_slot]
                row_invalid = (q_slot >= q_tile_count) or (h >= logical_q_heads_i32)
                for c in cutlass.range_constexpr(ncol_s):
                    col = tScS_mn[r, c][1]
                    tok_col = sTok[col, stage]
                    if row_invalid or tok_col < Int32(0) or tok_col > q_pos:
                        acc_S_mn[r, c] = -Float32.inf

    @cute.jit
    def _apply_exact_union_mask(
        self,
        acc_S,
        tScS_mn,
        sTok,
        sExactQMask,
        stage: Int32,
        q_base: Int32,
        positions: cute.Tensor,
        q_local_base: Int32,
        q_tile_count: Int32,
        wg_slot_base: Int32,
        logical_q_heads_i32: Int32,
    ) -> None:
        acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
        nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
        ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
        for c in cutlass.range_constexpr(ncol_s):
            col = tScS_mn[0, c][1]
            query_mask = sExactQMask[col, stage]
            tok_col = sTok[col, stage]
            for r in cutlass.range_constexpr(nrow_s):
                row = tScS_mn[r, c][0]
                slot_rel = row // Int32(self.GROUP_SIZE_PAD_MMA)
                h = row - slot_rel * Int32(self.GROUP_SIZE_PAD_MMA)
                q_slot = wg_slot_base + slot_rel
                q_slot_safe = q_slot if q_slot < Int32(self.q_group) else Int32(0)
                q_pos = q_base + q_slot
                if const_expr(self.per_query_positions):
                    q_pos = Int32(0)
                    if q_slot < q_tile_count:
                        q_pos = positions[q_local_base + q_slot]
                row_invalid = (q_slot >= q_tile_count) or (h >= logical_q_heads_i32)
                selected = (query_mask & (Int32(1) << q_slot_safe)) != Int32(0)
                if (
                    row_invalid
                    or (not selected)
                    or tok_col < Int32(0)
                    or tok_col > q_pos
                ):
                    acc_S_mn[r, c] = -Float32.inf

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.WGMMA_ROWS, self.block_n),
        )
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.WGMMA_ROWS, self.HEAD_DIM_V),
            a_source=sm90_utils.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_smem_layouts(self):
        qk_tiler_mnk = (self.WGMMA_ROWS, self.block_n, self.HEAD_DIM)
        pv_tiler_mnk = (self.WGMMA_ROWS, self.HEAD_DIM_V, self.block_n)
        sQ_layout = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR,
            qk_tiler_mnk,
            self.dtype,
            self.num_math_warpgroups,
        )
        sK_layout = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR,
            qk_tiler_mnk,
            self.dtype,
            self.num_stages,
        )
        sV_layout = sm90_utils.make_smem_layout_b(
            LayoutEnum.COL_MAJOR,
            pv_tiler_mnk,
            self.dtype,
            self.num_stages,
        )
        return sQ_layout, sK_layout, sV_layout

    def _get_shared_storage_cls(self):
        sQ_layout, sK_layout, sV_layout = self._get_smem_layouts()
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)],
            1024,
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)],
            1024,
        ]
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)],
            1024,
        ]
        mbar_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages]
        sTok_struct = cute.struct.MemRange[Int32, self.block_n * self.num_stages]
        sExactQMask_struct = cute.struct.MemRange[
            Int32,
            self.block_n * self.num_stages,
        ]

        @cute.struct
        class SharedStorage:
            mbar_ptr_free: mbar_struct
            mbar_ptr_k_stage: mbar_struct
            mbar_ptr_v_stage: mbar_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sTok: sTok_struct
            sExactQMask: sExactQMask_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: Optional[cute.Tensor],
        mUnionCount: cute.Tensor,
        mExactMaskBits: Optional[cute.Tensor],
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mCausalLimits: Optional[cute.Tensor],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cutlass_torch.cuda.CUstream,
    ) -> None:
        softmax_scale_log2 = Float32(softmax_scale * math.log2(math.e))

        def _assume_stride_divisible(stride_elem, *, element_width: int):
            if const_expr(isinstance(stride_elem, int)):
                return stride_elem
            return cute.assume(stride_elem, divby=128 // element_width)

        new_stride = lambda t: (
            *(
                _assume_stride_divisible(s, element_width=t.element_type.width)
                for s in t.stride[:-1]
            ),
            t.stride[-1],
        )
        mQ, mKCache, mVCache, mO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            for t in (mQ, mKCache, mVCache, mO)
        ]
        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(
                mLSE.iterator,
                cute.make_layout(mLSE.shape, stride=new_stride(mLSE)),
            )

        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, [0, 2, 1]))
        mKCache = cute.make_tensor(
            mKCache.iterator, cute.select(mKCache.layout, [1, 3, 2, 0])
        )
        mVtCache = cute.make_tensor(
            mVCache.iterator, cute.select(mVCache.layout, [3, 1, 2, 0])
        )
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, [0, 2, 1]))
        mKCache = cute.make_tensor(
            mKCache.iterator,
            cute.make_layout(
                (
                    mKCache.shape[0] * mKCache.shape[3],
                    mKCache.shape[1],
                    mKCache.shape[2],
                ),
                stride=(mKCache.stride[0], mKCache.stride[1], mKCache.stride[2]),
            ),
        )
        mVtCache = cute.make_tensor(
            mVtCache.iterator,
            cute.make_layout(
                (
                    mVtCache.shape[0],
                    mVtCache.shape[1] * mVtCache.shape[3],
                    mVtCache.shape[2],
                ),
                stride=(mVtCache.stride[0], mVtCache.stride[1], mVtCache.stride[2]),
            ),
        )
        mUnionPhys = cute.make_tensor(mUnionPhys.iterator, mUnionPhys.layout)
        if const_expr(mUnionLogical is not None):
            mUnionLogical = cute.make_tensor(
                mUnionLogical.iterator, mUnionLogical.layout
            )
        mUnionCount = cute.make_tensor(mUnionCount.iterator, mUnionCount.layout)
        if const_expr(mExactMaskBits is not None):
            mExactMaskBits = cute.make_tensor(
                mExactMaskBits.iterator, mExactMaskBits.layout
            )
        mWorkQGlobal = cute.make_tensor(mWorkQGlobal.iterator, mWorkQGlobal.layout)
        mWorkQLocal = cute.make_tensor(mWorkQLocal.iterator, mWorkQLocal.layout)
        mWorkQLen = cute.make_tensor(mWorkQLen.iterator, mWorkQLen.layout)
        if const_expr(mCausalLimits is not None):
            mCausalLimits = cute.make_tensor(
                mCausalLimits.iterator, mCausalLimits.layout
            )
        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, [0, 1]))

        shape_Q_packed = (
            (self.GROUP_SIZE_PAD, mQ.shape[0]),
            mQ.shape[1],
            mKCache.shape[2],
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),
            mQ.stride[1],
            mQ.stride[2] * self.GROUP_SIZE_PAD,
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
        )

        shape_O_packed = (
            (self.GROUP_SIZE_PAD, mO.shape[0]),
            mO.shape[1],
            mKCache.shape[2],
        )
        stride_O_packed = (
            (mO.stride[2], mO.stride[0]),
            mO.stride[1],
            mO.stride[2] * self.GROUP_SIZE_PAD,
        )
        mO = cute.make_tensor(
            mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
        )

        work_tiles = Int32(cute.size(mWorkQLocal.shape[0]))
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        sQ_layout, sK_layout, sV_layout = self._get_smem_layouts()
        SharedStorage = self._get_shared_storage_cls()

        self.kernel(
            mQ,
            mKCache,
            mVtCache,
            mUnionPhys,
            mUnionLogical,
            mUnionCount,
            mExactMaskBits,
            mWorkQGlobal,
            mWorkQLocal,
            mWorkQLen,
            mCausalLimits,
            mO,
            mLSE,
            softmax_scale_log2,
            tiled_mma_qk,
            tiled_mma_pv,
            sQ_layout,
            sK_layout,
            sV_layout,
            SharedStorage,
        ).launch(
            grid=(work_tiles, 1, 1),
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVtCache: cute.Tensor,
        mUnionPhys: cute.Tensor,
        mUnionLogical: Optional[cute.Tensor],
        mUnionCount: cute.Tensor,
        mExactMaskBits: Optional[cute.Tensor],
        mWorkQGlobal: cute.Tensor,
        mWorkQLocal: cute.Tensor,
        mWorkQLen: cute.Tensor,
        mCausalLimits: Optional[cute.Tensor],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout,
        sK_layout,
        sV_layout,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        work_q_block, _, _ = cute.arch.block_idx()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        is_producer = warp_idx < Int32(self.num_producer_warps)
        is_compute = warp_idx >= Int32(self.num_producer_warps)
        math_thread = tidx - Int32(self.producer_threads)
        math_wg = cute.arch.make_warp_uniform(math_thread // Int32(128))
        idx_in_wg = math_thread - math_wg * Int32(128)
        wg_slot_base = math_wg * Int32(self.slots_per_warpgroup)

        region_tokens_i32 = Int32(8)
        block_n_i32 = Int32(self.block_n)
        stage_count = Int32(self.num_stages)
        q_local_base = mWorkQLocal[work_q_block]
        q_req_len = mWorkQLen[work_q_block]
        q_tile_count = q_req_len - q_local_base
        if q_tile_count < Int32(0):
            q_tile_count = Int32(0)
        if q_tile_count > Int32(self.q_group):
            q_tile_count = Int32(self.q_group)
        q_base = Int32(0)
        if const_expr(not self.per_query_positions):
            q_base = mWorkQGlobal[work_q_block]
        union_windows = mUnionCount[work_q_block]
        if union_windows > Int32(self.max_union_windows):
            union_windows = Int32(self.max_union_windows)
        count_tokens = union_windows * region_tokens_i32
        tile_split_i32 = (count_tokens + block_n_i32 - Int32(1)) // block_n_i32

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_free = storage.mbar_ptr_free.data_ptr()
        mbar_k_stage = storage.mbar_ptr_k_stage.data_ptr()
        mbar_v_stage = storage.mbar_ptr_v_stage.data_ptr()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sTok = storage.sTok.get_tensor(
            cute.make_layout(
                (self.block_n, self.num_stages), stride=(self.num_stages, 1)
            )
        )
        sExactQMask = storage.sExactQMask.get_tensor(
            cute.make_layout(
                (self.block_n, self.num_stages), stride=(self.num_stages, 1)
            )
        )

        if tidx < Int32(self.num_stages):
            cute.arch.mbarrier_init(mbar_free + tidx, Int32(self.num_math_warpgroups))
            cute.arch.mbarrier_init(
                mbar_k_stage + tidx, Int32(self.k_stage_arrive_threads)
            )
            cute.arch.mbarrier_init(
                mbar_v_stage + tidx, Int32(self.v_stage_arrive_threads)
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        vec_k = Int32(self.PRODUCER_ASYNC_COPY_ELEMS)
        producer_col_slab = Int32(64)
        producer_head_slabs = self.HEAD_DIM // 64
        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        elem_bytes_i64 = Int64(self.dtype.width // 8)
        producer_copy_align = self.REQUIRED_POINTER_ALIGN
        mKCache_base_i64 = mKCache.iterator.toint()
        mVtCache_base_i64 = mVtCache.iterator.toint()
        mK_tok_stride_bytes = Int64(mKCache.stride[0]) * elem_bytes_i64
        mK_col_stride_bytes = Int64(mKCache.stride[1]) * elem_bytes_i64
        mVt_tok_stride_bytes = Int64(mVtCache.stride[1]) * elem_bytes_i64
        mVt_col_stride_bytes = Int64(mVtCache.stride[0]) * elem_bytes_i64
        producer_subgroup_lanes = Int32(self.producer_subgroup_lanes)
        producer_threads = Int32(self.producer_threads)
        producer_groups = Int32(self.producer_groups)

        if is_producer:
            cute.arch.warpgroup_reg_dealloc(self.PRODUCER_REGS)
            group_idx = tidx // producer_subgroup_lanes
            lane_in_group = tidx - group_idx * producer_subgroup_lanes
            group_leader_lane = lane - lane_in_group
            col_base = lane_in_group * vec_k
            k_g_lane_col_bytes = Int64(col_base) * mK_col_stride_bytes
            v_g_lane_col_bytes = Int64(col_base) * mVt_col_stride_bytes
            row_pre_buf = cute.make_rmem_tensor(self.max_rows_per_group, Int32)
            phys_tok_buf = cute.make_rmem_tensor(self.max_rows_per_group, Int32)

            for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                row_pre_buf[local_row] = group_idx + Int32(local_row) * producer_groups

            for local_idx in cutlass.range(tile_split_i32, unroll=1):
                stage = local_idx % stage_count
                if local_idx >= stage_count and tidx == Int32(0):
                    wait_phase = (
                        (local_idx - stage_count - stage) // stage_count
                    ) & Int32(1)
                    cute.arch.mbarrier_wait(mbar_free + stage, wait_phase)
                if local_idx >= stage_count:
                    cute.arch.barrier(barrier_id=4, number_of_threads=producer_threads)

                for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                    row_pre = row_pre_buf[local_row]
                    token_idx = local_idx * block_n_i32 + row_pre
                    phys_tok_leader = Int32(-1)
                    if lane_in_group == Int32(0) and row_pre < block_n_i32:
                        logical_tok_leader = Int32(-1)
                        exact_qmask_leader = Int32(0)
                        if token_idx < count_tokens:
                            window_idx = token_idx // region_tokens_i32
                            region_lane = token_idx - window_idx * region_tokens_i32
                            phys_region = mUnionPhys[work_q_block, window_idx]
                            valid_phys = (phys_region >= Int32(0)) and (
                                phys_region
                                < Int32(cute.size(mKCache.shape[0])) // Int32(8)
                            )
                            if valid_phys:
                                phys_tok_leader = (
                                    phys_region * region_tokens_i32 + region_lane
                                )
                            if const_expr(mExactMaskBits is not None):
                                exact_qmask_leader = mExactMaskBits[
                                    work_q_block, window_idx
                                ]
                            if const_expr(mUnionLogical is not None):
                                logical_region = mUnionLogical[work_q_block, window_idx]
                                # Invalid physical slots load token zero as a placeholder.
                                if logical_region >= Int32(0) and valid_phys:
                                    logical_tok_leader = (
                                        logical_region * region_tokens_i32 + region_lane
                                    )
                            else:
                                logical_tok_leader = phys_tok_leader
                        if const_expr(
                            (mExactMaskBits is not None) or (mCausalLimits is None)
                        ):
                            sTok[row_pre, stage] = logical_tok_leader
                        if const_expr(mExactMaskBits is not None):
                            sExactQMask[row_pre, stage] = exact_qmask_leader
                    phys_tok_cur = cute.arch.shuffle_sync(
                        phys_tok_leader, group_leader_lane
                    )
                    phys_tok_buf[local_row] = phys_tok_cur
                    row_valid = row_pre < block_n_i32
                    k_valid = row_valid and (phys_tok_cur >= Int32(0))
                    if row_valid:
                        for head_slab in cutlass.range_constexpr(producer_head_slabs):
                            slab_col_i32 = Int32(head_slab) * producer_col_slab
                            slab_col_i64 = Int64(slab_col_i32)
                            s_ptr = elem_pointer(
                                sK, (row_pre, slab_col_i32 + col_base, stage)
                            )
                            s_ptr = cute.make_ptr(
                                self.dtype,
                                s_ptr.toint(),
                                sK.memspace,
                                assumed_align=producer_copy_align,
                            )
                            s_ptr = cute.recast_ptr(s_ptr, sK_layout.inner)
                            s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                            phys_tok_safe = phys_tok_cur if k_valid else Int32(0)
                            g_ptr = cute.make_ptr(
                                self.dtype,
                                mKCache_base_i64
                                + Int64(phys_tok_safe) * mK_tok_stride_bytes
                                + slab_col_i64 * mK_col_stride_bytes
                                + k_g_lane_col_bytes,
                                mKCache.memspace,
                                assumed_align=producer_copy_align,
                            )
                            g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                            utils.vector_copy_with_explicit_width(
                                g_vec,
                                s_vec,
                                num_copy_elems=self.PRODUCER_ASYNC_COPY_ELEMS,
                                is_async=True,
                            )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar_k_stage + stage)

                for local_row in cutlass.range_constexpr(self.max_rows_per_group):
                    row_pre = row_pre_buf[local_row]
                    phys_tok_cur = phys_tok_buf[local_row]
                    row_valid = row_pre < block_n_i32
                    v_valid = row_valid and (phys_tok_cur >= Int32(0))
                    if row_valid:
                        for head_slab in cutlass.range_constexpr(producer_head_slabs):
                            slab_col_i32 = Int32(head_slab) * producer_col_slab
                            slab_col_i64 = Int64(slab_col_i32)
                            s_ptr = elem_pointer(
                                sV, (slab_col_i32 + col_base, row_pre, stage)
                            )
                            s_ptr = cute.make_ptr(
                                self.dtype,
                                s_ptr.toint(),
                                sV.memspace,
                                assumed_align=producer_copy_align,
                            )
                            s_ptr = cute.recast_ptr(s_ptr, sV_layout.inner)
                            s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                            phys_tok_safe = phys_tok_cur if v_valid else Int32(0)
                            g_ptr = cute.make_ptr(
                                self.dtype,
                                mVtCache_base_i64
                                + Int64(phys_tok_safe) * mVt_tok_stride_bytes
                                + slab_col_i64 * mVt_col_stride_bytes
                                + v_g_lane_col_bytes,
                                mVtCache.memspace,
                                assumed_align=producer_copy_align,
                            )
                            g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                            utils.vector_copy_with_explicit_width(
                                g_vec,
                                s_vec,
                                num_copy_elems=self.PRODUCER_ASYNC_COPY_ELEMS,
                                is_async=True,
                            )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar_v_stage + stage)

        if is_compute:
            cute.arch.warpgroup_reg_alloc(self.math_regs)
            logical_q_heads_i32 = Int32(self.logical_q_heads)
            q_vec_elems = Int32(8)
            q_vec_count = (self.WGMMA_ROWS * self.HEAD_DIM) // 8
            mQ_cur = mQ[None, None, Int32(0)]
            mO_cur = mO[None, None, Int32(0)]
            sQ_wg = sQ[None, None, math_wg]

            for vec_idx in cutlass.range(idx_in_wg, q_vec_count, 128, unroll=1):
                elem_idx = vec_idx * q_vec_elems
                row = elem_idx // self.HEAD_DIM
                d = elem_idx - row * self.HEAD_DIM
                slot_rel = row // Int32(self.GROUP_SIZE_PAD_MMA)
                h = row - slot_rel * Int32(self.GROUP_SIZE_PAD_MMA)
                q_slot = wg_slot_base + slot_rel
                q_valid = (q_slot < q_tile_count) and (h < logical_q_heads_i32)
                q_local = q_local_base + q_slot
                q_local_safe = q_local if q_valid else Int32(0)
                gQ_slot = cute.local_tile(
                    mQ_cur,
                    (self.GROUP_SIZE_PAD, self.HEAD_DIM),
                    (q_local_safe, 0),
                )
                s_ptr = elem_pointer(sQ_wg, (row, d))
                s_ptr = cute.make_ptr(
                    self.dtype,
                    s_ptr.toint(),
                    sQ_wg.memspace,
                    assumed_align=self.REQUIRED_POINTER_ALIGN,
                )
                s_ptr = cute.recast_ptr(s_ptr, sQ_layout.inner)
                s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                g_ptr = utils.elem_pointer_i64_offset(gQ_slot, (h, d))
                g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                utils.vector_copy_with_explicit_width(
                    g_vec, s_vec, num_copy_elems=self.Q_LOAD_COPY_ELEMS
                )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(
                barrier_id=6, number_of_threads=Int32(self.num_compute_warps * 32)
            )

            thr_mma_qk = tiled_mma_qk.get_slice(math_thread)
            thr_mma_pv = tiled_mma_pv.get_slice(math_thread)

            cS = cute.make_identity_tensor((self.WGMMA_ROWS, self.block_n))
            tScS_mn = utils.make_acc_tensor_mn_view_from_mma(thr_mma_qk.partition_C(cS))
            acc_shape_S = thr_mma_qk.partition_shape_C((self.WGMMA_ROWS, self.block_n))

            cO = cute.make_identity_tensor((self.WGMMA_ROWS, self.HEAD_DIM_V))
            tOcO_mn = utils.make_acc_tensor_mn_view_from_mma(thr_mma_pv.partition_C(cO))
            acc_shape_O = thr_mma_pv.partition_shape_C(
                (self.WGMMA_ROWS, self.HEAD_DIM_V)
            )
            acc_O = thr_mma_pv.make_fragment_C(acc_shape_O)
            acc_O.fill(Float32.zero)
            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            )
            softmax.reset()
            nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
            ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
            nrow_o = const_expr(cute.size(tOcO_mn.shape[0]))
            ncol_o = const_expr(cute.size(tOcO_mn.shape[1]))
            acc_O_mn = utils.make_acc_tensor_mn_view_from_mma(acc_O)
            tCrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ_wg))
            if const_expr(mCausalLimits is not None):
                wg_first_limit = Int32(0)
                if wg_slot_base < Int32(self.q_group):
                    wg_first_limit = mCausalLimits[work_q_block, wg_slot_base]
                causal_row_limit = cute.make_rmem_tensor(nrow_s, Int32)
                causal_row_invalid = cute.make_rmem_tensor(nrow_s, Int32)
                for r in cutlass.range_constexpr(nrow_s):
                    row = tScS_mn[r, 0][0]
                    slot_rel = row // Int32(self.GROUP_SIZE_PAD_MMA)
                    h = row - slot_rel * Int32(self.GROUP_SIZE_PAD_MMA)
                    q_slot = wg_slot_base + slot_rel
                    q_slot_safe = q_slot if q_slot < Int32(self.q_group) else Int32(0)
                    causal_row_limit[r] = mCausalLimits[work_q_block, q_slot_safe]
                    causal_row_invalid[r] = (
                        Int32(1)
                        if (q_slot >= q_tile_count) or (h >= logical_q_heads_i32)
                        else Int32(0)
                    )
            if tile_split_i32 > Int32(0):
                stage = Int32(0)
                wait_phase = Int32(0)
                cute.arch.mbarrier_wait(mbar_k_stage + stage, wait_phase)
                cute.arch.fence_proxy("async.shared", space="cta")

                sK_stage = sK[None, None, stage]
                tCrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK_stage))
                acc_S = thr_mma_qk.make_fragment_C(acc_shape_S)
                hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                    tiled_mma_qk,
                    acc_S,
                    tCrQ,
                    tCrK,
                    zero_init=True,
                    wg_wait=0,
                )
                if const_expr(mExactMaskBits is not None):
                    self._apply_exact_union_mask(
                        acc_S,
                        tScS_mn,
                        sTok,
                        sExactQMask,
                        stage,
                        q_base,
                        mWorkQGlobal,
                        q_local_base,
                        q_tile_count,
                        wg_slot_base,
                        logical_q_heads_i32,
                    )
                elif const_expr(mCausalLimits is not None):
                    acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
                    tile_col_base = stage * block_n_i32
                    tile_col_end = tile_col_base + block_n_i32
                    tile_before_wg_min_q = (
                        q_tile_count >= (wg_slot_base + Int32(self.slots_per_warpgroup))
                    ) and (wg_first_limit >= tile_col_end)
                    if not tile_before_wg_min_q:
                        for r in cutlass.range_constexpr(nrow_s):
                            row_invalid = causal_row_invalid[r] != Int32(0)
                            col_limit = causal_row_limit[r]
                            for c in cutlass.range_constexpr(ncol_s):
                                col = tScS_mn[r, c][1]
                                global_col = tile_col_base + col
                                if row_invalid or global_col >= col_limit:
                                    acc_S_mn[r, c] = -Float32.inf
                else:
                    self._apply_causal_token_mask(
                        acc_S,
                        tScS_mn,
                        sTok,
                        stage,
                        q_base,
                        mWorkQGlobal,
                        q_local_base,
                        q_tile_count,
                        wg_slot_base,
                        logical_q_heads_i32,
                    )
                softmax.online_softmax(
                    acc_S,
                    is_first=True,
                    check_inf=True,
                )
                rP_step = self._make_acc_into_pv_operand(
                    acc_S, tiled_mma_pv.tv_layout_A, self.dtype
                )

                cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                cute.arch.fence_proxy("async.shared", space="cta")
                sV_stage = sV[None, None, stage]
                tCrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV_stage))
                hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                    tiled_mma_pv,
                    acc_O,
                    rP_step,
                    tCrV,
                    zero_init=False,
                    wg_wait=0,
                )
                if idx_in_wg == Int32(0) and stage_count < tile_split_i32:
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.mbarrier_arrive(mbar_free + stage)

            for local_idx in cutlass.range(Int32(1), tile_split_i32, unroll=1):
                stage = local_idx % stage_count
                wait_phase = ((local_idx - stage) // stage_count) & Int32(1)
                cute.arch.mbarrier_wait(mbar_k_stage + stage, wait_phase)
                cute.arch.fence_proxy("async.shared", space="cta")

                sK_stage = sK[None, None, stage]
                tCrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK_stage))
                acc_S = thr_mma_qk.make_fragment_C(acc_shape_S)
                hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                    tiled_mma_qk,
                    acc_S,
                    tCrQ,
                    tCrK,
                    zero_init=True,
                    wg_wait=0,
                )
                if const_expr(mExactMaskBits is not None):
                    self._apply_exact_union_mask(
                        acc_S,
                        tScS_mn,
                        sTok,
                        sExactQMask,
                        stage,
                        q_base,
                        mWorkQGlobal,
                        q_local_base,
                        q_tile_count,
                        wg_slot_base,
                        logical_q_heads_i32,
                    )
                elif const_expr(mCausalLimits is not None):
                    acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
                    tile_col_base = local_idx * block_n_i32
                    tile_col_end = tile_col_base + block_n_i32
                    tile_before_wg_min_q = (
                        q_tile_count >= (wg_slot_base + Int32(self.slots_per_warpgroup))
                    ) and (wg_first_limit >= tile_col_end)
                    if not tile_before_wg_min_q:
                        for r in cutlass.range_constexpr(nrow_s):
                            row_invalid = causal_row_invalid[r] != Int32(0)
                            col_limit = causal_row_limit[r]
                            for c in cutlass.range_constexpr(ncol_s):
                                col = tScS_mn[r, c][1]
                                global_col = tile_col_base + col
                                if row_invalid or global_col >= col_limit:
                                    acc_S_mn[r, c] = -Float32.inf
                else:
                    self._apply_causal_token_mask(
                        acc_S,
                        tScS_mn,
                        sTok,
                        stage,
                        q_base,
                        mWorkQGlobal,
                        q_local_base,
                        q_tile_count,
                        wg_slot_base,
                        logical_q_heads_i32,
                    )

                row_scale = softmax.online_softmax(
                    acc_S,
                    is_first=False,
                    check_inf=True,
                )
                softmax.rescale_O(acc_O, row_scale)
                rP_step = self._make_acc_into_pv_operand(
                    acc_S, tiled_mma_pv.tv_layout_A, self.dtype
                )

                cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                cute.arch.fence_proxy("async.shared", space="cta")
                sV_stage = sV[None, None, stage]
                tCrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV_stage))
                hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                    tiled_mma_pv,
                    acc_O,
                    rP_step,
                    tCrV,
                    zero_init=False,
                    wg_wait=0,
                )
                if idx_in_wg == Int32(0) and (local_idx + stage_count) < tile_split_i32:
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.mbarrier_arrive(mbar_free + stage)

            row_scale_final = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale_final)
            for r in cutlass.range_constexpr(nrow_o):
                for c in cutlass.range_constexpr(ncol_o):
                    row = tOcO_mn[r, c][0]
                    d_out = tOcO_mn[r, c][1]
                    slot_rel = row // Int32(self.GROUP_SIZE_PAD_MMA)
                    h_out = row - slot_rel * Int32(self.GROUP_SIZE_PAD_MMA)
                    q_slot = wg_slot_base + slot_rel
                    q_valid = (q_slot < q_tile_count) and (h_out < logical_q_heads_i32)
                    q_local = q_local_base + q_slot
                    q_local_safe = q_local if q_valid else Int32(0)
                    gO_slot = cute.local_tile(
                        mO_cur,
                        (self.GROUP_SIZE_PAD, self.HEAD_DIM_V),
                        (q_local_safe, 0),
                    )
                    if q_valid and d_out < self.HEAD_DIM_V:
                        gO_slot[h_out, d_out] = self.dtype(acc_O_mn[r, c])

            if const_expr(mLSE is not None):
                for r in cutlass.range_constexpr(nrow_s):
                    row = tScS_mn[r, 0][0]
                    col = tScS_mn[r, 0][1]
                    if col == Int32(0):
                        slot_rel = row // Int32(self.GROUP_SIZE_PAD_MMA)
                        h_lse = row - slot_rel * Int32(self.GROUP_SIZE_PAD_MMA)
                        q_slot = wg_slot_base + slot_rel
                        if (q_slot < q_tile_count) and (h_lse < logical_q_heads_i32):
                            mLSE[q_local_base + q_slot, h_lse] = softmax.row_sum[r]


@functools.cache
def _get_prefill_union_kernel_variant(
    dtype,
    logical_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int,
    num_threads: int,
    q_group: int,
    max_union_windows: int,
) -> _TokenWiseSparsePagedPrefillUnionGQA:
    return _TokenWiseSparsePagedPrefillUnionGQA(
        dtype=dtype,
        logical_q_heads=int(logical_q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        block_n=int(block_n),
        num_threads=int(num_threads),
        q_group=int(q_group),
        max_union_windows=int(max_union_windows),
    )


@cached_compile_function
def _get_compiled_prefill_union_kernel(
    dtype,
    logical_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int,
    num_threads: int,
    q_group: int,
    max_union_windows: int,
    has_union_logical: bool,
    has_exact_mask: bool,
    has_lse: bool,
    q_signature: tuple[object, ...],
    k_signature: tuple[object, ...],
    v_signature: tuple[object, ...],
    union_phys_signature: tuple[object, ...],
    union_logical_signature: tuple[object, ...] | None,
    union_counts_signature: tuple[object, ...],
    exact_mask_signature: tuple[object, ...] | None,
    work_q_global_signature: tuple[object, ...],
    work_q_local_signature: tuple[object, ...],
    work_q_len_signature: tuple[object, ...],
    causal_limits_signature: tuple[object, ...],
    out_signature: tuple[object, ...],
    lse_signature: tuple[object, ...] | None,
    q_align: int,
    k_align: int,
    v_align: int,
    union_phys_align: int,
    union_logical_align: int | None,
    union_counts_align: int,
    exact_mask_align: int | None,
    work_q_global_align: int,
    work_q_local_align: int,
    work_q_len_align: int,
    causal_limits_align: int,
    out_align: int,
    lse_align: int | None,
    softmax_scale: float,
    device_key: tuple[str, int | None],
):
    device = utils.device_from_cache_key(device_key)
    kernel_impl = _get_prefill_union_kernel_variant(
        dtype,
        int(logical_q_heads),
        int(head_dim),
        int(head_dim_v),
        int(block_n),
        int(num_threads),
        int(q_group),
        int(max_union_windows),
    )
    q = utils.placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=kernel_impl.q_group
    )
    k_cache = utils.placeholder_from_signature(
        k_signature, device=device, dynamic_shape_fill=1
    )
    v_cache = utils.placeholder_from_signature(
        v_signature, device=device, dynamic_shape_fill=1
    )
    union_phys = utils.placeholder_from_signature(
        union_phys_signature, device=device, dynamic_shape_fill=1
    )
    union_logical = (
        utils.placeholder_from_signature(
            union_logical_signature, device=device, dynamic_shape_fill=1
        )
        if has_union_logical
        else None
    )
    union_counts = utils.placeholder_from_signature(
        union_counts_signature, device=device, dynamic_shape_fill=1
    )
    exact_mask_bits = (
        utils.placeholder_from_signature(
            exact_mask_signature, device=device, dynamic_shape_fill=1
        )
        if has_exact_mask
        else None
    )
    work_q_global = utils.placeholder_from_signature(
        work_q_global_signature, device=device, dynamic_shape_fill=1
    )
    work_q_local = utils.placeholder_from_signature(
        work_q_local_signature, device=device, dynamic_shape_fill=1
    )
    work_q_len = utils.placeholder_from_signature(
        work_q_len_signature, device=device, dynamic_shape_fill=1
    )
    causal_limits = utils.placeholder_from_signature(
        causal_limits_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=kernel_impl.q_group
    )
    lse = (
        utils.placeholder_from_signature(
            lse_signature, device=device, dynamic_shape_fill=kernel_impl.q_group
        )
        if has_lse
        else None
    )
    fQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=q_align, dynamic_shape_dim=0
    )
    fK = utils.make_fake_tensor_like_with_dynamic_dim(
        k_cache, alignment=k_align, dynamic_shape_dim=0
    )
    fV = utils.make_fake_tensor_like_with_dynamic_dim(
        v_cache, alignment=v_align, dynamic_shape_dim=0
    )
    fUnionPhys = utils.make_fake_tensor_like_with_dynamic_dim(
        union_phys, alignment=union_phys_align, dynamic_shape_dim=0
    )
    fUnionLogical = (
        utils.make_fake_tensor_like_with_dynamic_dim(
            union_logical, alignment=union_logical_align, dynamic_shape_dim=0
        )
        if union_logical is not None
        else None
    )
    fUnionCount = utils.make_fake_tensor_like_with_dynamic_dim(
        union_counts, alignment=union_counts_align, dynamic_shape_dim=0
    )
    fExactMaskBits = (
        utils.make_fake_tensor_like_with_dynamic_dim(
            exact_mask_bits, alignment=exact_mask_align, dynamic_shape_dim=0
        )
        if exact_mask_bits is not None
        else None
    )
    fWG = utils.make_fake_tensor_like_with_dynamic_dim(
        work_q_global, alignment=work_q_global_align, dynamic_shape_dim=0
    )
    fWL = utils.make_fake_tensor_like_with_dynamic_dim(
        work_q_local, alignment=work_q_local_align, dynamic_shape_dim=0
    )
    fWQ = utils.make_fake_tensor_like_with_dynamic_dim(
        work_q_len, alignment=work_q_len_align, dynamic_shape_dim=0
    )
    fCausalLimits = utils.make_fake_tensor_like_with_dynamic_dim(
        causal_limits, alignment=causal_limits_align, dynamic_shape_dim=0
    )
    fO = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=out_align, dynamic_shape_dim=0
    )
    fLSE = (
        utils.make_fake_tensor_like_with_dynamic_dim(
            lse, alignment=lse_align, dynamic_shape_dim=0
        )
        if lse is not None
        else None
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_impl,
        fQ,
        fK,
        fV,
        fUnionPhys,
        fUnionLogical,
        fUnionCount,
        fExactMaskBits,
        fWG,
        fWL,
        fWQ,
        fCausalLimits,
        fO,
        fLSE,
        cutlass.Float32(softmax_scale),
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


class TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillUnionGQA:
    """Wrapper for GQA grouped-union sparse prefill."""

    arch: int = 90
    REQUIRED_POINTER_ALIGN = _TokenWiseSparsePagedPrefillUnionGQA.REQUIRED_POINTER_ALIGN

    def __init__(
        self,
        *,
        dtype,
        logical_q_heads: int,
        head_dim: int,
        head_dim_v: int,
        block_n: int = 64,
        num_threads: int = 512,
        q_group: Optional[int] = None,
    ) -> None:
        self.dtype = dtype
        self.logical_q_heads = int(logical_q_heads)
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.q_group = (
            self._expected_q_group(self.logical_q_heads)
            if q_group is None
            else int(q_group)
        )
        _TokenWiseSparsePagedPrefillUnionGQA._validate_q_group_contract(
            logical_q_heads=self.logical_q_heads,
            q_group=self.q_group,
        )
        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("Only fp16/bf16 are supported")
        if (
            self.logical_q_heads
            not in _TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_LOGICAL_Q_HEADS
        ):
            raise ValueError(
                "GQA grouped-union prefill wrapper supports only logical_q_heads in "
                f"{_TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_LOGICAL_Q_HEADS}, "
                f"got {self.logical_q_heads}"
            )
        if (
            self.HEAD_DIM
            not in _TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_HEAD_DIMS
            or self.HEAD_DIM_V
            not in _TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_HEAD_DIMS
        ):
            raise ValueError(
                "GQA grouped-union prefill wrapper supports only head_dim/head_dim_v in "
                f"{_TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_HEAD_DIMS}, "
                f"got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "GQA grouped-union prefill keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if (
            self.q_group,
            self.block_n,
            self.num_threads,
        ) not in _TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_TOPOLOGIES:
            raise ValueError(
                "GQA grouped-union prefill supports only "
                f"{sorted(_TokenWiseSparsePagedPrefillUnionGQA.SUPPORTED_TOPOLOGIES)}"
            )

    def _validate_128b_aligned_strides(self, t, *, name: str):
        align_elems = 128 // t.element_size() // 8
        for i, s in enumerate(t.stride()[:-1]):
            if int(s) % align_elems != 0:
                raise ValueError(
                    f"{name}.stride[{i}]={int(s)} is not divisible by {align_elems}; "
                    "this kernel requires 128-bit aligned logical strides"
                )

    def _validate_min_pointer_alignment(self, t, *, name: str, min_align: int) -> None:
        ptr = int(t.data_ptr())
        if ptr % int(min_align) != 0:
            raise ValueError(
                f"{name}.data_ptr()={ptr} is not {min_align}-byte aligned; "
                "this kernel requires the 16B vector copy configuration"
            )

    def _validate_inputs(
        self,
        q,
        k_cache,
        v_cache,
        o,
        lse,
        union_phys_indices,
        union_logical_indices,
        union_counts,
        exact_mask_bits,
        work_q_global,
        work_q_local,
        work_q_len,
        causal_limits,
    ):
        import torch

        total_q, Hq, D = q.shape
        num_pages_k, page_size_k, Hkv, Dk = k_cache.shape
        num_pages_v, page_size_v, Hkv_v, Dv = v_cache.shape

        if (Hq, D) != (self.logical_q_heads, self.HEAD_DIM):
            raise ValueError(
                f"Expected q shape (*, {self.logical_q_heads}, {self.HEAD_DIM}), got {tuple(q.shape)}"
            )
        if Dk != self.HEAD_DIM or Dv != self.HEAD_DIM_V:
            raise ValueError(
                "Input head dimensions do not match GQA grouped-union prefill config "
                f"({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if Hkv != 1 or Hkv_v != 1:
            raise ValueError("GQA grouped-union prefill expects a single local kv head")
        if num_pages_k != num_pages_v:
            raise ValueError(
                f"Paged K/V page count mismatch: {num_pages_k} vs {num_pages_v}"
            )
        if page_size_k != page_size_v:
            raise ValueError(
                f"Paged K/V page size mismatch: {page_size_k} vs {page_size_v}"
            )
        if page_size_k != 16:
            raise ValueError(f"Expected page size 16, got {page_size_k}")
        if o.shape != (total_q, Hq, self.HEAD_DIM_V):
            raise ValueError(
                f"Expected o shape {(total_q, Hq, self.HEAD_DIM_V)}, got {tuple(o.shape)}"
            )
        self._validate_128b_aligned_strides(q, name="q")
        self._validate_128b_aligned_strides(k_cache, name="k_cache")
        self._validate_128b_aligned_strides(v_cache, name="v_cache")
        self._validate_128b_aligned_strides(o, name="o")
        self._validate_min_pointer_alignment(
            q, name="q", min_align=self.REQUIRED_POINTER_ALIGN
        )
        self._validate_min_pointer_alignment(
            k_cache, name="k_cache", min_align=self.REQUIRED_POINTER_ALIGN
        )
        self._validate_min_pointer_alignment(
            v_cache, name="v_cache", min_align=self.REQUIRED_POINTER_ALIGN
        )
        self._validate_min_pointer_alignment(
            o, name="o", min_align=self.REQUIRED_POINTER_ALIGN
        )

        if lse is not None:
            if lse.shape != (total_q, Hq):
                raise ValueError(
                    f"Expected lse shape {(total_q, Hq)}, got {tuple(lse.shape)}"
                )
            if lse.dtype != torch.float32:
                raise ValueError("lse must be float32")
            if lse.device != q.device:
                raise ValueError("lse must be on the same device as q")

        if union_phys_indices is None or union_counts is None:
            raise ValueError(
                "GQA grouped-union prefill requires union_phys_indices and union_counts"
            )
        if union_phys_indices.dtype != torch.int32:
            raise ValueError(
                f"union_phys_indices must be int32, got {union_phys_indices.dtype}"
            )
        if union_counts.dtype != torch.int32:
            raise ValueError("union_counts must be int32")
        if union_phys_indices.device != q.device or union_counts.device != q.device:
            raise ValueError("union metadata must be on the same device as q")
        if union_phys_indices.ndim != 2 or union_counts.ndim != 1:
            raise ValueError(
                "union_phys_indices must be 2D and union_counts must be 1D"
            )
        if int(union_phys_indices.shape[0]) != int(union_counts.shape[0]):
            raise ValueError("union metadata first dimensions must match")
        if not union_phys_indices.is_contiguous():
            raise ValueError(
                f"union_phys_indices must be contiguous, got stride={tuple(union_phys_indices.stride())}"
            )
        if not union_counts.is_contiguous():
            raise ValueError(
                f"union_counts must be contiguous, got stride={tuple(union_counts.stride())}"
            )

        exact_enabled = exact_mask_bits is not None or union_logical_indices is not None
        if exact_enabled:
            if exact_mask_bits is None or union_logical_indices is None:
                raise ValueError(
                    "GQA grouped-union exact mask requires both exact_mask_bits "
                    "and union_logical_indices"
                )
            if union_logical_indices.dtype != torch.int32:
                raise ValueError("union_logical_indices must be int32")
            if exact_mask_bits.dtype != torch.int32:
                raise ValueError("exact_mask_bits must be int32")
            if (
                union_logical_indices.device != q.device
                or exact_mask_bits.device != q.device
            ):
                raise ValueError("exact mask metadata must be on the same device as q")
            if tuple(union_logical_indices.shape) != tuple(union_phys_indices.shape):
                raise ValueError(
                    "union_logical_indices must match union_phys_indices shape, "
                    f"got {tuple(union_logical_indices.shape)} vs {tuple(union_phys_indices.shape)}"
                )
            expected_exact_shape = (
                int(union_counts.numel()),
                int(union_phys_indices.shape[1]),
            )
            if tuple(exact_mask_bits.shape) != expected_exact_shape:
                raise ValueError(
                    f"Expected exact_mask_bits shape {expected_exact_shape}, "
                    f"got {tuple(exact_mask_bits.shape)}"
                )
            if not union_logical_indices.is_contiguous():
                raise ValueError(
                    "union_logical_indices must be contiguous, "
                    f"got stride={tuple(union_logical_indices.stride())}"
                )
            if not exact_mask_bits.is_contiguous():
                raise ValueError(
                    f"exact_mask_bits must be contiguous, got stride={tuple(exact_mask_bits.stride())}"
                )

        work_tensors = {
            "work_q_global": work_q_global,
            "work_q_local": work_q_local,
            "work_q_len": work_q_len,
        }
        work_size = int(union_counts.numel())
        for name, tensor in work_tensors.items():
            if tensor is None:
                raise ValueError(f"{name} is required for GQA grouped-union prefill")
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32")
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be 1D, got shape={tuple(tensor.shape)}")
            if int(tensor.numel()) != work_size:
                raise ValueError("work metadata tensors must match union_counts length")
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous, got stride={tuple(tensor.stride())}"
                )
        if work_size <= 0:
            raise ValueError(
                "GQA grouped-union prefill requires at least one work tile"
            )

        if causal_limits is None:
            raise ValueError("GQA grouped-union prefill requires causal_limits")
        expected_shape = (work_size, self.q_group)
        if causal_limits.dtype != torch.int32:
            raise ValueError("causal_limits must be int32")
        if causal_limits.device != q.device:
            raise ValueError("causal_limits must be on the same device as q")
        if tuple(causal_limits.shape) != expected_shape:
            raise ValueError(
                f"Expected causal_limits shape {expected_shape}, got {tuple(causal_limits.shape)}"
            )
        if not causal_limits.is_contiguous():
            raise ValueError(
                f"causal_limits must be contiguous, got stride={tuple(causal_limits.stride())}"
            )

        return (
            union_phys_indices,
            union_logical_indices,
            union_counts,
            exact_mask_bits,
            work_tensors["work_q_global"],
            work_tensors["work_q_local"],
            work_tensors["work_q_len"],
            causal_limits,
        )

    def run(
        self,
        q,
        k_cache,
        v_cache,
        o,
        *,
        lse=None,
        union_phys_indices=None,
        union_logical_indices=None,
        union_counts=None,
        exact_mask_bits=None,
        work_q_global=None,
        work_q_local=None,
        work_q_len=None,
        causal_limits,
        softmax_scale: Optional[float] = None,
        stream: Optional[cutlass_torch.cuda.CUstream] = None,
    ):
        import torch

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(float(self.HEAD_DIM))
        if stream is not None:
            raise ValueError(
                "prefill sparse attention uses the TVM-FFI environment stream"
            )
        record_torch_stream = torch.cuda.current_stream(device=q.device)

        (
            union_phys_indices,
            union_logical_indices,
            union_counts,
            exact_mask_bits,
            work_q_global,
            work_q_local,
            work_q_len,
            causal_limits,
        ) = self._validate_inputs(
            q,
            k_cache,
            v_cache,
            o,
            lse,
            union_phys_indices,
            union_logical_indices,
            union_counts,
            exact_mask_bits,
            work_q_global,
            work_q_local,
            work_q_len,
            causal_limits,
        )
        is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        can_record_stream = not (is_capturing is not None and bool(is_capturing()))
        if can_record_stream:
            # TVM-FFI launches bypass the ATen dispatcher, so the caching
            # allocator cannot infer that temporary union/work metadata remains
            # live until this asynchronous launch completes.
            for tensor in (
                q,
                k_cache,
                v_cache,
                o,
                lse,
                union_phys_indices,
                union_logical_indices,
                union_counts,
                exact_mask_bits,
                work_q_global,
                work_q_local,
                work_q_len,
                causal_limits,
            ):
                if tensor is not None:
                    tensor.record_stream(record_torch_stream)

        self._validate_min_pointer_alignment(
            union_phys_indices, name="union_phys_indices", min_align=4
        )
        if union_logical_indices is not None:
            self._validate_min_pointer_alignment(
                union_logical_indices, name="union_logical_indices", min_align=4
            )
        self._validate_min_pointer_alignment(
            union_counts, name="union_counts", min_align=4
        )
        if exact_mask_bits is not None:
            self._validate_min_pointer_alignment(
                exact_mask_bits, name="exact_mask_bits", min_align=4
            )
        self._validate_min_pointer_alignment(
            work_q_global, name="work_q_global", min_align=4
        )
        self._validate_min_pointer_alignment(
            work_q_local, name="work_q_local", min_align=4
        )
        self._validate_min_pointer_alignment(work_q_len, name="work_q_len", min_align=4)
        self._validate_min_pointer_alignment(
            causal_limits, name="causal_limits", min_align=4
        )
        if lse is not None:
            self._validate_min_pointer_alignment(lse, name="lse", min_align=4)

        q_align = self.REQUIRED_POINTER_ALIGN
        k_align = self.REQUIRED_POINTER_ALIGN
        v_align = self.REQUIRED_POINTER_ALIGN
        o_align = self.REQUIRED_POINTER_ALIGN
        union_phys_align = 4
        union_logical_align = 4 if union_logical_indices is not None else None
        union_counts_align = 4
        exact_mask_align = 4 if exact_mask_bits is not None else None
        work_q_global_align = 4
        work_q_local_align = 4
        work_q_len_align = 4
        causal_limits_align = 4
        lse_align = 4 if lse is not None else None

        compiled = _get_compiled_prefill_union_kernel(
            self.dtype,
            int(self.logical_q_heads),
            int(self.HEAD_DIM),
            int(self.HEAD_DIM_V),
            int(self.block_n),
            int(self.num_threads),
            int(self.q_group),
            int(union_phys_indices.shape[1]),
            bool(union_logical_indices is not None),
            bool(exact_mask_bits is not None),
            bool(lse is not None),
            utils.tensor_signature_dynamic(q, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(k_cache, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(v_cache, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(union_phys_indices, dynamic_shape_dims=(0,)),
            (
                utils.tensor_signature_dynamic(
                    union_logical_indices, dynamic_shape_dims=(0,)
                )
                if union_logical_indices is not None
                else None
            ),
            utils.tensor_signature_dynamic(union_counts, dynamic_shape_dims=(0,)),
            (
                utils.tensor_signature_dynamic(exact_mask_bits, dynamic_shape_dims=(0,))
                if exact_mask_bits is not None
                else None
            ),
            utils.tensor_signature_dynamic(work_q_global, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(work_q_local, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(work_q_len, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(causal_limits, dynamic_shape_dims=(0,)),
            utils.tensor_signature_dynamic(o, dynamic_shape_dims=(0,)),
            None
            if lse is None
            else utils.tensor_signature_dynamic(lse, dynamic_shape_dims=(0,)),
            int(q_align),
            int(k_align),
            int(v_align),
            int(union_phys_align),
            None if union_logical_indices is None else int(union_logical_align),
            int(union_counts_align),
            None if exact_mask_bits is None else int(exact_mask_align),
            int(work_q_global_align),
            int(work_q_local_align),
            int(work_q_len_align),
            int(causal_limits_align),
            int(o_align),
            None if lse is None else int(lse_align),
            float(softmax_scale),
            utils.device_cache_key(q.device),
        )

        compiled(
            q,
            k_cache,
            v_cache,
            union_phys_indices,
            union_logical_indices,
            union_counts,
            exact_mask_bits,
            work_q_global,
            work_q_local,
            work_q_len,
            causal_limits,
            o,
            lse,
            cutlass.Float32(softmax_scale),
        )
        return o


__all__ = ["TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillUnionGQA"]
