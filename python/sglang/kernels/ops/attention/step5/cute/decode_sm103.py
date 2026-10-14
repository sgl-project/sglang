# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe SM103 hardware port of Step5's SM90 sparse GQA decode.

Source: vllm/models/step5/nvidia/ops/cute_dsl/sparse_gqa/token_sparse_attn/
decode_sm90_gqa.py, class _TokenWiseSparseDecodeWGMMA64GQA.
"""

import math
import operator
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import tcgen05

from sglang.kernels.ops.attention.flash_attn.cute.utils import warp_prefix_sum

from . import cute_utils as utils
from .decode_sm103_utils import (
    elem_pointer,
    elem_pointer_i64_offset,
    tcgen_after_sync,
    tcgen_before_sync,
    vector_copy_with_explicit_width,
)


class SparseDecodeSm103:
    """SM103 port of the Step5 SM90 sparse decode producer/consumer kernel.

    Preserve 64-token swapAB tiles, staged sparse K/V gathers and device split
    scheduling. TMEM copy coordinates replace WGMMA register ownership.
    """

    arch: int = 103
    SUPPORTED_HEAD_DIMS = (128, 192)
    PAGE_SIZE = 16
    KV_HEADS = 1
    SUPPORTED_LOGICAL_Q_HEADS = (8, 16)
    MMA_Q_HEADS = 16
    BLOCK_N = 64
    NUM_STAGES = 2
    PRODUCER_SUBGROUP_LANES = 8
    PRODUCER_REGS = 24
    MATH_REGS = 240
    PRODUCER_REGS_FP8 = 96
    MATH_REGS_FP8 = 160
    REQUIRED_POINTER_ALIGN = 16
    DYNAMIC_SPLIT_CAPACITY = 16
    MIN_WINDOWS_PER_DYNAMIC_SPLIT = 32
    MIN_DYNAMIC_SPLIT_WAVE_NUMERATOR = 7
    MIN_DYNAMIC_SPLIT_WAVE_DENOMINATOR = 8

    def __init__(
        self,
        *,
        dtype,
        logical_q_heads: int,
        head_dim: int,
        head_dim_v: int,
        block_n: int,
        num_threads: int,
        topk_windows: int,
        mtp_q_len: int = 1,
        producer_k_warps: Optional[int] = None,
        split_capacity: int = 1,
        sm_count: Optional[int] = None,
        kv_fp8: bool = False,
    ) -> None:
        self.dtype = dtype
        self.kv_fp8 = bool(kv_fp8)
        self.logical_q_heads = int(logical_q_heads)
        # SM103 supports an N8 MMA tile. Avoid padded-head score/accumulator
        # work for TP8 while retaining N16 for the supported TP4 layout.
        self.MMA_Q_HEADS = self.logical_q_heads
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.topk_windows = int(topk_windows)
        self.mtp_q_len = int(mtp_q_len)
        self.split_capacity = int(split_capacity)
        self.sm_count = None if sm_count is None else int(sm_count)
        self.dynamic_split = self.split_capacity == self.DYNAMIC_SPLIT_CAPACITY
        self.num_stages = int(self.NUM_STAGES)
        # Ld16x256b distributes adjacent lanes across heads 0/2/4/6.
        # Four padding words put their score stores in distinct bank groups
        # while each softmax warp still loads consecutive token addresses.
        self.score_stride = self.block_n + 4
        self.head_dim_slabs = self.HEAD_DIM // 64
        self.head_dim_v_slabs = self.HEAD_DIM_V // 64
        self.windows_per_tile = self.block_n // 8
        self.num_warps = self.num_threads // 32
        self.num_compute_warps = 4
        self.num_producer_warps = self.num_warps - self.num_compute_warps
        producer_k_warps_cfg = (
            max(1, self.num_producer_warps // 2)
            if producer_k_warps is None
            else int(producer_k_warps)
        )
        self.producer_k_warps = producer_k_warps_cfg
        self.producer_v_warps = self.num_producer_warps - self.producer_k_warps
        self.producer_subgroup_lanes = int(self.PRODUCER_SUBGROUP_LANES)
        self.producer_k_groups = (
            self.producer_k_warps * 32
        ) // self.producer_subgroup_lanes
        self.producer_v_groups = (
            self.producer_v_warps * 32
        ) // self.producer_subgroup_lanes
        self.max_rows_per_k_group = (
            self.block_n + self.producer_k_groups - 1
        ) // self.producer_k_groups
        self.max_rows_per_v_group = (
            self.block_n + self.producer_v_groups - 1
        ) // self.producer_v_groups

        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("Only fp16/bf16 are supported")
        if self.logical_q_heads not in self.SUPPORTED_LOGICAL_Q_HEADS:
            raise ValueError(
                "CuTe sparse decode supports only logical_q_heads in "
                f"{self.SUPPORTED_LOGICAL_Q_HEADS}, "
                f"got {self.logical_q_heads}"
            )
        if (
            self.HEAD_DIM not in self.SUPPORTED_HEAD_DIMS
            or self.HEAD_DIM_V not in self.SUPPORTED_HEAD_DIMS
        ):
            raise ValueError(
                "CuTe sparse decode supports only head_dim/head_dim_v in "
                f"{self.SUPPORTED_HEAD_DIMS}, got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "CuTe sparse decode keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if self.HEAD_DIM % 64 != 0 or self.HEAD_DIM_V % 64 != 0:
            raise ValueError(
                "CuTe sparse decode requires head_dim/head_dim_v to be 64-aligned, "
                f"got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.block_n != self.BLOCK_N:
            raise ValueError(
                f"CuTe sparse decode is fixed to block_n=64, got {self.block_n}"
            )
        if self.num_threads != 256:
            raise ValueError(
                "CuTe sparse decode is fixed to num_threads=256, "
                f"got {self.num_threads}"
            )
        if self.topk_windows <= 0:
            raise ValueError("CuTe sparse decode requires topk_windows > 0")
        if self.mtp_q_len < 1 or self.mtp_q_len > 16:
            raise ValueError(f"mtp_q_len must be in [1, 16], got {self.mtp_q_len}")
        if self.split_capacity not in (1, self.DYNAMIC_SPLIT_CAPACITY):
            raise ValueError(
                "CuTe sparse decode kernel split capacity must be 1 or "
                f"{self.DYNAMIC_SPLIT_CAPACITY}, got {self.split_capacity}"
            )
        if self.dynamic_split and (self.sm_count is None or self.sm_count <= 0):
            raise ValueError("dynamic split decode requires a positive SM count")
        if self.num_producer_warps != 4:
            raise ValueError("CuTe sparse decode requires four producer warps")
        if (
            self.producer_k_warps <= 0
            or self.producer_k_warps >= self.num_producer_warps
        ):
            raise ValueError(
                "producer_k_warps must be in [1, num_producer_warps-1], got "
                f"{self.producer_k_warps} with num_producer_warps={self.num_producer_warps}"
            )

    def _get_tiled_mma(self):
        qk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (self.block_n, self.MMA_Q_HEADS),
        )
        pv = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, self.MMA_Q_HEADS),
        )
        return qk, pv

    @cute.jit
    def _dynamic_split_parts(
        self,
        region_count: Int32,
        target_parts: Int32,
    ) -> Int32:
        parts = Int32(0)
        if region_count > Int32(0):
            parts = Int32(1)
            if target_parts >= Int32(2) and region_count >= Int32(
                2 * self.MIN_WINDOWS_PER_DYNAMIC_SPLIT
            ):
                parts = Int32(2)
            if target_parts >= Int32(4) and region_count >= Int32(
                4 * self.MIN_WINDOWS_PER_DYNAMIC_SPLIT
            ):
                parts = Int32(4)
            if target_parts >= Int32(8) and region_count >= Int32(
                8 * self.MIN_WINDOWS_PER_DYNAMIC_SPLIT
            ):
                parts = Int32(8)
            if target_parts >= Int32(16) and region_count >= Int32(
                16 * self.MIN_WINDOWS_PER_DYNAMIC_SPLIT
            ):
                parts = Int32(16)
        return parts

    @cute.jit
    def _target_split_parts(self, active_count: Int32) -> Int32:
        target = Int32(1)
        if active_count > Int32(0):
            min_wave_work = Int32(self.sm_count * self.MIN_DYNAMIC_SPLIT_WAVE_NUMERATOR)
            scale = Int32(self.MIN_DYNAMIC_SPLIT_WAVE_DENOMINATOR)
            if active_count * scale < min_wave_work:
                target = Int32(16)
                if active_count * Int32(2) * scale >= min_wave_work:
                    target = Int32(2)
                elif active_count * Int32(4) * scale >= min_wave_work:
                    target = Int32(4)
                elif active_count * Int32(8) * scale >= min_wave_work:
                    target = Int32(8)
        return target

    @cute.jit
    def _schedule_dynamic(
        self,
        counts: cute.Tensor,
        request_starts: cute.Tensor,
        valid_rows: Int32,
        launch_idx: Int32,
        schedule: cute.Tensor,
        lane: Int32,
    ):
        # One warp preserves the source's launch order and logical partial
        # offsets. No CTA-local thread serially scans the entire query batch.
        capacity = Int32(cute.size(counts.shape[0]))
        if lane == 0:
            schedule[0] = Int32(-1)
            schedule[1] = Int32(0)
            schedule[2] = Int32(1)
            schedule[3] = Int32(-1)
        # Four MTP verify rows provide enough independent work to use the full
        # 16-way split capacity on a 148-SM L20D.  Eight-way splitting leaves
        # most SMs idle while every CTA still processes eight K/V tiles.
        target = Int32(16)
        if const_expr(self.mtp_q_len == 1):
            active = Int32(0)
            for row in cutlass.range(lane, capacity, 32, unroll=1):
                if row < valid_rows and counts[row] > Int32(0):
                    active += Int32(1)
            active = cute.arch.warp_reduction(active, operator.add)
            target = self._target_split_parts(active)
        cute.arch.sync_warp()

        entries = capacity
        if const_expr(self.mtp_q_len > 1):
            entries = Int32(cute.size(request_starts.shape[0])) - Int32(1)
        groups = (entries + Int32(31)) // Int32(32)
        cursor = Int32(0)
        for slot in cutlass.range_constexpr(self.mtp_q_len):
            for group in cutlass.range(groups, unroll=1):
                entry = group * Int32(32) + lane
                row = entry
                live = entry < entries and row < valid_rows
                if const_expr(self.mtp_q_len > 1):
                    live = cutlass.Boolean(False)
                    if entry < entries:
                        row = Int32(request_starts[entry]) + Int32(slot)
                        end = Int32(request_starts[entry + Int32(1)])
                        live = row < end and row < valid_rows
                count = Int32(0)
                if live:
                    count = counts[row]
                parts = self._dynamic_split_parts(count, target)
                prefix = warp_prefix_sum(parts, lane)
                first = cursor + prefix - parts
                if launch_idx >= first and launch_idx < cursor + prefix:
                    schedule[0] = row
                    schedule[1] = launch_idx - first
                    schedule[2] = parts
                    schedule[3] = launch_idx
                cursor += cute.arch.shuffle_sync(prefix, Int32(31))
        cute.arch.sync_warp()

        if const_expr(self.mtp_q_len > 1):
            # Launch order is slot-major for ragged MTP, but source merge reads
            # row-major partials. Reconstruct that offset with a warp reduction.
            selected_q = schedule[0]
            offset = Int32(0)
            for row in cutlass.range(lane, capacity, 32, unroll=1):
                if row < selected_q and row < valid_rows:
                    offset += self._dynamic_split_parts(counts[row], target)
            offset = cute.arch.warp_reduction(offset, operator.add)
            if lane == 0 and selected_q >= Int32(0):
                schedule[3] = offset + schedule[1]
        cute.arch.sync_warp()

    def _get_smem_layouts(self):
        qk, pv = self._get_tiled_mma()
        qk_tile = (self.block_n, self.MMA_Q_HEADS, self.HEAD_DIM)
        pv_tile = (64, self.MMA_Q_HEADS, self.block_n)
        return (
            sm100_utils.make_smem_layout_a(qk, qk_tile, self.dtype, self.num_stages),
            sm100_utils.make_smem_layout_b(qk, qk_tile, self.dtype, 1),
            sm100_utils.make_smem_layout_a(
                pv, pv_tile, self.dtype, self.num_stages * self.head_dim_v_slabs
            ),
            sm100_utils.make_smem_layout_b(pv, pv_tile, self.dtype, 1),
        )

    def _get_shared_storage_cls(self):
        sK_layout, sQ_layout, sV_layout, sP_layout = self._get_smem_layouts()
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)],
            1024,
        ]
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)],
            1024,
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)],
            1024,
        ]
        sP_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sP_layout)],
            1024,
        ]
        stage_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages]
        scalar_struct = cute.struct.MemRange[Int32, 1]
        schedule_struct = cute.struct.MemRange[Int32, 4]
        packed_struct = cute.struct.MemRange[
            cutlass.Int64,
            self.windows_per_tile * self.num_stages,
        ]
        head_f32_struct = cute.struct.MemRange[Float32, self.logical_q_heads]
        score_struct = cute.struct.MemRange[
            Float32, self.score_stride * self.MMA_Q_HEADS
        ]
        mma_barrier_struct = cute.struct.MemRange[cutlass.Int64, 1]
        k_scale_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, self.num_stages * self.block_n if self.kv_fp8 else 1
            ],
            16,
        ]
        v_scale_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, self.num_stages * self.block_n if self.kv_fp8 else 1
            ],
            16,
        ]

        @cute.struct
        class SharedStorage:
            mbar_ptr_free: stage_struct
            mbar_ptr_k_stage: stage_struct
            mbar_ptr_v_stage: stage_struct
            mbar_ptr_metadata: stage_struct
            sK: sK_struct
            sQ: sQ_struct
            sV: sV_struct
            sP: sP_struct
            sPacked: packed_struct
            sCount: scalar_struct
            sKVSeqLen: scalar_struct
            sSchedule: schedule_struct
            sHeadMax: head_f32_struct
            sHeadSum: head_f32_struct
            sHeadScale: head_f32_struct
            sScores: score_struct
            mbarMma: mma_barrier_struct
            tmemBase: scalar_struct
            sKScaleStage: k_scale_struct
            sVScaleStage: v_scale_struct
            sVTileMax: head_f32_struct

        return SharedStorage

    @cute.jit
    def _convert_k_stage(
        self,
        sK: cute.Tensor,
        sK_layout: cute.ComposedLayout,
        k_stage: Int32,
        tid: Int32,
        threads: cutlass.Constexpr,
        barrier_id: cutlass.Constexpr,
    ):
        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        vectors = self.block_n * self.HEAD_DIM // 8
        k_raw = elem_pointer(sK, ((0, 0), 0, 0, k_stage)).toint()
        loaded = []
        for j in cutlass.range_constexpr(vectors // threads):
            loaded.append(
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float8E4M3FN,
                        k_raw + (tid + Int32(j * threads)) * Int32(8),
                        sK.memspace,
                        assumed_align=8,
                    ),
                    vec_layout_8,
                ).load()
            )
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)
        for j in cutlass.range_constexpr(vectors // threads):
            vec = tid + Int32(j * threads)
            kb = vec // Int32(2 * self.block_n)
            rem = vec - kb * Int32(2 * self.block_n)
            tok = rem // Int32(2)
            half = rem - tok * Int32(2)
            dst_ptr = elem_pointer(sK, ((tok, half * Int32(8)), 0, kb, k_stage))
            dst_ptr = cute.make_ptr(
                self.dtype,
                dst_ptr.toint(),
                sK.memspace,
                assumed_align=self.REQUIRED_POINTER_ALIGN,
            )
            cute.make_tensor(
                cute.recast_ptr(dst_ptr, sK_layout.inner), vec_layout_8
            ).store(loaded[j].to(self.dtype))
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _convert_v_stage(
        self,
        sV: cute.Tensor,
        sV_layout: cute.ComposedLayout,
        v_stage: Int32,
        stage_count: Int32,
        tid: Int32,
        threads: cutlass.Constexpr,
        barrier_id: cutlass.Constexpr,
    ):
        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        per_slab = self.block_n * 64 // 8
        vectors = per_slab * self.head_dim_v_slabs
        loaded = []
        for j in cutlass.range_constexpr(vectors // threads):
            vec = tid + Int32(j * threads)
            slab = vec // Int32(per_slab)
            rem = vec - slab * Int32(per_slab)
            v_raw = elem_pointer(
                sV, ((0, 0), 0, 0, v_stage + stage_count * slab)
            ).toint()
            loaded.append(
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float8E4M3FN,
                        v_raw + rem * Int32(8),
                        sV.memspace,
                        assumed_align=8,
                    ),
                    vec_layout_8,
                ).load()
            )
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)
        for j in cutlass.range_constexpr(vectors // threads):
            vec = tid + Int32(j * threads)
            slab = vec // Int32(per_slab)
            rem = vec - slab * Int32(per_slab)
            tok = rem // Int32(8)
            dchunk = rem - tok * Int32(8)
            dst_ptr = elem_pointer(
                sV,
                (
                    (dchunk * Int32(8), tok % 16),
                    0,
                    tok // 16,
                    v_stage + stage_count * slab,
                ),
            )
            dst_ptr = cute.make_ptr(
                self.dtype,
                dst_ptr.toint(),
                sV.memspace,
                assumed_align=self.REQUIRED_POINTER_ALIGN,
            )
            cute.make_tensor(
                cute.recast_ptr(dst_ptr, sV_layout.inner), vec_layout_8
            ).store(loaded[j].to(self.dtype))
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _softmax_heads(
        self,
        scores: cute.Tensor,
        probabilities: cute.Tensor,
        head_max: cute.Tensor,
        head_sum: cute.Tensor,
        head_scale: cute.Tensor,
        scale: Float32,
        warp: Int32,
        lane: Int32,
        v_scales: cute.Tensor,
        v_offset: Int32,
        v_tile_max: cute.Tensor,
    ):
        # TMEM's register ownership differs from WGMMA. A shared score tile
        # makes the mapping explicit; each warp reduces complete heads.
        v_weight_0 = Float32(1)
        v_weight_1 = Float32(1)
        if const_expr(self.kv_fp8):
            inv_max = v_tile_max[1]
            v_weight_0 = v_scales[v_offset + lane] * inv_max
            v_weight_1 = v_scales[v_offset + lane + Int32(32)] * inv_max
        for slot in cutlass.range_constexpr(self.logical_q_heads // 4):
            head = warp * Int32(self.logical_q_heads // 4) + Int32(slot)
            maximum = cute.arch.fmax(scores[lane, head], scores[lane + 32, head])
            maximum = cute.arch.warp_redux_sync(maximum, "max")
            old_max = head_max[head]
            maximum = cute.arch.fmax(old_max, maximum)
            # A masked tile must not replace a finite negative running maximum
            # with zero. Keep the empty-state guard out of persistent state.
            safe_max = Float32(0) if maximum == -Float32.inf else maximum
            alpha = utils.exp2f((old_max - safe_max) * scale)
            total = Float32(0)
            for half in cutlass.range_constexpr(2):
                row = lane + Int32(half * 32)
                probability = utils.exp2f(scores[row, head] * scale - safe_max * scale)
                # Preserve the FP32 probability and its BF16 MMA-input boundary,
                # without a shared FP32 store/reload between those operations.
                weighted = probability
                if const_expr(self.kv_fp8):
                    weighted = probability * (v_weight_0 if half == 0 else v_weight_1)
                probabilities[((head, row % 16), 0, row // 16)] = self.dtype(weighted)
                total += probability
            total = cute.arch.warp_reduction(total, operator.add)
            cute.arch.sync_warp()
            if lane == 0:
                head_max[head] = maximum
                head_scale[head] = alpha
                head_sum[head] = head_sum[head] * alpha + total

    def _run_impl(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mKScale: Optional[cute.Tensor],
        mVScale: Optional[cute.Tensor],
        mPackedIndices: cute.Tensor,
        mCount: cute.Tensor,
        mKVSeqLen: cute.Tensor,
        mQueryStartLoc: cute.Tensor,
        mValidRows: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        n_split4: Int32,
        n_split2: Int32,
        stream: cuda.CUstream,
    ) -> None:
        softmax_scale_log2 = Float32(softmax_scale * math.log2(math.e))

        def _assume_stride_divisible(stride_elem, *, element_width: int):
            if const_expr(isinstance(stride_elem, int)):
                return stride_elem
            return cute.assume(stride_elem, divby=128 // element_width)

        def new_stride(t):
            return (
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

        mKCache = cute.make_tensor(
            mKCache.iterator, cute.select(mKCache.layout, [1, 3, 2, 0])
        )
        mVtCache = cute.make_tensor(
            mVCache.iterator, cute.select(mVCache.layout, [3, 1, 2, 0])
        )
        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, [0, 1]))

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        sK_layout, sQ_layout, sV_layout, sP_layout = self._get_smem_layouts()
        SharedStorage = self._get_shared_storage_cls()

        self.kernel(
            mQ,
            mKCache,
            mVtCache,
            mKScale,
            mVScale,
            mPackedIndices,
            mCount,
            mKVSeqLen,
            mQueryStartLoc,
            mValidRows,
            mO,
            mLSE,
            softmax_scale_log2,
            tiled_mma_qk,
            tiled_mma_pv,
            sK_layout,
            sQ_layout,
            sV_layout,
            sP_layout,
            SharedStorage,
            n_split4,
            n_split2,
        ).launch(
            grid=(cute.size(mO.shape[0]), 1, 1),
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mKScale: Optional[cute.Tensor],
        mVScale: Optional[cute.Tensor],
        mPackedIndices: cute.Tensor,
        mCount: cute.Tensor,
        mKVSeqLen: cute.Tensor,
        mQueryStartLoc: cute.Tensor,
        mValidRows: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        n_split4: Int32,
        n_split2: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self._run_impl(
            mQ,
            mKCache,
            mVCache,
            mKScale,
            mVScale,
            mPackedIndices,
            mCount,
            mKVSeqLen,
            mQueryStartLoc,
            mValidRows,
            mO,
            mLSE,
            softmax_scale,
            n_split4,
            n_split2,
            stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVtCache: cute.Tensor,
        mKScale: Optional[cute.Tensor],
        mVScale: Optional[cute.Tensor],
        mPackedIndices: cute.Tensor,
        mCount: cute.Tensor,
        mKVSeqLen: cute.Tensor,
        mQueryStartLoc: cute.Tensor,
        mValidRows: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sK_layout,
        sQ_layout,
        sV_layout,
        sP_layout,
        SharedStorage: cutlass.Constexpr,
        n_split4: Int32,
        n_split2: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        launch_idx, _, _ = cute.arch.block_idx()
        work_idx = launch_idx
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        q_pos = Int32(0)
        split_idx = Int32(0)
        split_parts = Int32(1)
        if const_expr(not self.dynamic_split):
            split4_work = n_split4 * Int32(4)
            split2_work = n_split2 * Int32(2)
            if work_idx < split4_work:
                q_pos = work_idx // Int32(4)
                split_idx = work_idx - q_pos * Int32(4)
                split_parts = Int32(4)
            elif work_idx < split4_work + split2_work:
                local_work = work_idx - split4_work
                local_q = local_work // Int32(2)
                q_pos = n_split4 + local_q
                split_idx = local_work - local_q * Int32(2)
                split_parts = Int32(2)
            else:
                local_work = work_idx - split4_work - split2_work
                q_pos = n_split4 + n_split2 + local_work
                split_idx = Int32(0)

        producer_warps = Int32(self.num_producer_warps)
        producer_k_warps = Int32(self.producer_k_warps)
        producer_v_warps = Int32(self.producer_v_warps)
        producer_k_threads = producer_k_warps * Int32(32)
        producer_v_threads = producer_v_warps * Int32(32)
        producer_threads = producer_warps * Int32(32)
        is_producer = warp_idx < producer_warps
        is_compute = warp_idx >= producer_warps
        idx_in_wg = tidx - producer_threads
        warp_in_wg = idx_in_wg // Int32(32)

        block_n_i32 = Int32(self.block_n)
        stage_count = Int32(self.num_stages)
        windows_per_tile_i32 = Int32(self.windows_per_tile)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_free = storage.mbar_ptr_free.data_ptr()
        mbar_k_stage = storage.mbar_ptr_k_stage.data_ptr()
        mbar_v_stage = storage.mbar_ptr_v_stage.data_ptr()
        mbar_metadata = storage.mbar_ptr_metadata.data_ptr()

        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sVTileMax = storage.sVTileMax.get_tensor(
            cute.make_layout((self.logical_q_heads,), stride=(1,))
        )
        sKScaleStage = storage.sKScaleStage.get_tensor(
            cute.make_layout((self.num_stages * self.block_n if self.kv_fp8 else 1,))
        )
        sVScaleStage = storage.sVScaleStage.get_tensor(
            cute.make_layout((self.num_stages * self.block_n if self.kv_fp8 else 1,))
        )
        sPacked = storage.sPacked.get_tensor(
            cute.make_layout(
                (self.windows_per_tile, self.num_stages),
                stride=(1, self.windows_per_tile),
            )
        )
        sCount = storage.sCount.get_tensor(cute.make_layout((1,), stride=(1,)))
        sKVSeqLen = storage.sKVSeqLen.get_tensor(cute.make_layout((1,), stride=(1,)))
        sSchedule = storage.sSchedule.get_tensor(cute.make_layout((4,), stride=(1,)))
        sHeadMax = storage.sHeadMax.get_tensor(
            cute.make_layout((self.logical_q_heads,), stride=(1,))
        )
        sHeadSum = storage.sHeadSum.get_tensor(
            cute.make_layout((self.logical_q_heads,), stride=(1,))
        )
        sHeadScale = storage.sHeadScale.get_tensor(
            cute.make_layout((self.logical_q_heads,), stride=(1,))
        )
        sScores = storage.sScores.get_tensor(
            cute.make_layout(
                (self.block_n, self.MMA_Q_HEADS), stride=(1, self.score_stride)
            )
        )
        mbar_mma = storage.mbarMma.data_ptr()
        tmem_base = storage.tmemBase.data_ptr()

        if const_expr(self.dynamic_split):
            if warp_idx == Int32(0):
                self._schedule_dynamic(
                    mCount,
                    mQueryStartLoc,
                    Int32(mValidRows[0]),
                    launch_idx,
                    sSchedule,
                    lane,
                )
            cute.arch.sync_threads()
            q_pos = sSchedule[0]
            split_idx = sSchedule[1]
            split_parts = sSchedule[2]
            work_idx = sSchedule[3]

        # The packed metadata has one extra current-region window, so 513 is
        # common. Proportional boundaries keep every split disjoint while
        # preserving that odd tail window.
        split_base_win_i32 = (Int32(self.topk_windows) * split_idx) // split_parts
        split_end_win_i32 = (
            Int32(self.topk_windows) * (split_idx + Int32(1))
        ) // split_parts
        work_windows_i32 = split_end_win_i32 - split_base_win_i32
        work_token_cap_i32 = work_windows_i32 * Int32(8)

        # Only CTAs owning a partial-output slot need TMEM or compute warps.
        # Dynamic Graph capacity includes idle CTAs with work_idx=-1; static
        # padded queries still own slots and retain their zero/-inf stores.
        work_valid = work_idx >= Int32(0) and work_idx < Int32(cute.size(mO.shape[0]))
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_mma, Int32(1))
        if work_valid and warp_idx == producer_warps:
            cute.arch.alloc_tmem(64, tmem_base, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        if tidx < Int32(self.num_stages):
            cute.arch.mbarrier_init(mbar_free + tidx, Int32(128))
            cute.arch.mbarrier_init(mbar_metadata + tidx, producer_threads)
            cute.arch.mbarrier_init(mbar_k_stage + tidx, producer_k_threads + 1)
            cute.arch.mbarrier_init(mbar_v_stage + tidx, producer_v_threads + 1)
        if tidx < Int32(self.logical_q_heads):
            sHeadMax[tidx] = -Float32.inf
            sHeadSum[tidx] = Float32(0)
            sHeadScale[tidx] = Float32(1)
        cute.arch.mbarrier_init_fence()

        q_valid = (
            q_pos >= Int32(0)
            and q_pos < Int32(cute.size(mQ.shape[0]))
            and q_pos < Int32(mValidRows[0])
        )
        if tidx == 0:
            total_count = (mCount[q_pos] * Int32(8)) if q_valid else Int32(0)
            split_token_start = split_base_win_i32 * Int32(8)
            count = total_count - split_token_start
            kv_seq_len = mKVSeqLen[q_pos] if q_valid else Int32(0)
            if count < Int32(0):
                count = Int32(0)
            if count > work_token_cap_i32:
                count = work_token_cap_i32
            if kv_seq_len < Int32(0):
                kv_seq_len = Int32(0)
            sCount[0] = count
            sKVSeqLen[0] = kv_seq_len
        cute.arch.sync_threads()
        count_slot = sCount[0]
        kv_seq_len_slot = sKVSeqLen[0]
        slot_active = q_valid and work_valid and (count_slot > Int32(0))
        work_split_tiles_i32 = (count_slot + block_n_i32 - Int32(1)) // block_n_i32
        num_phys_pages = Int32(cute.size(mKCache.shape[3]))
        max_phys_regions = num_phys_pages * Int32(2)

        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        producer_subgroup_lanes = Int32(self.producer_subgroup_lanes)
        is_producer_k = is_producer and (warp_idx < producer_k_warps)
        is_producer_v = is_producer and (warp_idx >= producer_k_warps)
        tidx_k = tidx
        tidx_v = tidx - producer_k_threads
        producer_k_groups = Int32(self.producer_k_groups)
        producer_v_groups = Int32(self.producer_v_groups)

        if is_producer:
            cute.arch.warpgroup_reg_dealloc(
                self.PRODUCER_REGS_FP8 if self.kv_fp8 else self.PRODUCER_REGS
            )
            for local_idx in cutlass.range(work_split_tiles_i32, unroll=1):
                stage = local_idx % stage_count
                if local_idx >= stage_count:
                    wait_phase = (
                        (local_idx - stage_count - stage) // stage_count
                    ) & Int32(1)
                    cute.arch.mbarrier_wait(mbar_free + stage, wait_phase)

                if warp_idx == Int32(0) and lane < windows_per_tile_i32:
                    local_win = local_idx * windows_per_tile_i32 + lane
                    base_win = split_base_win_i32 + local_win
                    packed_meta = cutlass.Int64(-1)
                    if (
                        base_win < Int32(self.topk_windows)
                        and (local_win * Int32(8)) < count_slot
                    ):
                        raw = mPackedIndices[q_pos, base_win]
                        phys_region_meta = Int32(
                            (raw >> cutlass.Int64(32)) & cutlass.Int64(0xFFFF_FFFF)
                        )
                        if (
                            raw >= cutlass.Int64(0)
                            and phys_region_meta >= Int32(0)
                            and phys_region_meta < max_phys_regions
                        ):
                            packed_meta = raw
                    sPacked[lane, stage] = packed_meta
                cute.arch.barrier(barrier_id=5, number_of_threads=producer_threads)
                # Publish ordinary metadata stores independently of cp.async.
                cute.arch.mbarrier_arrive(mbar_metadata + stage)

                if is_producer_k:
                    group_idx_k = tidx_k // producer_subgroup_lanes
                    lane_in_group_k = tidx_k - group_idx_k * producer_subgroup_lanes
                    col_base_k = lane_in_group_k * Int32(8)
                    base_tok_idx_k = local_idx * block_n_i32
                    for local_row in cutlass.range_constexpr(self.max_rows_per_k_group):
                        row = group_idx_k + Int32(local_row) * producer_k_groups
                        if row < block_n_i32:
                            win = row >> Int32(3)
                            packed = sPacked[win, stage]
                            phys_region = Int32(
                                (packed >> cutlass.Int64(32))
                                & cutlass.Int64(0xFFFF_FFFF)
                            )
                            valid = (
                                (base_tok_idx_k + row) < count_slot
                                and packed >= cutlass.Int64(0)
                                and phys_region >= Int32(0)
                                and phys_region < max_phys_regions
                                and Int32(packed & cutlass.Int64(0xFFFF_FFFF))
                                + (row & Int32(7))
                                < kv_seq_len_slot
                            )
                            phys_block = phys_region >> Int32(1)
                            page_off = (phys_region & Int32(1)) * Int32(8) + (
                                row & Int32(7)
                            )
                            for slab in cutlass.range_constexpr(self.head_dim_slabs):
                                col_base = col_base_k + Int32(slab * 64)
                                s_ptr = elem_pointer(
                                    sK, ((row, col_base % 16), 0, col_base // 16, stage)
                                )
                                s_ptr = cute.make_ptr(
                                    self.dtype,
                                    s_ptr.toint(),
                                    sK.memspace,
                                    assumed_align=self.REQUIRED_POINTER_ALIGN,
                                )
                                s_ptr = cute.recast_ptr(s_ptr, sK_layout.inner)
                                s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                                if valid:
                                    g_ptr = elem_pointer_i64_offset(
                                        mKCache,
                                        (page_off, col_base, Int32(0), phys_block),
                                    )
                                    if const_expr(self.kv_fp8):
                                        stage_tok = stage * block_n_i32 + row
                                        vector_copy_with_explicit_width(
                                            cute.make_tensor(
                                                cute.make_ptr(
                                                    mKCache.element_type,
                                                    g_ptr.toint(),
                                                    mKCache.memspace,
                                                    assumed_align=8,
                                                ),
                                                vec_layout_8,
                                            ),
                                            cute.make_tensor(
                                                cute.make_ptr(
                                                    mKCache.element_type,
                                                    elem_pointer(
                                                        sK, ((0, 0), 0, 0, stage)
                                                    ).toint()
                                                    + (
                                                        (col_base // Int32(16))
                                                        * block_n_i32
                                                        + row
                                                    )
                                                    * Int32(16)
                                                    + col_base % Int32(16),
                                                    sK.memspace,
                                                    assumed_align=8,
                                                ),
                                                vec_layout_8,
                                            ),
                                            num_copy_elems=8,
                                            is_async=True,
                                        )
                                        if const_expr(slab == 0):
                                            if lane_in_group_k == Int32(0):
                                                vector_copy_with_explicit_width(
                                                    cute.make_tensor(
                                                        cute.make_ptr(
                                                            Float32,
                                                            elem_pointer(
                                                                mKScale,
                                                                (
                                                                    phys_block
                                                                    * Int32(16)
                                                                    + page_off,
                                                                ),
                                                            ).toint(),
                                                            mKScale.memspace,
                                                            assumed_align=4,
                                                        ),
                                                        cute.make_layout((1,)),
                                                    ),
                                                    cute.make_tensor(
                                                        cute.make_ptr(
                                                            Float32,
                                                            elem_pointer(
                                                                sKScaleStage,
                                                                (stage_tok,),
                                                            ).toint(),
                                                            sKScaleStage.memspace,
                                                            assumed_align=4,
                                                        ),
                                                        cute.make_layout((1,)),
                                                    ),
                                                    num_copy_elems=1,
                                                    is_async=True,
                                                )
                                    else:
                                        g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                                        vector_copy_with_explicit_width(
                                            g_vec,
                                            s_vec,
                                            num_copy_elems=8,
                                            is_async=True,
                                        )
                                else:
                                    if const_expr(self.kv_fp8):
                                        stage_tok = stage * block_n_i32 + row
                                        zero = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Int64,
                                                elem_pointer(
                                                    sK, ((0, 0), 0, 0, stage)
                                                ).toint()
                                                + (
                                                    (col_base // Int32(16))
                                                    * block_n_i32
                                                    + row
                                                )
                                                * Int32(16)
                                                + col_base % Int32(16),
                                                sK.memspace,
                                                assumed_align=8,
                                            ),
                                            cute.make_layout((1,)),
                                        )
                                        zero[0] = cutlass.Int64(0)
                                        if const_expr(slab == 0):
                                            if lane_in_group_k == Int32(0):
                                                sKScaleStage[stage_tok] = Float32(0)
                                    else:
                                        for vi in cutlass.range_constexpr(8):
                                            s_vec[vi] = self.dtype(0)
                    if const_expr(self.kv_fp8):
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.barrier(
                            barrier_id=6, number_of_threads=producer_k_threads
                        )
                        self._convert_k_stage(
                            sK,
                            sK_layout,
                            stage,
                            tidx_k,
                            self.producer_k_warps * 32,
                            6,
                        )
                        cute.arch.mbarrier_arrive(mbar_k_stage + stage)
                    else:
                        cute.arch.cp_async_commit_group()
                        cute.arch.barrier(
                            barrier_id=6, number_of_threads=producer_k_threads
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_k_stage + stage)
                    if tidx_k == 0:
                        cute.arch.mbarrier_arrive(mbar_k_stage + stage)

                if is_producer_v:
                    group_idx_v = tidx_v // producer_subgroup_lanes
                    lane_in_group_v = tidx_v - group_idx_v * producer_subgroup_lanes
                    col_base_v = lane_in_group_v * Int32(8)
                    base_tok_idx_v = local_idx * block_n_i32
                    for local_row in cutlass.range_constexpr(self.max_rows_per_v_group):
                        row = group_idx_v + Int32(local_row) * producer_v_groups
                        if row < block_n_i32:
                            win = row >> Int32(3)
                            packed = sPacked[win, stage]
                            phys_region = Int32(
                                (packed >> cutlass.Int64(32))
                                & cutlass.Int64(0xFFFF_FFFF)
                            )
                            valid = (
                                (base_tok_idx_v + row) < count_slot
                                and packed >= cutlass.Int64(0)
                                and phys_region >= Int32(0)
                                and phys_region < max_phys_regions
                                # Mask before loading V: 0 * NaN in the PV MMA
                                # would otherwise poison a valid decode row.
                                and Int32(packed & cutlass.Int64(0xFFFF_FFFF))
                                + (row & Int32(7))
                                < kv_seq_len_slot
                            )
                            phys_block = phys_region >> Int32(1)
                            page_off = (phys_region & Int32(1)) * Int32(8) + (
                                row & Int32(7)
                            )
                            for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                                col_base = col_base_v + Int32(slab * 64)
                                s_stage = stage + stage_count * Int32(slab)
                                s_ptr = elem_pointer(
                                    sV, ((col_base_v, row % 16), 0, row // 16, s_stage)
                                )
                                s_ptr = cute.make_ptr(
                                    self.dtype,
                                    s_ptr.toint(),
                                    sV.memspace,
                                    assumed_align=self.REQUIRED_POINTER_ALIGN,
                                )
                                s_ptr = cute.recast_ptr(s_ptr, sV_layout.inner)
                                s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                                if valid:
                                    g_ptr = elem_pointer_i64_offset(
                                        mVtCache,
                                        (col_base, page_off, Int32(0), phys_block),
                                    )
                                    if const_expr(self.kv_fp8):
                                        stage_tok = stage * block_n_i32 + row
                                        vector_copy_with_explicit_width(
                                            cute.make_tensor(
                                                cute.make_ptr(
                                                    mVtCache.element_type,
                                                    g_ptr.toint(),
                                                    mVtCache.memspace,
                                                    assumed_align=8,
                                                ),
                                                vec_layout_8,
                                            ),
                                            cute.make_tensor(
                                                cute.make_ptr(
                                                    mVtCache.element_type,
                                                    elem_pointer(
                                                        sV,
                                                        (
                                                            (0, 0),
                                                            0,
                                                            0,
                                                            stage
                                                            + stage_count * Int32(slab),
                                                        ),
                                                    ).toint()
                                                    + row * Int32(64)
                                                    + col_base_v,
                                                    sV.memspace,
                                                    assumed_align=8,
                                                ),
                                                vec_layout_8,
                                            ),
                                            num_copy_elems=8,
                                            is_async=True,
                                        )
                                        if const_expr(slab == 0):
                                            if lane_in_group_v == Int32(0):
                                                vector_copy_with_explicit_width(
                                                    cute.make_tensor(
                                                        cute.make_ptr(
                                                            Float32,
                                                            elem_pointer(
                                                                mVScale,
                                                                (
                                                                    phys_block
                                                                    * Int32(16)
                                                                    + page_off,
                                                                ),
                                                            ).toint(),
                                                            mVScale.memspace,
                                                            assumed_align=4,
                                                        ),
                                                        cute.make_layout((1,)),
                                                    ),
                                                    cute.make_tensor(
                                                        cute.make_ptr(
                                                            Float32,
                                                            elem_pointer(
                                                                sVScaleStage,
                                                                (stage_tok,),
                                                            ).toint(),
                                                            sVScaleStage.memspace,
                                                            assumed_align=4,
                                                        ),
                                                        cute.make_layout((1,)),
                                                    ),
                                                    num_copy_elems=1,
                                                    is_async=True,
                                                )
                                    else:
                                        g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                                        vector_copy_with_explicit_width(
                                            g_vec,
                                            s_vec,
                                            num_copy_elems=8,
                                            is_async=True,
                                        )
                                else:
                                    if const_expr(self.kv_fp8):
                                        stage_tok = stage * block_n_i32 + row
                                        zero = cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Int64,
                                                elem_pointer(
                                                    sV,
                                                    (
                                                        (0, 0),
                                                        0,
                                                        0,
                                                        stage
                                                        + stage_count * Int32(slab),
                                                    ),
                                                ).toint()
                                                + row * Int32(64)
                                                + col_base_v,
                                                sV.memspace,
                                                assumed_align=8,
                                            ),
                                            cute.make_layout((1,)),
                                        )
                                        zero[0] = cutlass.Int64(0)
                                        if const_expr(slab == 0):
                                            if lane_in_group_v == Int32(0):
                                                sVScaleStage[stage_tok] = Float32(0)
                                    else:
                                        for vi in cutlass.range_constexpr(8):
                                            s_vec[vi] = self.dtype(0)
                    if const_expr(self.kv_fp8):
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.barrier(
                            barrier_id=7, number_of_threads=producer_v_threads
                        )
                        cute.arch.mbarrier_arrive(mbar_v_stage + stage)
                    else:
                        cute.arch.cp_async_commit_group()
                        cute.arch.barrier(
                            barrier_id=7, number_of_threads=producer_v_threads
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_v_stage + stage)
                    if tidx_v == 0:
                        cute.arch.mbarrier_arrive(mbar_v_stage + stage)

        if is_compute and work_valid:
            cute.arch.warpgroup_reg_alloc(
                self.MATH_REGS_FP8 if self.kv_fp8 else self.MATH_REGS
            )
            q_pos_safe = cutlass.select_(q_valid, q_pos, Int32(0))
            q_vec_count = (self.MMA_Q_HEADS * self.HEAD_DIM) // 8
            sQ_tile = sQ[None, None, None, Int32(0)]
            for vec_idx in cutlass.range(idx_in_wg, q_vec_count, 128, unroll=1):
                elem_idx = vec_idx * Int32(8)
                head = elem_idx // self.HEAD_DIM
                d = elem_idx - head * self.HEAD_DIM
                s_ptr = elem_pointer(sQ_tile, ((head, d % 16), 0, d // 16))
                s_ptr = cute.make_ptr(
                    self.dtype,
                    s_ptr.toint(),
                    sQ_tile.memspace,
                    assumed_align=self.REQUIRED_POINTER_ALIGN,
                )
                s_ptr = cute.recast_ptr(s_ptr, sQ_layout.inner)
                s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                head_valid = head < Int32(self.logical_q_heads)
                head_safe = head if head_valid else Int32(0)
                g_ptr = elem_pointer_i64_offset(mQ, (q_pos_safe, head_safe, d))
                g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                if q_valid and head_valid:
                    if const_expr(self.kv_fp8):
                        s_vec.store(g_vec.load().to(self.dtype))
                    else:
                        vector_copy_with_explicit_width(g_vec, s_vec, num_copy_elems=8)
                else:
                    for vi in cutlass.range_constexpr(8):
                        s_vec[vi] = self.dtype(0)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_base)
            thr_mma = tiled_mma_qk.get_slice(0)
            shape = thr_mma.partition_shape_C((self.block_n, self.MMA_Q_HEADS))
            acc_fake = thr_mma.make_fragment_C(shape)
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            acc_mn = acc[((None, None), 0, 0)]
            copy_atom = cute.make_copy_atom(tcgen05.Ld16x256bOp(), Float32)
            t2r = tcgen05.make_tmem_copy(copy_atom, acc_mn)
            thr_copy = t2r.get_slice(idx_in_wg)
            coords = thr_copy.partition_D(
                cute.make_identity_tensor((self.block_n, self.MMA_Q_HEADS))
            )
            rScore = cute.make_rmem_tensor(coords.shape, Float32)
            rPV = cute.make_rmem_tensor(
                cute.append(coords.shape, self.head_dim_v_slabs), Float32
            )
            rOut = cute.make_rmem_tensor(
                cute.append(coords.shape, self.head_dim_v_slabs), Float32
            )
            rOut.fill(Float32(0))
            acc_columns = const_expr(cute.size(acc_mn.shape[1]))
            assert self.head_dim_v_slabs * acc_columns <= 64
            fQ = tiled_mma_qk.make_fragment_B(sQ_tile)
            fK = tiled_mma_qk.make_fragment_A(sK)
            fV = tiled_mma_pv.make_fragment_A(sV)
            sP_tile = sP[None, None, None, Int32(0)]
            fP = tiled_mma_pv.make_fragment_B(sP_tile)
            p_coords = thr_mma.partition_B(
                cute.make_identity_tensor((self.MMA_Q_HEADS, self.block_n))
            )
            # Softmax writes only logical heads. Padded MMA heads stay zero for
            # every tile; publish these stores with the first probability tile.
            for i in cutlass.range(idx_in_wg, cute.size(p_coords), 128, unroll=1):
                head, _ = p_coords[i]
                if head >= self.logical_q_heads:
                    sP_tile[i] = self.dtype(0)

            mma_phase = Int32(0)

            for local_idx in cutlass.range(work_split_tiles_i32, unroll=1):
                stage = local_idx % stage_count
                wait_phase = ((local_idx - stage) // stage_count) & Int32(1)
                k_start = local_idx * block_n_i32
                cute.arch.mbarrier_wait(mbar_metadata + stage, wait_phase)
                cute.arch.mbarrier_wait(mbar_k_stage + stage, wait_phase)
                cute.arch.fence_view_async_shared()
                if warp_in_wg == 0:
                    tcgen_after_sync()
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(self.HEAD_DIM // 16):
                        cute.gemm(
                            tiled_mma_qk,
                            acc,
                            fK[None, None, kk, stage],
                            fQ[None, None, kk],
                            acc,
                        )
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_mma)
                if const_expr(self.kv_fp8):
                    cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                    self._convert_v_stage(
                        sV, sV_layout, stage, stage_count, idx_in_wg, 128, 8
                    )
                cute.arch.mbarrier_wait(mbar_mma, mma_phase)
                mma_phase = mma_phase ^ 1
                tcgen_after_sync()
                cute.copy(t2r, thr_copy.partition_S(acc_mn), rScore)
                cute.arch.fence_view_async_tmem_load()
                tcgen_before_sync()
                for i in cutlass.range_constexpr(cute.size(rScore)):
                    tok, head = coords[i]
                    packed = sPacked[tok >> Int32(3), stage]
                    physical = Int32(
                        (packed >> cutlass.Int64(32)) & cutlass.Int64(0xFFFFFFFF)
                    )
                    logical = Int32(packed & cutlass.Int64(0xFFFFFFFF)) + (
                        tok & Int32(7)
                    )
                    value = -Float32.inf
                    if (
                        slot_active
                        and head < self.logical_q_heads
                        and k_start + tok < count_slot
                        and packed >= cutlass.Int64(0)
                        and physical >= 0
                        and physical < max_phys_regions
                        and logical < kv_seq_len_slot
                    ):
                        value = rScore[i]
                        if const_expr(self.kv_fp8):
                            value = value * sKScaleStage[stage * block_n_i32 + tok]
                    sScores[tok, head] = value
                if const_expr(self.kv_fp8):
                    if warp_in_wg == 0:
                        v_max = cute.arch.fmax(
                            sVScaleStage[stage * block_n_i32 + lane],
                            sVScaleStage[stage * block_n_i32 + lane + Int32(32)],
                        )
                        v_max = cute.arch.warp_redux_sync(v_max, "max")
                        if lane == 0:
                            v_max = v_max if v_max > Float32(0) else Float32(1)
                            sVTileMax[0] = v_max
                            sVTileMax[1] = cute.arch.rcp_approx(v_max)
                cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
                self._softmax_heads(
                    sScores,
                    sP_tile,
                    sHeadMax,
                    sHeadSum,
                    sHeadScale,
                    softmax_scale_log2,
                    warp_in_wg,
                    lane,
                    sVScaleStage,
                    stage * block_n_i32,
                    sVTileMax,
                )
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
                cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                if warp_in_wg == 0:
                    tcgen_after_sync()
                    for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                        acc_pv = cute.make_tensor(
                            tmem_ptr + slab * acc_columns, acc.layout
                        )
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range_constexpr(self.block_n // 16):
                            cute.gemm(
                                tiled_mma_pv,
                                acc_pv,
                                fV[None, None, kk, stage + stage_count * Int32(slab)],
                                fP[None, None, kk],
                                acc_pv,
                            )
                            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_mma)
                cute.arch.mbarrier_wait(mbar_mma, mma_phase)
                mma_phase = mma_phase ^ 1
                tcgen_after_sync()
                for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                    acc_pv_mn = cute.make_tensor(
                        tmem_ptr + slab * acc_columns, acc_mn.layout
                    )
                    cute.copy(
                        t2r,
                        thr_copy.partition_S(acc_pv_mn),
                        rPV[None, None, None, slab],
                    )
                cute.arch.fence_view_async_tmem_load()
                tcgen_before_sync()
                tile_max = Float32(1)
                if const_expr(self.kv_fp8):
                    tile_max = sVTileMax[0]
                for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                    out_slab = rOut[None, None, None, slab]
                    pv_slab = rPV[None, None, None, slab]
                    for i in cutlass.range_constexpr(cute.size(rScore)):
                        _, head = coords[i]
                        if head < self.logical_q_heads:
                            if const_expr(self.kv_fp8):
                                out_slab[i] = (
                                    out_slab[i] * sHeadScale[head]
                                    + pv_slab[i] * tile_max
                                )
                            else:
                                out_slab[i] = (
                                    out_slab[i] * sHeadScale[head] + pv_slab[i]
                                )
                # Every consumer has retired its TMEM loads before QK reuses
                # TMEM and before producers overwrite this shared K/V stage.
                cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
                # Each consumer releases its own shared metadata reads before
                # every producer acquires the stage for reuse.
                if local_idx + stage_count < work_split_tiles_i32:
                    cute.arch.mbarrier_arrive(mbar_free + stage)

            # Static Graph padding owns real output slots and must clear them.
            # Inactive dynamic CTAs have work_idx=-1 and own no output slot.
            if work_valid:
                if warp_in_wg == 0 and lane < self.logical_q_heads:
                    denom = sHeadSum[lane]
                    valid_sum = denom != Float32(0) and denom == denom
                    sHeadScale[lane] = cute.arch.rcp_approx(
                        denom if valid_sum else Float32(1)
                    )
                    if const_expr(mLSE is not None):
                        lse = -Float32.inf
                        if valid_sum:
                            lse = (
                                sHeadMax[lane] * softmax_scale_log2 + utils.log2f(denom)
                            ) * math.log(2.0)
                        mLSE[work_idx, lane] = lse
                cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
                for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                    out_slab = rOut[None, None, None, slab]
                    for i in cutlass.range_constexpr(cute.size(rScore)):
                        dim, head = coords[i]
                        if head < self.logical_q_heads:
                            mO[work_idx, head, dim + Int32(slab * 64)] = (
                                mO.element_type(out_slab[i] * sHeadScale[head])
                            )
            cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
            if warp_in_wg == 0:
                tcgen_after_sync()
                cute.arch.dealloc_tmem(tmem_ptr, 64, is_two_cta=False)
