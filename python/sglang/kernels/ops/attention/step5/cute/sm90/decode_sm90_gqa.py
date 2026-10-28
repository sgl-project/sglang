# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 GQA fixed-spec sparse decode on paged KV with swapAB WGMMA."""

import functools
import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, const_expr
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
from sglang.kernels.ops.attention.step5.cute.sm90.utils import elem_pointer


def _gqa_cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


@functools.cache
def _get_decode_kernel_variant(
    dtype,
    logical_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int,
    num_threads: int,
    topk_windows: int,
    mtp_q_len: int,
    batch_invariant: bool,
    producer_k_warps: Optional[int],
    split_capacity: int,
    sm_count: Optional[int],
) -> "_TokenWiseSparseDecodeWGMMA64GQA":
    return _TokenWiseSparseDecodeWGMMA64GQA(
        dtype=dtype,
        logical_q_heads=int(logical_q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        block_n=int(block_n),
        num_threads=int(num_threads),
        topk_windows=int(topk_windows),
        mtp_q_len=int(mtp_q_len),
        batch_invariant=bool(batch_invariant),
        producer_k_warps=producer_k_warps,
        split_capacity=int(split_capacity),
        sm_count=None if sm_count is None else int(sm_count),
    )


@cached_compile_function
def _get_compiled_decode_kernel(
    dtype,
    logical_q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int,
    num_threads: int,
    topk_windows: int,
    mtp_q_len: int,
    batch_invariant: bool,
    producer_k_warps: Optional[int],
    split_capacity: int,
    sm_count: Optional[int],
    q_signature: tuple[object, ...],
    k_signature: tuple[object, ...],
    v_signature: tuple[object, ...],
    packed_signature: tuple[object, ...],
    counts_signature: tuple[object, ...],
    seq_lens_signature: tuple[object, ...],
    query_start_loc_signature: tuple[object, ...],
    valid_rows_signature: tuple[object, ...],
    out_signature: tuple[object, ...],
    lse_signature: tuple[object, ...] | None,
    q_align: int,
    k_align: int,
    v_align: int,
    packed_align: int,
    counts_align: int,
    seq_lens_align: int,
    out_align: int,
    lse_align: int | None,
    softmax_scale: float,
    device_key: tuple[str, int | None],
):
    device = utils.device_from_cache_key(device_key)
    kernel_impl = _get_decode_kernel_variant(
        dtype,
        int(logical_q_heads),
        int(head_dim),
        int(head_dim_v),
        int(block_n),
        int(num_threads),
        int(topk_windows),
        int(mtp_q_len),
        bool(batch_invariant),
        producer_k_warps,
        int(split_capacity),
        None if sm_count is None else int(sm_count),
    )
    q = utils.placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=1
    )
    k_cache = utils.placeholder_from_signature(
        k_signature, device=device, dynamic_shape_fill=1
    )
    v_cache = utils.placeholder_from_signature(
        v_signature, device=device, dynamic_shape_fill=1
    )
    packed = utils.placeholder_from_signature(
        packed_signature, device=device, dynamic_shape_fill=1
    )
    counts = utils.placeholder_from_signature(
        counts_signature, device=device, dynamic_shape_fill=1
    )
    seq_lens = utils.placeholder_from_signature(
        seq_lens_signature, device=device, dynamic_shape_fill=1
    )
    query_start_loc = utils.placeholder_from_signature(
        query_start_loc_signature, device=device, dynamic_shape_fill=1
    )
    valid_rows = utils.placeholder_from_signature(
        valid_rows_signature, device=device, dynamic_shape_fill=1
    )
    out = utils.placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=1
    )
    lse = (
        utils.placeholder_from_signature(
            lse_signature, device=device, dynamic_shape_fill=1
        )
        if lse_signature is not None
        else None
    )
    fQ = utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=q_align, dynamic_shape_dims=(0,)
    )
    fK = utils.make_fake_tensor_like_with_dynamic_dim(
        k_cache, alignment=k_align, dynamic_shape_dims=(0,)
    )
    fV = utils.make_fake_tensor_like_with_dynamic_dim(
        v_cache, alignment=v_align, dynamic_shape_dims=(0,)
    )
    fP = utils.make_fake_tensor_like_with_dynamic_dim(
        packed, alignment=packed_align, dynamic_shape_dims=(0,)
    )
    fC = utils.make_fake_tensor_like_with_dynamic_dim(
        counts, alignment=counts_align, dynamic_shape_dims=(0,)
    )
    fS = utils.make_fake_tensor_like_with_dynamic_dim(
        seq_lens, alignment=seq_lens_align, dynamic_shape_dims=(0,)
    )
    fQueryStartLoc = utils.make_fake_tensor_like_with_dynamic_dim(
        query_start_loc, alignment=4, dynamic_shape_dims=(0,)
    )
    fValidRows = utils.make_fake_tensor_like_with_dynamic_dim(
        valid_rows, alignment=4, dynamic_shape_dims=(0,)
    )
    fO = utils.make_fake_tensor_like_with_dynamic_dim(
        out, alignment=out_align, dynamic_shape_dims=(0,)
    )
    fLSE = (
        utils.make_fake_tensor_like_with_dynamic_dim(
            lse, alignment=lse_align, dynamic_shape_dims=(0,)
        )
        if lse is not None
        else None
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _gqa_cute_compile(
        kernel_impl,
        fQ,
        fK,
        fV,
        None,
        None,
        fP,
        fC,
        fS,
        fQueryStartLoc,
        fValidRows,
        fO,
        fLSE,
        cutlass.Float32(softmax_scale),
        Int32(0),
        Int32(0),
        stream_fake,
    )


def _valid_variable_split_max(value: int) -> int:
    value = int(value)
    if value not in (1, 2, 4, 16):
        raise ValueError(
            f"variable_split_max must be one of {{1, 2, 4, 16}}, got {value}"
        )
    return value


def _decode_compile_tensor_signatures(
    q,
    k_cache,
    v_cache,
    packed,
    counts,
    seq_lens,
    query_start_loc,
    valid_rows,
    out,
    lse,
):
    dynamic_dim0 = {"dynamic_shape_dims": (0,)}
    return (
        utils.tensor_signature_dynamic(q, **dynamic_dim0),
        utils.tensor_signature_dynamic(k_cache, **dynamic_dim0),
        utils.tensor_signature_dynamic(v_cache, **dynamic_dim0),
        utils.tensor_signature_dynamic(packed, **dynamic_dim0),
        utils.tensor_signature_dynamic(counts, **dynamic_dim0),
        utils.tensor_signature_dynamic(seq_lens, **dynamic_dim0),
        utils.tensor_signature_dynamic(query_start_loc, **dynamic_dim0),
        utils.tensor_signature_dynamic(valid_rows, **dynamic_dim0),
        utils.tensor_signature_dynamic(out, **dynamic_dim0),
        None if lse is None else utils.tensor_signature_dynamic(lse, **dynamic_dim0),
    )


def _add_periodic_work(
    sm_work: list[int], *, start: int, tasks: int, weight: int
) -> None:
    if tasks <= 0:
        return
    sm_count = len(sm_work)
    full, rem = divmod(int(tasks), sm_count)
    if full:
        add = full * int(weight)
        for idx in range(sm_count):
            sm_work[idx] += add
    for idx in range(rem):
        sm_work[(start + idx) % sm_count] += int(weight)


def _estimate_max_sm_work(
    *,
    batch: int,
    sm_count: int,
    n_split4: int,
    n_split2: int,
) -> int:
    n_split1 = int(batch) - int(n_split4) - int(n_split2)
    if n_split4 < 0 or n_split2 < 0 or n_split1 < 0:
        raise ValueError(
            "invalid split plan: "
            f"batch={batch}, n_split4={n_split4}, n_split2={n_split2}"
        )
    sm_work = [0] * int(sm_count)
    offset = 0
    _add_periodic_work(sm_work, start=offset, tasks=int(n_split4) * 4, weight=1)
    offset += int(n_split4) * 4
    _add_periodic_work(sm_work, start=offset, tasks=int(n_split2) * 2, weight=2)
    offset += int(n_split2) * 2
    _add_periodic_work(sm_work, start=offset, tasks=n_split1, weight=4)
    return max(sm_work) if sm_work else 0


def _variable_split_balance_metrics(
    *,
    batch: int,
    sm_count: int,
    n_split4: int,
    n_split2: int,
) -> dict[str, float | int]:
    batch = int(batch)
    sm_count = int(sm_count)
    n_split4 = int(n_split4)
    n_split2 = int(n_split2)
    n_split1 = batch - n_split4 - n_split2
    if batch < 0 or sm_count <= 0 or n_split4 < 0 or n_split2 < 0 or n_split1 < 0:
        raise ValueError(
            "invalid split balance input: "
            f"batch={batch}, sm_count={sm_count}, n_split4={n_split4}, n_split2={n_split2}"
        )
    sm_work = [0] * sm_count
    offset = 0
    _add_periodic_work(sm_work, start=offset, tasks=n_split4 * 4, weight=1)
    offset += n_split4 * 4
    _add_periodic_work(sm_work, start=offset, tasks=n_split2 * 2, weight=2)
    offset += n_split2 * 2
    _add_periodic_work(sm_work, start=offset, tasks=n_split1, weight=4)
    total_work = sum(sm_work)
    max_work = max(sm_work) if sm_work else 0
    min_work = min(sm_work) if sm_work else 0
    nonzero_sms = sum(1 for work in sm_work if work > 0)
    avg_work = total_work / float(sm_count) if sm_count else 0.0
    efficiency = (
        total_work / float(sm_count * max_work)
        if sm_count > 0 and max_work > 0
        else 1.0
    )
    return {
        "total_sm_work": total_work,
        "max_sm_work": max_work,
        "min_sm_work": min_work,
        "avg_sm_work": avg_work,
        "nonzero_sms": nonzero_sms,
        "sm_efficiency": efficiency,
        "max_over_avg": (max_work / avg_work) if avg_work > 0 else 1.0,
    }


def _split_plan_key(
    *,
    batch: int,
    sm_count: int,
    n_split4: int,
    n_split2: int,
) -> tuple[int, int, int, int]:
    max_work = _estimate_max_sm_work(
        batch=batch,
        sm_count=sm_count,
        n_split4=n_split4,
        n_split2=n_split2,
    )
    cta_count = int(batch) + int(n_split2) + int(n_split4) * 3
    split_queries = int(n_split2) + int(n_split4)
    return max_work, cta_count, split_queries, int(n_split4)


def _best_variable_split_plan(
    *,
    batch: int,
    sm_count: int,
    max_split: int,
) -> tuple[int, int]:
    batch = int(batch)
    sm_count = int(sm_count)
    max_split = _valid_variable_split_max(max_split)
    if batch <= 0:
        return 0, 0
    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")
    if max_split == 1:
        return 0, 0

    candidates: set[tuple[int, int]] = {(0, 0)}
    lower_bound = (4 * batch + sm_count - 1) // sm_count
    min_cta_for_lb = max(batch, (4 * batch + lower_bound - 1) // lower_bound)

    def add_candidates_for_cta(cta: int) -> None:
        if cta < batch or cta > 4 * batch:
            return
        if max_split >= 2:
            n_split2 = cta - batch
            if 0 <= n_split2 <= batch:
                candidates.add((0, n_split2))
        if max_split >= 4:
            low_n4 = max(0, (cta - 2 * batch + 1) // 2)
            high_n4 = min(batch, (cta - batch) // 3)
            for n_split4 in (
                low_n4,
                low_n4 + 1,
                (low_n4 + high_n4) // 2,
                high_n4 - 1,
                high_n4,
            ):
                if n_split4 < low_n4 or n_split4 > high_n4:
                    continue
                n_split2 = cta - batch - 3 * n_split4
                if 0 <= n_split2 <= batch - n_split4:
                    candidates.add((n_split4, n_split2))

    for base in (
        batch,
        min_cta_for_lb,
        sm_count,
        2 * sm_count,
        3 * sm_count,
        4 * sm_count,
        4 * batch,
    ):
        for delta in range(-32, 33):
            add_candidates_for_cta(base + delta)
        for delta in range(-2 * sm_count, 2 * sm_count + 1, max(1, sm_count // 16)):
            add_candidates_for_cta(base + delta)
    if max_split >= 4:
        candidates.add((batch, 0))
    if max_split >= 2:
        candidates.add((0, batch))

    best = min(
        candidates,
        key=lambda item: _split_plan_key(
            batch=batch,
            sm_count=sm_count,
            n_split4=item[0],
            n_split2=item[1],
        ),
    )
    return int(best[0]), int(best[1])


def _build_variable_split_plan_table(
    *,
    max_batch: int,
    sm_count: int,
    max_split: int,
) -> list[tuple[int, int]]:
    return [
        _best_variable_split_plan(
            batch=batch,
            sm_count=sm_count,
            max_split=max_split,
        )
        for batch in range(int(max_batch) + 1)
    ]


class _TokenWiseSparseDecodeWGMMA64GQA:
    """SM90 small-M sparse decode using K/Q swapAB WGMMA.

    The runtime contract is intentionally narrow: Step-style GQA ratio 8,
    paged KV page=16, head_dim=128, block8 topk metadata, and one KV head per
    rank. K/V are still loaded per selected region token; TMA is not used because
    selected windows are sparse and request-local.
    """

    arch: int = 90
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
    PRODUCER_REGS_FP8 = 56
    MATH_REGS_FP8 = 224
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
        batch_invariant: bool = False,
        producer_k_warps: Optional[int] = None,
        split_capacity: int = 1,
        sm_count: Optional[int] = None,
        kv_fp8: bool = False,
    ) -> None:
        self.dtype = dtype
        self.kv_fp8 = bool(kv_fp8)
        self.logical_q_heads = int(logical_q_heads)
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.topk_windows = int(topk_windows)
        self.mtp_q_len = int(mtp_q_len)
        self.batch_invariant = bool(batch_invariant)
        self.split_capacity = int(split_capacity)
        self.sm_count = None if sm_count is None else int(sm_count)
        self.dynamic_split = self.split_capacity == self.DYNAMIC_SPLIT_CAPACITY
        self.num_stages = int(self.NUM_STAGES)
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
        if self.kv_fp8 and self.dtype != cutlass.Float16:
            raise TypeError("FP8 K/V decode computes in FP16")
        if self.logical_q_heads not in self.SUPPORTED_LOGICAL_Q_HEADS:
            raise ValueError(
                "WGMMA sparse decode supports only logical_q_heads in "
                f"{self.SUPPORTED_LOGICAL_Q_HEADS}, "
                f"got {self.logical_q_heads}"
            )
        if (
            self.HEAD_DIM not in self.SUPPORTED_HEAD_DIMS
            or self.HEAD_DIM_V not in self.SUPPORTED_HEAD_DIMS
        ):
            raise ValueError(
                "WGMMA sparse decode supports only head_dim/head_dim_v in "
                f"{self.SUPPORTED_HEAD_DIMS}, got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "WGMMA sparse decode keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if self.HEAD_DIM % 64 != 0 or self.HEAD_DIM_V % 64 != 0:
            raise ValueError(
                "WGMMA sparse decode requires head_dim/head_dim_v to be 64-aligned, "
                f"got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.block_n != self.BLOCK_N:
            raise ValueError(
                f"WGMMA sparse decode is fixed to block_n=64, got {self.block_n}"
            )
        if self.num_threads != 256:
            raise ValueError(
                "WGMMA sparse decode is fixed to num_threads=256, "
                f"got {self.num_threads}"
            )
        if self.topk_windows <= 0:
            raise ValueError("WGMMA sparse decode requires topk_windows > 0")
        if self.mtp_q_len < 1 or self.mtp_q_len > 16:
            raise ValueError(f"mtp_q_len must be in [1, 16], got {self.mtp_q_len}")
        if self.split_capacity not in (1, self.DYNAMIC_SPLIT_CAPACITY):
            raise ValueError(
                "WGMMA sparse decode kernel split capacity must be 1 or "
                f"{self.DYNAMIC_SPLIT_CAPACITY}, got {self.split_capacity}"
            )
        if self.dynamic_split and (self.sm_count is None or self.sm_count <= 0):
            raise ValueError("dynamic split decode requires a positive SM count")
        if self.num_producer_warps != 4:
            raise ValueError("WGMMA sparse decode requires four producer warps")
        if (
            self.producer_k_warps <= 0
            or self.producer_k_warps >= self.num_producer_warps
        ):
            raise ValueError(
                "producer_k_warps must be in [1, num_producer_warps-1], got "
                f"{self.producer_k_warps} with num_producer_warps={self.num_producer_warps}"
            )

    def _get_tiled_mma(self):
        qk_tiler_mnk = (self.block_n, self.MMA_Q_HEADS, self.HEAD_DIM)
        pv_tiler_mnk = (64, self.MMA_Q_HEADS, self.block_n)
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=qk_tiler_mnk[:2],
        )
        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=pv_tiler_mnk[:2],
        )
        return tiled_mma_qk, tiled_mma_pv

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

    def _get_smem_layouts(self):
        qk_tiler_mnk = (self.block_n, self.MMA_Q_HEADS, self.HEAD_DIM)
        pv_tiler_mnk = (64, self.MMA_Q_HEADS, self.block_n)
        sK_layout = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR,
            qk_tiler_mnk,
            self.dtype,
            self.num_stages,
        )
        sQ_layout = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR,
            qk_tiler_mnk,
            self.dtype,
            1,
        )
        sV_layout = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR,
            pv_tiler_mnk,
            self.dtype,
            self.num_stages * self.head_dim_v_slabs,
        )
        sP_layout = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR,
            pv_tiler_mnk,
            self.dtype,
            1,
        )
        return sK_layout, sQ_layout, sV_layout, sP_layout

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
        warp_head_f32_struct = cute.struct.MemRange[
            Float32,
            self.num_compute_warps * self.logical_q_heads,
        ]
        fp8_tokens = self.num_stages * self.block_n if self.kv_fp8 else 1
        token_scale_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, fp8_tokens], 16
        ]
        k_raw_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, fp8_tokens * self.HEAD_DIM],
            16,
        ]
        v_raw_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, fp8_tokens * self.HEAD_DIM_V],
            16,
        ]
        tile_max_struct = cute.struct.MemRange[Float32, 3]

        @cute.struct
        class SharedStorage:
            mbar_ptr_free: stage_struct
            mbar_ptr_k_stage: stage_struct
            mbar_ptr_v_stage: stage_struct
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
            sWarpMax: warp_head_f32_struct
            sWarpSum: warp_head_f32_struct
            sKScale: token_scale_struct
            sVScale: token_scale_struct
            sKRaw: k_raw_struct
            sVRaw: v_raw_struct
            sVTileMax: tile_max_struct

        return SharedStorage

    @cute.jit
    def _mask_scores(
        self,
        acc_S: cute.Tensor,
        tScS_mn: cute.Tensor,
        sPacked: cute.Tensor,
        stage: Int32,
        k_start: Int32,
        count_slot: Int32,
        kv_seq_len_slot: Int32,
        max_phys_regions: Int32,
        sKScale: cute.Tensor,
    ) -> None:
        acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
        nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
        ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
        for r in cutlass.range_constexpr(nrow_s):
            for c in cutlass.range_constexpr(ncol_s):
                tok = tScS_mn[r, c][0]
                head = tScS_mn[r, c][1]
                win = tok >> Int32(3)
                packed = sPacked[win, stage]
                phys_region = Int32(
                    (packed >> cutlass.Int64(32)) & cutlass.Int64(0xFFFF_FFFF)
                )
                start_tok = Int32(packed & cutlass.Int64(0xFFFF_FFFF))
                logical_tok = start_tok + (tok & Int32(7))
                invalid = (
                    (head >= Int32(self.logical_q_heads))
                    or ((k_start + tok) >= count_slot)
                    or (packed < cutlass.Int64(0))
                    or (phys_region < Int32(0))
                    or (phys_region >= max_phys_regions)
                    or (logical_tok >= kv_seq_len_slot)
                )
                if const_expr(self.kv_fp8):
                    acc_S_mn[r, c] = (
                        acc_S_mn[r, c] * sKScale[stage * Int32(self.block_n) + tok]
                    )
                if invalid:
                    acc_S_mn[r, c] = -Float32.inf

    @cute.jit
    def _warp_reduce_lane_mod4_max(self, value: Float32, lane: Int32) -> Float32:
        other = cute.arch.shuffle_sync_down(value, offset=Int32(4))
        if lane < Int32(28):
            value = utils.fmax_f32(value, other)
        other = cute.arch.shuffle_sync_down(value, offset=Int32(8))
        if lane < Int32(24):
            value = utils.fmax_f32(value, other)
        other = cute.arch.shuffle_sync_down(value, offset=Int32(16))
        if lane < Int32(16):
            value = utils.fmax_f32(value, other)
        return value

    @cute.jit
    def _warp_reduce_lane_mod4_sum(self, value: Float32, lane: Int32) -> Float32:
        other = cute.arch.shuffle_sync_down(value, offset=Int32(4))
        if lane < Int32(28):
            value += other
        other = cute.arch.shuffle_sync_down(value, offset=Int32(8))
        if lane < Int32(24):
            value += other
        other = cute.arch.shuffle_sync_down(value, offset=Int32(16))
        if lane < Int32(16):
            value += other
        return value

    @cute.jit
    def _online_softmax_heads(
        self,
        acc_S: cute.Tensor,
        tScS_mn: cute.Tensor,
        sHeadMax: cute.Tensor,
        sHeadSum: cute.Tensor,
        sHeadScale: cute.Tensor,
        sWarpMax: cute.Tensor,
        sWarpSum: cute.Tensor,
        softmax_scale_log2: Float32,
        warp_in_wg: Int32,
        lane: Int32,
    ) -> None:
        acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
        nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
        ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
        lane_head_base = (lane % Int32(4)) * Int32(2)
        local_max0 = -Float32.inf
        local_max1 = -Float32.inf
        local_max2 = -Float32.inf
        local_max3 = -Float32.inf

        elem = 0
        for r in cutlass.range_constexpr(nrow_s):
            for c in cutlass.range_constexpr(ncol_s):
                if const_expr(elem % 4 == 0):
                    local_max0 = utils.fmax_f32(local_max0, acc_S_mn[r, c])
                elif const_expr(elem % 4 == 1):
                    local_max1 = utils.fmax_f32(local_max1, acc_S_mn[r, c])
                elif const_expr(elem % 4 == 2):
                    local_max2 = utils.fmax_f32(local_max2, acc_S_mn[r, c])
                else:
                    local_max3 = utils.fmax_f32(local_max3, acc_S_mn[r, c])
                elem += 1

        warp_max0 = self._warp_reduce_lane_mod4_max(local_max0, lane)
        warp_max1 = self._warp_reduce_lane_mod4_max(local_max1, lane)
        warp_max2 = self._warp_reduce_lane_mod4_max(local_max2, lane)
        warp_max3 = self._warp_reduce_lane_mod4_max(local_max3, lane)
        if lane < Int32(4):
            head0 = lane_head_base
            head1 = lane_head_base + Int32(1)
            head2 = lane_head_base + Int32(8)
            head3 = lane_head_base + Int32(9)
            if head0 < Int32(self.logical_q_heads):
                sWarpMax[warp_in_wg, head0] = warp_max0
            if head1 < Int32(self.logical_q_heads):
                sWarpMax[warp_in_wg, head1] = warp_max1
            if head2 < Int32(self.logical_q_heads):
                sWarpMax[warp_in_wg, head2] = warp_max2
            if head3 < Int32(self.logical_q_heads):
                sWarpMax[warp_in_wg, head3] = warp_max3
        cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

        if warp_in_wg == Int32(0) and lane < Int32(self.logical_q_heads):
            red_max = sWarpMax[Int32(0), lane]
            for w in cutlass.range_constexpr(1, self.num_compute_warps):
                red_max = utils.fmax_f32(red_max, sWarpMax[Int32(w), lane])
            old_max = sHeadMax[lane]
            new_max = utils.fmax_f32(old_max, red_max)
            safe_max = Float32(0) if new_max == -Float32.inf else new_max
            sHeadScale[lane] = utils.exp2f((old_max - safe_max) * softmax_scale_log2)
            sHeadMax[lane] = new_max
        cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

        local_sum0 = Float32(0)
        local_sum1 = Float32(0)
        local_sum2 = Float32(0)
        local_sum3 = Float32(0)
        elem = 0
        for r in cutlass.range_constexpr(nrow_s):
            for c in cutlass.range_constexpr(ncol_s):
                if const_expr(elem % 4 == 0):
                    head = lane_head_base
                    if head < Int32(self.logical_q_heads):
                        head_max = sHeadMax[head]
                        if head_max == -Float32.inf:
                            head_max = Float32(0)
                        head_max_scaled = head_max * softmax_scale_log2
                        prob = utils.exp2f(
                            acc_S_mn[r, c] * softmax_scale_log2 - head_max_scaled
                        )
                        acc_S_mn[r, c] = prob
                        local_sum0 += prob
                elif const_expr(elem % 4 == 1):
                    head = lane_head_base + Int32(1)
                    if head < Int32(self.logical_q_heads):
                        head_max = sHeadMax[head]
                        if head_max == -Float32.inf:
                            head_max = Float32(0)
                        head_max_scaled = head_max * softmax_scale_log2
                        prob = utils.exp2f(
                            acc_S_mn[r, c] * softmax_scale_log2 - head_max_scaled
                        )
                        acc_S_mn[r, c] = prob
                        local_sum1 += prob
                elif const_expr(elem % 4 == 2):
                    head = lane_head_base + Int32(8)
                    if head < Int32(self.logical_q_heads):
                        head_max = sHeadMax[head]
                        if head_max == -Float32.inf:
                            head_max = Float32(0)
                        head_max_scaled = head_max * softmax_scale_log2
                        prob = utils.exp2f(
                            acc_S_mn[r, c] * softmax_scale_log2 - head_max_scaled
                        )
                        acc_S_mn[r, c] = prob
                        local_sum2 += prob
                else:
                    head = lane_head_base + Int32(9)
                    if head < Int32(self.logical_q_heads):
                        head_max = sHeadMax[head]
                        if head_max == -Float32.inf:
                            head_max = Float32(0)
                        head_max_scaled = head_max * softmax_scale_log2
                        prob = utils.exp2f(
                            acc_S_mn[r, c] * softmax_scale_log2 - head_max_scaled
                        )
                        acc_S_mn[r, c] = prob
                        local_sum3 += prob
                elem += 1

        warp_sum0 = self._warp_reduce_lane_mod4_sum(local_sum0, lane)
        warp_sum1 = self._warp_reduce_lane_mod4_sum(local_sum1, lane)
        warp_sum2 = self._warp_reduce_lane_mod4_sum(local_sum2, lane)
        warp_sum3 = self._warp_reduce_lane_mod4_sum(local_sum3, lane)
        if lane < Int32(4):
            head0 = lane_head_base
            head1 = lane_head_base + Int32(1)
            head2 = lane_head_base + Int32(8)
            head3 = lane_head_base + Int32(9)
            if head0 < Int32(self.logical_q_heads):
                sWarpSum[warp_in_wg, head0] = warp_sum0
            if head1 < Int32(self.logical_q_heads):
                sWarpSum[warp_in_wg, head1] = warp_sum1
            if head2 < Int32(self.logical_q_heads):
                sWarpSum[warp_in_wg, head2] = warp_sum2
            if head3 < Int32(self.logical_q_heads):
                sWarpSum[warp_in_wg, head3] = warp_sum3
        cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

        if warp_in_wg == Int32(0) and lane < Int32(self.logical_q_heads):
            tile_sum = sWarpSum[Int32(0), lane]
            for w in cutlass.range_constexpr(1, self.num_compute_warps):
                tile_sum += sWarpSum[Int32(w), lane]
            old_sum = sHeadSum[lane]
            sHeadSum[lane] = old_sum * sHeadScale[lane] + tile_sum
        cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

    @cute.jit
    def _rescale_o_by_head(
        self,
        acc_O: cute.Tensor,
        tOcO_mn: cute.Tensor,
        sHeadScale: cute.Tensor,
        factor: Float32,
    ) -> None:
        acc_O_mn = utils.make_acc_tensor_mn_view_from_mma(acc_O)
        nrow_o = const_expr(cute.size(tOcO_mn.shape[0]))
        ncol_o = const_expr(cute.size(tOcO_mn.shape[1]))
        for r in cutlass.range_constexpr(nrow_o):
            for c in cutlass.range_constexpr(ncol_o):
                head = tOcO_mn[r, c][1]
                if head < Int32(self.logical_q_heads):
                    if const_expr(self.kv_fp8):
                        acc_O_mn[r, c] = acc_O_mn[r, c] * (sHeadScale[head] * factor)
                    else:
                        acc_O_mn[r, c] = acc_O_mn[r, c] * sHeadScale[head]

    @cute.jit
    def _store_probs_to_smem(
        self,
        acc_S: cute.Tensor,
        tScS_mn: cute.Tensor,
        sP_tile: cute.Tensor,
        sVScale: cute.Tensor,
        stage: Int32,
        inv_tile_max: Float32,
    ) -> None:
        acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
        nrow_s = const_expr(cute.size(tScS_mn.shape[0]))
        ncol_s = const_expr(cute.size(tScS_mn.shape[1]))
        for r in cutlass.range_constexpr(nrow_s):
            for c in cutlass.range_constexpr(ncol_s):
                tok = tScS_mn[r, c][0]
                head = tScS_mn[r, c][1]
                if head < Int32(self.logical_q_heads) and tok < Int32(self.block_n):
                    src = cute.make_rmem_tensor((1,), self.dtype)
                    if const_expr(self.kv_fp8):
                        src[0] = self.dtype(
                            acc_S_mn[r, c]
                            * (
                                sVScale[stage * Int32(self.block_n) + tok]
                                * inv_tile_max
                            )
                        )
                    else:
                        src[0] = self.dtype(acc_S_mn[r, c])
                    dst_ptr = elem_pointer(sP_tile, (head, tok))
                    dst = cute.make_tensor(dst_ptr, cute.make_layout((1,), stride=(1,)))
                    copy_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.dtype,
                    )
                    cute.copy(copy_atom, src, dst)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

    @cute.jit
    def _fp8_row_meta(
        self,
        sPacked: cute.Tensor,
        stage: Int32,
        row: Int32,
        base_tok: Int32,
        count_slot: Int32,
        kv_seq_len_slot: Int32,
        max_phys_regions: Int32,
    ):
        packed = sPacked[row >> Int32(3), stage]
        phys_region = Int32((packed >> cutlass.Int64(32)) & cutlass.Int64(0xFFFF_FFFF))
        # FP8 has NaN encodings; skip tokens past the sequence before loading.
        valid = (
            (base_tok + row) < count_slot
            and packed >= cutlass.Int64(0)
            and phys_region >= Int32(0)
            and phys_region < max_phys_regions
            and Int32(packed & cutlass.Int64(0xFFFF_FFFF)) + (row & Int32(7))
            < kv_seq_len_slot
        )
        phys_block = phys_region >> Int32(1)
        page_off = (phys_region & Int32(1)) * Int32(8) + (row & Int32(7))
        return valid, phys_block, page_off

    @cute.jit
    def _fp8_copy_async(self, src_addr, src_space, dst_addr, dst_space, elems, dtype):
        align = elems * (dtype.width // 8)
        utils.vector_copy_with_explicit_width(
            cute.make_tensor(
                cute.make_ptr(dtype, src_addr, src_space, assumed_align=align),
                cute.make_layout((elems,)),
            ),
            cute.make_tensor(
                cute.make_ptr(dtype, dst_addr, dst_space, assumed_align=align),
                cute.make_layout((elems,)),
            ),
            num_copy_elems=elems,
            is_async=True,
        )

    @cute.jit
    def _stage_fp8_k(
        self,
        mKCache: cute.Tensor,
        mKScale: cute.Tensor,
        sK: cute.Tensor,
        sK_layout: cute.ComposedLayout,
        sKRaw: cute.Tensor,
        sKScale: cute.Tensor,
        sPacked: cute.Tensor,
        stage: Int32,
        base_tok: Int32,
        tid: Int32,
        count_slot: Int32,
        kv_seq_len_slot: Int32,
        max_phys_regions: Int32,
    ):
        lanes = Int32(self.producer_subgroup_lanes)
        groups = Int32(self.producer_k_groups)
        group_idx = tid // lanes
        lane_in_group = tid - group_idx * lanes
        col_base_k = lane_in_group * Int32(8)
        block_n = Int32(self.block_n)
        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        for local_row in cutlass.range_constexpr(self.max_rows_per_k_group):
            row = group_idx + Int32(local_row) * groups
            if row < block_n:
                valid, phys_block, page_off = self._fp8_row_meta(
                    sPacked,
                    stage,
                    row,
                    base_tok,
                    count_slot,
                    kv_seq_len_slot,
                    max_phys_regions,
                )
                stage_tok = stage * block_n + row
                if valid:
                    for slab in cutlass.range_constexpr(self.head_dim_slabs):
                        col = col_base_k + Int32(slab * 64)
                        self._fp8_copy_async(
                            utils.elem_pointer_i64_offset(
                                mKCache, (page_off, col, Int32(0), phys_block)
                            ).toint(),
                            mKCache.memspace,
                            elem_pointer(
                                sKRaw, (stage_tok * Int32(self.HEAD_DIM) + col,)
                            ).toint(),
                            sKRaw.memspace,
                            8,
                            cutlass.Float8E4M3FN,
                        )
                if lane_in_group == Int32(0):
                    if valid:
                        self._fp8_copy_async(
                            elem_pointer(
                                mKScale, (phys_block * Int32(16) + page_off,)
                            ).toint(),
                            mKScale.memspace,
                            elem_pointer(sKScale, (stage_tok,)).toint(),
                            sKScale.memspace,
                            1,
                            Float32,
                        )
                    else:
                        sKScale[stage_tok] = Float32(0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        for local_row in cutlass.range_constexpr(self.max_rows_per_k_group):
            row = group_idx + Int32(local_row) * groups
            if row < block_n:
                valid, _, _ = self._fp8_row_meta(
                    sPacked,
                    stage,
                    row,
                    base_tok,
                    count_slot,
                    kv_seq_len_slot,
                    max_phys_regions,
                )
                stage_tok = stage * block_n + row
                for slab in cutlass.range_constexpr(self.head_dim_slabs):
                    col = col_base_k + Int32(slab * 64)
                    s_ptr = elem_pointer(sK, (row, col, stage))
                    s_ptr = cute.make_ptr(
                        self.dtype,
                        s_ptr.toint(),
                        sK.memspace,
                        assumed_align=self.REQUIRED_POINTER_ALIGN,
                    )
                    s_vec = cute.make_tensor(
                        cute.recast_ptr(s_ptr, sK_layout.inner), vec_layout_8
                    )
                    if valid:
                        raw = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                elem_pointer(
                                    sKRaw, (stage_tok * Int32(self.HEAD_DIM) + col,)
                                ).toint(),
                                sKRaw.memspace,
                                assumed_align=8,
                            ),
                            vec_layout_8,
                        )
                        s_vec.store(raw.load().to(self.dtype))
                    else:
                        for vi in cutlass.range_constexpr(8):
                            s_vec[vi] = self.dtype(0)
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def _stage_fp8_v(
        self,
        mVtCache: cute.Tensor,
        mVScale: cute.Tensor,
        sV: cute.Tensor,
        sV_layout: cute.ComposedLayout,
        sVRaw: cute.Tensor,
        sVScale: cute.Tensor,
        sPacked: cute.Tensor,
        stage: Int32,
        base_tok: Int32,
        tid: Int32,
        count_slot: Int32,
        kv_seq_len_slot: Int32,
        max_phys_regions: Int32,
    ):
        lanes = Int32(self.producer_subgroup_lanes)
        groups = Int32(self.producer_v_groups)
        group_idx = tid // lanes
        lane_in_group = tid - group_idx * lanes
        col_base_v = lane_in_group * Int32(8)
        block_n = Int32(self.block_n)
        stage_count = Int32(self.num_stages)
        vec_layout_8 = cute.make_layout((8,), stride=(1,))
        for local_row in cutlass.range_constexpr(self.max_rows_per_v_group):
            row = group_idx + Int32(local_row) * groups
            if row < block_n:
                valid, phys_block, page_off = self._fp8_row_meta(
                    sPacked,
                    stage,
                    row,
                    base_tok,
                    count_slot,
                    kv_seq_len_slot,
                    max_phys_regions,
                )
                stage_tok = stage * block_n + row
                if valid:
                    for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                        col = col_base_v + Int32(slab * 64)
                        self._fp8_copy_async(
                            utils.elem_pointer_i64_offset(
                                mVtCache, (col, page_off, Int32(0), phys_block)
                            ).toint(),
                            mVtCache.memspace,
                            elem_pointer(
                                sVRaw, (stage_tok * Int32(self.HEAD_DIM_V) + col,)
                            ).toint(),
                            sVRaw.memspace,
                            8,
                            cutlass.Float8E4M3FN,
                        )
                if lane_in_group == Int32(0):
                    if valid:
                        self._fp8_copy_async(
                            elem_pointer(
                                mVScale, (phys_block * Int32(16) + page_off,)
                            ).toint(),
                            mVScale.memspace,
                            elem_pointer(sVScale, (stage_tok,)).toint(),
                            sVScale.memspace,
                            1,
                            Float32,
                        )
                    else:
                        sVScale[stage_tok] = Float32(0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        for local_row in cutlass.range_constexpr(self.max_rows_per_v_group):
            row = group_idx + Int32(local_row) * groups
            if row < block_n:
                valid, _, _ = self._fp8_row_meta(
                    sPacked,
                    stage,
                    row,
                    base_tok,
                    count_slot,
                    kv_seq_len_slot,
                    max_phys_regions,
                )
                stage_tok = stage * block_n + row
                for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                    col = col_base_v + Int32(slab * 64)
                    s_stage = stage + stage_count * Int32(slab)
                    s_ptr = elem_pointer(sV, (col_base_v, row, s_stage))
                    s_ptr = cute.make_ptr(
                        self.dtype,
                        s_ptr.toint(),
                        sV.memspace,
                        assumed_align=self.REQUIRED_POINTER_ALIGN,
                    )
                    s_vec = cute.make_tensor(
                        cute.recast_ptr(s_ptr, sV_layout.inner), vec_layout_8
                    )
                    if valid:
                        raw = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                elem_pointer(
                                    sVRaw, (stage_tok * Int32(self.HEAD_DIM_V) + col,)
                                ).toint(),
                                sVRaw.memspace,
                                assumed_align=8,
                            ),
                            vec_layout_8,
                        )
                        s_vec.store(raw.load().to(self.dtype))
                    else:
                        for vi in cutlass.range_constexpr(8):
                            s_vec[vi] = self.dtype(0)
        cute.arch.fence_proxy("async.shared", space="cta")

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

        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
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
        sWarpMax = storage.sWarpMax.get_tensor(
            cute.make_layout(
                (self.num_compute_warps, self.logical_q_heads),
                stride=(self.logical_q_heads, 1),
            )
        )
        sWarpSum = storage.sWarpSum.get_tensor(
            cute.make_layout(
                (self.num_compute_warps, self.logical_q_heads),
                stride=(self.logical_q_heads, 1),
            )
        )
        fp8_tokens = self.num_stages * self.block_n if self.kv_fp8 else 1
        sKScale = storage.sKScale.get_tensor(cute.make_layout((fp8_tokens,)))
        sVScale = storage.sVScale.get_tensor(cute.make_layout((fp8_tokens,)))
        sKRaw = storage.sKRaw.get_tensor(
            cute.make_layout((fp8_tokens * self.HEAD_DIM,))
        )
        sVRaw = storage.sVRaw.get_tensor(
            cute.make_layout((fp8_tokens * self.HEAD_DIM_V,))
        )
        # [0] previous/current tile max, [1] 1/current, [2] current tile max.
        sVTileMax = storage.sVTileMax.get_tensor(cute.make_layout((3,)))

        if const_expr(self.dynamic_split):
            if tidx == 0:
                batch_capacity = Int32(cute.size(mQ.shape[0]))
                valid_rows = Int32(mValidRows[0])
                active_count = Int32(0)
                for batch_idx in cutlass.range(batch_capacity, unroll=1):
                    if (batch_idx < valid_rows) and (mCount[batch_idx] > Int32(0)):
                        active_count += Int32(1)

                target_parts = Int32(1)
                if active_count > Int32(0):
                    min_wave_work = Int32(self.sm_count) * Int32(
                        self.MIN_DYNAMIC_SPLIT_WAVE_NUMERATOR
                    )
                    split_scale = Int32(self.MIN_DYNAMIC_SPLIT_WAVE_DENOMINATOR)
                    if active_count * split_scale < min_wave_work:
                        target_parts = Int32(16)
                        if active_count * Int32(2) * split_scale >= min_wave_work:
                            target_parts = Int32(2)
                        elif active_count * Int32(4) * split_scale >= min_wave_work:
                            target_parts = Int32(4)
                        elif active_count * Int32(8) * split_scale >= min_wave_work:
                            target_parts = Int32(8)
                if const_expr(self.mtp_q_len > 1):
                    # Must match merge_dynamic_split_nat_lse_states_sm90_gqa.
                    target_parts = Int32(16)
                if const_expr(self.batch_invariant):
                    target_parts = Int32(self.DYNAMIC_SPLIT_CAPACITY)

                selected_q = Int32(-1)
                selected_split = Int32(0)
                selected_parts = Int32(1)
                logical_work_idx = Int32(-1)
                launch_cursor = Int32(0)
                if const_expr(self.mtp_q_len == 1):
                    for batch_idx in cutlass.range(batch_capacity, unroll=1):
                        region_count = mCount[batch_idx]
                        if batch_idx >= valid_rows:
                            region_count = Int32(0)
                        parts = self._dynamic_split_parts(
                            region_count,
                            target_parts,
                        )
                        next_cursor = launch_cursor + parts
                        if (
                            selected_q < Int32(0)
                            and launch_idx >= launch_cursor
                            and launch_idx < next_cursor
                        ):
                            selected_q = batch_idx
                            selected_split = launch_idx - launch_cursor
                            selected_parts = parts
                            logical_work_idx = launch_idx
                        launch_cursor = next_cursor
                else:
                    request_capacity = Int32(
                        cute.size(mQueryStartLoc.shape[0])
                    ) - Int32(1)
                    for mtp_slot in cutlass.range_constexpr(self.mtp_q_len):
                        for request_idx in cutlass.range(request_capacity, unroll=1):
                            request_start = Int32(mQueryStartLoc[request_idx])
                            request_end = Int32(mQueryStartLoc[request_idx + Int32(1)])
                            batch_idx = request_start + Int32(mtp_slot)
                            row_valid = (batch_idx < request_end) & (
                                batch_idx < valid_rows
                            )
                            safe_batch_idx = cutlass.select_(
                                row_valid, batch_idx, Int32(0)
                            )
                            region_count = cutlass.select_(
                                row_valid, mCount[safe_batch_idx], Int32(0)
                            )
                            parts = self._dynamic_split_parts(
                                region_count,
                                target_parts,
                            )
                            next_cursor = launch_cursor + parts
                            if (
                                selected_q < Int32(0)
                                and launch_idx >= launch_cursor
                                and launch_idx < next_cursor
                            ):
                                selected_q = batch_idx
                                selected_split = launch_idx - launch_cursor
                                selected_parts = parts
                            launch_cursor = next_cursor

                    if selected_q >= Int32(0):
                        logical_work_idx = selected_split
                        for batch_idx in cutlass.range(batch_capacity, unroll=1):
                            if batch_idx < selected_q:
                                region_count = mCount[batch_idx]
                                if batch_idx >= valid_rows:
                                    region_count = Int32(0)
                                logical_work_idx += self._dynamic_split_parts(
                                    region_count,
                                    target_parts,
                                )
                sSchedule[0] = selected_q
                sSchedule[1] = selected_split
                sSchedule[2] = selected_parts
                sSchedule[3] = logical_work_idx
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

        if tidx < Int32(self.num_stages):
            cute.arch.mbarrier_init(mbar_free + tidx, Int32(1))
            cute.arch.mbarrier_init(mbar_k_stage + tidx, producer_k_threads)
            cute.arch.mbarrier_init(mbar_v_stage + tidx, producer_v_threads)
        if tidx < Int32(self.logical_q_heads):
            sHeadMax[tidx] = -Float32.inf
            sHeadSum[tidx] = Float32(0)
            sHeadScale[tidx] = Float32(1)
        if tidx < Int32(3):
            sVTileMax[tidx] = Float32(1)
        cute.arch.mbarrier_init_fence()

        q_valid = (
            q_pos >= Int32(0)
            and q_pos < Int32(cute.size(mQ.shape[0]))
            and q_pos < Int32(mValidRows[0])
        )
        work_valid = work_idx >= Int32(0) and work_idx < Int32(cute.size(mO.shape[0]))
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

                if is_producer_k:
                    if const_expr(self.kv_fp8):
                        self._stage_fp8_k(
                            mKCache,
                            mKScale,
                            sK,
                            sK_layout,
                            sKRaw,
                            sKScale,
                            sPacked,
                            stage,
                            local_idx * block_n_i32,
                            tidx_k,
                            count_slot,
                            kv_seq_len_slot,
                            max_phys_regions,
                        )
                        cute.arch.mbarrier_arrive(mbar_k_stage + stage)
                    else:
                        group_idx_k = tidx_k // producer_subgroup_lanes
                        lane_in_group_k = tidx_k - group_idx_k * producer_subgroup_lanes
                        col_base_k = lane_in_group_k * Int32(8)
                        base_tok_idx_k = local_idx * block_n_i32
                        for local_row in cutlass.range_constexpr(
                            self.max_rows_per_k_group
                        ):
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
                                )
                                phys_block = phys_region >> Int32(1)
                                page_off = (phys_region & Int32(1)) * Int32(8) + (
                                    row & Int32(7)
                                )
                                for slab in cutlass.range_constexpr(
                                    self.head_dim_slabs
                                ):
                                    col_base = col_base_k + Int32(slab * 64)
                                    s_ptr = elem_pointer(sK, (row, col_base, stage))
                                    s_ptr = cute.make_ptr(
                                        self.dtype,
                                        s_ptr.toint(),
                                        sK.memspace,
                                        assumed_align=self.REQUIRED_POINTER_ALIGN,
                                    )
                                    s_ptr = cute.recast_ptr(s_ptr, sK_layout.inner)
                                    s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                                    if valid:
                                        g_ptr = utils.elem_pointer_i64_offset(
                                            mKCache,
                                            (page_off, col_base, Int32(0), phys_block),
                                        )
                                        g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                                        utils.vector_copy_with_explicit_width(
                                            g_vec,
                                            s_vec,
                                            num_copy_elems=8,
                                            is_async=True,
                                        )
                                    else:
                                        for vi in cutlass.range_constexpr(8):
                                            s_vec[vi] = self.dtype(0)
                        cute.arch.cp_async_commit_group()
                        cute.arch.barrier(
                            barrier_id=6, number_of_threads=producer_k_threads
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_k_stage + stage)

                if is_producer_v:
                    if const_expr(self.kv_fp8):
                        self._stage_fp8_v(
                            mVtCache,
                            mVScale,
                            sV,
                            sV_layout,
                            sVRaw,
                            sVScale,
                            sPacked,
                            stage,
                            local_idx * block_n_i32,
                            tidx_v,
                            count_slot,
                            kv_seq_len_slot,
                            max_phys_regions,
                        )
                        cute.arch.mbarrier_arrive(mbar_v_stage + stage)
                    else:
                        group_idx_v = tidx_v // producer_subgroup_lanes
                        lane_in_group_v = tidx_v - group_idx_v * producer_subgroup_lanes
                        col_base_v = lane_in_group_v * Int32(8)
                        base_tok_idx_v = local_idx * block_n_i32
                        for local_row in cutlass.range_constexpr(
                            self.max_rows_per_v_group
                        ):
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
                                )
                                phys_block = phys_region >> Int32(1)
                                page_off = (phys_region & Int32(1)) * Int32(8) + (
                                    row & Int32(7)
                                )
                                for slab in cutlass.range_constexpr(
                                    self.head_dim_v_slabs
                                ):
                                    col_base = col_base_v + Int32(slab * 64)
                                    s_stage = stage + stage_count * Int32(slab)
                                    s_ptr = elem_pointer(sV, (col_base_v, row, s_stage))
                                    s_ptr = cute.make_ptr(
                                        self.dtype,
                                        s_ptr.toint(),
                                        sV.memspace,
                                        assumed_align=self.REQUIRED_POINTER_ALIGN,
                                    )
                                    s_ptr = cute.recast_ptr(s_ptr, sV_layout.inner)
                                    s_vec = cute.make_tensor(s_ptr, vec_layout_8)
                                    if valid:
                                        g_ptr = utils.elem_pointer_i64_offset(
                                            mVtCache,
                                            (col_base, page_off, Int32(0), phys_block),
                                        )
                                        g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                                        utils.vector_copy_with_explicit_width(
                                            g_vec,
                                            s_vec,
                                            num_copy_elems=8,
                                            is_async=True,
                                        )
                                    else:
                                        for vi in cutlass.range_constexpr(8):
                                            s_vec[vi] = self.dtype(0)
                        cute.arch.cp_async_commit_group()
                        cute.arch.barrier(
                            barrier_id=7, number_of_threads=producer_v_threads
                        )
                        cute.arch.cp_async_mbarrier_arrive_noinc(mbar_v_stage + stage)

        if is_compute:
            cute.arch.warpgroup_reg_alloc(
                self.MATH_REGS_FP8 if self.kv_fp8 else self.MATH_REGS
            )
            q_pos_safe = cutlass.select_(q_valid, q_pos, Int32(0))
            q_vec_count = (self.MMA_Q_HEADS * self.HEAD_DIM) // 8
            sQ_tile = sQ[None, None, Int32(0)]
            for vec_idx in cutlass.range(idx_in_wg, q_vec_count, 128, unroll=1):
                elem_idx = vec_idx * Int32(8)
                head = elem_idx // self.HEAD_DIM
                d = elem_idx - head * self.HEAD_DIM
                s_ptr = elem_pointer(sQ_tile, (head, d))
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
                g_ptr = utils.elem_pointer_i64_offset(mQ, (q_pos_safe, head_safe, d))
                g_vec = cute.make_tensor(g_ptr, vec_layout_8)
                if q_valid and head_valid:
                    if const_expr(self.kv_fp8):
                        s_vec.store(g_vec.load().to(self.dtype))
                    else:
                        utils.vector_copy_with_explicit_width(
                            g_vec, s_vec, num_copy_elems=8
                        )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))

            thr_mma_qk = tiled_mma_qk.get_slice(idx_in_wg)
            thr_mma_pv = tiled_mma_pv.get_slice(idx_in_wg)
            tCrQ = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sQ_tile))
            cS = cute.make_identity_tensor((self.block_n, self.MMA_Q_HEADS))
            tScS_mn = utils.make_acc_tensor_mn_view_from_mma(thr_mma_qk.partition_C(cS))
            cO = cute.make_identity_tensor((64, self.MMA_Q_HEADS))
            tOcO_mn = utils.make_acc_tensor_mn_view_from_mma(thr_mma_pv.partition_C(cO))
            acc_shape_S = thr_mma_qk.partition_shape_C((self.block_n, self.MMA_Q_HEADS))
            acc_shape_O = thr_mma_pv.partition_shape_C((64, self.MMA_Q_HEADS))
            acc_O = [
                thr_mma_pv.make_fragment_C(acc_shape_O)
                for _ in range(self.head_dim_v_slabs)
            ]
            for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                acc_O[slab].fill(Float32.zero)
            any_valid_block = Int32(0)

            for local_idx in cutlass.range(work_split_tiles_i32, unroll=1):
                stage = local_idx % stage_count
                wait_phase = ((local_idx - stage) // stage_count) & Int32(1)
                k_start = local_idx * block_n_i32
                block_active = slot_active and (k_start < count_slot)

                cute.arch.mbarrier_wait(mbar_k_stage + stage, wait_phase)
                if const_expr(self.kv_fp8):
                    # Probability weights need this tile's V scales.
                    cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                cute.arch.fence_proxy("async.shared", space="cta")

                if block_active:
                    sK_stage = sK[None, None, stage]
                    tCrK = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sK_stage))
                    acc_S = thr_mma_qk.make_fragment_C(acc_shape_S)
                    hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                        tiled_mma_qk,
                        acc_S,
                        tCrK,
                        tCrQ,
                        zero_init=True,
                        wg_wait=0,
                    )
                    self._mask_scores(
                        acc_S,
                        tScS_mn,
                        sPacked,
                        stage,
                        k_start,
                        count_slot,
                        kv_seq_len_slot,
                        max_phys_regions,
                        sKScale,
                    )
                    if const_expr(self.kv_fp8):
                        if warp_in_wg == Int32(0):
                            scale_base = stage * block_n_i32
                            v_max = utils.fmax_f32(
                                sVScale[scale_base + lane],
                                sVScale[scale_base + lane + Int32(32)],
                            )
                            v_max = utils.warp_reduce(v_max, utils.fmax_f32)
                            if lane == Int32(0):
                                v_max = v_max if v_max > Float32(0) else Float32(1)
                                inv_v_max = cute.arch.rcp_approx(v_max)
                                sVTileMax[0] = sVTileMax[2] * inv_v_max
                                sVTileMax[1] = inv_v_max
                                sVTileMax[2] = v_max
                        cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
                    self._online_softmax_heads(
                        acc_S,
                        tScS_mn,
                        sHeadMax,
                        sHeadSum,
                        sHeadScale,
                        sWarpMax,
                        sWarpSum,
                        softmax_scale_log2,
                        warp_in_wg,
                        lane,
                    )
                    any_valid_block = Int32(1)
                    # FP8 keeps acc_O in units of the current tile's V-scale max.
                    tile_ratio = Float32(1)
                    inv_tile_max = Float32(1)
                    if const_expr(self.kv_fp8):
                        tile_ratio = sVTileMax[0]
                        inv_tile_max = sVTileMax[1]
                    for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                        self._rescale_o_by_head(
                            acc_O[slab], tOcO_mn, sHeadScale, tile_ratio
                        )
                    sP_tile = sP[None, None, Int32(0)]
                    self._store_probs_to_smem(
                        acc_S, tScS_mn, sP_tile, sVScale, stage, inv_tile_max
                    )

                if const_expr(not self.kv_fp8):
                    cute.arch.mbarrier_wait(mbar_v_stage + stage, wait_phase)
                cute.arch.fence_proxy("async.shared", space="cta")

                if block_active:
                    sP_tile = sP[None, None, Int32(0)]
                    tCrP = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sP_tile))
                    for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                        s_stage = stage + stage_count * Int32(slab)
                        sV_stage = sV[None, None, s_stage]
                        tCrV = thr_mma_pv.make_fragment_A(
                            thr_mma_pv.partition_A(sV_stage)
                        )
                        hop_helpers.warpgroup_gemm_with_optional_swap_wait(
                            tiled_mma_pv,
                            acc_O[slab],
                            tCrV,
                            tCrP,
                            zero_init=False,
                            wg_wait=0,
                        )

                if (
                    idx_in_wg == Int32(0)
                    and (local_idx + stage_count) < work_split_tiles_i32
                ):
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.mbarrier_arrive(mbar_free + stage)

            # Static graph padding rows own output slots and must be cleared.
            if work_valid:
                if slot_active and (any_valid_block > Int32(0)):
                    if warp_in_wg == Int32(0) and lane < Int32(self.logical_q_heads):
                        denom = sHeadSum[lane]
                        valid_sum = denom != Float32(0) and denom == denom
                        inv = cute.arch.rcp_approx(denom if valid_sum else Float32(1))
                        if const_expr(self.kv_fp8):
                            inv = inv * sVTileMax[2]
                        sHeadScale[lane] = inv
                        if const_expr(mLSE is not None):
                            lse_val = (
                                (
                                    sHeadMax[lane] * softmax_scale_log2
                                    + utils.log2f(denom)
                                )
                                * math.log(2.0)
                                if valid_sum
                                else -Float32.inf
                            )
                            mLSE[work_idx, lane] = lse_val
                    cute.arch.barrier(barrier_id=8, number_of_threads=Int32(128))
                    for slab in cutlass.range_constexpr(self.head_dim_v_slabs):
                        self._rescale_o_by_head(
                            acc_O[slab], tOcO_mn, sHeadScale, Float32(1)
                        )
                    acc_O_mn = [
                        utils.make_acc_tensor_mn_view_from_mma(acc_O[slab])
                        for slab in range(self.head_dim_v_slabs)
                    ]
                    nrow_o = const_expr(cute.size(tOcO_mn.shape[0]))
                    ncol_o = const_expr(cute.size(tOcO_mn.shape[1]))
                    for r in cutlass.range_constexpr(nrow_o):
                        for c in cutlass.range_constexpr(ncol_o):
                            d_out = tOcO_mn[r, c][0]
                            h_out = tOcO_mn[r, c][1]
                            if h_out < Int32(self.logical_q_heads) and d_out < Int32(
                                64
                            ):
                                for slab in cutlass.range_constexpr(
                                    self.head_dim_v_slabs
                                ):
                                    mO[
                                        work_idx,
                                        h_out,
                                        d_out + Int32(slab * 64),
                                    ] = mO.element_type(acc_O_mn[slab][r, c])
                else:
                    for idx_zero in cutlass.range(
                        idx_in_wg, self.logical_q_heads * self.HEAD_DIM_V, 128, unroll=1
                    ):
                        head_zero = idx_zero // self.HEAD_DIM_V
                        dim_zero = idx_zero - head_zero * self.HEAD_DIM_V
                        mO[work_idx, head_zero, dim_zero] = mO.element_type(0)
                    if const_expr(mLSE is not None):
                        if warp_in_wg == Int32(0) and lane < Int32(
                            self.logical_q_heads
                        ):
                            mLSE[work_idx, lane] = -Float32.inf


class TokenWiseFlashAttnFwdSm90GQADecode:
    """Fixed-spec decode wrapper for the GQA sparse paged-KV kernel."""

    arch: int = 90
    REQUIRED_POINTER_ALIGN = _TokenWiseSparseDecodeWGMMA64GQA.REQUIRED_POINTER_ALIGN

    def __init__(
        self,
        *,
        dtype,
        logical_q_heads: int,
        head_dim: int = 128,
        head_dim_v: int = 128,
        block_n: int = 64,
        num_threads: int = 256,
        topk_windows: int,
        mtp_q_len: int = 1,
        batch_invariant: bool = False,
        variable_split_max: int = 1,
        variable_split_max_batch: int = 2048,
        sm_count: int | None = None,
        producer_k_warps: Optional[int] = None,
    ) -> None:
        self.dtype = dtype
        self.logical_q_heads = int(logical_q_heads)
        self.HEAD_DIM = int(head_dim)
        self.HEAD_DIM_V = int(head_dim_v)
        self.block_n = int(block_n)
        self.num_threads = int(num_threads)
        self.topk_windows = int(topk_windows)
        self.mtp_q_len = int(mtp_q_len)
        self.batch_invariant = bool(batch_invariant)
        self.variable_split_max = _valid_variable_split_max(variable_split_max)
        self.variable_split_max_batch = int(variable_split_max_batch)
        self.sm_count = None if sm_count is None else int(sm_count)
        self.producer_k_warps = (
            None if producer_k_warps is None else int(producer_k_warps)
        )
        self._variable_split_plan_table: list[tuple[int, int]] | None = None

        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("Only fp16/bf16 are supported")
        if (
            self.logical_q_heads
            not in _TokenWiseSparseDecodeWGMMA64GQA.SUPPORTED_LOGICAL_Q_HEADS
        ):
            raise ValueError(
                "GQA decode wrapper supports only logical_q_heads in "
                f"{_TokenWiseSparseDecodeWGMMA64GQA.SUPPORTED_LOGICAL_Q_HEADS}, "
                f"got {self.logical_q_heads}"
            )
        if (
            self.HEAD_DIM not in _TokenWiseSparseDecodeWGMMA64GQA.SUPPORTED_HEAD_DIMS
            or self.HEAD_DIM_V
            not in _TokenWiseSparseDecodeWGMMA64GQA.SUPPORTED_HEAD_DIMS
        ):
            raise ValueError(
                "GQA decode wrapper supports only head_dim/head_dim_v in "
                f"{_TokenWiseSparseDecodeWGMMA64GQA.SUPPORTED_HEAD_DIMS}, got ({self.HEAD_DIM}, {self.HEAD_DIM_V})"
            )
        if self.HEAD_DIM != self.HEAD_DIM_V:
            raise ValueError(
                "GQA decode keeps the equal-dim contract, "
                f"got head_dim={self.HEAD_DIM}, head_dim_v={self.HEAD_DIM_V}"
            )
        if self.block_n != 64:
            raise ValueError("GQA decode WGMMA specialization requires block_n == 64")
        if self.num_threads != 256:
            raise ValueError(
                "GQA decode WGMMA specialization requires num_threads == 256"
            )
        if self.mtp_q_len < 1 or self.mtp_q_len > 16:
            raise ValueError(f"mtp_q_len must be in [1, 16], got {self.mtp_q_len}")
        if self.variable_split_max > 1 and self.sm_count is None:
            raise ValueError("variable split requires sm_count")
        if self.variable_split_max_batch < 1:
            raise ValueError(
                "variable_split_max_batch must be positive, "
                f"got {self.variable_split_max_batch}"
            )
        if 1 < self.variable_split_max <= 4 and not self.batch_invariant:
            self._variable_split_plan_table = _build_variable_split_plan_table(
                max_batch=self.variable_split_max_batch,
                sm_count=int(self.sm_count),
                max_split=self.variable_split_max,
            )

    def get_variable_split_plan(self, batch: int) -> tuple[int, int, int]:
        batch = int(batch)
        if batch < 0:
            raise ValueError(f"batch must be non-negative, got {batch}")
        if self.variable_split_max == 1:
            return 0, 0, batch
        if self.batch_invariant and self.variable_split_max == 2:
            return 0, batch, batch * 2
        if self.batch_invariant and self.variable_split_max == 4:
            return batch, 0, batch * 4
        if (
            self.variable_split_max
            == _TokenWiseSparseDecodeWGMMA64GQA.DYNAMIC_SPLIT_CAPACITY
        ):
            if self._dynamic_split_capacity_can_run_direct(batch):
                return 0, 0, batch
            return 0, 0, batch * self.variable_split_max
        if self._variable_split_plan_table is None:
            raise ValueError("variable split plan table is not initialized")
        if batch >= len(self._variable_split_plan_table):
            raise ValueError(
                "variable split decode batch exceeds the precomputed plan table: "
                f"batch={batch}, max_batch={self.variable_split_max_batch}"
            )
        n_split4, n_split2 = self._variable_split_plan_table[batch]
        n_split1 = batch - n_split4 - n_split2
        if n_split1 < 0:
            raise ValueError(
                "invalid variable split plan: "
                f"batch={batch}, n_split4={n_split4}, n_split2={n_split2}"
            )
        return (
            int(n_split4),
            int(n_split2),
            int(n_split4) * 4 + int(n_split2) * 2 + n_split1,
        )

    def _dynamic_split_capacity_can_run_direct(self, batch: int) -> bool:
        if self.batch_invariant:
            return False
        # Dynamic16 is for underfilled batches. Once the declared graph
        # capacity already fills the target SM wave, the extra 16x work slots
        # are graph padding CTAs, not useful split-KV work.
        return (
            int(batch)
            * _TokenWiseSparseDecodeWGMMA64GQA.MIN_DYNAMIC_SPLIT_WAVE_DENOMINATOR
            >= int(self.sm_count)
            * _TokenWiseSparseDecodeWGMMA64GQA.MIN_DYNAMIC_SPLIT_WAVE_NUMERATOR
        )

    def get_variable_split_balance_metrics(self, batch: int) -> dict[str, float | int]:
        batch = int(batch)
        if self.sm_count is None:
            raise ValueError("variable split balance metrics require sm_count")
        n_split4, n_split2, work_count = self.get_variable_split_plan(batch)
        planned = _variable_split_balance_metrics(
            batch=batch,
            sm_count=int(self.sm_count),
            n_split4=n_split4,
            n_split2=n_split2,
        )
        baseline = _variable_split_balance_metrics(
            batch=batch,
            sm_count=int(self.sm_count),
            n_split4=0,
            n_split2=0,
        )
        planned["baseline_max_sm_work"] = baseline["max_sm_work"]
        planned["baseline_sm_efficiency"] = baseline["sm_efficiency"]
        planned["max_sm_work_reduction"] = int(baseline["max_sm_work"]) - int(
            planned["max_sm_work"]
        )
        planned["work_count"] = int(work_count)
        planned["n_split4"] = int(n_split4)
        planned["n_split2"] = int(n_split2)
        return planned

    @staticmethod
    def _validate_pointer_alignment(t, *, name: str, min_align: int) -> None:
        ptr = int(t.data_ptr())
        if ptr % int(min_align) != 0:
            raise ValueError(
                f"{name}.data_ptr()={ptr} is not {int(min_align)}-byte aligned; "
                "Step3p5 sparse decode requires stable pointer alignment for "
                "precompiled CuTeDSL kernels"
            )

    def _validate_inputs(
        self,
        q,
        k_cache,
        v_cache,
        out,
        lse,
        region_counts,
        region_packed_indices,
        kv_seqlens,
        query_start_loc,
    ):
        import torch

        if k_cache is None or v_cache is None:
            raise ValueError("decode requires k_cache and v_cache")
        if (
            q.device.type != "cuda"
            or k_cache.device.type != "cuda"
            or v_cache.device.type != "cuda"
        ):
            raise ValueError("q/k_cache/v_cache must be CUDA tensors")
        if q.ndim != 3:
            raise ValueError(
                f"Expected q shape (Tq, {self.logical_q_heads}, {self.HEAD_DIM}), got {tuple(q.shape)}"
            )
        if self.mtp_q_len > 1:
            if (
                query_start_loc is None
                or query_start_loc.device != q.device
                or query_start_loc.dtype != torch.int32
                or query_start_loc.ndim != 1
                or int(query_start_loc.numel()) < 2
                or not query_start_loc.is_contiguous()
            ):
                raise ValueError(
                    "ragged MTP decode requires contiguous CUDA int32 "
                    "query_start_loc with shape [requests + 1]"
                )
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                "k_cache/v_cache must be paged tensors with shape (num_pages, page, heads, dim)"
            )
        if tuple(q.shape[1:]) != (self.logical_q_heads, self.HEAD_DIM):
            raise ValueError(
                f"Expected q shape (*, {self.logical_q_heads}, {self.HEAD_DIM}), got {tuple(q.shape)}"
            )
        if tuple(k_cache.shape[1:]) != (
            _TokenWiseSparseDecodeWGMMA64GQA.PAGE_SIZE,
            _TokenWiseSparseDecodeWGMMA64GQA.KV_HEADS,
            self.HEAD_DIM,
        ):
            raise ValueError(
                f"Expected k_cache shape (*, 16, 1, {self.HEAD_DIM}), got {tuple(k_cache.shape)}"
            )
        if tuple(v_cache.shape[1:]) != (
            _TokenWiseSparseDecodeWGMMA64GQA.PAGE_SIZE,
            _TokenWiseSparseDecodeWGMMA64GQA.KV_HEADS,
            self.HEAD_DIM_V,
        ):
            raise ValueError(
                f"Expected v_cache shape (*, 16, 1, {self.HEAD_DIM_V}), got {tuple(v_cache.shape)}"
            )
        if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
            raise ValueError("q/k_cache/v_cache must have the same dtype")
        n_split4, n_split2, expected_work_count = self.get_variable_split_plan(
            int(q.shape[0])
        )
        expected_out_shape = (
            expected_work_count,
            self.logical_q_heads,
            self.HEAD_DIM_V,
        )
        if out.shape != expected_out_shape:
            raise ValueError(
                f"Expected out shape {expected_out_shape}, got {tuple(out.shape)}"
            )
        if lse is not None:
            expected_lse_shape = (
                expected_work_count,
                self.logical_q_heads,
            )
            if lse.shape != expected_lse_shape:
                raise ValueError(
                    f"Expected lse shape {expected_lse_shape}, got {tuple(lse.shape)}"
                )
            if lse.dtype != torch.float32:
                raise ValueError("lse must be float32")
            if lse.device != q.device:
                raise ValueError("lse must be on the same device as q")
        if region_counts is None or region_packed_indices is None or kv_seqlens is None:
            raise ValueError(
                "GQA decode requires region_counts, region_packed_indices, and kv_seqlens"
            )
        if region_counts.dtype != torch.int32:
            raise ValueError("region_counts must be int32")
        if region_packed_indices.dtype != torch.int64:
            raise ValueError("region_packed_indices must be int64")
        if region_counts.device != q.device or region_packed_indices.device != q.device:
            raise ValueError(
                "region_counts/region_packed_indices must be on the same device as q"
            )
        if tuple(region_counts.shape) != (q.shape[0],):
            raise ValueError(
                f"Expected region_counts shape {(q.shape[0],)}, got {tuple(region_counts.shape)}"
            )
        counts = region_counts
        if tuple(region_packed_indices.shape) != (q.shape[0], self.topk_windows):
            raise ValueError(
                "Expected region_packed_indices shape "
                f"{(q.shape[0], self.topk_windows)}, got {tuple(region_packed_indices.shape)}"
            )
        packed = region_packed_indices
        if kv_seqlens.device != q.device:
            raise ValueError("kv_seqlens must be on the same device as q")
        if kv_seqlens.dtype != torch.int32:
            raise ValueError(f"kv_seqlens must be int32, got {kv_seqlens.dtype}")
        if tuple(kv_seqlens.shape) != (q.shape[0],):
            raise ValueError(
                f"Expected kv_seqlens shape {(q.shape[0],)}, got {tuple(kv_seqlens.shape)}"
            )
        seq_lens = kv_seqlens
        if 1 < self.variable_split_max <= 4 and n_split4 + n_split2 > int(q.shape[0]):
            raise ValueError(
                "invalid variable split plan: "
                f"batch={int(q.shape[0])}, n_split4={n_split4}, n_split2={n_split2}"
            )
        if not counts.is_contiguous():
            raise ValueError(
                f"region_counts must be contiguous, got stride={tuple(counts.stride())}"
            )
        if not packed.is_contiguous():
            raise ValueError(
                f"region_packed_indices must be contiguous, got stride={tuple(packed.stride())}"
            )
        if not seq_lens.is_contiguous():
            raise ValueError(
                f"kv_seqlens must be contiguous, got stride={tuple(seq_lens.stride())}"
            )
        return k_cache, v_cache, counts, packed, seq_lens, n_split4, n_split2

    def run(
        self,
        q,
        k_cache,
        v_cache,
        out,
        *,
        lse=None,
        region_counts,
        region_packed_indices,
        kv_seqlens=None,
        query_start_loc=None,
        valid_rows=None,
        softmax_scale: Optional[float] = None,
        stream: Optional[cuda.CUstream] = None,
    ):
        import torch

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(float(self.HEAD_DIM))
        if stream is not None:
            raise ValueError(
                "decode sparse attention uses the TVM-FFI environment stream"
            )

        k_cache, v_cache, counts, packed, seq_lens, n_split4, n_split2 = (
            self._validate_inputs(
                q,
                k_cache,
                v_cache,
                out,
                lse,
                region_counts,
                region_packed_indices,
                kv_seqlens,
                query_start_loc,
            )
        )
        if valid_rows is None:
            valid_rows = torch.full(
                (1,), int(q.shape[0]), dtype=torch.int32, device=q.device
            )
        if (
            valid_rows.device != q.device
            or valid_rows.dtype != torch.int32
            or valid_rows.ndim != 1
            or int(valid_rows.numel()) != 1
            or not valid_rows.is_contiguous()
        ):
            raise ValueError(
                "valid_rows must be a contiguous CUDA int32 tensor with shape [1]"
            )
        if query_start_loc is None:
            query_start_loc = valid_rows
        self._validate_pointer_alignment(
            q, name="q", min_align=self.REQUIRED_POINTER_ALIGN
        )
        self._validate_pointer_alignment(
            k_cache, name="k_cache", min_align=self.REQUIRED_POINTER_ALIGN
        )
        self._validate_pointer_alignment(
            v_cache, name="v_cache", min_align=self.REQUIRED_POINTER_ALIGN
        )
        self._validate_pointer_alignment(
            packed, name="region_packed_indices", min_align=8
        )
        self._validate_pointer_alignment(counts, name="region_counts", min_align=4)
        self._validate_pointer_alignment(seq_lens, name="kv_seqlens", min_align=4)
        self._validate_pointer_alignment(
            out, name="out", min_align=self.REQUIRED_POINTER_ALIGN
        )
        if lse is not None:
            self._validate_pointer_alignment(lse, name="lse", min_align=4)
        q_align = self.REQUIRED_POINTER_ALIGN
        k_align = self.REQUIRED_POINTER_ALIGN
        v_align = self.REQUIRED_POINTER_ALIGN
        packed_align = 8
        counts_align = 4
        seq_lens_align = 4
        out_align = self.REQUIRED_POINTER_ALIGN
        lse_align = 4 if lse is not None else None
        use_dynamic_split_kernel = (
            self.variable_split_max
            == _TokenWiseSparseDecodeWGMMA64GQA.DYNAMIC_SPLIT_CAPACITY
            and int(out.shape[0]) != int(q.shape[0])
        )
        tensor_signatures = _decode_compile_tensor_signatures(
            q,
            k_cache,
            v_cache,
            packed,
            counts,
            seq_lens,
            query_start_loc,
            valid_rows,
            out,
            lse,
        )
        compiled = _get_compiled_decode_kernel(
            self.dtype,
            int(self.logical_q_heads),
            int(self.HEAD_DIM),
            int(self.HEAD_DIM_V),
            int(self.block_n),
            int(self.num_threads),
            int(self.topk_windows),
            int(self.mtp_q_len),
            bool(self.batch_invariant),
            self.producer_k_warps,
            (
                _TokenWiseSparseDecodeWGMMA64GQA.DYNAMIC_SPLIT_CAPACITY
                if use_dynamic_split_kernel
                else 1
            ),
            (int(self.sm_count) if use_dynamic_split_kernel else None),
            *tensor_signatures,
            int(q_align),
            int(k_align),
            int(v_align),
            int(packed_align),
            int(counts_align),
            int(seq_lens_align),
            int(out_align),
            None if lse is None else int(lse_align),
            float(softmax_scale),
            utils.device_cache_key(q.device),
        )

        compiled_args = (
            q,
            k_cache,
            v_cache,
            None,
            None,
            packed,
            counts,
            seq_lens,
            query_start_loc,
            valid_rows,
            out,
            lse,
        )
        compiled(
            *compiled_args,
            cutlass.Float32(softmax_scale),
            int(n_split4),
            int(n_split2),
        )
        return out


__all__ = ["TokenWiseFlashAttnFwdSm90GQADecode"]
