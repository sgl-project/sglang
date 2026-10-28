# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float8E4M3FN, Float32, Int32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from sglang.kernels.ops.attention.flash_attn.cute import (
    blackwell_helpers as sm100_flash_utils,
)
from sglang.kernels.ops.attention.flash_attn.cute import (
    copy_utils,
)
from sglang.kernels.ops.attention.flash_attn.cute.utils import elem_pointer

from . import cute_utils as sparse_utils
from .compile_cache import cached_compile_function
from .decode_sm103_utils import elem_pointer_i64_offset

# NOTE(TP4): this decoding specialization is the TP4/H4 path.  N=16
# keeps one compact Gen5 output tile for the four logical heads.  HEAD_DIM is
# split into two exact K=128 segments; no dimension is padded or discarded.
Q_HEADS_PER_KV = 4
HEAD_DIM = 256
GQA_BLOCK_QH = 16
# Keep the logical q-slot stride at 32 positions.  The four H4 heads are
# encoded in the first 16 physical rows by the existing 8-D subgroup mapping.
GQA_BLOCK_Q = 32
MTP_QUERY_CAPACITY = 4
GQA_NUM_TMA_REGS = 32
GQA_NUM_MATH_REGS = 112
# One Gen5 swizzle period spans 8 rows. The decode cache uses one native M128
# region tile per physical indexer page.
PAGED_TMA_ROWS = 8
PAGED_TMA_PAIR_BYTES = PAGED_TMA_ROWS * HEAD_DIM
PAGED_Gen5_ATOM_BYTES = 128
PAGED_TMA_CHUNK_BYTES = PAGED_TMA_ROWS * PAGED_Gen5_ATOM_BYTES
PAGED_BULK_ROWS = 8
PAGED_SUPPORTED_SUMMARIES_PER_PAGE = (8, 32, 128)
# Each persistent CTA consumes a balanced contiguous chunk of native M128 Gen5
# units from the flattened batch. Shared memory retains a two-stage 256-region
# ring for producer/consumer overlap.
PAGED_PIPELINE_KV = 256
PAGED_STAGE_KV = 128
PAGED_NUM_K_STAGES = PAGED_PIPELINE_KV // PAGED_STAGE_KV
PAGED_K_SEGMENTS = 2
PAGED_K_SEGMENT = HEAD_DIM // PAGED_K_SEGMENTS
PAGED_Gen5_ATOM_STRIDE = PAGED_STAGE_KV * PAGED_Gen5_ATOM_BYTES
PAGED_TMA_PAIRS_PER_BULK = PAGED_BULK_ROWS // PAGED_TMA_ROWS
PAGED_BULKS_PER_STAGE = PAGED_STAGE_KV // PAGED_BULK_ROWS
PAGED_STAGE_BYTES = PAGED_STAGE_KV * HEAD_DIM
PAGED_BULK_SEGMENT_BYTES = PAGED_BULK_ROWS * PAGED_K_SEGMENT
PAGED_NUM_MATH_THREADS = 128
PAGED_NUM_PRODUCER_THREADS = 128
PAGED_NUM_MATH_WARPS = PAGED_NUM_MATH_THREADS // 32
PAGED_NUM_THREADS = PAGED_NUM_MATH_THREADS + PAGED_NUM_PRODUCER_THREADS
# Bound the persistent grid at one CTA/SM. The launch still takes the
# runtime minimum with max_work_items, so short/mixed varlen batches do not
# launch empty CTAs while long batches get one work item per CTA when possible.
PAGED_PERSISTENT_CTAS_PER_SM = 1
PAGED_MAX_SPLIT_N_FACTOR = 128
PAGED_SUPPORTED_SPLIT_N_FACTORS = tuple(range(1, PAGED_MAX_SPLIT_N_FACTOR + 1))
PAGED_DEFAULT_SPLIT_N_FACTOR: Optional[int] = None


def _cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


def _decode_validate_split_n_factor(split_n_factor: int) -> int:
    factor = int(split_n_factor)
    if factor < 1 or factor > PAGED_MAX_SPLIT_N_FACTOR:
        raise ValueError(
            f"split_n_factor must be in [1,{PAGED_MAX_SPLIT_N_FACTOR}], got {factor}"
        )
    return factor


def _decode_native_region_tiles_per_row(max_regions: int) -> int:
    return (int(max_regions) + PAGED_STAGE_KV - 1) // PAGED_STAGE_KV


def _decode_split_n_task_multiplier(max_regions: int, split_n_factor: int) -> int:
    factor = _decode_validate_split_n_factor(split_n_factor)
    return min(factor, _decode_native_region_tiles_per_row(max_regions))


def _decode_select_split_n_factor(
    request_capacity: int, max_regions: int, sm_count: int
) -> int:
    if request_capacity <= 0 or max_regions <= 0 or sm_count <= 0:
        raise ValueError("request_capacity, max_regions, and sm_count must be positive")
    target_workers = max(1, (sm_count + request_capacity - 1) // request_capacity)
    max_workers = _decode_native_region_tiles_per_row(max_regions)
    return min(target_workers, max_workers, PAGED_MAX_SPLIT_N_FACTOR)


def _load_f32_global(tensor: cute.Tensor, coord: tuple[Int32, Int32, Int32]) -> Float32:
    src = cute.make_tensor(
        elem_pointer(tensor, coord),
        cute.make_layout((1,), stride=(1,)),
    )
    return Float32(src[0])


@cute.jit
def _device_contract_trap() -> None:
    # Note(wangbojun/codex): Runtime metadata stays device-resident under CUDA
    # Graph, so malformed active metadata must fail here without a host read.
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _mul_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _tmem_load_f32x2(tmem_addr: Int32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(tmem_addr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.32x32b.x2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(
            llvm.bitcast(
                T.f32(),
                llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        )
        for i in range(2)
    )


@cute.jit
def _gen5_gemm_fragment_accumulator(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
) -> None:
    # The Gen5 fragment count is derived from the operand layout. SM100 FP8
    # layouts do not retain SM90's fixed four-fragment decomposition.
    rank = cute.rank(tCrA)
    if cutlass.const_expr(rank == 4):
        k_fragments = cute.size(tCrA.shape[3])
    else:
        k_fragments = cute.size(tCrA.shape[2])
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    for k in cutlass.range_constexpr(k_fragments):
        mma_atom.set(
            tcgen05.Field.ACCUMULATE,
            not cutlass.const_expr(k == 0),
        )
        if cutlass.const_expr(rank == 4):
            rA = tCrA[None, None, None, k]
            rB = tCrB[None, None, None, k]
        else:
            rA = tCrA[None, None, k]
            rB = tCrB[None, None, k]
        cute.gemm(mma_atom, acc, rA, rB, acc)


def _tensor_signature_dynamic(
    x: torch.Tensor,
    *,
    dynamic_shape_dims: tuple[int, ...],
    dynamic_stride_dims: tuple[int, ...] = (),
) -> tuple[torch.dtype, tuple[object, ...], tuple[object, ...]]:
    shape: list[object] = [int(v) for v in x.shape]
    stride: list[object] = [int(v) for v in x.stride()]
    for dim in dynamic_shape_dims:
        shape[int(dim)] = None
    for dim in dynamic_stride_dims:
        stride[int(dim)] = None
    return x.dtype, tuple(shape), tuple(stride)


def _placeholder_from_signature(
    signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    *,
    device: torch.device,
    dynamic_shape_fill: int,
    dynamic_stride_fill: int = 1,
) -> torch.Tensor:
    dtype, shape, stride = signature
    concrete_shape = tuple(
        int(dynamic_shape_fill) if dim is None else int(dim) for dim in shape
    )
    concrete_stride = tuple(
        int(dynamic_stride_fill) if dim is None else int(dim) for dim in stride
    )
    return torch.empty_strided(
        concrete_shape,
        concrete_stride,
        device=device,
        dtype=dtype,
    )


@cute.kernel
def _batch_decode_logits_paged_sm100_kernel(
    tma_atom_Q: cute.CopyAtom,
    tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom,
    tma_tensor_K: cute.Tensor,
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    mRowReqIdx: cute.Tensor,
    mRowTableIdx: cute.Tensor,
    mRowStarts: cute.Tensor,
    mRowEnds: cute.Tensor,
    mRequestLimits: cute.Tensor,
    mValidTokens: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sQ_layout_fp8: cute.ComposedLayout,
    sK_layout_fp8: cute.ComposedLayout,
    mW: cute.Tensor,
    mO: cute.Tensor,
    max_regions: Int32,
    query_rows: Int32,
    num_reqs: Int32,
    block_table_cols: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    max_work_items: Int32,
    output_group_rows: cutlass.Constexpr[int],
):
    """Persistent varlen TP4 FP8 QK over paged FP8 summaries.

    Work is flattened over the maximum live M128 tiles and partitioned into
    balanced contiguous grid-wide chunks. Split-N controls the task budget,
    while the persistent grid caps active tasks at one CTA per SM.
    ``row_req_idx`` maps every query row to its request owner for validity;
    ``row_table_idx`` maps it to the row used for page-table lookup. Runtime
    row bounds reject unused graph capacity before any TMA or Gen5. A two-stage
    ring loads 128 regions per stage in native 8-row page fragments. Q and K
    retain their native K256 SW128 layout, while Gen5 consumes two K128
    fragments.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    work_id = cute.arch.block_idx()[0]
    work_stride = cute.arch.grid_dim()[0]

    if work_id < max_work_items:
        if warp_idx == 0 and (tidx % 32) == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)

        smem = cutlass.utils.SmemAllocator()
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8)], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8)], 1024
        ]
        mbar_ptr_q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_k_struct = cute.struct.MemRange[cutlass.Int64, PAGED_NUM_K_STAGES * 2]
        mbar_ptr_mma_struct = cute.struct.MemRange[cutlass.Int64, 1]

        @cute.struct
        class SharedStorage:
            mbar_ptr_q: mbar_ptr_q_struct
            mbar_ptr_k: mbar_ptr_k_struct
            mbar_ptr_mma: mbar_ptr_mma_struct
            sQ: sQ_struct
            sK: sK_struct

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout_fp8.outer, swizzle=sQ_layout_fp8.inner)
        sK = storage.sK.get_tensor(sK_layout_fp8.outer, swizzle=sK_layout_fp8.inner)
        # The K page cache is physically interleaved by the producer of the
        # cache. Expose the same allocation as a flat [bytes, pages] tensor for
        # TMA while retaining the Gen5-composed tensor above for MMA.
        sK_pages = cute.make_tensor(
            cute.recast_ptr(storage.sK.data_ptr(), dtype=Uint8),
            cute.make_layout(
                (PAGED_STAGE_BYTES * PAGED_NUM_K_STAGES,),
                stride=(1,),
            ),
        )
        gQ = cute.local_tile(tma_tensor_Q, (GQA_BLOCK_QH, HEAD_DIM), (None, 0))
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, cute.rank(sQ) - 1),
            cute.group_modes(gQ, 0, cute.rank(gQ) - 1),
        )
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, PAGED_NUM_MATH_WARPS
        )
        pipeline_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_q.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=GQA_BLOCK_QH * HEAD_DIM,
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_k.data_ptr(),
            num_stages=PAGED_NUM_K_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=PAGED_STAGE_BYTES,
        )
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, PAGED_NUM_K_STAGES
        )
        k_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PAGED_NUM_K_STAGES
        )
        mma_bar_ptr = storage.mbar_ptr_mma.data_ptr()
        request_capacity = query_rows // Int32(MTP_QUERY_CAPACITY)
        capacity_tiles_per_request = cutlass.max(
            Int32(1),
            (Int32(mRequestLimits[1]) + Int32(PAGED_STAGE_KV - 1))
            // Int32(PAGED_STAGE_KV),
        )
        total_capacity_tiles = request_capacity * capacity_tiles_per_request
        if tidx == Int32(0):
            cute.arch.mbarrier_init(mma_bar_ptr, Int32(1))
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if tidx >= PAGED_NUM_MATH_THREADS:
            cute.arch.warpgroup_reg_dealloc(GQA_NUM_TMA_REGS)
            producer_idx = tidx - Int32(PAGED_NUM_MATH_THREADS)
            lane_idx_tma = producer_idx % Int32(32)
            is_tma_leader = (producer_idx // Int32(32)) == Int32(
                0
            ) and lane_idx_tma == Int32(0)
            if is_tma_leader:
                producer_work_id = work_id
                while producer_work_id < max_work_items:
                    task_tile_begin = (
                        producer_work_id * total_capacity_tiles
                    ) // max_work_items
                    task_tile_end = (
                        (producer_work_id + Int32(1)) * total_capacity_tiles
                    ) // max_work_items
                    global_tile_idx = task_tile_begin
                    while global_tile_idx < task_tile_end:
                        request_idx = global_tile_idx // capacity_tiles_per_request
                        request_tile_base = request_idx * capacity_tiles_per_request
                        request_tile_end = (
                            request_tile_base + capacity_tiles_per_request
                        )
                        segment_tile_end = task_tile_end
                        if segment_tile_end > request_tile_end:
                            segment_tile_end = request_tile_end
                        region_tile_begin = global_tile_idx - request_tile_base
                        region_tile_limit = segment_tile_end - request_tile_base
                        q_row = request_idx * Int32(MTP_QUERY_CAPACITY)
                        valid_request_count = Int32(mRequestLimits[0])
                        valid_token_count = Int32(mValidTokens[0])
                        scalar_contract_ok = (
                            (valid_request_count >= Int32(0))
                            & (
                                valid_request_count
                                <= query_rows // Int32(MTP_QUERY_CAPACITY)
                            )
                            & (valid_token_count >= Int32(0))
                            & (valid_token_count <= query_rows)
                        )
                        if scalar_contract_ok == Int32(0):
                            _device_contract_trap()
                        q_valid = (q_row < query_rows) & (q_row < valid_token_count)
                        req_idx = Int32(-1)
                        table_idx = Int32(-1)
                        row_start = Int32(0)
                        row_end = Int32(0)
                        if q_valid:
                            req_idx = Int32(mRowReqIdx[q_row])
                            table_idx = Int32(mRowTableIdx[q_row])
                            leader_active = req_idx >= Int32(0)
                            if leader_active:
                                row_start = Int32(mRowStarts[q_row])
                                row_end = Int32(mRowEnds[q_row])
                                leader_contract_ok = (
                                    (req_idx < valid_request_count)
                                    & (table_idx >= Int32(0))
                                    & (table_idx < num_reqs)
                                    & (row_start >= Int32(0))
                                    & (row_start % Int32(PAGED_BULK_ROWS) == Int32(0))
                                    & (row_end >= row_start)
                                    & (row_end <= max_regions)
                                )
                                if leader_contract_ok == Int32(0):
                                    _device_contract_trap()
                                for q_slot in cutlass.range_constexpr(
                                    1, MTP_QUERY_CAPACITY
                                ):
                                    slot_row = q_row + Int32(q_slot)
                                    if slot_row < valid_token_count:
                                        slot_req_idx = Int32(mRowReqIdx[slot_row])
                                        slot_table_idx = Int32(mRowTableIdx[slot_row])
                                        slot_inactive = (slot_req_idx < Int32(0)) & (
                                            slot_table_idx < Int32(0)
                                        )
                                        slot_row_start = Int32(mRowStarts[slot_row])
                                        slot_row_end = Int32(mRowEnds[slot_row])
                                        slot_matches = (
                                            (slot_req_idx == req_idx)
                                            & (slot_table_idx == table_idx)
                                            & (slot_row_start >= Int32(0))
                                            & (
                                                slot_row_start % Int32(PAGED_BULK_ROWS)
                                                == Int32(0)
                                            )
                                            & (slot_row_end >= slot_row_start)
                                            & (slot_row_end <= max_regions)
                                        )
                                        if (slot_inactive | slot_matches) == Int32(0):
                                            _device_contract_trap()
                                        if slot_matches:
                                            if slot_row_start < row_start:
                                                row_start = slot_row_start
                                            if slot_row_end > row_end:
                                                row_end = slot_row_end
                                q_valid = leader_contract_ok
                            else:
                                if table_idx >= Int32(0):
                                    _device_contract_trap()
                                if region_tile_begin == Int32(0):
                                    for q_slot in cutlass.range_constexpr(
                                        1, MTP_QUERY_CAPACITY
                                    ):
                                        slot_row = q_row + Int32(q_slot)
                                        if slot_row < valid_token_count:
                                            slot_req_idx = Int32(mRowReqIdx[slot_row])
                                            slot_table_idx = Int32(
                                                mRowTableIdx[slot_row]
                                            )
                                            if (slot_req_idx >= Int32(0)) | (
                                                slot_table_idx >= Int32(0)
                                            ):
                                                _device_contract_trap()
                                q_valid = leader_active
                        req_len = row_end - row_start
                        num_region_tiles = (
                            req_len + Int32(PAGED_STAGE_KV - 1)
                        ) // Int32(PAGED_STAGE_KV)
                        worker_found = q_valid & (region_tile_begin < num_region_tiles)
                        if worker_found == Int32(1):
                            pipeline_q.producer_acquire(q_prod)
                            cute.copy(
                                tma_atom_Q,
                                tQgQ[None, request_idx],
                                tQsQ[None, q_prod.index],
                                tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                            )
                            q_prod.advance()
                            active_tile_limit = region_tile_limit
                            if active_tile_limit > num_region_tiles:
                                active_tile_limit = num_region_tiles
                            region_tile_idx = region_tile_begin
                            while region_tile_idx < active_tile_limit:
                                tile_base = row_start + region_tile_idx * Int32(
                                    PAGED_STAGE_KV
                                )
                                for stage_iter in cutlass.range_constexpr(1):
                                    stage_base = tile_base + Int32(
                                        stage_iter * PAGED_STAGE_KV
                                    )
                                    if stage_base < row_end:
                                        pipeline_k.producer_acquire(k_prod)
                                        stage_offset = Int32(k_prod.index) * Int32(
                                            PAGED_STAGE_BYTES
                                        )
                                        for local_bulk in cutlass.range_constexpr(
                                            PAGED_BULKS_PER_STAGE
                                        ):
                                            logical_region = stage_base + Int32(
                                                local_bulk * PAGED_BULK_ROWS
                                            )
                                            valid_page = logical_region < row_end
                                            logical_page = (
                                                logical_region // summaries_per_page
                                            )
                                            if valid_page & (
                                                logical_page >= block_table_cols
                                            ):
                                                _device_contract_trap()
                                            safe_col = cutlass.select_(
                                                valid_page, logical_page, Int32(0)
                                            )
                                            physical_page = mBlockTable[
                                                cutlass.Int64(table_idx)
                                                * block_table_cols
                                                + safe_col
                                            ]
                                            physical_page_valid = (
                                                physical_page >= Int32(0)
                                            ) & (physical_page < num_pages)
                                            if valid_page & (
                                                physical_page_valid == Int32(0)
                                            ):
                                                _device_contract_trap()
                                            valid_page = (
                                                valid_page & physical_page_valid
                                            )
                                            safe_page = cutlass.select_(
                                                valid_page, physical_page, Int32(0)
                                            )
                                            page_slot = (
                                                logical_region
                                                - logical_page * summaries_per_page
                                            )
                                            pair_in_page = page_slot // Int32(
                                                PAGED_TMA_ROWS
                                            )
                                            safe_pair_in_page = cutlass.select_(
                                                valid_page, pair_in_page, Int32(0)
                                            )
                                            row_base = stage_offset + Int32(
                                                local_bulk * PAGED_BULK_SEGMENT_BYTES
                                            )
                                            gK_page = cute.domain_offset(
                                                (
                                                    Int32(0),
                                                    Int32(0),
                                                    Int32(0),
                                                    safe_pair_in_page,
                                                    safe_page,
                                                ),
                                                tma_tensor_K,
                                            )
                                            gK_page = cute.make_tensor(
                                                gK_page.iterator,
                                                cute.make_layout(
                                                    (
                                                        PAGED_Gen5_ATOM_BYTES,
                                                        PAGED_TMA_ROWS,
                                                        PAGED_K_SEGMENTS,
                                                        PAGED_TMA_PAIRS_PER_BULK,
                                                    ),
                                                    stride=(
                                                        tma_tensor_K.stride[0],
                                                        tma_tensor_K.stride[1],
                                                        tma_tensor_K.stride[2],
                                                        tma_tensor_K.stride[3],
                                                    ),
                                                ),
                                            )
                                            sK_page_ptr0 = elem_pointer_i64_offset(
                                                sK_pages, (row_base,)
                                            )
                                            sK_page = cute.make_tensor(
                                                sK_page_ptr0,
                                                cute.make_layout(
                                                    (
                                                        PAGED_Gen5_ATOM_BYTES,
                                                        PAGED_TMA_ROWS,
                                                        PAGED_K_SEGMENTS,
                                                        PAGED_TMA_PAIRS_PER_BULK,
                                                    ),
                                                    stride=(
                                                        1,
                                                        PAGED_Gen5_ATOM_BYTES,
                                                        PAGED_STAGE_KV
                                                        * PAGED_K_SEGMENT,
                                                        PAGED_TMA_CHUNK_BYTES,
                                                    ),
                                                ),
                                            )
                                            load_K, _, _ = copy_utils.tma_get_copy_fn(
                                                tma_atom_K,
                                                0,
                                                cute.make_layout(1),
                                                gK_page,
                                                sK_page,
                                                single_stage=True,
                                            )
                                            load_K(
                                                tma_bar_ptr=pipeline_k.producer_get_barrier(
                                                    k_prod
                                                )
                                            )
                                        k_prod.advance()
                                region_tile_idx = region_tile_idx + Int32(1)
                        global_tile_idx = segment_tile_end
                    producer_work_id = producer_work_id + work_stride
        else:
            cute.arch.warpgroup_reg_alloc(GQA_NUM_MATH_REGS)
            warp_idx = tidx // Int32(32)
            warp_group_idx = cute.arch.make_warp_uniform(warp_idx // Int32(4))
            warp_group_thread_layout = cute.make_layout(
                PAGED_NUM_MATH_THREADS // 128, stride=128
            )
            wg_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
            tmem_base = cute.make_ptr(
                Float32,
                0,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            mma_phase = Int32(0)
            work_id = cute.arch.block_idx()[0]
            while work_id < max_work_items:
                task_tile_begin = (work_id * total_capacity_tiles) // max_work_items
                task_tile_end = (
                    (work_id + Int32(1)) * total_capacity_tiles
                ) // max_work_items
                global_tile_idx = task_tile_begin
                while global_tile_idx < task_tile_end:
                    request_idx = global_tile_idx // capacity_tiles_per_request
                    request_tile_base = request_idx * capacity_tiles_per_request
                    request_tile_end = request_tile_base + capacity_tiles_per_request
                    segment_tile_end = task_tile_end
                    if segment_tile_end > request_tile_end:
                        segment_tile_end = request_tile_end
                    region_tile_begin = global_tile_idx - request_tile_base
                    region_tile_limit = segment_tile_end - request_tile_base
                    q_row = request_idx * Int32(MTP_QUERY_CAPACITY)
                    q_valid = q_row < query_rows
                    req_idx = Int32(-1)
                    table_idx = Int32(-1)
                    row_start = Int32(0)
                    row_end = Int32(0)
                    if q_valid:
                        req_idx = Int32(mRowReqIdx[q_row])
                        table_idx = Int32(mRowTableIdx[q_row])
                        q_valid = (
                            q_valid
                            & (req_idx >= Int32(0))
                            & (req_idx < Int32(mRequestLimits[0]))
                        )
                        q_valid = q_valid & (q_row < Int32(mValidTokens[0]))
                        q_valid = (
                            q_valid & (table_idx >= Int32(0)) & (table_idx < num_reqs)
                        )
                    if q_valid:
                        row_start = Int32(mRowStarts[q_row])
                        row_end = Int32(mRowEnds[q_row])
                    slot_valid = []
                    slot_row_starts = []
                    slot_row_ends = []
                    for q_slot in cutlass.range_constexpr(MTP_QUERY_CAPACITY):
                        slot_row = q_row + Int32(q_slot)
                        current_slot_valid = (
                            q_valid
                            & (slot_row < Int32(mValidTokens[0]))
                            & (Int32(mRowReqIdx[slot_row]) == req_idx)
                            & (Int32(mRowTableIdx[slot_row]) == table_idx)
                        )
                        current_slot_start = Int32(mRowStarts[slot_row])
                        current_slot_end = Int32(mRowEnds[slot_row])
                        slot_valid.append(current_slot_valid)
                        slot_row_starts.append(current_slot_start)
                        slot_row_ends.append(current_slot_end)
                        if current_slot_valid:
                            if current_slot_start < row_start:
                                row_start = current_slot_start
                            if current_slot_end > row_end:
                                row_end = current_slot_end
                    req_len = row_end - row_start
                    found = (
                        q_valid
                        & (req_idx >= Int32(0))
                        & (table_idx < num_reqs)
                        & (row_start >= Int32(0))
                        & (row_start % Int32(PAGED_BULK_ROWS) == Int32(0))
                        & (row_end <= max_regions)
                        & (req_len > Int32(0))
                        & (
                            region_tile_begin
                            < (req_len + Int32(PAGED_STAGE_KV - 1))
                            // Int32(PAGED_STAGE_KV)
                        )
                    )
                    if found == Int32(1):
                        pipeline_q.consumer_wait(q_cons)
                        q_stage = q_cons.index
                        slot_weights = []
                        for q_slot in cutlass.range_constexpr(MTP_QUERY_CAPACITY):
                            slot_weights.append(
                                [
                                    _load_f32_global(
                                        mW,
                                        (request_idx, Int32(q_slot), Int32(head)),
                                    )
                                    for head in range(Q_HEADS_PER_KV)
                                ]
                            )

                        num_region_tiles = (
                            req_len + Int32(PAGED_STAGE_KV - 1)
                        ) // Int32(PAGED_STAGE_KV)
                        active_tile_limit = region_tile_limit
                        if active_tile_limit > num_region_tiles:
                            active_tile_limit = num_region_tiles
                        region_tile_idx = region_tile_begin
                        while region_tile_idx < active_tile_limit:
                            tile_base = row_start + region_tile_idx * Int32(
                                PAGED_STAGE_KV
                            )
                            tile_end = tile_base + Int32(PAGED_STAGE_KV)
                            if tile_end > row_end:
                                tile_end = row_end
                            for stage_iter in cutlass.range_constexpr(1):
                                kv_base = tile_base + Int32(stage_iter * PAGED_STAGE_KV)
                                if kv_base < row_end:
                                    pipeline_k.consumer_wait(k_cons)
                                    k_stage = k_cons.index
                                    sK_stage = sK[None, None, None, k_stage]
                                    sQ_stage = sQ[None, None, None, q_stage]
                                    tCrA = tiled_mma.make_fragment_A(
                                        wg_mma.partition_A(sK_stage)
                                    )
                                    tCrB = tiled_mma.make_fragment_B(
                                        wg_mma.partition_B(sQ_stage)
                                    )
                                    if cutlass.const_expr(cute.rank(tCrA) == 4):
                                        tCrA = cute.group_modes(tCrA, 0, 2)
                                    if cutlass.const_expr(cute.rank(tCrB) == 4):
                                        tCrB = cute.group_modes(tCrB, 0, 2)
                                    if warp_idx == Int32(0):
                                        sm100_flash_utils.gemm_ptx_partial(
                                            tiled_mma.op,
                                            Int32(0),
                                            tCrA,
                                            tCrB,
                                            sK_stage,
                                            sQ_stage,
                                            zero_init=True,
                                        )
                                        with cute.arch.elect_one():
                                            tcgen05.commit(mma_bar_ptr)
                                    cute.arch.mbarrier_wait(mma_bar_ptr, mma_phase)
                                    mma_phase ^= Int32(1)
                                    cute.arch.fence_acq_rel_cta()
                                    global_k = kv_base + tidx
                                    slot_pair01 = []
                                    slot_pair23 = []
                                    for q_slot in cutlass.range_constexpr(
                                        MTP_QUERY_CAPACITY
                                    ):
                                        slot_pair01.append(
                                            _tmem_load_f32x2(
                                                Int32(tmem_base.toint() + q_slot * 2)
                                            )
                                        )
                                        slot_pair23.append(
                                            _tmem_load_f32x2(
                                                Int32(
                                                    tmem_base.toint() + 8 + q_slot * 2
                                                )
                                            )
                                        )
                                    cute.arch.fence_view_async_tmem_load()
                                    pipeline_k.consumer_release(k_cons)
                                    k_cons.advance()
                                    for q_slot in cutlass.range_constexpr(
                                        MTP_QUERY_CAPACITY
                                    ):
                                        products = []
                                        for head in cutlass.range_constexpr(
                                            Q_HEADS_PER_KV
                                        ):
                                            raw = (
                                                slot_pair01[q_slot][head]
                                                if head < 2
                                                else slot_pair23[q_slot][head - 2]
                                            )
                                            products.append(
                                                _mul_rn_f32(
                                                    cute.arch.fmax(raw, Float32(0.0)),
                                                    slot_weights[q_slot][head],
                                                )
                                            )
                                        weighted = (products[0] + products[2]) + (
                                            products[1] + products[3]
                                        )
                                        slot_row = q_row + Int32(q_slot)
                                        if (
                                            (Int32(q_slot) < Int32(output_group_rows))
                                            & slot_valid[q_slot]
                                            & (global_k >= slot_row_starts[q_slot])
                                            & (global_k < slot_row_ends[q_slot])
                                        ):
                                            dst = cute.make_tensor(
                                                elem_pointer(
                                                    mO,
                                                    (
                                                        (
                                                            request_idx
                                                            * Int32(output_group_rows)
                                                            + Int32(q_slot)
                                                        )
                                                        * max_regions
                                                        + global_k,
                                                    ),
                                                ),
                                                cute.make_layout((1,), stride=(1,)),
                                            )
                                            dst[0] = mO.element_type(weighted)
                            region_tile_idx = region_tile_idx + Int32(1)
                        pipeline_q.consumer_release(q_cons)
                        q_cons.advance()
                    global_tile_idx = segment_tile_end
                work_id = work_id + work_stride


# Note(wangbojun/codex): CuTe's persistent compile cache does not include the
# bodies of called @cute.jit helpers in the entrypoint key.  Keep this
# entrypoint versioned whenever the Gen5 helper changes so serving warmup
# cannot load a cubin compiled with older accumulation semantics.
def _make_batch_decode_logits_paged_sm100_call_v32(
    split_n_factor: int, output_group_rows: int
):
    split_n_factor = _decode_validate_split_n_factor(split_n_factor)

    @cute.jit
    def _call_v32(
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mBlockTable: cute.Tensor,
        mRowReqIdx: cute.Tensor,
        mRowTableIdx: cute.Tensor,
        mRowStarts: cute.Tensor,
        mRowEnds: cute.Tensor,
        mRequestLimits: cute.Tensor,
        mValidTokens: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        stream,
        max_regions: Int32,
        query_rows: Int32,
        num_reqs: Int32,
        block_table_cols: Int32,
        num_pages: Int32,
        summaries_per_page: Int32,
        max_work_items: Int32,
        persistent_ctas: Int32,
    ):
        mma_tiler_mnk = (PAGED_STAGE_KV, GQA_BLOCK_QH, HEAD_DIM)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (PAGED_STAGE_KV, GQA_BLOCK_QH),
        )
        sQ_layout_fp8 = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, Float8E4M3FN, 1
        )
        sK_layout_fp8 = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, Float8E4M3FN, PAGED_NUM_K_STAGES
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(sQ_layout_fp8, mode=[0, 1, 2]),
            mma_tiler_mnk,
            tiled_mma,
            1,
        )
        mKCache_pages = cute.make_tensor(
            cute.recast_ptr(mKCache.iterator, dtype=Uint8),
            cute.make_layout(
                (
                    PAGED_Gen5_ATOM_BYTES,
                    PAGED_TMA_ROWS,
                    PAGED_K_SEGMENTS,
                    summaries_per_page // Int32(PAGED_TMA_ROWS),
                    mKCache.shape[0],
                ),
                stride=(
                    1,
                    PAGED_Gen5_ATOM_BYTES,
                    PAGED_TMA_CHUNK_BYTES,
                    Int32(PAGED_TMA_PAIR_BYTES),
                    summaries_per_page * Int32(HEAD_DIM),
                ),
            ),
        )
        sK_page_layout = cute.make_layout(
            (
                PAGED_Gen5_ATOM_BYTES,
                PAGED_TMA_ROWS,
                PAGED_K_SEGMENTS,
                PAGED_TMA_PAIRS_PER_BULK,
            ),
            stride=(
                1,
                PAGED_Gen5_ATOM_BYTES,
                PAGED_STAGE_KV * PAGED_K_SEGMENT,
                PAGED_TMA_CHUNK_BYTES,
            ),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mKCache_pages,
            sK_page_layout,
            (
                PAGED_Gen5_ATOM_BYTES,
                PAGED_TMA_ROWS,
                PAGED_K_SEGMENTS,
                PAGED_TMA_PAIRS_PER_BULK,
            ),
        )
        _batch_decode_logits_paged_sm100_kernel(
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            mQ,
            mKCache,
            mBlockTable,
            mRowReqIdx,
            mRowTableIdx,
            mRowStarts,
            mRowEnds,
            mRequestLimits,
            mValidTokens,
            tiled_mma,
            sQ_layout_fp8,
            sK_layout_fp8,
            mW,
            mO,
            max_regions,
            query_rows,
            num_reqs,
            block_table_cols,
            num_pages,
            summaries_per_page,
            max_work_items,
            output_group_rows,
        ).launch(
            grid=(persistent_ctas, 1, 1), block=[PAGED_NUM_THREADS, 1, 1], stream=stream
        )

    return _call_v32


@cached_compile_function
def _compile_paged_for_signature(
    device_key: tuple[str, Optional[int]],
    split_n_factor: int,
    output_group_rows: int,
    q_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_cache_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    block_table_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_req_idx_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_table_idx_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_starts_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    row_ends_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    request_limits_signature: tuple[
        torch.dtype, tuple[object, ...], tuple[object, ...]
    ],
    valid_tokens_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
):
    device = sparse_utils.device_from_cache_key(device_key)
    q_runtime = _placeholder_from_signature(
        q_runtime_signature,
        device=device,
        dynamic_shape_fill=GQA_BLOCK_QH,
    )
    k_cache = _placeholder_from_signature(
        k_cache_signature,
        device=device,
        dynamic_shape_fill=1,
        # Keep a valid compact stride order while the outer page stride is
        # marked symbolic below.  The value is only a fake descriptor value.
        dynamic_stride_fill=8192,
    )
    block_table_flat = _placeholder_from_signature(
        block_table_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_req_idx = _placeholder_from_signature(
        row_req_idx_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_table_idx = _placeholder_from_signature(
        row_table_idx_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_starts = _placeholder_from_signature(
        row_starts_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    row_ends = _placeholder_from_signature(
        row_ends_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    request_limits = _placeholder_from_signature(
        request_limits_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    valid_tokens = _placeholder_from_signature(
        valid_tokens_signature,
        device=device,
        dynamic_shape_fill=1,
    )
    w_mat = _placeholder_from_signature(
        w_signature,
        device=device,
        dynamic_shape_fill=GQA_BLOCK_Q,
    )
    out = _placeholder_from_signature(
        out_signature,
        device=device,
        dynamic_shape_fill=PAGED_PIPELINE_KV,
    )
    fQ = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_runtime,
        alignment=16,
        dynamic_shape_dim=0,
        divisibility=GQA_BLOCK_QH,
    )
    fK = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_cache,
        alignment=16,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    fBlockTable = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        block_table_flat,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowReqIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_req_idx,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowTableIdx = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_table_idx,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowStarts = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_starts,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRowEnds = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        row_ends,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fRequestLimits = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        request_limits,
        alignment=4,
        dynamic_shape_dim=0,
    )
    fValidTokens = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        valid_tokens,
        alignment=4,
        dynamic_shape_dim=0,
    )
    fW = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_mat,
        alignment=16,
        dynamic_shape_dim=0,
    )
    fO = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out,
        alignment=16,
        dynamic_shape_dim=0,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _make_batch_decode_logits_paged_sm100_call_v32(
            split_n_factor, output_group_rows
        ),
        fQ,
        fK,
        fBlockTable,
        fRowReqIdx,
        fRowTableIdx,
        fRowStarts,
        fRowEnds,
        fRequestLimits,
        fValidTokens,
        fW,
        fO,
        stream_fake,
        Int32(PAGED_PIPELINE_KV),
        Int32(1),
        Int32(1),
        Int32(2),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
    )


def launch_paged_scoring(
    summary,
    block_table,
    row_req_idx,
    row_table_idx,
    row_starts,
    row_ends,
    q_runtime,
    kernel_weights,
    out,
    request_limits,
    valid_tokens,
    output_group_rows,
):
    request_capacity = q_runtime.shape[0] // GQA_BLOCK_QH
    query_rows = request_capacity * MTP_QUERY_CAPACITY
    max_regions = out.shape[1]
    summaries_per_page = summary.shape[1]
    k_input = summary.view(torch.float8_e4m3fn)
    sm_count = torch.cuda.get_device_properties(out.device).multi_processor_count
    split_n_factor = _decode_select_split_n_factor(
        request_capacity, max_regions, sm_count
    )
    paged = _compile_paged_for_signature(
        sparse_utils.device_cache_key(out.device),
        split_n_factor,
        output_group_rows,
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(
            k_input, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        _tensor_signature_dynamic(block_table.reshape(-1), dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_req_idx, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_table_idx, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_starts, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(row_ends, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(request_limits, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(valid_tokens, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(out.reshape(-1), dynamic_shape_dims=(0,)),
    )
    max_work_items = min(
        request_capacity * _decode_split_n_task_multiplier(max_regions, split_n_factor),
        request_capacity * _decode_native_region_tiles_per_row(max_regions),
        sm_count * PAGED_PERSISTENT_CTAS_PER_SM,
    )
    paged(
        q_runtime,
        k_input,
        block_table.reshape(-1),
        row_req_idx,
        row_table_idx,
        row_starts,
        row_ends,
        request_limits,
        valid_tokens,
        kernel_weights,
        out.reshape(-1),
        Int32(max_regions),
        Int32(query_rows),
        Int32(block_table.shape[0]),
        Int32(block_table.shape[1]),
        Int32(summary.shape[0]),
        Int32(summaries_per_page),
        Int32(max_work_items),
        Int32(max_work_items),
    )
