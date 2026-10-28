# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
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
from .sm100_union.sparse_gqa.kernel_utils.fp_compat import cvt_f32_to_e4m3

HEAD_DIM = 256
SUMMARY_KV_HEADS = 1
COMPILE_PLACEHOLDER_DIM = 1
SUPPORTED_Q_HEADS_PER_KV = (2, 4, 8)
PREFILL_BLOCK_QH = 128
PREFILL_BLOCK_KV = 256
PREFILL_K_SEGMENT = HEAD_DIM // 2
PREFILL_TMA_ROWS = 8
PREFILL_SUPPORTED_SUMMARIES_PER_PAGE = (8, 32, 128)
PREFILL_GEN5_ATOM_BYTES = 128
PREFILL_TMA_CHUNK_BYTES = PREFILL_TMA_ROWS * PREFILL_K_SEGMENT
PREFILL_TMA_PAGE_STRIDE_BYTES = PREFILL_TMA_ROWS * HEAD_DIM
PREFILL_Q_STAGES = 2
PREFILL_WEIGHT_STAGE_BYTES = PREFILL_BLOCK_QH * 4
# Each K stage stores both exact D=128 segments.  Stage depth changes must be
# revalidated together with occupancy and the barrier protocol.
PREFILL_K_STAGES = 2
PREFILL_S_STAGES = 3
PREFILL_NUM_CONSUMER_THREADS = 256
PREFILL_CONSUMER_GROUP_THREADS = 128
PREFILL_NUM_ROLE_THREADS = 128
PREFILL_NUM_THREADS = PREFILL_NUM_CONSUMER_THREADS + PREFILL_NUM_ROLE_THREADS
PREFILL_NUM_WARPS = PREFILL_NUM_THREADS // 32
LANES_PER_WARP = 32
WARPS_PER_WARPGROUP = 4
REGION_TOKENS = 8
M_BARRIER_PTRS_PER_STAGE = 2
PREFILL_NUM_CONSUMER_GROUPS = 2
PREFILL_CONSUMER_WARPS_PER_GROUP = WARPS_PER_WARPGROUP
PREFILL_NUM_ROLE_REGS = 56
PREFILL_NUM_CONSUMER_REGS = 224
PREFILL_PREPARE_THREADS = 256
PREFILL_MMA_TILE_M = 128
PREFILL_SCORE_ROWS = PREFILL_MMA_TILE_M
# DeepGEMM SM100 role order: two math warpgroups, then Q producer, K producer,
# UMMA producer, and the remaining specialized warp for output ownership.
PREFILL_TMA_Q_LOAD_WARP = PREFILL_NUM_CONSUMER_THREADS // 32
PREFILL_TMA_K_LOAD_WARP = PREFILL_TMA_Q_LOAD_WARP + 1
PREFILL_MMA_WARP = PREFILL_TMA_Q_LOAD_WARP + 2
PREFILL_TMA_STORE_WARP = 11
PREFILL_TMEM_REQUIRED_COLUMNS = PREFILL_S_STAGES * PREFILL_BLOCK_QH
PREFILL_TMEM_COLUMNS = 1 << (PREFILL_TMEM_REQUIRED_COLUMNS - 1).bit_length()
assert PREFILL_TMEM_COLUMNS <= 512
PREFILL_PERSISTENT_CTAS_PER_SM = 1
PREFILL_GRID_K = 8
PREFILL_DEFAULT_WORK_K_GROUP = 1
TMA_COPY_ALIGN_BYTES = 16
PREFILL_COMPILED_Q_HEADS_PER_KV = 4
PREFILL_COMPILED_BLOCK_Q = PREFILL_BLOCK_QH // PREFILL_COMPILED_Q_HEADS_PER_KV
PREFILL_COMPILED_DEBUG_MODE = 0
PREFILL_COMPILED_WORK_K_GROUP = PREFILL_DEFAULT_WORK_K_GROUP
PREFILL_COMPILED_SUMMARIES_PER_PAGE = PREFILL_SUPPORTED_SUMMARIES_PER_PAGE[0]
PREFILL_COMPILED_TMA_PAIRS_PER_PAGE = (
    PREFILL_COMPILED_SUMMARIES_PER_PAGE // PREFILL_TMA_ROWS
)
PREFILL_COMPILED_PAGES_PER_K_TILE = (
    PREFILL_BLOCK_KV // PREFILL_COMPILED_SUMMARIES_PER_PAGE
)
PREFILL_COMPILED_PAGE_SEGMENT_BYTES = (
    PREFILL_COMPILED_SUMMARIES_PER_PAGE * PREFILL_K_SEGMENT
)


def _cute_compile(func, *args):
    return cute.compile(func, *args, options="--enable-tvm-ffi --opt-level 2")


def _prefill_block_q_for_heads(q_heads_per_kv: int) -> int:
    h = int(q_heads_per_kv)
    if h in SUPPORTED_Q_HEADS_PER_KV and PREFILL_BLOCK_QH % h == 0:
        return PREFILL_BLOCK_QH // h
    raise ValueError(
        "paged prefill logits supports q_heads_per_kv in "
        f"{SUPPORTED_Q_HEADS_PER_KV}, got {h}"
    )


def _prefill_validate_summaries_per_page(summaries_per_page: int) -> int:
    if type(summaries_per_page) is not int:
        raise TypeError(
            "paged prefill logits page_size must be an int, "
            f"got {type(summaries_per_page).__name__}"
        )
    page_rows = summaries_per_page
    if page_rows not in PREFILL_SUPPORTED_SUMMARIES_PER_PAGE:
        raise ValueError(
            "paged prefill logits page_size must be in "
            f"{PREFILL_SUPPORTED_SUMMARIES_PER_PAGE}, got {page_rows}"
        )
    return page_rows


def _prefill_grid_k_work_items(work_k_group: int) -> int:
    group = int(work_k_group)
    if group <= 0 or PREFILL_GRID_K % group:
        raise ValueError(
            f"work_k_group must be a positive divisor of {PREFILL_GRID_K}, got {group}"
        )
    return PREFILL_GRID_K // group


def _prefill_work_k_group_for_shape(
    query_rows: int, batch_size: int, block_q: int
) -> int:
    max_q_blocks = (
        int(query_rows) + int(batch_size) * (int(block_q) - 1) + int(block_q) - 1
    ) // int(block_q)
    return 2 if max_q_blocks >= 256 else PREFILL_DEFAULT_WORK_K_GROUP


@functools.cache
def _sm_count(device_key: tuple[str, Optional[int]]) -> int:
    device = sparse_utils.device_from_cache_key(device_key)
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


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


def _load_f32_2d(tensor: cute.Tensor, row: Int32, col: Int32) -> Float32:
    src = cute.make_tensor(
        elem_pointer(tensor, (row, col)),
        cute.make_layout((1,), stride=(1,)),
    )
    return Float32(src[0])


@dsl_user_op
def _tcgen05_after_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tcgen05_before_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::before_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
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


@dsl_user_op
def _tmem_load_f32x4(
    tmem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(tmem_addr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
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
        for i in range(4)
    )


@dsl_user_op
def _tmem_load_f32x8(
    tmem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal(
            [T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32()]
        ),
        [Int32(tmem_addr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,r",
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
        for i in range(8)
    )


@cute.jit
def _tmem_load_f32_heads(
    tmem_addr: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
):
    # Keep the address collective and let the SM100 .32x32b fragment map one
    # K row to one lane. This is the same interleaved local-load contract used
    # by DeepGEMM; a CuTe epilogue layout must not rewrite the TMEM address.
    if cutlass.const_expr(q_heads_per_kv == 2):
        return _tmem_load_f32x2(tmem_addr)
    elif cutlass.const_expr(q_heads_per_kv == 4):
        return _tmem_load_f32x4(tmem_addr)
    elif cutlass.const_expr(q_heads_per_kv == 8):
        return _tmem_load_f32x8(tmem_addr)
    else:
        raise ValueError("SM100 prefill TMEM local reduction supports H=2/4/8")


@cute.jit
def _device_contract_trap() -> None:
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _prefill_persistent_work_info(
    work_idx: Int32,
    mCuSeqlensQ: cute.Tensor,
    mCuSeqlensK: cute.Tensor,
    batch_size: Int32,
    block_q: cutlass.Constexpr[int],
):
    grid_k_work_items = PREFILL_GRID_K // PREFILL_COMPILED_WORK_K_GROUP
    q_block_ordinal = work_idx // Int32(grid_k_work_items)
    k_partition = (work_idx % Int32(grid_k_work_items)) * Int32(
        PREFILL_COMPILED_WORK_K_GROUP
    )
    batch_idx = Int32(0)
    q_start = Int32(0)
    q_global_row = Int32(0)
    q_local_block = Int32(0)
    valid_seq_q = Int32(0)
    valid_seq_k = Int32(0)
    request_block_prefix = Int32(0)
    found = False
    for batch in cutlass.range(batch_size, unroll=1):
        batch_q_start = mCuSeqlensQ[batch]
        batch_q_end = mCuSeqlensQ[batch + Int32(1)]
        request_q = batch_q_end - batch_q_start
        request_q_blocks = cute.ceil_div(request_q, block_q)
        owns_q = (q_block_ordinal >= request_block_prefix) & (
            q_block_ordinal < request_block_prefix + request_q_blocks
        )
        if owns_q:
            batch_idx = batch
            q_start = batch_q_start
            q_local_block = q_block_ordinal - request_block_prefix
            q_global_row = batch_q_start + q_local_block * Int32(block_q)
            valid_seq_q = request_q
            valid_seq_k = mCuSeqlensK[batch + Int32(1)] - mCuSeqlensK[batch]
            found = True
        request_block_prefix = request_block_prefix + request_q_blocks

    q_block_last = cutlass.min(
        (q_local_block + Int32(1)) * Int32(block_q), valid_seq_q
    ) - Int32(1)
    min_actual_q = cutlass.max(
        valid_seq_q - Int32(block_q - 1), q_block_last + Int32(1)
    )
    max_query_position = (
        valid_seq_k * Int32(REGION_TOKENS) - min_actual_q + q_block_last
    )
    causal_regions = cutlass.max(Int32(0), max_query_position // Int32(REGION_TOKENS))
    causal_regions = cutlass.min(causal_regions, valid_seq_k)
    valid_k_blocks = cute.ceil_div(causal_regions, PREFILL_BLOCK_KV)
    valid = (
        found
        & (valid_seq_q > Int32(0))
        & (valid_seq_k > Int32(0))
        & (k_partition < valid_k_blocks)
    )
    return (
        valid,
        q_global_row,
        q_local_block,
        k_partition,
        batch_idx,
        q_start,
        valid_seq_q,
        valid_seq_k,
        causal_regions,
        valid_k_blocks,
    )


@cute.kernel
def _pack_prefill_qw_kernel(
    mQIn: cute.Tensor,
    mWIn: cute.Tensor,
    mQOut: cute.Tensor,
    mWOut: cute.Tensor,
    total_q_elems: Int32,
    total_w_elems: Int32,
    batch_q: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
    block_q: cutlass.Constexpr[int],
):
    tid = (
        cute.arch.block_idx()[0] * Int32(PREFILL_PREPARE_THREADS)
        + cute.arch.thread_idx()[0]
    )
    total = total_q_elems + total_w_elems
    if tid < total_q_elems:
        dim = tid % Int32(HEAD_DIM)
        packed_row = tid // Int32(HEAD_DIM)
        q_idx = packed_row // Int32(q_heads_per_kv)
        head = packed_row % Int32(q_heads_per_kv)
        valid = q_idx < batch_q
        safe_q = cutlass.select_(valid, q_idx, Int32(0))
        safe_head = cutlass.select_(valid, head, Int32(0))
        value = mQIn[safe_q, Int32(0), safe_head, dim]
        mQOut[packed_row, dim] = cutlass.select_(
            valid, cvt_f32_to_e4m3(Float32(value)).to(Uint8), Uint8(0)
        )
    if (tid >= total_q_elems) & (tid < total):
        weight_tid = tid - total_q_elems
        head = weight_tid % Int32(q_heads_per_kv)
        q_idx = weight_tid // Int32(q_heads_per_kv)
        valid = q_idx < batch_q
        safe_q = cutlass.select_(valid, q_idx, Int32(0))
        value = mWIn[safe_q, Int32(0), head]
        mWOut[q_idx, head] = cutlass.select_(valid, Float32(value), Float32(0.0))


def _make_pack_prefill_qw_call_v2(q_heads_per_kv: int, block_q: int):
    @cute.jit
    def _call_v2(
        mQIn: cute.Tensor,
        mWIn: cute.Tensor,
        mQOut: cute.Tensor,
        mWOut: cute.Tensor,
        stream,
        grid_x: Int32,
        total_q_elems: Int32,
        total_w_elems: Int32,
        batch_q: Int32,
    ):
        _pack_prefill_qw_kernel(
            mQIn,
            mWIn,
            mQOut,
            mWOut,
            total_q_elems,
            total_w_elems,
            batch_q,
            q_heads_per_kv,
            block_q,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[PREFILL_PREPARE_THREADS, 1, 1],
            stream=stream,
        )

    return _call_v2


@cached_compile_function
def _compile_pack_prefill_qw_for_signature(
    device_key: tuple[str, Optional[int]],
    q_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    w_out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_heads_per_kv: int,
    block_q: int,
):
    device = sparse_utils.device_from_cache_key(device_key)
    q = _placeholder_from_signature(
        q_signature, device=device, dynamic_shape_fill=block_q
    )
    weights = _placeholder_from_signature(
        w_signature, device=device, dynamic_shape_fill=block_q
    )
    q_out = _placeholder_from_signature(
        q_out_signature, device=device, dynamic_shape_fill=PREFILL_BLOCK_QH
    )
    w_out = _placeholder_from_signature(
        w_out_signature, device=device, dynamic_shape_fill=block_q
    )
    f_q = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q, alignment=TMA_COPY_ALIGN_BYTES, dynamic_shape_dim=0, divisibility=block_q
    )
    f_w = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        weights,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=block_q,
    )
    f_q_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_out,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=PREFILL_BLOCK_QH,
    )
    f_w_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        w_out,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=block_q,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return _cute_compile(
        _make_pack_prefill_qw_call_v2(q_heads_per_kv, block_q),
        f_q,
        f_w,
        f_q_out,
        f_w_out,
        stream_fake,
        Int32(
            (q_out.numel() + w_out.numel() + PREFILL_PREPARE_THREADS - 1)
            // PREFILL_PREPARE_THREADS
        ),
        Int32(q_out.numel()),
        Int32(w_out.numel()),
        Int32(q.shape[0]),
    )


@cute.jit
def _prefill_work_barrier(
    work_done_ptr: cute.Pointer,
    phase: Int32,
) -> Int32:
    lane_idx = cute.arch.thread_idx()[0] % Int32(LANES_PER_WARP)
    if lane_idx == Int32(0):
        cute.arch.mbarrier_arrive(work_done_ptr)
    cute.arch.mbarrier_wait(work_done_ptr, phase)
    return phase ^ Int32(1)


@cute.jit
def _prefill_consumer_loop(
    tmem_base: cute.Pointer,
    sW: cute.Tensor,
    mO: cute.Tensor,
    mCuSeqlensQ: cute.Tensor,
    mCuSeqlensK: cute.Tensor,
    w_full_ptr: cute.Pointer,
    w_empty_ptr: cute.Pointer,
    s_full_ptr: cute.Pointer,
    s_empty_ptr: cute.Pointer,
    work_done_ptr: cute.Pointer,
    total_work: Int32,
    batch_size: Int32,
    consumer_stage: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
    block_q: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    consumer_tidx = tidx % Int32(128)
    w_stage = Int32(0)
    w_phase = Int32(0)
    tmem_stage = consumer_stage
    tmem_phase = Int32(0)
    work_done_phase = Int32(0)
    work_idx = cute.arch.block_idx()[0]
    while work_idx < total_work:
        (
            valid_work,
            q_global_row,
            q_local_block,
            k_partition,
            _,
            _,
            valid_seq_q,
            _,
            causal_regions,
            valid_k_blocks,
        ) = _prefill_persistent_work_info(
            work_idx, mCuSeqlensQ, mCuSeqlensK, batch_size, block_q
        )
        if valid_work:
            q_local_base = q_local_block * Int32(block_q)
            cute.arch.mbarrier_wait(w_full_ptr + w_stage, w_phase)
            weight_regs = cute.make_rmem_tensor((block_q, q_heads_per_kv), Float32)
            for q_slot in cutlass.range_constexpr(block_q):
                valid_q_slot = (q_local_base + Int32(q_slot)) < valid_seq_q
                for head in cutlass.range_constexpr(q_heads_per_kv):
                    weight = Float32(sW[Int32(q_slot), Int32(head), w_stage])
                    weight_regs[q_slot, head] = cutlass.select_(
                        valid_q_slot, weight, Float32(0.0)
                    )

            for k_group_offset in cutlass.range_constexpr(
                PREFILL_COMPILED_WORK_K_GROUP
            ):
                kv_block = k_partition + Int32(k_group_offset)
                while kv_block < valid_k_blocks:
                    cute.arch.mbarrier_wait(s_full_ptr + tmem_stage, tmem_phase)
                    _tcgen05_after_thread_sync()
                    kv_base = kv_block * Int32(
                        PREFILL_BLOCK_KV
                    ) + consumer_stage * Int32(PREFILL_MMA_TILE_M)
                    for q_slot in cutlass.range_constexpr(block_q):
                        tmem_addr = Int32(
                            tmem_base.toint()
                            + tmem_stage * Int32(PREFILL_BLOCK_QH)
                            + Int32(q_slot * q_heads_per_kv)
                        )
                        acc_token_val = _tmem_load_f32_heads(tmem_addr, q_heads_per_kv)
                        cute.arch.fence_view_async_tmem_load()
                        weighted = Float32(0.0)
                        row = consumer_tidx
                        for head in cutlass.range_constexpr(q_heads_per_kv):
                            raw = acc_token_val[head]
                            value = cute.arch.fmax(raw, Float32(0.0))
                            weighted = weighted + value * weight_regs[q_slot, head]
                        if q_slot == block_q - 1:
                            _tcgen05_before_thread_sync()
                            cute.arch.mbarrier_arrive(s_empty_ptr + tmem_stage)
                        key = kv_base + row
                        q_local_idx = q_local_base + Int32(q_slot)
                        if (q_local_idx < valid_seq_q) & (key < causal_regions):
                            dst = cute.make_tensor(
                                elem_pointer_i64_offset(
                                    mO, (q_global_row + Int32(q_slot), key)
                                ),
                                cute.make_layout((1,), stride=(1,)),
                            )
                            dst[0] = mO.element_type(weighted)
                    next_tmem_stage = tmem_stage + Int32(PREFILL_NUM_CONSUMER_GROUPS)
                    if next_tmem_stage >= Int32(PREFILL_S_STAGES):
                        next_tmem_stage = next_tmem_stage - Int32(PREFILL_S_STAGES)
                        tmem_phase = tmem_phase ^ Int32(1)
                    tmem_stage = next_tmem_stage
                    kv_block = kv_block + Int32(PREFILL_GRID_K)
            cute.arch.mbarrier_arrive(w_empty_ptr + w_stage)
            next_w_stage = w_stage + Int32(1)
            if next_w_stage >= Int32(PREFILL_Q_STAGES):
                next_w_stage = Int32(0)
                w_phase = w_phase ^ Int32(1)
            w_stage = next_w_stage
        if q_heads_per_kv == 2:
            work_done_phase = _prefill_work_barrier(work_done_ptr, work_done_phase)
        work_idx = work_idx + cute.arch.grid_dim()[0]


@cute.kernel
def _prefill_paged_logits_kernel(
    tma_atom_Q0: cute.CopyAtom,
    tma_tensor_Q0: cute.Tensor,
    tma_atom_Q1: cute.CopyAtom,
    tma_tensor_Q1: cute.Tensor,
    tma_atom_K0: cute.CopyAtom,
    tma_tensor_K0: cute.Tensor,
    tma_atom_K1: cute.CopyAtom,
    tma_tensor_K1: cute.Tensor,
    tma_atom_W: cute.CopyAtom,
    tma_tensor_W: cute.Tensor,
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sQ_layout_fp8_0: cute.ComposedLayout,
    sQ_layout_fp8_1: cute.ComposedLayout,
    sK_layout_fp8_0: cute.ComposedLayout,
    sK_layout_fp8_1: cute.ComposedLayout,
    sW_layout: cute.Layout,
    mW: cute.Tensor,
    mO: cute.Tensor,
    mCuSeqlensQ: cute.Tensor,
    mCuSeqlensK: cute.Tensor,
    seq_q: Int32,
    seq_k: Int32,
    block_table_cols: Int32,
    batch_size: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    q_heads_per_kv = PREFILL_COMPILED_Q_HEADS_PER_KV
    block_q = PREFILL_COMPILED_BLOCK_Q
    debug_mode = PREFILL_COMPILED_DEBUG_MODE
    max_q_blocks = cute.ceil_div(seq_q + batch_size * Int32(block_q - 1), block_q)
    grid_k_work_items = PREFILL_GRID_K // PREFILL_COMPILED_WORK_K_GROUP
    total_work = max_q_blocks * Int32(grid_k_work_items)

    if warp_idx == Int32(0) and (tidx % Int32(LANES_PER_WARP)) == Int32(0):
        cpasync.prefetch_descriptor(tma_atom_Q0)
        cpasync.prefetch_descriptor(tma_atom_Q1)
        cpasync.prefetch_descriptor(tma_atom_K0)
        cpasync.prefetch_descriptor(tma_atom_K1)
        cpasync.prefetch_descriptor(tma_atom_W)

    smem = cutlass.utils.SmemAllocator()
    sQ0_struct = cute.struct.Align[
        cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8_0)], 1024
    ]
    sQ1_struct = cute.struct.Align[
        cute.struct.MemRange[Float8E4M3FN, cute.cosize(sQ_layout_fp8_1)], 1024
    ]
    sK0_struct = cute.struct.Align[
        cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8_0)], 1024
    ]
    sK1_struct = cute.struct.Align[
        cute.struct.MemRange[Float8E4M3FN, cute.cosize(sK_layout_fp8_1)], 1024
    ]
    sW_struct = cute.struct.Align[
        cute.struct.MemRange[Float32, cute.cosize(sW_layout)], 1024
    ]
    mbar_ptr_q_struct = cute.struct.MemRange[
        cutlass.Int64, PREFILL_Q_STAGES * M_BARRIER_PTRS_PER_STAGE
    ]
    mbar_ptr_k_struct = cute.struct.MemRange[
        cutlass.Int64, PREFILL_K_STAGES * M_BARRIER_PTRS_PER_STAGE
    ]
    mbar_w_full_struct = cute.struct.MemRange[cutlass.Int64, PREFILL_Q_STAGES]
    mbar_w_empty_struct = cute.struct.MemRange[cutlass.Int64, PREFILL_Q_STAGES]
    mbar_s_full_struct = cute.struct.MemRange[cutlass.Int64, PREFILL_S_STAGES]
    mbar_s_empty_struct = cute.struct.MemRange[cutlass.Int64, PREFILL_S_STAGES]
    mbar_work_done_struct = cute.struct.MemRange[cutlass.Int64, 1]

    @cute.struct
    class SharedStorage:
        mbar_ptr_q: mbar_ptr_q_struct
        mbar_ptr_k: mbar_ptr_k_struct
        mbar_w_full: mbar_w_full_struct
        mbar_w_empty: mbar_w_empty_struct
        mbar_s_full: mbar_s_full_struct
        mbar_s_empty: mbar_s_empty_struct
        mbar_work_done: mbar_work_done_struct
        tmem_holding_buf: Int32
        sQ0: sQ0_struct
        sQ1: sQ1_struct
        sK0: sK0_struct
        sK1: sK1_struct
        sW: sW_struct

    storage = smem.allocate(SharedStorage)
    sQ0 = storage.sQ0.get_tensor(sQ_layout_fp8_0.outer, swizzle=sQ_layout_fp8_0.inner)
    sQ1 = storage.sQ1.get_tensor(sQ_layout_fp8_1.outer, swizzle=sQ_layout_fp8_1.inner)
    sK0 = storage.sK0.get_tensor(sK_layout_fp8_0.outer, swizzle=sK_layout_fp8_0.inner)
    sK1 = storage.sK1.get_tensor(sK_layout_fp8_1.outer, swizzle=sK_layout_fp8_1.inner)
    sW = storage.sW.get_tensor(sW_layout)
    sK_pages0 = cute.make_tensor(
        cute.recast_ptr(storage.sK0.data_ptr(), dtype=Uint8),
        cute.make_layout(
            (PREFILL_K_STAGES * PREFILL_BLOCK_KV * PREFILL_K_SEGMENT,),
            stride=(1,),
        ),
    )
    sK_pages1 = cute.make_tensor(
        cute.recast_ptr(storage.sK1.data_ptr(), dtype=Uint8),
        cute.make_layout(
            (PREFILL_K_STAGES * PREFILL_BLOCK_KV * PREFILL_K_SEGMENT,),
            stride=(1,),
        ),
    )

    w_full_ptr = storage.mbar_w_full.data_ptr()
    w_empty_ptr = storage.mbar_w_empty.data_ptr()
    s_full_ptr = storage.mbar_s_full.data_ptr()
    s_empty_ptr = storage.mbar_s_empty.data_ptr()
    work_done_ptr = storage.mbar_work_done.data_ptr()
    if tidx == Int32(0):
        for stage in cutlass.range_constexpr(PREFILL_Q_STAGES):
            cute.arch.mbarrier_init(w_full_ptr + Int32(stage), Int32(1))
            cute.arch.mbarrier_init(
                w_empty_ptr + Int32(stage),
                Int32(PREFILL_NUM_CONSUMER_THREADS),
            )
        for stage in cutlass.range_constexpr(PREFILL_S_STAGES):
            cute.arch.mbarrier_init(s_full_ptr + Int32(stage), Int32(1))
            cute.arch.mbarrier_init(
                s_empty_ptr + Int32(stage),
                Int32(PREFILL_CONSUMER_GROUP_THREADS),
            )
        cute.arch.mbarrier_init(work_done_ptr, Int32(PREFILL_NUM_WARPS))
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if tidx < Int32(PREFILL_NUM_CONSUMER_THREADS):
        for stage in cutlass.range_constexpr(PREFILL_Q_STAGES):
            cute.arch.mbarrier_arrive(w_empty_ptr + Int32(stage))
    if tidx < Int32(PREFILL_CONSUMER_GROUP_THREADS):
        for stage in cutlass.range_constexpr(PREFILL_S_STAGES):
            cute.arch.mbarrier_arrive(s_empty_ptr + Int32(stage))
    cute.arch.sync_threads()

    tma_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    umma_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    pipeline_q = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.mbar_ptr_q.data_ptr(),
        num_stages=PREFILL_Q_STAGES,
        producer_group=tma_group,
        consumer_group=umma_group,
        tx_count=PREFILL_BLOCK_QH * HEAD_DIM,
    )
    pipeline_k = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.mbar_ptr_k.data_ptr(),
        num_stages=PREFILL_K_STAGES,
        producer_group=tma_group,
        consumer_group=umma_group,
        tx_count=PREFILL_BLOCK_KV * HEAD_DIM,
    )
    gQ0 = cute.local_tile(
        tma_tensor_Q0, (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT), (None, 0)
    )
    gQ1 = cute.local_tile(
        tma_tensor_Q1, (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT), (None, 0)
    )
    tQsQ0, tQgQ0 = cpasync.tma_partition(
        tma_atom_Q0,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ0, 0, cute.rank(sQ0) - 1),
        cute.group_modes(gQ0, 0, cute.rank(gQ0) - 1),
    )
    tQsQ1, tQgQ1 = cpasync.tma_partition(
        tma_atom_Q1,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ1, 0, cute.rank(sQ1) - 1),
        cute.group_modes(gQ1, 0, cute.rank(gQ1) - 1),
    )
    gW = cute.local_tile(tma_tensor_W, (block_q, q_heads_per_kv), (None, 0))
    tWsW, tWgW = cpasync.tma_partition(
        tma_atom_W,
        0,
        cute.make_layout(1),
        cute.group_modes(sW, 0, cute.rank(sW) - 1),
        cute.group_modes(gW, 0, cute.rank(gW) - 1),
    )
    if cutlass.const_expr(debug_mode != 3 and debug_mode != 6):
        if tidx < Int32(PREFILL_NUM_CONSUMER_THREADS):
            cute.arch.warpgroup_reg_alloc(PREFILL_NUM_CONSUMER_REGS)
        else:
            cute.arch.warpgroup_reg_dealloc(PREFILL_NUM_ROLE_REGS)
    if (warp_idx == Int32(PREFILL_MMA_WARP)) & cutlass.const_expr(
        debug_mode != 3 and debug_mode != 5
    ):
        cute.arch.alloc_tmem(Int32(PREFILL_TMEM_COLUMNS), storage.tmem_holding_buf)
        cute.arch.sync_warp()
    cute.arch.sync_threads()
    if cutlass.const_expr(debug_mode != 3 and debug_mode != 5):
        tmem_base = cute.arch.retrieve_tmem_ptr(
            Float32,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
    else:
        tmem_base = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )

    if (warp_idx == Int32(PREFILL_TMA_Q_LOAD_WARP)) & cutlass.const_expr(
        debug_mode == 0 or debug_mode == 1 or debug_mode == 4
    ):
        lane_idx = tidx % Int32(LANES_PER_WARP)
        if lane_idx >= Int32(0):
            q_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, PREFILL_Q_STAGES
            )
            w_stage = Int32(0)
            w_phase = Int32(0)
            work_done_phase = Int32(0)
            work_idx = cute.arch.block_idx()[0]
            while work_idx < total_work:
                (
                    valid_work,
                    q_global_row,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = _prefill_persistent_work_info(
                    work_idx, mCuSeqlensQ, mCuSeqlensK, batch_size, block_q
                )
                if valid_work:
                    if cutlass.const_expr(debug_mode == 0 or debug_mode == 4):
                        cute.arch.mbarrier_wait(w_empty_ptr + w_stage, w_phase)
                    pipeline_q.producer_acquire(q_prod)
                    q_row_offset = q_global_row * Int32(q_heads_per_kv)
                    gQ0_row = cute.local_tile(
                        cute.domain_offset((q_row_offset, Int32(0)), tma_tensor_Q0),
                        (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT),
                        (None, 0),
                    )
                    gQ1_row = cute.local_tile(
                        cute.domain_offset((q_row_offset, Int32(0)), tma_tensor_Q1),
                        (PREFILL_BLOCK_QH, PREFILL_K_SEGMENT),
                        (None, 0),
                    )
                    _, tQgQ0_row = cpasync.tma_partition(
                        tma_atom_Q0,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQ0, 0, cute.rank(sQ0) - 1),
                        cute.group_modes(gQ0_row, 0, cute.rank(gQ0_row) - 1),
                    )
                    _, tQgQ1_row = cpasync.tma_partition(
                        tma_atom_Q1,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQ1, 0, cute.rank(sQ1) - 1),
                        cute.group_modes(gQ1_row, 0, cute.rank(gQ1_row) - 1),
                    )
                    if cutlass.const_expr(debug_mode == 0 or debug_mode == 4):
                        gW_row = cute.local_tile(
                            cute.domain_offset((q_global_row, Int32(0)), tma_tensor_W),
                            (block_q, q_heads_per_kv),
                            (None, 0),
                        )
                        _, tWgW_row = cpasync.tma_partition(
                            tma_atom_W,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sW, 0, cute.rank(sW) - 1),
                            cute.group_modes(gW_row, 0, cute.rank(gW_row) - 1),
                        )
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                w_full_ptr + w_stage,
                                Int32(PREFILL_WEIGHT_STAGE_BYTES),
                            )
                        cute.copy(
                            tma_atom_W,
                            tWgW_row[None, 0],
                            tWsW[None, w_stage],
                            tma_bar_ptr=w_full_ptr + w_stage,
                        )
                    cute.copy(
                        tma_atom_Q0,
                        tQgQ0_row[None, 0],
                        tQsQ0[None, q_prod.index],
                        tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                    )
                    cute.copy(
                        tma_atom_Q1,
                        tQgQ1_row[None, 0],
                        tQsQ1[None, q_prod.index],
                        tma_bar_ptr=pipeline_q.producer_get_barrier(q_prod),
                    )
                    q_prod.advance()
                    if cutlass.const_expr(debug_mode == 0 or debug_mode == 4):
                        next_w_stage = w_stage + Int32(1)
                        if next_w_stage >= Int32(PREFILL_Q_STAGES):
                            next_w_stage = Int32(0)
                            w_phase = w_phase ^ Int32(1)
                        w_stage = next_w_stage
                if q_heads_per_kv == 2:
                    work_done_phase = _prefill_work_barrier(
                        work_done_ptr, work_done_phase
                    )
                work_idx = work_idx + cute.arch.grid_dim()[0]

    if (warp_idx == Int32(PREFILL_TMA_K_LOAD_WARP)) & cutlass.const_expr(
        debug_mode == 0 or debug_mode == 2 or debug_mode == 4
    ):
        lane_idx = tidx % Int32(LANES_PER_WARP)
        if lane_idx >= Int32(0):
            k_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, PREFILL_K_STAGES
            )
            work_done_phase = Int32(0)
            work_idx = cute.arch.block_idx()[0]
            while work_idx < total_work:
                (
                    valid_work,
                    _,
                    _,
                    k_partition,
                    batch_idx,
                    _,
                    _,
                    _,
                    causal_regions,
                    valid_k_blocks,
                ) = _prefill_persistent_work_info(
                    work_idx, mCuSeqlensQ, mCuSeqlensK, batch_size, block_q
                )
                if valid_work:
                    for k_group_offset in cutlass.range_constexpr(
                        PREFILL_COMPILED_WORK_K_GROUP
                    ):
                        kv_block = k_partition + Int32(k_group_offset)
                        while kv_block < valid_k_blocks:
                            pipeline_k.producer_acquire(k_prod)
                            stage_offset = k_prod.index * Int32(
                                PREFILL_BLOCK_KV * PREFILL_K_SEGMENT
                            )
                            for page in cutlass.range_constexpr(
                                PREFILL_COMPILED_PAGES_PER_K_TILE
                            ):
                                logical_page = kv_block * Int32(
                                    PREFILL_COMPILED_PAGES_PER_K_TILE
                                ) + Int32(page)
                                active_page = (
                                    logical_page
                                    * Int32(PREFILL_COMPILED_SUMMARIES_PER_PAGE)
                                    < causal_regions
                                )
                                if active_page & (logical_page >= block_table_cols):
                                    _device_contract_trap()
                                table_entry_valid = active_page & (
                                    logical_page < block_table_cols
                                )
                                safe_col = cutlass.select_(
                                    table_entry_valid, logical_page, Int32(0)
                                )
                                physical_page = mBlockTable[
                                    batch_idx * block_table_cols + safe_col
                                ]
                                physical_page_valid = (physical_page >= Int32(0)) & (
                                    physical_page < num_pages
                                )
                                if active_page & (physical_page_valid == Int32(0)):
                                    _device_contract_trap()
                                load_page = table_entry_valid & physical_page_valid
                                safe_page = cutlass.select_(
                                    load_page, physical_page, Int32(0)
                                )
                                dst_offset = stage_offset + Int32(
                                    page * PREFILL_COMPILED_PAGE_SEGMENT_BYTES
                                )
                                gK0 = cute.domain_offset(
                                    (Int32(0), Int32(0), Int32(0), safe_page),
                                    tma_tensor_K0,
                                )
                                gK1 = cute.domain_offset(
                                    (Int32(0), Int32(0), Int32(0), safe_page),
                                    tma_tensor_K1,
                                )
                                page_layout0 = cute.make_layout(
                                    (
                                        PREFILL_GEN5_ATOM_BYTES,
                                        PREFILL_TMA_ROWS,
                                        PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
                                    ),
                                    stride=(
                                        tma_tensor_K0.stride[0],
                                        tma_tensor_K0.stride[1],
                                        tma_tensor_K0.stride[2],
                                    ),
                                )
                                page_layout1 = cute.make_layout(
                                    (
                                        PREFILL_GEN5_ATOM_BYTES,
                                        PREFILL_TMA_ROWS,
                                        PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
                                    ),
                                    stride=(
                                        tma_tensor_K1.stride[0],
                                        tma_tensor_K1.stride[1],
                                        tma_tensor_K1.stride[2],
                                    ),
                                )
                                gK0 = cute.make_tensor(gK0.iterator, page_layout0)
                                gK1 = cute.make_tensor(gK1.iterator, page_layout1)
                                smem_page_layout = cute.make_layout(
                                    (
                                        PREFILL_GEN5_ATOM_BYTES,
                                        PREFILL_TMA_ROWS,
                                        PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
                                    ),
                                    stride=(
                                        1,
                                        PREFILL_GEN5_ATOM_BYTES,
                                        PREFILL_TMA_CHUNK_BYTES,
                                    ),
                                )
                                sK0_chunk = cute.make_tensor(
                                    elem_pointer_i64_offset(sK_pages0, (dst_offset,)),
                                    smem_page_layout,
                                )
                                sK1_chunk = cute.make_tensor(
                                    elem_pointer_i64_offset(sK_pages1, (dst_offset,)),
                                    smem_page_layout,
                                )
                                load_K0, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K0,
                                    0,
                                    cute.make_layout(1),
                                    gK0,
                                    sK0_chunk,
                                    single_stage=True,
                                )
                                load_K1, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K1,
                                    0,
                                    cute.make_layout(1),
                                    gK1,
                                    sK1_chunk,
                                    single_stage=True,
                                )
                                load_K0(
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod)
                                )
                                load_K1(
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(k_prod)
                                )
                            k_prod.advance()
                            kv_block = kv_block + Int32(PREFILL_GRID_K)
                if q_heads_per_kv == 2:
                    work_done_phase = _prefill_work_barrier(
                        work_done_ptr, work_done_phase
                    )
                work_idx = work_idx + cute.arch.grid_dim()[0]

    if (warp_idx == Int32(PREFILL_MMA_WARP)) & cutlass.const_expr(debug_mode == 1):
        q_cons_debug = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PREFILL_Q_STAGES
        )
        work_idx = cute.arch.block_idx()[0]
        while work_idx < total_work:
            valid_work = _prefill_persistent_work_info(
                work_idx, mCuSeqlensQ, mCuSeqlensK, batch_size, block_q
            )[0]
            if valid_work:
                pipeline_q.consumer_wait(q_cons_debug)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        pipeline_q.sync_object_empty.get_barrier(q_cons_debug.index)
                    )
                q_cons_debug.advance()
            work_idx = work_idx + cute.arch.grid_dim()[0]

    if (warp_idx == Int32(PREFILL_MMA_WARP)) & cutlass.const_expr(debug_mode == 2):
        k_cons_debug = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PREFILL_K_STAGES
        )
        work_idx = cute.arch.block_idx()[0]
        while work_idx < total_work:
            (
                valid_work,
                _,
                _,
                k_partition,
                _,
                _,
                _,
                _,
                _,
                valid_k_blocks,
            ) = _prefill_persistent_work_info(
                work_idx, mCuSeqlensQ, mCuSeqlensK, batch_size, block_q
            )
            if valid_work:
                for k_group_offset in cutlass.range_constexpr(
                    PREFILL_COMPILED_WORK_K_GROUP
                ):
                    kv_block = k_partition + Int32(k_group_offset)
                    while kv_block < valid_k_blocks:
                        pipeline_k.consumer_wait(k_cons_debug)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                pipeline_k.sync_object_empty.get_barrier(
                                    k_cons_debug.index
                                )
                            )
                        k_cons_debug.advance()
                        kv_block = kv_block + Int32(PREFILL_GRID_K)
            work_idx = work_idx + cute.arch.grid_dim()[0]

    if (warp_idx == Int32(PREFILL_MMA_WARP)) & cutlass.const_expr(
        debug_mode == 0 or debug_mode == 4
    ):
        lane_idx = tidx % Int32(LANES_PER_WARP)
        q_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PREFILL_Q_STAGES
        )
        k_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PREFILL_K_STAGES
        )
        tCrA0_all = tiled_mma.make_fragment_A(sK0)
        tCrA1_all = tiled_mma.make_fragment_A(sK1)
        tCrB0_all = tiled_mma.make_fragment_B(sQ0)
        tCrB1_all = tiled_mma.make_fragment_B(sQ1)
        # The UMMA producer owns one persistent three-slot TMEM ring.  The two
        # math groups consume alternating slots (+2 modulo 3), so their state
        # must survive persistent work-item boundaries.
        score_stage = Int32(0)
        score_phase = Int32(0)
        work_done_phase = Int32(0)
        work_idx = cute.arch.block_idx()[0]
        while work_idx < total_work:
            (
                valid_work,
                _,
                _,
                k_partition,
                _,
                _,
                _,
                _,
                _,
                valid_k_blocks,
            ) = _prefill_persistent_work_info(
                work_idx, mCuSeqlensQ, mCuSeqlensK, batch_size, block_q
            )
            if valid_work:
                pipeline_q.consumer_wait(q_cons)
                q_stage = q_cons.index
                sQ0_stage = sQ0[None, None, None, q_stage]
                sQ1_stage = sQ1[None, None, None, q_stage]
                tCrB0_stage = tCrB0_all[None, None, None, q_stage]
                tCrB1_stage = tCrB1_all[None, None, None, q_stage]
                for k_group_offset in cutlass.range_constexpr(
                    PREFILL_COMPILED_WORK_K_GROUP
                ):
                    kv_block = k_partition + Int32(k_group_offset)
                    while kv_block < valid_k_blocks:
                        pipeline_k.consumer_wait(k_cons)
                        k_stage = k_cons.index
                        sK0_stage = sK0[None, None, None, k_stage]
                        sK1_stage = sK1[None, None, None, k_stage]
                        for math_group in cutlass.range_constexpr(
                            PREFILL_NUM_CONSUMER_GROUPS
                        ):
                            cute.arch.mbarrier_wait(
                                s_empty_ptr + score_stage, score_phase
                            )
                            _tcgen05_after_thread_sync()
                            tmem_offset = tmem_base.toint() + score_stage * Int32(
                                PREFILL_BLOCK_QH
                            )
                            k0_group = sK0_stage
                            k1_group = sK1_stage
                            if math_group == 1:
                                k0_group = cute.make_tensor(
                                    sK0_stage.iterator + sK0_stage.stride[1],
                                    sK0_stage.layout,
                                )
                                k1_group = cute.make_tensor(
                                    sK1_stage.iterator + sK1_stage.stride[1],
                                    sK1_stage.layout,
                                )
                            sm100_flash_utils.gemm_ptx_partial(
                                tiled_mma.op,
                                tmem_offset,
                                tCrA0_all[None, None, None, k_stage],
                                tCrB0_stage,
                                k0_group,
                                sQ0_stage,
                                zero_init=True,
                            )
                            sm100_flash_utils.gemm_ptx_partial(
                                tiled_mma.op,
                                tmem_offset,
                                tCrA1_all[None, None, None, k_stage],
                                tCrB1_stage,
                                k1_group,
                                sQ1_stage,
                                zero_init=False,
                            )
                            with cute.arch.elect_one():
                                tcgen05.commit(s_full_ptr + score_stage)
                            next_score_stage = score_stage + Int32(1)
                            if next_score_stage >= Int32(PREFILL_S_STAGES):
                                next_score_stage = Int32(0)
                                score_phase = score_phase ^ Int32(1)
                            score_stage = next_score_stage
                        pipeline_k.consumer_release(k_cons)
                        k_cons.advance()
                        kv_block = kv_block + Int32(PREFILL_GRID_K)
                pipeline_q.consumer_release(q_cons)
                q_cons.advance()
            if q_heads_per_kv == 2:
                work_done_phase = _prefill_work_barrier(work_done_ptr, work_done_phase)
            work_idx = work_idx + cute.arch.grid_dim()[0]

    if (tidx < Int32(PREFILL_NUM_CONSUMER_THREADS)) & cutlass.const_expr(
        debug_mode == 0 or debug_mode == 4
    ):
        _prefill_consumer_loop(
            tmem_base,
            sW,
            mO,
            mCuSeqlensQ,
            mCuSeqlensK,
            w_full_ptr,
            w_empty_ptr,
            s_full_ptr,
            s_empty_ptr,
            work_done_ptr,
            total_work,
            batch_size,
            warp_idx // Int32(WARPS_PER_WARPGROUP),
            q_heads_per_kv,
            block_q,
        )

    cute.arch.sync_threads()
    if (warp_idx == Int32(PREFILL_MMA_WARP)) & cutlass.const_expr(
        debug_mode != 3 and debug_mode != 5
    ):
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(tmem_base, Int32(PREFILL_TMEM_COLUMNS))


def _prefill_paged_logits_call_impl_v131(
    mQ: cute.Tensor,
    mKCache: cute.Tensor,
    mBlockTable: cute.Tensor,
    mW: cute.Tensor,
    mO: cute.Tensor,
    mCuSeqlensQ: cute.Tensor,
    mCuSeqlensK: cute.Tensor,
    stream,
    seq_q: Int32,
    seq_k: Int32,
    block_table_cols: Int32,
    batch_size: Int32,
    num_pages: Int32,
    summaries_per_page: Int32,
    persistent_grid: Int32,
    q_heads_per_kv: cutlass.Constexpr[int],
    block_q: cutlass.Constexpr[int],
):
    mma_tiler_mnk = (PREFILL_BLOCK_KV, PREFILL_BLOCK_QH, PREFILL_K_SEGMENT)
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        Float8E4M3FN,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        Float32,
        tcgen05.CtaGroup.ONE,
        (PREFILL_MMA_TILE_M, PREFILL_BLOCK_QH),
    )
    sQ_layout_fp8_0 = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, Float8E4M3FN, PREFILL_Q_STAGES
    )
    sQ_layout_fp8_1 = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, Float8E4M3FN, PREFILL_Q_STAGES
    )
    sK_layout_fp8_0 = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, Float8E4M3FN, PREFILL_K_STAGES
    )
    sK_layout_fp8_1 = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, Float8E4M3FN, PREFILL_K_STAGES
    )
    q_segment_layout = cute.make_layout(
        (mQ.shape[0], PREFILL_K_SEGMENT),
        stride=(mQ.stride[0], mQ.stride[1]),
    )
    mQ_segment0 = cute.make_tensor(mQ.iterator, q_segment_layout)
    mQ_segment1_ptr = cute.domain_offset((Int32(0), Int32(PREFILL_K_SEGMENT)), mQ)
    mQ_segment1 = cute.make_tensor(mQ_segment1_ptr.iterator, q_segment_layout)
    tma_atom_Q0, tma_tensor_Q0 = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ_segment0,
        cute.select(sQ_layout_fp8_0, mode=[0, 1, 2]),
        mma_tiler_mnk,
        tiled_mma,
        1,
    )
    tma_atom_Q1, tma_tensor_Q1 = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(),
        mQ_segment1,
        cute.select(sQ_layout_fp8_1, mode=[0, 1, 2]),
        mma_tiler_mnk,
        tiled_mma,
        1,
    )
    mKCache_pages = cute.make_tensor(
        cute.recast_ptr(mKCache.iterator, dtype=Uint8),
        cute.make_layout(
            (
                PREFILL_GEN5_ATOM_BYTES,
                PREFILL_TMA_ROWS,
                HEAD_DIM // PREFILL_K_SEGMENT,
                PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
                mKCache.shape[0],
            ),
            stride=(
                1,
                PREFILL_GEN5_ATOM_BYTES,
                PREFILL_TMA_CHUNK_BYTES,
                Int32(PREFILL_TMA_PAGE_STRIDE_BYTES),
                summaries_per_page * Int32(HEAD_DIM),
            ),
        ),
    )
    k_page_layout = cute.make_layout(
        (
            PREFILL_GEN5_ATOM_BYTES,
            PREFILL_TMA_ROWS,
            PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
            mKCache.shape[0],
        ),
        stride=(
            1,
            PREFILL_GEN5_ATOM_BYTES,
            Int32(PREFILL_TMA_PAGE_STRIDE_BYTES),
            summaries_per_page * Int32(HEAD_DIM),
        ),
    )
    mKCache_pages0 = cute.make_tensor(mKCache_pages.iterator, k_page_layout)
    mKCache_pages1_ptr = cute.domain_offset(
        (Int32(0), Int32(0), Int32(1), Int32(0), Int32(0)), mKCache_pages
    )
    mKCache_pages1 = cute.make_tensor(mKCache_pages1_ptr.iterator, k_page_layout)
    sK_page_layout = cute.make_layout(
        (
            PREFILL_GEN5_ATOM_BYTES,
            PREFILL_TMA_ROWS,
            PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
        ),
        stride=(1, PREFILL_GEN5_ATOM_BYTES, PREFILL_TMA_CHUNK_BYTES),
    )
    tma_atom_K0, tma_tensor_K0 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_pages0,
        sK_page_layout,
        (
            PREFILL_GEN5_ATOM_BYTES,
            PREFILL_TMA_ROWS,
            PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
        ),
    )
    tma_atom_K1, tma_tensor_K1 = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mKCache_pages1,
        sK_page_layout,
        (
            PREFILL_GEN5_ATOM_BYTES,
            PREFILL_TMA_ROWS,
            PREFILL_COMPILED_TMA_PAIRS_PER_PAGE,
        ),
    )
    sW_layout_single = cute.make_layout(
        (block_q, q_heads_per_kv), stride=(q_heads_per_kv, 1)
    )
    sW_layout = cute.make_layout(
        (block_q, q_heads_per_kv, PREFILL_Q_STAGES),
        stride=(q_heads_per_kv, 1, PREFILL_BLOCK_QH),
    )
    tma_atom_W, tma_tensor_W = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mW,
        sW_layout_single,
        (block_q, q_heads_per_kv),
    )
    _prefill_paged_logits_kernel(
        tma_atom_Q0,
        tma_tensor_Q0,
        tma_atom_Q1,
        tma_tensor_Q1,
        tma_atom_K0,
        tma_tensor_K0,
        tma_atom_K1,
        tma_tensor_K1,
        tma_atom_W,
        tma_tensor_W,
        mQ,
        mKCache,
        mBlockTable,
        tiled_mma,
        sQ_layout_fp8_0,
        sQ_layout_fp8_1,
        sK_layout_fp8_0,
        sK_layout_fp8_1,
        sW_layout,
        mW,
        mO,
        mCuSeqlensQ,
        mCuSeqlensK,
        seq_q,
        seq_k,
        block_table_cols,
        batch_size,
        num_pages,
        summaries_per_page,
    ).launch(
        grid=(persistent_grid, 1, 1),
        block=[PREFILL_NUM_THREADS, 1, 1],
        stream=stream,
    )


# Note(wangbojun/codex): Version this CuTe entrypoint when a called Gen5 helper
# changes because the persistent cache key does not include helper bodies.
def _make_prefill_paged_logits_call_v152(
    q_heads_per_kv: int,
    block_q: int,
    debug_mode: int,
    work_k_group: int,
    summaries_per_page: int,
):
    global PREFILL_COMPILED_Q_HEADS_PER_KV
    global PREFILL_COMPILED_BLOCK_Q
    global PREFILL_COMPILED_DEBUG_MODE
    global PREFILL_COMPILED_WORK_K_GROUP
    global PREFILL_COMPILED_SUMMARIES_PER_PAGE
    global PREFILL_COMPILED_TMA_PAIRS_PER_PAGE
    global PREFILL_COMPILED_PAGES_PER_K_TILE
    global PREFILL_COMPILED_PAGE_SEGMENT_BYTES
    _prefill_grid_k_work_items(work_k_group)
    summaries_per_page = _prefill_validate_summaries_per_page(summaries_per_page)
    PREFILL_COMPILED_Q_HEADS_PER_KV = q_heads_per_kv
    PREFILL_COMPILED_BLOCK_Q = block_q
    PREFILL_COMPILED_DEBUG_MODE = debug_mode
    PREFILL_COMPILED_WORK_K_GROUP = work_k_group
    PREFILL_COMPILED_SUMMARIES_PER_PAGE = summaries_per_page
    PREFILL_COMPILED_TMA_PAIRS_PER_PAGE = summaries_per_page // PREFILL_TMA_ROWS
    PREFILL_COMPILED_PAGES_PER_K_TILE = PREFILL_BLOCK_KV // summaries_per_page
    PREFILL_COMPILED_PAGE_SEGMENT_BYTES = summaries_per_page * PREFILL_K_SEGMENT

    @cute.jit
    def _call_v152(
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mBlockTable: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        stream,
        seq_q: Int32,
        seq_k: Int32,
        block_table_cols: Int32,
        batch_size: Int32,
        num_pages: Int32,
        summaries_per_page: Int32,
        persistent_grid: Int32,
    ):
        _prefill_paged_logits_call_impl_v131(
            mQ,
            mKCache,
            mBlockTable,
            mW,
            mO,
            mCuSeqlensQ,
            mCuSeqlensK,
            stream,
            seq_q,
            seq_k,
            block_table_cols,
            batch_size,
            num_pages,
            summaries_per_page,
            persistent_grid,
            q_heads_per_kv,
            block_q,
        )

    return _call_v152


@cached_compile_function
def _compile_prefill_paged_logits_for_signature(
    device_key: tuple[str, Optional[int]],
    q_runtime_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    k_cache_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    block_table_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    weights_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    out_signature: tuple[torch.dtype, tuple[object, ...], tuple[object, ...]],
    q_heads_per_kv: int,
    block_q: int,
    summaries_per_page: int,
    debug_mode: int = 0,
    work_k_group: int = PREFILL_DEFAULT_WORK_K_GROUP,
):
    device = sparse_utils.device_from_cache_key(device_key)
    q_runtime = _placeholder_from_signature(
        q_runtime_signature, device=device, dynamic_shape_fill=PREFILL_BLOCK_QH
    )
    k_cache = _placeholder_from_signature(
        k_cache_signature,
        device=device,
        dynamic_shape_fill=summaries_per_page,
        dynamic_stride_fill=PREFILL_TMA_PAGE_STRIDE_BYTES,
    )
    block_table_flat = _placeholder_from_signature(
        block_table_signature,
        device=device,
        dynamic_shape_fill=COMPILE_PLACEHOLDER_DIM,
    )
    weights = _placeholder_from_signature(
        weights_signature, device=device, dynamic_shape_fill=block_q
    )
    out = _placeholder_from_signature(
        out_signature, device=device, dynamic_shape_fill=PREFILL_SCORE_ROWS
    )
    f_q = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        q_runtime,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=PREFILL_BLOCK_QH,
    )
    f_k = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        k_cache,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
    )
    f_block_table = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        block_table_flat,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
    )
    f_weights = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        weights,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dim=0,
        divisibility=block_q,
    )
    f_out = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        out,
        alignment=TMA_COPY_ALIGN_BYTES,
        dynamic_shape_dims=(0, 1),
        dynamic_stride_dims=(0,),
        divisibility=block_q,
    )
    cu_q = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        torch.empty((COMPILE_PLACEHOLDER_DIM + 1,), device=device, dtype=torch.int32),
        alignment=4,
        dynamic_shape_dim=0,
    )
    cu_k = sparse_utils.make_fake_tensor_like_with_dynamic_dim(
        torch.empty((COMPILE_PLACEHOLDER_DIM + 1,), device=device, dtype=torch.int32),
        alignment=4,
        dynamic_shape_dim=0,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return _cute_compile(
        _make_prefill_paged_logits_call_v152(
            q_heads_per_kv,
            block_q,
            debug_mode,
            work_k_group,
            summaries_per_page,
        ),
        f_q,
        f_k,
        f_block_table,
        f_weights,
        f_out,
        cu_q,
        cu_k,
        stream_fake,
        Int32(block_q),
        Int32(PREFILL_SCORE_ROWS),
        Int32(COMPILE_PLACEHOLDER_DIM),
        Int32(COMPILE_PLACEHOLDER_DIM),
        Int32(COMPILE_PLACEHOLDER_DIM),
        Int32(summaries_per_page),
        Int32(COMPILE_PLACEHOLDER_DIM),
    )


def _validate_inputs(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    page_size = _prefill_validate_summaries_per_page(page_size)
    q_heads_per_kv = int(index_q.shape[2])
    block_q = _prefill_block_q_for_heads(q_heads_per_kv)
    if tuple(index_q.shape[1:]) != (SUMMARY_KV_HEADS, q_heads_per_kv, HEAD_DIM):
        raise ValueError(
            f"paged prefill logits index_q must be [Q,{SUMMARY_KV_HEADS},H,{HEAD_DIM}], got "
            f"{tuple(index_q.shape)}"
        )
    query_rows = int(index_q.shape[0])
    max_regions = int(out.shape[1])
    if query_rows <= 0 or query_rows % block_q:
        raise ValueError(
            f"paged prefill logits Q capacity must be a positive multiple of "
            f"{block_q}, got {query_rows}"
        )
    if max_regions <= 0 or max_regions % PREFILL_SCORE_ROWS:
        raise ValueError(
            f"paged prefill logits region capacity must be a positive multiple of "
            f"{PREFILL_SCORE_ROWS}, got {max_regions}"
        )
    if tuple(weights.shape) != (query_rows, SUMMARY_KV_HEADS, q_heads_per_kv):
        raise ValueError(
            f"paged prefill logits weights must be [Q,{SUMMARY_KV_HEADS},H], got "
            f"{tuple(weights.shape)}"
        )
    if summary.dim() != 4 or tuple(summary.shape[2:]) != (SUMMARY_KV_HEADS, HEAD_DIM):
        raise ValueError(
            "paged prefill logits summary must be "
            f"[pages,page_rows,{SUMMARY_KV_HEADS},{HEAD_DIM}] with page_rows in "
            f"{PREFILL_SUPPORTED_SUMMARIES_PER_PAGE}, "
            f"got {tuple(summary.shape)}"
        )
    if int(summary.shape[1]) != page_size:
        raise ValueError(
            "paged prefill logits page_size must match summary.shape[1], "
            f"got page_size={page_size}, summary.shape[1]={int(summary.shape[1])}"
        )
    batch_size = int(block_table.shape[0]) if block_table.dim() == 2 else 0
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or batch_size <= 0
        or not block_table.is_contiguous()
    ):
        raise ValueError(
            "paged prefill logits block_table must be contiguous int32 [B,pages]"
        )
    if tuple(q_runtime.shape) != (query_rows * q_heads_per_kv, HEAD_DIM):
        raise ValueError(
            f"paged prefill logits q_runtime must be [Q*H,{HEAD_DIM}], got "
            f"{tuple(q_runtime.shape)}"
        )
    if tuple(kernel_weights.shape) != (query_rows, q_heads_per_kv):
        raise ValueError(
            "paged prefill logits kernel_weights must be [Q,H], got "
            f"{tuple(kernel_weights.shape)}"
        )
    if tuple(out.shape) != (query_rows, max_regions):
        raise ValueError("paged prefill logits out must be [Q,padded_regions]")
    for name, tensor, dtype in (
        ("index_q", index_q, (torch.float16, torch.bfloat16, torch.float32)),
        ("summary", summary, (torch.float8_e4m3fn, torch.uint8)),
        ("q_runtime", q_runtime, (torch.float8_e4m3fn,)),
        ("kernel_weights", kernel_weights, (torch.float32,)),
        ("out", out, (torch.float32,)),
    ):
        if tensor.dtype not in dtype or not tensor.is_contiguous():
            raise ValueError(
                f"paged prefill logits {name} has unsupported dtype/layout: "
                f"dtype={tensor.dtype}, contiguous={tensor.is_contiguous()}"
            )
    for name, tensor in (
        ("weights", weights),
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if tensor.device != index_q.device or not tensor.is_contiguous():
            raise ValueError(
                f"paged prefill logits {name} must be contiguous on index_q.device"
            )
    if weights.dtype != torch.float32:
        raise ValueError(
            f"paged prefill logits weights must be float32, got {weights.dtype}"
        )
    if (
        tuple(cu_seqlens_q.shape) != (batch_size + 1,)
        or tuple(cu_seqlens_k.shape) != (batch_size + 1,)
        or cu_seqlens_q.dtype != torch.int32
        or cu_seqlens_k.dtype != torch.int32
    ):
        raise ValueError(
            "paged prefill logits cu_seqlens_q/k must be contiguous int32 [B+1] "
            f"for B={batch_size}"
        )
    return query_rows, max_regions, q_heads_per_kv, block_q, batch_size


def _prefill_paged_weighted_relu_logits_impl(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
) -> None:
    query_rows, max_regions, q_heads_per_kv, block_q, batch_size = _validate_inputs(
        index_q,
        weights,
        summary,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        page_size,
        q_runtime,
        kernel_weights,
        out,
    )
    summary_page_size = int(summary.shape[1])
    if index_q.device.type != "cuda":
        raise RuntimeError("paged prefill WGMMA logits requires CUDA tensors")
    q_input = index_q
    k_input = (
        summary.view(torch.float8_e4m3fn) if summary.dtype == torch.uint8 else summary
    )
    pack = _compile_pack_prefill_qw_for_signature(
        sparse_utils.device_cache_key(index_q.device),
        _tensor_signature_dynamic(q_input, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(q_runtime.view(torch.uint8), dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
        q_heads_per_kv,
        block_q,
    )
    total_q = q_runtime.numel()
    total_w = kernel_weights.numel()
    pack(
        q_input,
        weights,
        q_runtime.view(torch.uint8),
        kernel_weights,
        Int32(
            (total_q + total_w + PREFILL_PREPARE_THREADS - 1) // PREFILL_PREPARE_THREADS
        ),
        Int32(total_q),
        Int32(total_w),
        Int32(query_rows),
    )
    work_k_group = _prefill_work_k_group_for_shape(query_rows, batch_size, block_q)
    compiled = _compile_prefill_paged_logits_for_signature(
        sparse_utils.device_cache_key(index_q.device),
        _tensor_signature_dynamic(q_runtime, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(
            k_input,
            dynamic_shape_dims=(0, 1),
            dynamic_stride_dims=(0,),
        ),
        _tensor_signature_dynamic(block_table.reshape(-1), dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(kernel_weights, dynamic_shape_dims=(0,)),
        _tensor_signature_dynamic(
            out, dynamic_shape_dims=(0, 1), dynamic_stride_dims=(0,)
        ),
        q_heads_per_kv,
        block_q,
        summary_page_size,
        0,
        work_k_group,
    )
    sm_count = _sm_count(sparse_utils.device_cache_key(index_q.device))
    total_work = (
        (query_rows + batch_size * (block_q - 1) + block_q - 1) // block_q
    ) * _prefill_grid_k_work_items(work_k_group)
    persistent_grid = min(
        total_work,
        max(1, int(sm_count) * PREFILL_PERSISTENT_CTAS_PER_SM),
    )
    compiled(
        q_runtime,
        k_input,
        block_table.reshape(-1),
        kernel_weights,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        Int32(query_rows),
        Int32(max_regions),
        Int32(block_table.shape[1]),
        Int32(batch_size),
        Int32(summary.shape[0]),
        Int32(summary_page_size),
        Int32(persistent_grid),
    )


def launch_prefill_paged_scoring(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    summary: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    *,
    q_runtime: torch.Tensor,
    kernel_weights: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _prefill_paged_weighted_relu_logits_impl(
        index_q,
        weights,
        summary,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        page_size,
        q_runtime,
        kernel_weights,
        out,
    )
    return out
