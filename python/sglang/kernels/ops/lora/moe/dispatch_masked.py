"""Dispatch rows into [expert, m_max, hidden] slabs, optionally quantizing to FP8.

Negative expert IDs write pair_to_row == -1; consumers must skip those rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.lora.moe.dispatch_checks import (
    check_fp8_width,
    check_source_rows,
)
from sglang.kernels.ops.lora.moe.fp8_quant import quantize_fp8_groups
from sglang.kernels.ops.lora.moe.quant_blockscaled import (
    BlockScaledRows,
    div_rn_ieee,
    quant_column_chunk,
    quantize_tile,
    store_quantized_rows,
)


@triton.jit
def _dispatch_fill_masked_bf16_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    gateup_input_ptr,  # [E_local, m_max, hidden] bf16 rows, viewed flat
    topk_ids_ptr,  # [num_tokens * topk]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_tokens * topk] int32 out; -1 for pairs without an expert
    masked_m_ptr,  # [E_local] int32 count and atomic cursor; caller zeroes it
    m_max,
    hidden_size,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.program_id(1)
    pair = t * TOPK + k
    expert = tl.load(topk_ids_ptr + pair)
    if expert < 0:
        # The finalize gates on the row map: no expert reads as -1.
        tl.store(pair_to_row_ptr + pair, -1)
    else:
        slot = tl.atomic_add(masked_m_ptr + expert, 1)
        dst = expert.to(tl.int64) * m_max + slot
        tl.store(pair_to_row_ptr + pair, dst.to(tl.int32))
        src = input_ptr + t.to(tl.int64) * hidden_size
        out = gateup_input_ptr + dst * hidden_size
        vec = tl.arange(0, BLOCK_H)
        for off in tl.range(0, hidden_size, BLOCK_H):
            mask = off + vec < hidden_size
            tl.store(out + off + vec, tl.load(src + off + vec, mask=mask), mask=mask)


def dispatch_fill_masked_bf16(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    *,
    masked_m_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
    rows_out: torch.Tensor,
) -> None:
    check_source_rows(hidden_states, topk_ids, top_k)
    num_tokens = hidden_states.size(0)
    # Clear atomic cursors on every forward, including graph replay.
    masked_m_out.zero_()
    if num_tokens > 0:
        _dispatch_fill_masked_bf16_kernel[(num_tokens, top_k)](
            hidden_states,
            rows_out,
            topk_ids.view(-1),
            pair_to_row_out,
            masked_m_out,
            rows_out.size(1),
            hidden_states.size(1),
            TOPK=top_k,
            BLOCK_H=1024,
        )


@triton.jit
def _dispatch_fill_masked_fp8_kernel(
    input_ptr,  # bf16 [num_tokens, K] source rows
    out_ptr,  # fp8e4m3 [E, m_max, K] contiguous
    scale_ptr,  # fp32 [E, m_max, GROUPS] contiguous
    topk_ids_ptr,  # [num_tokens * top_k]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_tokens * top_k] int32 out; -1 for pairs without an expert
    masked_m_ptr,  # [E] int32 count and atomic cursor; caller zeroes it
    m_max,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,  # 128, or BLOCK_K for one scale per token
    GROUPS: tl.constexpr,  # K // GROUP, or 1 per token
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.program_id(1)
    pair = t * TOPK + k
    expert = tl.load(topk_ids_ptr + pair)
    if expert < 0:
        tl.store(pair_to_row_ptr + pair, -1)
        return
    slot = tl.atomic_add(masked_m_ptr + expert, 1)
    dst = expert.to(tl.int64) * m_max + slot
    tl.store(pair_to_row_ptr + pair, dst.to(tl.int32))

    offs = tl.arange(0, BLOCK_K)
    x = tl.load(input_ptr + t.to(tl.int64) * K + offs, mask=offs < K, other=0).to(
        tl.float32
    )
    grouped = tl.reshape(x, (BLOCK_K // GROUP, GROUP))
    q, scale = quantize_fp8_groups(grouped, GROUPS == 1)
    tl.store(
        out_ptr + dst * K + offs,
        tl.reshape(q, (BLOCK_K,)).to(out_ptr.dtype.element_ty),
        mask=offs < K,
    )
    groups = tl.arange(0, BLOCK_K // GROUP)
    tl.store(scale_ptr + dst * GROUPS + groups, scale, mask=groups < GROUPS)


def dispatch_fill_masked_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    *,
    masked_m_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
    rows_fp8_out: torch.Tensor,
    scale_out: torch.Tensor,
    per_token: bool = False,
) -> None:
    """Dispatch FP8 rows with FP32 scales per group of 128 values or per token."""
    check_source_rows(hidden_states, topk_ids, top_k)
    num_tokens, k = hidden_states.shape
    check_fp8_width(k)
    masked_m_out.zero_()
    if num_tokens > 0:
        block_k = triton.next_power_of_2(k)
        _dispatch_fill_masked_fp8_kernel[(num_tokens, top_k)](
            hidden_states,
            rows_fp8_out,
            scale_out,
            topk_ids.view(-1),
            pair_to_row_out,
            masked_m_out,
            rows_fp8_out.size(1),
            TOPK=top_k,
            K=k,
            GROUP=block_k if per_token else 128,
            GROUPS=1 if per_token else k // 128,
            BLOCK_K=block_k,
            num_warps=4 if k <= 4096 else 8,
        )


@triton.jit
def _dispatch_fill_masked_blockscaled_kernel(
    input_ptr,  # bf16 [num_tokens, K] source rows
    out_ptr,  # [E, m_max, K] E4M3 or [E, m_max, K // 2] E2M1 pairs, viewed flat
    sf_ptr,  # uint8 scale atoms
    topk_ids_ptr,  # [num_tokens * top_k]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_tokens * top_k] int32 out; -1 for pairs without an expert
    masked_m_ptr,  # [E] int32 count and atomic cursor; caller zeroes it
    stride_out_row,
    rows_per_slab,
    row_atoms,
    k_atoms,
    global_scale,
    row_scale_ptr,  # [E * m_max] fp32 stored per-row scales (PER_ROW)
    row_denominator,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VEC: tl.constexpr,
    ELEM_MAX: tl.constexpr,
    ELEM_MAX_LOG2: tl.constexpr,
    PACK_E2M1: tl.constexpr,
    NVFP4: tl.constexpr,
    PER_ROW: tl.constexpr,
    RECIPE_SELECT: tl.constexpr,
    ERR_MSE: tl.constexpr,
    ERR_FAST: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.program_id(1)
    pair = t * TOPK + k
    expert = tl.load(topk_ids_ptr + pair)
    if expert < 0:
        tl.store(pair_to_row_ptr + pair, -1)
        return
    slot = tl.atomic_add(masked_m_ptr + expert, 1)
    dst = expert.to(tl.int64) * rows_per_slab + slot
    tl.store(pair_to_row_ptr + pair, dst.to(tl.int32))

    # Whole column chunks: no lane is masked, and the loop pipelines the loads.
    offs = tl.arange(0, BLOCK_K)
    row_scale = tl.zeros((1,), tl.float32)
    if PER_ROW:
        # First pass: the row's stored scale from its amax over every chunk.
        amax = tl.zeros((BLOCK_K,), tl.float32)
        for col0 in range(0, K, BLOCK_K):
            x = tl.load(input_ptr + t.to(tl.int64) * K + col0 + offs)
            amax = tl.maximum(amax, tl.abs(x.to(tl.float32)))
        row_scale = div_rn_ieee(
            tl.max(amax, axis=0) + row_scale, row_scale + row_denominator
        )
        tl.store(row_scale_ptr + dst + tl.arange(0, 1), row_scale)
    for col0 in range(0, K, BLOCK_K):
        x = tl.load(input_ptr + t.to(tl.int64) * K + col0 + offs)
        codes, sf_bytes = quantize_tile(
            x.to(tl.float32)[None, :],
            global_scale,
            row_scale,
            VEC,
            ELEM_MAX,
            ELEM_MAX_LOG2,
            PACK_E2M1,
            NVFP4,
            PER_ROW,
            RECIPE_SELECT,
            ERR_MSE,
            ERR_FAST,
            row_denominator,
        )
        store_quantized_rows(
            codes,
            sf_bytes,
            dst + tl.zeros((1,), tl.int64),
            tl.arange(0, 1) == 0,
            col0,
            K,
            out_ptr,
            stride_out_row,
            sf_ptr,
            rows_per_slab,
            row_atoms,
            k_atoms,
            VEC,
            PACK_E2M1,
        )


def dispatch_fill_masked_blockscaled(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    *,
    masked_m_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
    rows_out: BlockScaledRows,
) -> None:
    """Dispatch rows straight into block-scaled codes and scale atoms."""
    check_source_rows(hidden_states, topk_ids, top_k)
    num_tokens, k = hidden_states.shape
    assert k == rows_out.k and k % 128 == 0
    masked_m_out.zero_()
    if num_tokens > 0:
        _dispatch_fill_masked_blockscaled_kernel[(num_tokens, top_k)](
            hidden_states,
            rows_out.data,
            rows_out.scales,
            topk_ids.view(-1),
            pair_to_row_out,
            masked_m_out,
            TOPK=top_k,
            K=k,
            BLOCK_K=quant_column_chunk(k),
            num_warps=4,
            **rows_out.kernel_args(),
        )
