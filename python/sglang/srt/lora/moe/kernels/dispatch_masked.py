"""Dispatch rows into [expert, m_max, hidden] slabs, optionally quantizing to FP8.

Negative expert IDs leave pair_to_row untouched; consumers must mask them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.kernels.dispatch_checks import (
    check_fp8_width,
    check_source_rows,
)
from sglang.srt.lora.moe.kernels.fp8_quant import quantize_fp8_groups


@triton.jit
def _dispatch_fill_masked_bf16_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    gateup_input_ptr,  # [E_local, m_max, hidden] bf16 rows, viewed flat
    topk_ids_ptr,  # [num_tokens * topk]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_tokens * topk] int32 out; valid pairs only
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
    if expert >= 0:
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
    scale_ptr,  # fp32 [E, m_max, K // 128] contiguous
    topk_ids_ptr,  # [num_tokens * top_k]; < 0 = padding or EP-unrouted
    pair_to_row_ptr,  # [num_tokens * top_k] int32 out; valid pairs only
    masked_m_ptr,  # [E] int32 count and atomic cursor; caller zeroes it
    m_max,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,  # K // 128
):
    t = tl.program_id(0)
    k = tl.program_id(1)
    pair = t * TOPK + k
    expert = tl.load(topk_ids_ptr + pair)
    if expert < 0:
        return
    slot = tl.atomic_add(masked_m_ptr + expert, 1)
    dst = expert.to(tl.int64) * m_max + slot
    tl.store(pair_to_row_ptr + pair, dst.to(tl.int32))

    offs = tl.arange(0, K)
    x = tl.load(input_ptr + t.to(tl.int64) * K + offs).to(tl.float32)
    grouped = tl.reshape(x, (GROUPS, 128))
    q, scale = quantize_fp8_groups(grouped, GROUPS == 1)
    tl.store(out_ptr + dst * K + offs, tl.reshape(q, (K,)).to(out_ptr.dtype.element_ty))
    tl.store(scale_ptr + dst * GROUPS + tl.arange(0, GROUPS), scale)


def dispatch_fill_masked_fp8(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    *,
    masked_m_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
    rows_fp8_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    """Dispatch FP8 rows with FP32 scales per group of 128 values."""
    check_source_rows(hidden_states, topk_ids, top_k)
    num_tokens, k = hidden_states.shape
    check_fp8_width(k)
    masked_m_out.zero_()
    if num_tokens > 0:
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
            GROUPS=k // 128,
            num_warps=4 if k <= 4096 else 8,
        )
