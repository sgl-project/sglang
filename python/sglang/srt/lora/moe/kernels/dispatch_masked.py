"""The masked row movers: one pair-grid launch fills the ``[E, m_max, *]``
slabs.

Negative expert IDs leave pair_to_row untouched; consumers must mask them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.kernels.dispatch_checks import check_source_rows


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
