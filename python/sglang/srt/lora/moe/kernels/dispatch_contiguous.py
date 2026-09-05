"""The contiguous row movers: routed rows sorted by expert into one compact
buffer whose segments start on aligned rows. The layout needs a launch
sequence, because segment bases are a prefix sum over all experts; the rows
then land in bf16.

Negative expert IDs leave pair_to_row untouched; consumers must mask them.
Atomic slot order may vary, so every stage must use the same pair_to_row map.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.kernels.dispatch_checks import check_source_rows

# Eight pairs per program outperformed 16/32 on 65k-pair prefill chunks.
PAIRS_PER_PROGRAM = 8


def contiguous_m_pad_ceiling(num_pairs: int, num_experts: int, alignment: int) -> int:
    """Bound padded rows by allowing alignment - 1 extra rows per expert."""
    raw = num_pairs + num_experts * (alignment - 1)
    return -(-raw // alignment) * alignment


@triton.jit
def _count_slots_contiguous_kernel(
    topk_ids_ptr,  # [num_pairs]; a value < 0 marks a pair with no expert
    slot_out_ptr,  # [num_pairs] int32 out: slot of the pair inside its expert
    seg_counts_ptr,  # [E_local] int32 count and cursor; the caller must zero it
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    pairs = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM + tl.arange(
        0, PAIRS_PER_PROGRAM
    )
    in_range = pairs < num_pairs
    experts = tl.load(topk_ids_ptr + pairs, mask=in_range, other=-1)
    valid = in_range & (experts >= 0)
    slots = tl.atomic_add(seg_counts_ptr + experts, 1, mask=valid)
    tl.store(slot_out_ptr + pairs, slots, mask=valid)


@triton.jit
def _seg_layout_contiguous_kernel(
    seg_counts_ptr,  # [E_local] int32 per-expert routed-pair counts
    seg_offsets_ptr,  # [E_local + 1] int32 out: first row of each segment
    ALIGN: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_EXPERTS)
    counts = tl.load(seg_counts_ptr + offs, mask=offs < EXPERTS, other=0)
    aligned = ((counts + (ALIGN - 1)) // ALIGN) * ALIGN
    inclusive = tl.cumsum(aligned, axis=0)
    tl.store(seg_offsets_ptr + offs, inclusive - aligned, mask=offs < EXPERTS)
    tl.store(seg_offsets_ptr + EXPERTS, tl.sum(aligned))


@triton.jit
def _fill_rows_contiguous_bf16_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    compact_ptr,  # [m_pad_ceiling, hidden] bf16, viewed flat
    topk_ids_ptr,  # [num_pairs]
    pair_to_row_ptr,  # [num_pairs] int32 finalized compact rows
    hidden_size,
    num_pairs,
    TOPK: tl.constexpr,
    PAIRS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Finalize pair_to_row in a separate launch: fusing it with this copy caused
    # out-of-bounds stores at 65536 pairs and hidden=2048.
    lane = tl.arange(0, PAIRS_PER_PROGRAM)
    base = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM
    pairs = base + lane
    in_range = pairs < num_pairs
    experts_vec = tl.load(topk_ids_ptr + pairs, mask=in_range, other=-1)
    if tl.sum((experts_vec >= 0).to(tl.int32), axis=0) == 0:
        return

    vec = tl.arange(0, BLOCK_H)
    for i in tl.static_range(PAIRS_PER_PROGRAM):
        pair = base + i
        if pair < num_pairs:
            expert = tl.load(topk_ids_ptr + pair)
            if expert >= 0:
                dst = tl.load(pair_to_row_ptr + pair).to(tl.int64)
                token = pair // TOPK
                src = input_ptr + token * hidden_size
                out = compact_ptr + dst * hidden_size
                for off in tl.range(0, hidden_size, BLOCK_H):
                    mask = off + vec < hidden_size
                    tl.store(
                        out + off + vec,
                        tl.load(src + off + vec, mask=mask),
                        mask=mask,
                    )


@triton.jit
def _finalize_pair_to_row_contiguous_kernel(
    topk_ids_ptr,  # [num_pairs]; a value < 0 marks a pair with no expert
    pair_to_row_ptr,  # [num_pairs] int32: dense slots IN, compact rows OUT
    seg_offsets_ptr,  # [E_local + 1] int32 first row of each segment
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM
    for i in tl.static_range(PAIRS_PER_PROGRAM):
        pair = base + i
        if pair < num_pairs:
            expert = tl.load(topk_ids_ptr + pair)
            if expert >= 0:
                slot = tl.load(pair_to_row_ptr + pair)
                seg_base = tl.load(seg_offsets_ptr + expert)
                tl.store(pair_to_row_ptr + pair, seg_base + slot)


def dispatch_layout_contiguous(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    top_k: int,
    alignment: int,
    *,
    seg_counts_out: torch.Tensor,
    seg_offsets_out: torch.Tensor,
    pair_to_row_out: torch.Tensor,
) -> None:
    """Write compact row indices to pair_to_row; row copies run separately."""
    check_source_rows(hidden_states, topk_ids, top_k)
    num_pairs = topk_ids.numel()
    pair_grid = (-(-num_pairs // PAIRS_PER_PROGRAM),)

    # Clear atomic cursors on every forward, including graph replay.
    seg_counts_out.zero_()
    if num_pairs > 0:
        _count_slots_contiguous_kernel[pair_grid](
            topk_ids.view(-1),
            pair_to_row_out,
            seg_counts_out,
            num_pairs,
            PAIRS_PER_PROGRAM=PAIRS_PER_PROGRAM,
        )
    # Initialize offsets even for an empty batch.
    _seg_layout_contiguous_kernel[(1,)](
        seg_counts_out,
        seg_offsets_out,
        ALIGN=alignment,
        EXPERTS=num_local_experts,
        BLOCK_EXPERTS=max(triton.next_power_of_2(num_local_experts), 2),
    )
    if num_pairs > 0:
        _finalize_pair_to_row_contiguous_kernel[pair_grid](
            topk_ids.view(-1),
            pair_to_row_out,
            seg_offsets_out,
            num_pairs,
            PAIRS_PER_PROGRAM=PAIRS_PER_PROGRAM,
        )


def dispatch_fill_rows_contiguous_bf16(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    pair_to_row: torch.Tensor,
    *,
    hidden_compact_out: torch.Tensor,
) -> None:
    num_tokens = hidden_states.size(0)
    num_pairs = topk_ids.numel()
    if num_pairs == 0:
        return
    check_source_rows(hidden_states, topk_ids, num_pairs // num_tokens)
    _fill_rows_contiguous_bf16_kernel[(-(-num_pairs // PAIRS_PER_PROGRAM),)](
        hidden_states,
        hidden_compact_out,
        topk_ids.view(-1),
        pair_to_row,
        hidden_states.size(1),
        num_pairs,
        TOPK=num_pairs // num_tokens,
        PAIRS_PER_PROGRAM=PAIRS_PER_PROGRAM,
        BLOCK_H=1024,
    )
