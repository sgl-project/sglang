"""The row movers, one prepare kernel set per row domain: masked fills the
``[E, m_max, *]`` slabs in one kernel; contiguous needs a sequence, because
its segment bases are a prefix sum over all experts.

A pair with ``topk_ids < 0`` gets no row and keeps old ``src2dst`` data, so
every consumer must test ``topk_ids >= 0`` first. Atomic arrival order picks
slots, and every stage reads the same ``src2dst``, so any order gives the
same bits. Grids are host-static, so CUDA graphs can capture everything.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl


@triton.jit
def _dispatch_fill_masked_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    gateup_input_ptr,  # [E_local, m_max, hidden] bf16 rows, viewed flat
    topk_ids_ptr,  # [num_tokens * topk]; < 0 = padding or EP-unrouted
    src2dst_ptr,  # [num_tokens * topk] int32 out; valid pairs only
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
        tl.store(src2dst_ptr + pair, dst.to(tl.int32))
        src = input_ptr + t.to(tl.int64) * hidden_size
        out = gateup_input_ptr + dst * hidden_size
        vec = tl.arange(0, BLOCK_H)
        for off in tl.range(0, hidden_size, BLOCK_H):
            mask = off + vec < hidden_size
            tl.store(out + off + vec, tl.load(src + off + vec, mask=mask), mask=mask)


def dispatch_fill_masked(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    hidden_states: torch.Tensor,
    top_k: int,
    *,
    masked_m_out: torch.Tensor | None = None,
    src2dst_out: torch.Tensor | None = None,
    gateup_input_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    # The fill kernel would silently downcast a non-BF16 input.
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("fused masked preprocess is BF16-only")
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous [num_tokens, hidden]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    num_tokens = hidden_states.size(0)
    num_pairs = topk_ids.numel()
    if num_pairs != num_tokens * top_k:
        raise ValueError(
            f"topk_ids carries {num_pairs} pairs for {num_tokens} tokens x "
            f"top_k={top_k}"
        )
    m_max = (hidden_states.size(0) // 256 + 1) * 256
    expected_m = (topk_ids.numel() - 1) // num_local_experts + 1

    masked_m = (
        torch.empty(num_local_experts, device=topk_ids.device, dtype=torch.int32)
        if masked_m_out is None
        else masked_m_out
    )
    src2dst = (
        torch.empty(num_pairs, device=topk_ids.device, dtype=torch.int32)
        if src2dst_out is None
        else src2dst_out
    )
    gateup_input = (
        torch.empty(
            (num_local_experts, m_max, hidden_states.size(1)),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        if gateup_input_out is None
        else gateup_input_out
    )

    # ``masked_m`` is also the atomic cursor, so it must start at zero. The
    # memset runs in stream order, and CUDA-graph capture accepts it.
    masked_m.zero_()
    if num_tokens > 0:
        _dispatch_fill_masked_kernel[(num_tokens, top_k)](
            hidden_states,
            gateup_input,
            topk_ids.view(-1),
            src2dst,
            masked_m,
            m_max,
            hidden_states.size(1),
            TOPK=top_k,
            BLOCK_H=1024,
        )
    return masked_m, expected_m, src2dst, gateup_input


# Swept at 8/16/32 on the 65k-pair prefill chunks: the pair kernels are
# program-count-bound, and 16/32 regress. ``tl.arange`` rejects a non-power-of-
# two override at compile time.
PAIRS_PER_PROGRAM = 8


def contiguous_m_pad_ceiling(num_pairs: int, num_experts: int, alignment: int) -> int:
    """Return a host-static upper bound on the padded row count.

    The true count is ``sum_e ceil(count_e / alignment) * alignment``, so each
    expert adds at most ``alignment - 1`` padding rows. This function rounds the
    bound up again, so every row lies inside a whole m-block.
    """
    if not isinstance(num_pairs, int) or num_pairs < 0:
        raise ValueError(f"num_pairs must be a non-negative int, got {num_pairs!r}")
    if not isinstance(num_experts, int) or num_experts < 1:
        raise ValueError(f"num_experts must be a positive int, got {num_experts!r}")
    if not isinstance(alignment, int) or alignment < 1:
        raise ValueError(f"alignment must be a positive int, got {alignment!r}")
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
    # Two lanes of one vector can hold the same expert. The atomic add is still
    # exact, because each lane does its own read-modify-write.
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
    grouped_layout_ptr,  # [m_pad_ceiling] int32 out; the caller does not prefill it
    m_pad_ceiling,  # host-static row ceiling; rows [M_pad, ceiling) get -1
    schedule1_ptr,  # BUILD_SCHEDULES only: [capacity1] int64 packed tiles
    tiles1_ptr,  # BUILD_SCHEDULES only: [1] int32 stage-1 tile count
    schedule2_ptr,  # BUILD_SCHEDULES only: [capacity2] int64 packed tiles
    tiles2_ptr,  # BUILD_SCHEDULES only: [1] int32 stage-2 tile count
    token_width,
    out_clusters1,
    out_clusters2,
    ALIGN: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BUILD_SCHEDULES: tl.constexpr,
    ENTRY_BLOCK: tl.constexpr,
    TOKEN_SHIFT: tl.constexpr,
    OUTPUT_SHIFT: tl.constexpr,
):
    # The grid is (EXPERTS + 1,), or (EXPERTS + 1, 2) when BUILD_SCHEDULES is
    # true. Every stage-0 program writes the same values into seg_offsets. That
    # race is safe because the values agree. Each program then labels its whole
    # aligned segment, padding rows included, so each m-block holds one group
    # only. Program EXPERTS labels the tail rows [M_pad, m_pad_ceiling) with -1
    # and so replaces a host-side fill_(-1). The two write ranges never overlap.
    expert = tl.program_id(0)
    stage = tl.program_id(1)
    offs = tl.arange(0, BLOCK_EXPERTS)
    counts = tl.load(seg_counts_ptr + offs, mask=offs < EXPERTS, other=0)
    aligned = ((counts + (ALIGN - 1)) // ALIGN) * ALIGN
    inclusive = tl.cumsum(aligned, axis=0)
    total_rows = tl.sum(aligned)

    if stage == 0:
        tl.store(seg_offsets_ptr + offs, inclusive - aligned, mask=offs < EXPERTS)
        tl.store(seg_offsets_ptr + EXPERTS, total_rows)

        is_tail = expert == EXPERTS
        seg_start = tl.where(
            is_tail,
            total_rows,
            tl.sum(tl.where(offs == expert, inclusive - aligned, 0)),
        ).to(tl.int64)
        seg_rows = tl.where(
            is_tail,
            m_pad_ceiling - total_rows,
            tl.sum(tl.where(offs == expert, aligned, 0)),
        )
        label = tl.where(is_tail, -1, expert)
        row_vec = tl.arange(0, BLOCK_ROWS)
        for base in tl.range(0, seg_rows, BLOCK_ROWS):
            rows = base + row_vec
            tl.store(
                grouped_layout_ptr + seg_start + rows,
                tl.full((BLOCK_ROWS,), 0, tl.int32) + label,
                mask=rows < seg_rows,
            )

    # Both schedule stages read the same device row counts in this one launch.
    # This kernel copies the arithmetic from _dual_schedule_kernel. This build
    # and the standalone build write the same entries.
    if BUILD_SCHEDULES:
        if expert < EXPERTS:
            out_clusters = tl.where(stage == 0, out_clusters1, out_clusters2)
            all_entries = tl.cdiv(counts, token_width) * out_clusters
            begin = tl.sum(tl.where(offs < expert, all_entries, 0))
            if expert == 0:
                total = tl.sum(tl.where(offs < EXPERTS, all_entries, 0))
                if stage == 0:
                    tl.store(tiles1_ptr, total)
                else:
                    tl.store(tiles2_ptr, total)
            rows = tl.load(seg_counts_ptr + expert)
            token_clusters = tl.cdiv(rows, token_width)
            safe_tc = tl.maximum(token_clusters, 1)
            entries = token_clusters * out_clusters
            token_major = token_clusters > out_clusters
            for base in range(0, entries, ENTRY_BLOCK):
                local = base + tl.arange(0, ENTRY_BLOCK)
                valid = local < entries
                oc_a = local // safe_tc
                tc_a = local - oc_a * safe_tc
                tc_b = local // out_clusters
                oc_b = local - tc_b * out_clusters
                tc_i = tl.where(token_major, tc_b, tc_a)
                oc_i = tl.where(token_major, oc_b, oc_a)
                # Widen to int64 before the shift. An int32 shift overflows
                # the upper fields and reports no error. See schedule_builder.
                packed = (
                    expert.to(tl.int64)
                    | (tc_i.to(tl.int64) << TOKEN_SHIFT)
                    | (oc_i.to(tl.int64) << OUTPUT_SHIFT)
                )
                if stage == 0:
                    tl.store(schedule1_ptr + begin + local, packed, mask=valid)
                else:
                    tl.store(schedule2_ptr + begin + local, packed, mask=valid)


@triton.jit
def _fill_rows_contiguous_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    compact_ptr,  # [m_pad_ceiling, hidden] bf16, viewed flat
    topk_ids_ptr,  # [num_pairs]
    src2dst_ptr,  # [num_pairs] int32 finalized compact rows
    seg_offsets_ptr,  # [E_local + 1] int32 first row of each segment
    hidden_size,
    num_pairs,
    TOPK: tl.constexpr,
    PAIRS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # This kernel only copies rows. src2dst must already hold the final compact
    # row of each pair. An earlier version computed that row here as well. At
    # num_pairs=65536 and hidden=2048 it stored past the end of the buffer. The
    # inputs were correct, and every form of the fused code failed the same way.
    # Keep the two stages in separate kernels.
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
                dst = tl.load(src2dst_ptr + pair).to(tl.int64)
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
def _finalize_src2dst_contiguous_kernel(
    topk_ids_ptr,  # [num_pairs]; a value < 0 marks a pair with no expert
    src2dst_ptr,  # [num_pairs] int32: dense slots IN, compact rows OUT
    seg_offsets_ptr,  # [E_local + 1] int32 first row of each segment
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    # This stage needs its own launch. See the note on _fill_rows_contiguous_kernel.
    base = tl.program_id(0).to(tl.int64) * PAIRS_PER_PROGRAM
    for i in tl.static_range(PAIRS_PER_PROGRAM):
        pair = base + i
        if pair < num_pairs:
            expert = tl.load(topk_ids_ptr + pair)
            if expert >= 0:
                slot = tl.load(src2dst_ptr + pair)
                seg_base = tl.load(seg_offsets_ptr + expert)
                tl.store(src2dst_ptr + pair, seg_base + slot)


class ContiguousSchedulePack(msgspec.Struct, kw_only=True):
    """The buffers and field shifts of the packed CuTeDSL tile schedule.

    Build this struct with
    ``cutedsl.schedule_builder.dual_stage_schedule_pack_contiguous``.
    That function checks the buffer capacities. This module reads the field
    shifts as plain data, so it does not depend on the GEMM engine.
    """

    schedule1: torch.Tensor  # [capacity1] int64 packed (expert, tc, oc)
    tiles1: torch.Tensor  # [1] int32 stage-1 tile count
    schedule2: torch.Tensor  # [capacity2] int64 packed (expert, tc, oc)
    tiles2: torch.Tensor  # [1] int32 stage-2 tile count
    token_width: int
    out_clusters1: int
    out_clusters2: int
    token_cluster_shift: int
    output_cluster_shift: int
    entry_block: int = 128


def dispatch_fill_contiguous(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    top_k: int,
    alignment: int,
    *,
    seg_counts_out: torch.Tensor,
    seg_offsets_out: torch.Tensor,
    src2dst_out: torch.Tensor,
    grouped_layout_out: torch.Tensor,
    hidden_compact_out: torch.Tensor,
    pairs_per_program: int = PAIRS_PER_PROGRAM,
    schedule_pack: ContiguousSchedulePack | None = None,
) -> None:
    """Count the pairs of each expert, place the segments, and gather the rows.

    On return ``src2dst[t * top_k + k]`` holds the compact row
    ``seg_offsets[e] + slot`` for each routed pair. A sentinel pair gets no
    store, as in the masked dispatch.

    Every grid and buffer shape comes from a host-static value, and the segment
    lengths stay on the device. A CUDA graph can therefore capture this call.
    """
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous [num_tokens, hidden]")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("contiguous dispatch is BF16-only")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    num_tokens = hidden_states.size(0)
    num_pairs = topk_ids.numel()
    if num_pairs != num_tokens * top_k:
        raise ValueError(
            f"topk_ids carries {num_pairs} pairs for {num_tokens} tokens x "
            f"top_k={top_k}"
        )
    m_pad_ceiling = grouped_layout_out.numel()
    pair_grid = (-(-num_pairs // pairs_per_program),)

    # The counts also serve as the atomic cursors, so they must start at zero.
    # A graph can capture this stream-ordered zeroing. grouped_layout needs no
    # prefill, because the seg-layout launch writes its -1 rows.
    seg_counts_out.zero_()
    if num_pairs > 0:
        _count_slots_contiguous_kernel[pair_grid](
            topk_ids.view(-1),
            src2dst_out,
            seg_counts_out,
            num_pairs,
            PAIRS_PER_PROGRAM=pairs_per_program,
        )
    # This launch runs even when there are no pairs. It defines seg_offsets and
    # the grouped_layout labels on every forward pass. With a pack it also
    # defines the packed tile counts.
    if schedule_pack is None:
        # The schedule arguments point at real tensors. The kernel reads and
        # writes none of them, because BUILD_SCHEDULES is a constexpr False.
        _seg_layout_contiguous_kernel[(num_local_experts + 1,)](
            seg_counts_out,
            seg_offsets_out,
            grouped_layout_out,
            m_pad_ceiling,
            seg_counts_out,
            seg_counts_out,
            seg_counts_out,
            seg_counts_out,
            1,
            1,
            1,
            ALIGN=alignment,
            EXPERTS=num_local_experts,
            BLOCK_EXPERTS=max(triton.next_power_of_2(num_local_experts), 2),
            BLOCK_ROWS=1024,
            BUILD_SCHEDULES=False,
            ENTRY_BLOCK=2,
            TOKEN_SHIFT=1,
            OUTPUT_SHIFT=1,
        )
    else:
        _seg_layout_contiguous_kernel[(num_local_experts + 1, 2)](
            seg_counts_out,
            seg_offsets_out,
            grouped_layout_out,
            m_pad_ceiling,
            schedule_pack.schedule1,
            schedule_pack.tiles1,
            schedule_pack.schedule2,
            schedule_pack.tiles2,
            schedule_pack.token_width,
            schedule_pack.out_clusters1,
            schedule_pack.out_clusters2,
            ALIGN=alignment,
            EXPERTS=num_local_experts,
            BLOCK_EXPERTS=max(triton.next_power_of_2(num_local_experts), 2),
            BLOCK_ROWS=1024,
            BUILD_SCHEDULES=True,
            ENTRY_BLOCK=schedule_pack.entry_block,
            TOKEN_SHIFT=schedule_pack.token_cluster_shift,
            OUTPUT_SHIFT=schedule_pack.output_cluster_shift,
        )
    if num_pairs > 0:
        _finalize_src2dst_contiguous_kernel[pair_grid](
            topk_ids.view(-1),
            src2dst_out,
            seg_offsets_out,
            num_pairs,
            PAIRS_PER_PROGRAM=pairs_per_program,
        )
        _fill_rows_contiguous_kernel[pair_grid](
            hidden_states,
            hidden_compact_out,
            topk_ids.view(-1),
            src2dst_out,
            seg_offsets_out,
            hidden_states.size(1),
            num_pairs,
            TOPK=top_k,
            PAIRS_PER_PROGRAM=pairs_per_program,
            BLOCK_H=1024,
        )
