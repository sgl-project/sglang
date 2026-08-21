"""The contiguous row domain: all routed rows in one compact 2-D buffer.

:mod:`masked_row_domain` gives each expert its own ``[m_max, ·]`` slab. This
module instead sorts the routed rows by expert into a single buffer. Each
expert's segment starts on a multiple of the DeepGEMM m-alignment. The buffer
then holds about ``num_tokens * top_k`` rows, not ``num_experts * m_max``.

Use this domain for prefill only. A decode port of it measured slower than the
masked domain on GB300, so decode keeps the masked domain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
)
from sglang.srt.lora.moe.kernels.activation_delta import (
    _activation_delta_masked_kernel,
)
from sglang.srt.lora.moe.kernels.fused_act import (
    MASKED_ACT_FAMILIES,
    _b_act_kernel,
    _is_power_of_two,
    _require_config,
    _validate_b_inputs,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


# The number of programs limits the dispatch kernels below. The work inside
# one program does not limit them. Each program therefore takes this many pairs.
PAIRS_PER_PROGRAM = 8


def _validate_pairs_per_program(pairs_per_program: int) -> None:
    if (
        not isinstance(pairs_per_program, int)
        or pairs_per_program < 1
        or pairs_per_program & (pairs_per_program - 1)
    ):
        raise ValueError(
            "pairs_per_program must be a positive power of two, got "
            f"{pairs_per_program!r}"
        )


def coarsened_pair_grid(num_pairs: int, pairs_per_program: int) -> int:
    _validate_pairs_per_program(pairs_per_program)
    if num_pairs < 0:
        raise ValueError(f"num_pairs must be non-negative, got {num_pairs}")
    return -(-num_pairs // pairs_per_program)


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
def _contig_count_slots_kernel(
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
def _contig_seg_layout_kernel(
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
def _contig_fill_rows_kernel(
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
def _contig_finalize_src2dst_kernel(
    topk_ids_ptr,  # [num_pairs]; a value < 0 marks a pair with no expert
    src2dst_ptr,  # [num_pairs] int32: dense slots IN, compact rows OUT
    seg_offsets_ptr,  # [E_local + 1] int32 first row of each segment
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    # This stage needs its own launch. See the note on _contig_fill_rows_kernel.
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
    ``cutedsl_masked.schedule_builder.contiguous_dual_stage_schedule_pack``.
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


def _validate_schedule_pack(
    pack: ContiguousSchedulePack,
    alignment: int,
    m_pad_ceiling: int,
    device: torch.device,
) -> None:
    if (
        not isinstance(pack.token_width, int)
        or pack.token_width < 1
        or alignment % pack.token_width
    ):
        raise ValueError(
            f"schedule_pack.token_width={pack.token_width!r} must be a "
            f"positive divisor of the segment alignment {alignment}"
        )
    for name, value in (
        ("out_clusters1", pack.out_clusters1),
        ("out_clusters2", pack.out_clusters2),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"schedule_pack.{name} must be a positive int, got {value!r}"
            )
    for name, value in (
        ("token_cluster_shift", pack.token_cluster_shift),
        ("output_cluster_shift", pack.output_cluster_shift),
    ):
        # The packed word is a signed int64, so bit 63 is the sign bit and no
        # field may reach it. See schedule_abi.
        if not isinstance(value, int) or not 0 < value < 63:
            raise ValueError(
                f"schedule_pack.{name} must be an int in (0, 63), got {value!r}"
            )
    if (
        not isinstance(pack.entry_block, int)
        or pack.entry_block < 1
        or pack.entry_block & (pack.entry_block - 1)
    ):
        raise ValueError(
            "schedule_pack.entry_block must be a positive power of two, got "
            f"{pack.entry_block!r}"
        )
    # The token tile divides the alignment. The build therefore writes at most
    # (m_pad_ceiling / token_width) * out_clusters entries for each stage.
    total_clusters = m_pad_ceiling // pack.token_width
    for name, tensor, needed, dtype in (
        ("schedule1", pack.schedule1, total_clusters * pack.out_clusters1, torch.int64),
        ("schedule2", pack.schedule2, total_clusters * pack.out_clusters2, torch.int64),
        ("tiles1", pack.tiles1, 1, torch.int32),
        ("tiles2", pack.tiles2, 1, torch.int32),
    ):
        if (
            tensor.ndim != 1
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.device != device
        ):
            raise ValueError(
                f"schedule_pack.{name} must be contiguous 1-D "
                f"{str(dtype).removeprefix('torch.')} on {device}"
            )
        if tensor.numel() < needed:
            raise ValueError(
                f"schedule_pack.{name} holds {tensor.numel()} entries but the "
                f"row ceiling implies up to {needed}"
            )
    if pack.tiles1.numel() != 1 or pack.tiles2.numel() != 1:
        raise ValueError("schedule_pack tile counts must be one-element int32")


def contiguous_dispatch_fill(
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
    if m_pad_ceiling != contiguous_m_pad_ceiling(
        num_pairs, num_local_experts, alignment
    ):
        raise ValueError(
            "grouped_layout does not match the host-static row ceiling: got "
            f"{m_pad_ceiling}, expected "
            f"{contiguous_m_pad_ceiling(num_pairs, num_local_experts, alignment)}"
        )
    for name, tensor, shape in (
        ("seg_counts_out", seg_counts_out, (num_local_experts,)),
        ("seg_offsets_out", seg_offsets_out, (num_local_experts + 1,)),
        ("src2dst_out", src2dst_out, (num_pairs,)),
        ("grouped_layout_out", grouped_layout_out, (m_pad_ceiling,)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32 {list(shape)}")
    if (
        hidden_compact_out.shape != (m_pad_ceiling, hidden_states.size(1))
        or hidden_compact_out.dtype != torch.bfloat16
        or not hidden_compact_out.is_contiguous()
    ):
        raise ValueError(
            "hidden_compact_out must be contiguous bf16 "
            f"[{m_pad_ceiling}, {hidden_states.size(1)}]"
        )

    if schedule_pack is not None:
        _validate_schedule_pack(
            schedule_pack, alignment, m_pad_ceiling, hidden_states.device
        )

    # The counts also serve as the atomic cursors, so they must start at zero.
    # A graph can capture this stream-ordered zeroing. grouped_layout needs no
    # prefill, because the seg-layout launch writes its -1 rows.
    seg_counts_out.zero_()
    if num_pairs > 0:
        _contig_count_slots_kernel[
            (coarsened_pair_grid(num_pairs, pairs_per_program),)
        ](
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
        _contig_seg_layout_kernel[(num_local_experts + 1,)](
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
        _contig_seg_layout_kernel[(num_local_experts + 1, 2)](
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
        _contig_finalize_src2dst_kernel[
            (coarsened_pair_grid(num_pairs, pairs_per_program),)
        ](
            topk_ids.view(-1),
            src2dst_out,
            seg_offsets_out,
            num_pairs,
            PAIRS_PER_PROGRAM=pairs_per_program,
        )
        _contig_fill_rows_kernel[(coarsened_pair_grid(num_pairs, pairs_per_program),)](
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


def act_delta_contiguous(
    gateup_output: torch.Tensor,  # [m_pad_ceiling, slices * inter] bf16
    gate_up_delta: torch.Tensor | None,  # [num_tokens, top_k, slices * inter]
    act_out: torch.Tensor,  # [m_pad_ceiling, inter] bf16
    activation_lora_input: torch.Tensor,  # [num_tokens, top_k, inter] bf16
    src2dst: torch.Tensor,  # [num_tokens * top_k] int32 COMPACT rows
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    num_local_experts: int,
    gate_first: bool = True,
    interleaved: bool = False,
    activation: str = "silu",
    consume_base_pdl: bool = False,
) -> None:
    """Run the masked activation kernel over the compact rows.

    The launch, the grid and the per-pair arithmetic match
    :func:`activation_delta.act_delta_masked`. Only the physical row behind
    each ``src2dst`` entry differs. The kernel writes a zero into
    ``activation_lora_input`` once for each invalid pair.

    ``num_local_experts`` is a parameter here because the compact buffer is 2-D
    and has no expert dimension to read it from.
    """
    ActivationFn.parse(activation)  # reject an unknown activation name
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    if gateup_output.ndim != 2:
        raise ValueError(
            f"base gate/up must be compact 2-D, got {tuple(gateup_output.shape)}"
        )
    # The slice count comes from the weight shape, not from the activation.
    num_slices = gateup_output.shape[-1] // inter
    if num_slices not in (1, 2) or num_slices * inter != gateup_output.shape[-1]:
        raise ValueError(
            f"gate/up width {gateup_output.shape[-1]} is not 1x or 2x "
            f"intermediate {inter}"
        )
    if act_out.ndim != 2 or act_out.shape[0] != gateup_output.shape[0]:
        raise ValueError("gate/up and activation compact buffers must share rows")
    if gate_up_delta is not None and gate_up_delta.shape != (
        *topk_ids.shape,
        num_slices * inter,
    ):
        raise ValueError(
            f"gate_up_delta must be {(*topk_ids.shape, num_slices * inter)}"
        )
    if activation_lora_input.shape != (*topk_ids.shape, inter):
        raise ValueError(f"activation_lora_input must be {(*topk_ids.shape, inter)}")
    if num_pairs == 0:
        return
    _activation_delta_masked_kernel[(num_pairs,)](
        gateup_output,
        gate_up_delta if gate_up_delta is not None else gateup_output,
        act_out,
        activation_lora_input,
        src2dst,
        topk_ids,
        topk_ids.shape[1],
        num_local_experts,
        inter,
        HAS_DELTA=gate_up_delta is not None,
        NUM_SLICES=num_slices,
        ACTIVATION_TYPE=activation,
        GATE_FIRST=gate_first,
        INTERLEAVED=interleaved,
        CONSUME_BASE_PDL=consume_base_pdl,
        BLOCK_SIZE=512,
        **({"launch_pdl": True} if consume_base_pdl else {}),
    )


def fused_b_act_contiguous(
    family: str,
    *,
    activation: str,
    base_gateup: torch.Tensor,  # [m_pad_ceiling, slices * inter] bf16
    act_compact: torch.Tensor,  # [m_pad_ceiling, inter] bf16
    act_pairs: torch.Tensor | None,  # [num_tokens, top_k, inter] or None
    src2dst: torch.Tensor,  # [num_tokens * top_k] int32 COMPACT rows
    routing: RouteView,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    config: Mapping[str, int],
    bridge_gateup: torch.Tensor | None = None,
    b_gate_up: torch.Tensor | None = None,
    bridge_top_k: int = 1,
    consume_base_pdl: bool = False,
) -> None:
    """Run the fused LoRA-B GEMM and the activation over the compact rows.

    The kernel, the grid and the per-pair arithmetic match
    :func:`fused_act.run_masked_fused_act`. Only the shape check
    differs, because the compact domain is one flat 2-D buffer. The kernel
    writes a zero into ``act_pairs`` once for each invalid pair.

    The kernel skips the segment padding rows, so they keep stale values. The
    down GEMM still reads them, but it treats every row separately. No consumer
    reads the output of a padding row.
    """
    if family not in MASKED_ACT_FAMILIES:
        raise ValueError(f"family={family!r} is not one of {MASKED_ACT_FAMILIES}")
    ActivationFn.parse(activation)
    pairs = routing.topk_ids.numel()
    if src2dst.dtype != torch.int32 or src2dst.numel() != pairs:
        raise ValueError(f"src2dst must be int32 with {pairs} entries")
    if act_compact.ndim != 2 or act_compact.shape[1] < 1:
        raise ValueError("act_compact must be compact [m_pad_ceiling, intermediate]")
    width = act_compact.shape[1]
    # The slice count comes from the weight shape, not from the activation.
    slices = base_gateup.shape[-1] // width
    if slices not in (1, 2) or slices * width != base_gateup.shape[-1]:
        raise ValueError(
            f"base gate/up width {base_gateup.shape[-1]} is not 1x or 2x "
            f"activation width {width}"
        )
    if base_gateup.shape != (act_compact.shape[0], slices * width):
        raise ValueError(
            "base_gateup must share the compact rows and carry "
            f"{slices * width} columns, got {tuple(base_gateup.shape)}"
        )
    if act_pairs is not None:
        if act_pairs.shape != (*routing.topk_ids.shape, width):
            raise ValueError(
                f"act_pairs must be {(*routing.topk_ids.shape, width)}, got "
                f"{tuple(act_pairs.shape)}"
            )
        if act_compact.dtype != act_pairs.dtype:
            raise TypeError(
                f"dual activation stores need one dtype, got "
                f"{act_compact.dtype} and {act_pairs.dtype}"
            )
    if base_gateup.dtype != torch.bfloat16 or act_compact.dtype != torch.bfloat16:
        raise TypeError("contiguous BF16 middle requires BF16 base and activation rows")
    tensors = (
        base_gateup,
        act_compact,
        src2dst,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
    )
    if act_pairs is not None:
        tensors += (act_pairs,)
    devices = {item.device for item in tensors}
    if len(devices) != 1:
        raise ValueError(f"contiguous-middle tensors span devices {devices}")
    if bridge_gateup is None or b_gate_up is None:
        raise ValueError(f"family {family!r} requires bridge_gateup and b_gate_up")
    gate_rank = _validate_b_inputs(
        bridge_gateup,
        b_gate_up,
        routing,
        pairs=pairs,
        slices=slices,
        width=width,
        bridge_top_k=bridge_top_k,
    )

    if pairs == 0:
        return
    block_w, block_k, group_m, num_warps, num_stages = _require_config(config)
    if routing.block_size < 16 or not _is_power_of_two(routing.block_size):
        raise ValueError(
            "aligned fused-act route block size must be a power of two >= 16"
        )
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    pair_target = act_pairs.view(-1, width) if act_pairs is not None else act_compact
    num_w_tiles = triton.cdiv(width, block_w)
    _b_act_kernel[(num_m_blocks * num_w_tiles,)](
        bridge_gateup,
        b_gate_up,
        base_gateup,
        act_compact,
        pair_target,
        src2dst,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        pairs,
        bridge_gateup.stride(0),
        bridge_gateup.stride(1),
        b_gate_up.stride(0),
        b_gate_up.stride(1),
        b_gate_up.stride(2),
        base_gateup.stride(0),
        base_gateup.stride(1),
        act_compact.stride(0),
        act_compact.stride(1),
        pair_target.stride(0),
        pair_target.stride(1),
        num_local_experts=num_local_experts,
        top_k=routing.topk_ids.shape[1],
        width=width,
        rank=gate_rank,
        num_slices=slices,
        activation_type=activation,
        gate_first=gate_first,
        interleaved=interleaved,
        bridge_token_major=bridge_top_k != 1,
        num_m_blocks=num_m_blocks,
        block_m=routing.block_size,
        block_w=block_w,
        block_k=block_k,
        group_m=group_m,
        store_pair_act=act_pairs is not None,
        consume_base_pdl=consume_base_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **({"launch_pdl": True} if consume_base_pdl else {}),
    )


class ContiguousRowState(msgspec.Struct, kw_only=True):
    hidden_compact: torch.Tensor  # [m_pad_ceiling, hidden] bf16
    seg_counts: torch.Tensor  # [E_local] int32
    seg_offsets: torch.Tensor  # [E_local + 1] int32 first row of each segment
    src2dst: torch.Tensor  # [num_tokens * top_k] int32 compact rows
    grouped_layout: torch.Tensor  # [m_pad_ceiling] int32; -1 = skipped rows
    m_pad_ceiling: int
    retained_inputs: bool


class ContiguousRowDomainProvider(MoeBaseProvider):
    """The prepare, activation and finalize stages over the contiguous domain.

    A subclass adds the two GEMM stages, gate/up and down. It also passes
    ``m_alignment``, the m-block alignment that its GEMM engine needs. DeepGEMM
    reports that value from ``get_m_alignment_for_contiguous_layout()``.
    """

    def __init__(self, quant_info: MoeLoraBf16QuantInfo, *, m_alignment: int):
        self.quant_info = quant_info
        if not isinstance(m_alignment, int) or m_alignment < 1:
            raise ValueError(f"m_alignment must be a positive int, got {m_alignment!r}")
        self._m_alignment = m_alignment
        if quant_info.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        expected_w2 = (
            quant_info.num_local_experts,
            quant_info.hidden_size,
            quant_info.intermediate_size,
        )
        if quant_info.w2_weight.shape != expected_w2:
            raise ValueError(
                f"w2_weight must be {expected_w2}, got "
                f"{tuple(quant_info.w2_weight.shape)}"
            )
        if (
            quant_info.w13_weight.ndim != 3
            or quant_info.w13_weight.shape[0] != quant_info.num_local_experts
            or quant_info.w13_weight.shape[2] != quant_info.hidden_size
        ):
            raise ValueError(
                "w13_weight must be [num_local_experts, slices*intermediate, hidden]"
            )
        gateup_width = quant_info.w13_weight.shape[1]
        if gateup_width % quant_info.intermediate_size:
            raise ValueError(
                "w13 output width must be an integer multiple of intermediate_size"
            )
        self._gate_up_slices = gateup_width // quant_info.intermediate_size
        if self._gate_up_slices not in (1, 2):
            raise ValueError(
                "contiguous BF16 provider supports one non-gated slice or two "
                f"gated gate/up slices, got {self._gate_up_slices}"
            )

    @property
    def m_alignment(self) -> int:
        return self._m_alignment

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
        *,
        schedule_pack: ContiguousSchedulePack | None = None,
    ) -> ContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        device = hidden_states.device
        if workspace is not None:
            # The row count depends on the alignment. The tag names the
            # alignment, so a provider with another alignment gets its own
            # buffers in this workspace.
            prefix = f"contig:a{self._m_alignment}"
            seg_counts = workspace.tensor(
                f"{prefix}:seg_counts",
                (num_experts,),
                dtype=torch.int32,
                device=device,
            )
            seg_offsets = workspace.tensor(
                f"{prefix}:seg_offsets",
                (num_experts + 1,),
                dtype=torch.int32,
                device=device,
            )
            src2dst = workspace.tensor(
                f"{prefix}:src2dst",
                (num_pairs,),
                dtype=torch.int32,
                device=device,
            )
            grouped_layout = workspace.tensor(
                f"{prefix}:grouped_layout",
                (m_pad_ceiling,),
                dtype=torch.int32,
                device=device,
            )
            # Zero the buffer on the first allocation, so a GEMM tile never
            # reads uninitialized memory. Stale rows from a later pass are safe.
            # Each row is independent, and nothing reads a padding row's output.
            hidden_compact = workspace.tensor(
                f"{prefix}:hidden_compact",
                (m_pad_ceiling, hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=device,
                zero_on_first_allocation=True,
            )
        else:
            seg_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
            seg_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
            src2dst = torch.empty(num_pairs, dtype=torch.int32, device=device)
            grouped_layout = torch.empty(
                m_pad_ceiling, dtype=torch.int32, device=device
            )
            hidden_compact = torch.zeros(
                (m_pad_ceiling, hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=device,
            )
        contiguous_dispatch_fill(
            hidden_states,
            topk_ids,
            num_experts,
            top_k,
            self._m_alignment,
            seg_counts_out=seg_counts,
            seg_offsets_out=seg_offsets,
            src2dst_out=src2dst,
            grouped_layout_out=grouped_layout,
            hidden_compact_out=hidden_compact,
            schedule_pack=schedule_pack,
        )
        return ContiguousRowState(
            hidden_compact=hidden_compact,
            seg_counts=seg_counts,
            seg_offsets=seg_offsets,
            src2dst=src2dst,
            grouped_layout=grouped_layout,
            m_pad_ceiling=m_pad_ceiling,
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, row_state: ContiguousRowState) -> None:
        # Nothing reads the compact hidden rows after the gate/up GEMM. A
        # workspace tensor must keep its address for graph replay, so the runner
        # frees it later instead.
        if row_state.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_compact)

    def act_with_delta(
        self,
        row_state: ContiguousRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        act_delta_contiguous(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.src2dst,
            topk_ids,
            self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def fused_act(
        self,
        row_state: ContiguousRowState,
        family: str,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_masked: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
        consume_base_pdl: bool = False,
    ) -> None:
        # ``act_masked`` is the parameter name in the provider interface. Here
        # it holds the compact 2-D activation buffer.
        fused_b_act_contiguous(
            family,
            activation=activation,
            base_gateup=base_gateup,
            act_compact=act_masked,
            act_pairs=act_pairs,
            src2dst=row_state.src2dst,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            consume_base_pdl=consume_base_pdl,
        )

    def gateup_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (
            row_state.m_pad_ceiling,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (row_state.m_pad_ceiling, self.quant_info.intermediate_size)

    def down_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (row_state.m_pad_ceiling, self.quant_info.hidden_size)
