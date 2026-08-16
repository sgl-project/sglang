"""The route-major (DeepGEMM "contiguous") row domain, GEMM-engine-agnostic.

This is the row-domain twin of :mod:`masked_row_domain`.  Where the masked
domain materializes expert-major ``[E_local, m_max, ·]`` slabs, the contiguous
domain sorts routed rows by expert into ONE compact 2-D buffer whose per-expert
segment starts are rounded up to the DeepGEMM contiguous m-alignment, exactly
the layout ``m_grouped_bf16_gemm_nt_contiguous`` consumes (mirroring the
upstream EP prefill convention of ``ep_scatter`` + ``grouped_gemm_nt_*_contig``
in ``sglang.srt.layers.moe.moe_runner.deep_gemm``).

Memory math per layer forward (P = T * top_k, a = m-alignment):

    masked      gather + gateup + act + down =
                E_local * m_max * (H + S*I + I + H)   rows scale O(E * chunk)
    contiguous  M_ceil  * (H + S*I + I + H)           rows scale O(T * top_k)
    M_ceil      = ceil((P + E_local * (a - 1)) / a) * a   (host-static ceiling
                  over the dynamic M_pad = sum_e ceil(count_e / a) * a)

``grouped_layout`` labels every row of an expert's ALIGNED segment (routed
rows and the partial-block padding tail alike) with the owning expert id and
the ceiling tail past ``M_pad`` with ``-1``.  Labeling in-segment padding with
the expert id keeps every m-block group-uniform under both the per-block and
per-row readings of the DeepGEMM contract — the exact convention upstream
``ep_scatter`` writes — while ``-1`` ceiling blocks are skipped work.  Padding
row CONTENT is never consumed: the GEMMs are row-independent, the pair-domain
activation join and finalize index only routed rows through ``src2dst``, and
``ensure_zero_padding`` (the wrapper default) zeroes ``-1`` output rows.

Stage mapping versus the masked twin:

* S1 dispatch: three launches (count/slot -> aligned prefix + segment labels
  -> row fill) instead of the masked single fused launch; that is the price
  of needing per-expert prefix sums before any row has a home.  The
  pair-domain launches are PAIR-COARSENED (``PAIRS_PER_PROGRAM`` consecutive
  pairs per program, vector prologue) — GB300 profiling showed these glue
  launches are program-count-bound (see the constants below).  The
  seg-layout launch owns the whole ``grouped_layout`` buffer: its extra tail
  program writes the ``-1`` ceiling rows, so no host-side ``fill_(-1)``
  memset precedes it, and a CuTeDSL caller can fold its dual-stage packed
  tile-schedule build into the same launch through a
  :class:`ContiguousSchedulePack` (both fusions cut one launch per layer
  prefill each).
* S3 activation join and S5 finalize REUSE the masked pair-domain kernels
  verbatim (``_activation_delta_masked_kernel``, ``post_reorder_deepgemm``):
  both index physical rows exclusively through ``src2dst``, so storing the
  compact row ``seg_offsets[e] + slot`` there instead of the masked
  ``e * m_max + slot`` re-targets them with zero kernel changes, including
  the exactly-once invalid-pair zero-write duty on the pair activation.
  For the same reason the LoRA A/B kernels need no contiguous variants: gate
  A is token-domain, down A consumes the canonical pair activation (or the
  ``src2dst`` row map through ``MappedLoraAInput``), and every B kernel is
  pair-domain.

The fused B+activation middle (:func:`fused_b_act_contiguous`) and the
mapped grouped down-A input reuse the masked pair-domain kernels through the
same lever: ``_b_act_kernel`` indexes base gate/up rows and activation rows
EXCLUSIVELY through ``src2dst`` over flattened views, so pointing it at the
compact 2-D buffers re-targets it with zero kernel changes, and the
pair-to-row ABI grouped down-A consumes is the same ``src2dst``.  That is
what admits the GB300 shared-outer SERIAL prefill winner (token-dedup shared
gate A + fused B+activation middle + one-launch shared down B + materialized
finalize): every LoRA factor kernel it uses is pair/token-domain, the JOINT
shared-outer route builder is pure pair-domain metadata whose ``route_pdl``
chain is internal to the three routing launches and independent of the row
domain, and the finalize is the same ``post_reorder``.

The shared-rank finalize family ports through the SAME lever: the rank
reduction is pure pair-domain (it reads the canonical pair-major down-A
bridge the mapped grouped down-A already emits — output rows are indexed by
pair id regardless of the input row map), and the from-scratch tail reads
base down rows exclusively through ``src2dst`` over a flat row view, so
pointing it at the compact 2-D down output re-targets it with zero kernel
changes (:mod:`masked_finalize` kernels, reused verbatim).

Eligibility is PREFILL-ONLY: a contiguous decode port measured slower than
masked on GB300 (CuTeDSL at 8-row alignment by 5-10%, DeepGEMM at its
128-row library alignment by 22-39%), so decode stays masked.  The
OVERLAPPED H200 winners also stay masked: the INDEXED down-A and the PDL
edges have no contiguous port, and overlap windows fail the serial
predicate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MappedLoraAInput,
    MoeBaseProvider,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_activation import (
    _activation_delta_masked_kernel,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_fused_middle import (
    MASKED_MIDDLE_ACTIVATIONS,
    MASKED_MIDDLE_FAMILIES,
    MASKED_MIDDLE_TRITON,
    _b_act_kernel,
    _is_power_of_two,
    _require_config,
    _validate_b_inputs,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo
from sglang.srt.lora.moe.routing import ROUTE_ALIGNED

if TYPE_CHECKING:
    from sglang.srt.lora.moe.routing import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


# Default pair-coarsening factor for the pair-domain S1 glue kernels.  These
# launches are program-count-bound on GB300 (~35us floor for 65,536 one-pair
# programs regardless of per-program work), so each program owns
# ``PAIRS_PER_PROGRAM`` consecutive pairs and the grid shrinks by that
# factor.  Must be a power of two (``tl.arange`` lanes); 8 cuts the
# T=8192/topk=8 program count from 65,536 to 8,192 while each program's
# vector prologue stays two loads (sweep: 16/32 regress).
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
    """``ceil(num_pairs / pairs_per_program)`` programs cover every pair.

    The ceiling keeps a partial tail program when ``num_pairs`` is not a
    multiple; its out-of-range lanes predicate off inside the kernel.  The
    count depends only on host-static values, so captured launches keep a
    static grid exactly like the one-pair-per-program grids they replace.
    """
    _validate_pairs_per_program(pairs_per_program)
    if num_pairs < 0:
        raise ValueError(f"num_pairs must be non-negative, got {num_pairs}")
    return -(-num_pairs // pairs_per_program)


def contiguous_m_pad_ceiling(num_pairs: int, num_experts: int, alignment: int) -> int:
    """Host-static row ceiling of the aligned per-expert segment layout.

    The dynamic total is ``M_pad = sum_e ceil(count_e / a) * a`` with
    ``sum_e count_e <= num_pairs``; each expert adds at most ``a - 1`` padding
    rows, so ``M_pad <= num_pairs + num_experts * (a - 1)``.  The ceiling is
    rounded up to a multiple of the alignment so the GEMM's whole row domain
    is m-block covered.  It depends only on host-static values, making every
    buffer shape and grid CUDA-graph capture-stable.
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
    topk_ids_ptr,  # [num_pairs]; < 0 = padding / EP-unrouted
    slot_out_ptr,  # [num_pairs] int32 out: dense per-expert slot (valid lanes)
    seg_counts_ptr,  # [E_local] int32 cursor + count, PRE-ZEROED by caller
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    # Pair-coarsened counting pass; the vectorized atomic is exact under
    # intra-vector expert collisions (each lane performs its own RMW), the
    # same property the masked metadata dispatch relies on.
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
    seg_offsets_ptr,  # [E_local + 1] int32 aligned exclusive prefix out
    grouped_layout_ptr,  # [m_pad_ceiling] int32 out; NO host prefill needed
    m_pad_ceiling,  # host-static row ceiling; [M_pad, ceiling) gets -1
    schedule1_ptr,  # BUILD_SCHEDULES only: [capacity1] int32 packed tiles
    tiles1_ptr,  # BUILD_SCHEDULES only: [1] int32 stage-1 tile count
    schedule2_ptr,  # BUILD_SCHEDULES only: [capacity2] int32 packed tiles
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
    # Grid (EXPERTS + 1,) plain, (EXPERTS + 1, 2) under BUILD_SCHEDULES.
    # Every program recomputes the (cheap, one tl.cumsum) aligned prefix from
    # seg_counts; each stage-0 program stores the shared seg_offsets entries
    # — a benign same-value race, exactly the upstream ep_scatter stage-1
    # pattern — then labels its FULL aligned segment (routed rows and the
    # partial-block padding tail) with its expert id so every m-block the
    # GEMM schedules is group-uniform.  Program EXPERTS owns the ceiling
    # tail: it writes -1 over [M_pad, m_pad_ceiling), replacing the
    # fill_(-1) memset this launch used to rely on; segment and tail writes
    # are disjoint by construction.
    #
    # BUILD_SCHEDULES additionally folds the CuTeDSL dual-stage packed
    # schedule build (the former standalone _dual_schedule_kernel launch)
    # into this grid: program (expert, stage) packs that expert's stage
    # entries from the SAME seg_counts, preserving the dual-ownership rule
    # (both stages' schedules derive from the one device row-count source,
    # in one launch).  The packing arithmetic is verbatim
    # _dual_schedule_kernel so the fused and standalone builds are
    # entry-identical.
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
                packed = expert | (tc_i << TOKEN_SHIFT) | (oc_i << OUTPUT_SHIFT)
                if stage == 0:
                    tl.store(schedule1_ptr + begin + local, packed, mask=valid)
                else:
                    tl.store(schedule2_ptr + begin + local, packed, mask=valid)


@triton.jit
def _contig_fill_rows_kernel(
    input_ptr,  # [num_tokens, hidden] bf16 source rows
    compact_ptr,  # [m_pad_ceiling, hidden] bf16, viewed flat
    topk_ids_ptr,  # [num_pairs]
    src2dst_ptr,  # [num_pairs] int32: dense slots IN, compact rows OUT
    seg_offsets_ptr,  # [E_local + 1] int32 aligned exclusive prefix
    hidden_size,
    num_pairs,
    TOPK: tl.constexpr,
    PAIRS_PER_PROGRAM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # PURE copy pass: src2dst must already hold FINALIZED compact rows
    # (written by _contig_finalize_src2dst_kernel). Fusing that rewrite
    # into this kernel made the compiled kernel emit compact-row stores
    # past the allocation at scale (T*topk=65536, hidden=2048) with
    # provably correct inputs, in every formulation tried (vector-extract
    # and per-pair scalar alike); the standalone split runs correctly, so
    # the two stages stay separate kernels by construction.
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
    topk_ids_ptr,  # [num_pairs]; < 0 = padding / EP-unrouted
    src2dst_ptr,  # [num_pairs] int32: dense slots IN, compact rows OUT
    seg_offsets_ptr,  # [E_local + 1] int32 aligned exclusive prefix
    num_pairs,
    PAIRS_PER_PROGRAM: tl.constexpr,
):
    # Rewrites src2dst in place from per-expert dense slots to compact rows
    # ``seg_offsets[expert] + slot``. Kept as its own launch — see the
    # hazard note on _contig_fill_rows_kernel.
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
    """CuTeDSL packed-schedule outputs folded into the S1 seg-layout launch.

    The contiguous CuTeDSL provider hands this to
    :func:`contiguous_dispatch_fill` so both stages' packed tile schedules
    are emitted by the SAME launch that computes ``seg_offsets`` from
    ``seg_counts`` — one launch fewer per layer prefill, and the
    dual-ownership rule (both schedules derive from the one device row-count
    source, in one launch) holds by construction.  Build instances through
    ``cutedsl_masked.schedule_builder.contiguous_dual_stage_schedule_pack``,
    which owns the packed-ABI capacity validation and the field-shift
    constants; this module stays engine-agnostic by taking the shifts as
    data instead of importing the CuTeDSL ABI.
    """

    schedule1: torch.Tensor  # [capacity1] int32 packed (expert, tc, oc)
    tiles1: torch.Tensor  # [1] int32 stage-1 tile count
    schedule2: torch.Tensor  # [capacity2] int32 packed (expert, tc, oc)
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
        if not isinstance(value, int) or not 0 < value < 32:
            raise ValueError(
                f"schedule_pack.{name} must be an int in (0, 32), got {value!r}"
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
    # With the token tile dividing the alignment, the fused build writes at
    # most (m_pad_ceiling / token_width) * out_clusters entries per stage —
    # the same bound the standalone builder's capacity validation proves.
    total_clusters = m_pad_ceiling // pack.token_width
    for name, tensor, needed in (
        ("schedule1", pack.schedule1, total_clusters * pack.out_clusters1),
        ("schedule2", pack.schedule2, total_clusters * pack.out_clusters2),
        ("tiles1", pack.tiles1, 1),
        ("tiles2", pack.tiles2, 1),
    ):
        if (
            tensor.ndim != 1
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tensor.device != device
        ):
            raise ValueError(
                f"schedule_pack.{name} must be contiguous 1-D int32 on {device}"
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
    """S1 of the contiguous domain: counts, aligned segments, and row gather.

    On return ``src2dst[t * top_k + k]`` holds the COMPACT row
    ``seg_offsets[e] + slot`` for every routed pair (sentinel pairs get no
    store anywhere, exactly like the masked dispatch), ``grouped_layout``
    labels aligned segments as documented on the module, and
    ``hidden_compact`` carries the gathered routed token rows.  With a
    ``schedule_pack``, the seg-layout launch additionally emits both CuTeDSL
    packed tile schedules from the same ``seg_counts``.  Grids and buffer
    shapes derive only from host-static values; segment lengths are read
    from device memory, so the whole sequence is CUDA-graph capturable.
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

    # Stream-ordered initialization (capture-legal): the counts double as the
    # atomic cursors.  grouped_layout needs no host prefill — the seg-layout
    # launch's tail program writes the -1 ceiling rows itself.
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
    # Runs even with zero pairs so seg_offsets, the grouped_layout labels,
    # and (with a pack) the packed tile counts are defined for every forward.
    if schedule_pack is None:
        # The schedule pointer/geometry arguments alias live tensors but are
        # never read or written: BUILD_SCHEDULES is constexpr-False, the same
        # dead-alias pattern as build_single_stage_schedule's stage-2 legs.
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


def silu_mul_delta_contiguous(
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
    """The masked S3 kernel re-targeted at compact rows through ``src2dst``.

    Same launch, same grid, same per-pair arithmetic and the same
    exactly-once invalid-pair zero write on ``activation_lora_input`` as
    :func:`masked_activation.silu_mul_delta_masked`; only the physical row
    behind each ``src2dst`` entry differs.  The only wrapper-level change is
    that ``num_local_experts`` is explicit — the masked wrapper reads it from
    the slab's leading dimension, which the compact 2-D buffer no longer
    carries.
    """
    if activation not in ("silu", "relu2"):
        raise ValueError(f"activation={activation!r} is not 'silu' or 'relu2'")
    num_pairs = topk_ids.numel()
    inter = act_out.shape[-1]
    if gateup_output.ndim != 2:
        raise ValueError(
            f"base gate/up must be compact 2-D, got {tuple(gateup_output.shape)}"
        )
    # Gating is a resident-shape property, independent of the activation.
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
            f"gate_up_delta must be canonical {(*topk_ids.shape, num_slices * inter)}"
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
    """The masked fused B+activation middle re-targeted at compact rows.

    Same kernel (``_b_act_kernel``), same grid, same per-pair arithmetic, and
    the same exactly-once invalid-pair zero-write duty on ``act_pairs`` as
    :func:`masked_fused_middle.run_masked_fused_middle`: the kernel addresses
    base and activation rows exclusively through ``src2dst`` over flattened
    row views, so storing the compact row ``seg_offsets[e] + slot`` there
    re-targets it with zero kernel changes.  Only the wrapper-level shape
    contract differs — the compact domain is one flat 2-D buffer instead of
    the masked ``[E, m_max, ·]`` slab, so the expert count cannot be read off
    a leading dimension and both row buffers validate against the one shared
    compact row count.  Segment-padding activation rows are untouched stale
    content, exactly like the materialized contiguous S3: the down GEMM
    computes over them row-independently and no consumer reads their output.
    """
    if family not in MASKED_MIDDLE_FAMILIES:
        raise ValueError(f"family={family!r} is not one of {MASKED_MIDDLE_FAMILIES}")
    if routing.view != ROUTE_ALIGNED:
        raise ValueError(
            f"contiguous fused middle needs route view {ROUTE_ALIGNED!r}, got "
            f"{routing.view!r}"
        )
    if activation not in MASKED_MIDDLE_ACTIVATIONS:
        raise ValueError(
            f"activation={activation!r} is not one of {MASKED_MIDDLE_ACTIVATIONS}"
        )
    pairs = routing.topk_ids.numel()
    if src2dst.dtype != torch.int32 or src2dst.numel() != pairs:
        raise ValueError(f"src2dst must be int32 with {pairs} entries")
    if act_compact.ndim != 2 or act_compact.shape[1] < 1:
        raise ValueError("act_compact must be compact [m_pad_ceiling, intermediate]")
    width = act_compact.shape[1]
    # Gating is a resident-shape property, independent of the activation.
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
            "aligned fused-middle route block size must be a power of two >= 16"
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


class ContiguousRowWorkspace(msgspec.Struct, kw_only=True):
    """Per-forward state of the contiguous row domain.

    Rows live in one compact buffer with
    ``src2dst[t * top_k + k] = seg_offsets[expert] + slot``; validity is
    carried by ``topk_ids >= 0`` exactly as in the masked domain.
    ``m_pad_ceiling`` is the host-static row ceiling every stage buffer is
    shaped by; the dynamic aligned total lives in ``seg_offsets[-1]`` on
    device and never reaches the host.
    """

    hidden_compact: torch.Tensor  # [m_pad_ceiling, hidden] bf16
    seg_counts: torch.Tensor  # [E_local] int32
    seg_offsets: torch.Tensor  # [E_local + 1] int32 aligned exclusive prefix
    src2dst: torch.Tensor  # [num_tokens * top_k] int32 compact rows
    grouped_layout: torch.Tensor  # [m_pad_ceiling] int32; -1 = skipped rows
    m_pad_ceiling: int
    retained_inputs: bool


class ContiguousRowDomainProvider(MoeBaseProvider):
    """S1/S3/S5 plus geometry over the contiguous row domain; S2/S4 abstract.

    ``m_alignment`` is the GEMM engine's contiguous m-block alignment
    (DeepGEMM's ``get_m_alignment_for_contiguous_layout()``); the engine
    subclass supplies it so this domain stays engine-agnostic.
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
                "contiguous BF16 provider supports one ReLU2 slice or two "
                f"gate/up slices, got {self._gate_up_slices}"
            )

        # Bind callees once at attach time, mirroring the masked domain: the
        # finalize is the SAME post_reorder kernel (it indexes rows only
        # through src2dst, which now carries compact rows).
        from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
        from sglang.srt.lora.moe.base_gemm_provider.down_b_scatter import (
            invoke_down_b_scatter,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_finalize import (
            MASKED_FINALIZE_TRITON,
            invoke_shared_from_scratch_finalize,
            invoke_shared_rank_reduce,
        )

        self._post_reorder = post_reorder_deepgemm
        # The down-B scatter epilogue is row-domain-agnostic for the same
        # reason post_reorder is: every physical row access goes through
        # src2dst over a flat row view.
        self._down_b_scatter = invoke_down_b_scatter
        # The shared-rank finalize pair is reused verbatim: the reduce is
        # pure pair-domain and the tail reads base rows only through
        # src2dst over a flat row view (compact rows here).
        self._shared_reduce_impls: dict[str, Callable] = {
            MASKED_FINALIZE_TRITON: invoke_shared_rank_reduce
        }
        self._shared_tail_impls: dict[str, Callable] = {
            MASKED_FINALIZE_TRITON: invoke_shared_from_scratch_finalize
        }

    @property
    def num_local_experts(self) -> int:
        return self.quant_info.num_local_experts

    @property
    def intermediate_size(self) -> int:
        return self.quant_info.intermediate_size

    @property
    def hidden_size(self) -> int:
        return self.quant_info.hidden_size

    @property
    def gate_up_slices(self) -> int:
        return self._gate_up_slices

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
    ) -> ContiguousRowWorkspace:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        device = hidden_states.device
        if workspace is not None:
            # The alignment tag keeps this instance's row-geometry buffers
            # distinct from any differently aligned sibling sharing the same
            # layer workspace: compact rows and ceilings are
            # alignment-dependent.
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
            # First-touch zero so uninitialized memory never enters a GEMM
            # tile; stale rows on reuse are benign (rows are independent and
            # padding outputs are never consumed).
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
        return ContiguousRowWorkspace(
            hidden_compact=hidden_compact,
            seg_counts=seg_counts,
            seg_offsets=seg_offsets,
            src2dst=src2dst,
            grouped_layout=grouped_layout,
            m_pad_ceiling=m_pad_ceiling,
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, ws: ContiguousRowWorkspace) -> None:
        # Same lifetime contract as the masked twin: the compact hidden rows
        # are dead after the gate/up GEMM; runner-workspace tensors remain
        # address-stable for graph replay and are reclaimed with the runner
        # workspace.
        if ws.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(ws.hidden_compact)

    def act_with_delta(
        self,
        ws: ContiguousRowWorkspace,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        if activation not in ("silu", "relu2"):
            raise ValueError(f"activation={activation!r} is not 'silu' or 'relu2'")
        silu_mul_delta_contiguous(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            ws.src2dst,
            topk_ids,
            self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def mapped_down_lora_a_input(
        self,
        ws: ContiguousRowWorkspace,
        activation: torch.Tensor,
    ) -> MappedLoraAInput | None:
        """Expose compact activation rows for standalone grouped down-A.

        The contiguous twin of the masked provider's mapping and the same
        semantic ABI: ``src2dst`` already holds one physical activation row
        per canonical routed pair.  Sentinel pairs' (uninitialized) entries
        are never read — the aligned route buckets them into ``-1``-labeled
        blocks the mapped grouped-A kernel skips, identical to the masked
        domain, whose dispatch also leaves them unwritten.
        """
        if not isinstance(ws, ContiguousRowWorkspace):
            raise TypeError("contiguous down-A input requires ContiguousRowWorkspace")
        expected = self.act_out_shape(ws)
        if tuple(activation.shape) != expected:
            raise ValueError(
                f"mapped down-A activation must be {expected}, got "
                f"{tuple(activation.shape)}"
            )
        if activation.dtype != self.contract.lora_activation_dtype:
            raise TypeError(
                "mapped down-A activation dtype must match the provider "
                f"contract {self.contract.lora_activation_dtype}"
            )
        if not activation.is_contiguous():
            raise ValueError("mapped down-A activation rows must be contiguous")
        if (
            ws.src2dst.ndim != 1
            or ws.src2dst.dtype != torch.int32
            or ws.src2dst.device != activation.device
            or not ws.src2dst.is_contiguous()
        ):
            raise ValueError(
                "mapped down-A pair-to-row metadata must be contiguous 1-D "
                "int32 on the activation device"
            )
        return MappedLoraAInput(
            rows=activation.view(-1, activation.shape[-1]),
            pair_to_row=ws.src2dst,
        )

    def fused_middle_implementations(self, family: str) -> tuple[str, ...]:
        if family in MASKED_MIDDLE_FAMILIES:
            return (MASKED_MIDDLE_TRITON,)
        return ()

    def supports_fused_middle(
        self,
        family: str,
        *,
        activation: str,
        implementation: str = "triton",
    ) -> bool:
        return (
            family in MASKED_MIDDLE_FAMILIES
            and activation in MASKED_MIDDLE_ACTIVATIONS
            and implementation == MASKED_MIDDLE_TRITON
        )

    def run_fused_middle(
        self,
        ws: ContiguousRowWorkspace,
        family: str,
        *,
        implementation: str,
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
        # ``act_masked`` keeps the provider-ABI parameter name; on this
        # domain it is the compact 2-D activation buffer.
        if not self.supports_fused_middle(
            family, activation=activation, implementation=implementation
        ):
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} fused-middle "
                f"implementation for {family!r}/{activation!r}"
            )
        fused_b_act_contiguous(
            family,
            activation=activation,
            base_gateup=base_gateup,
            act_compact=act_masked,
            act_pairs=act_pairs,
            src2dst=ws.src2dst,
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

    def install_fused_finalize_implementation(
        self,
        family: str,
        ownership: str,
        name: str,
        implementation: Callable | tuple[Callable, Callable],
    ) -> None:
        if not name:
            raise ValueError("a fused-finalize implementation needs a name")
        if ownership not in ("per_expert", "shared"):
            raise ValueError(f"unknown fused-finalize ownership {ownership!r}")
        if family == "shared_rank_reduce":
            if ownership != "shared":
                raise ValueError("shared_rank_reduce requires shared ownership")
            if (
                not isinstance(implementation, tuple)
                or len(implementation) != 2
                or not all(callable(item) for item in implementation)
            ):
                raise ValueError(
                    "shared_rank_reduce implementation must be a "
                    "(reduce, tail) callable pair"
                )
            self._shared_reduce_impls[name], self._shared_tail_impls[name] = (
                implementation
            )
        else:
            raise ValueError(f"unknown fused-finalize family {family!r}")

    def fused_finalize_implementations(
        self, family: str, ownership: str
    ) -> tuple[str, ...]:
        if family == "shared_rank_reduce" and ownership == "shared":
            return tuple(
                name
                for name in self._shared_reduce_impls
                if name in self._shared_tail_impls
            )
        return ()

    def supports_down_b_scatter(self) -> bool:
        return True

    def run_down_b_scatter(
        self,
        ws: ContiguousRowWorkspace,
        *,
        down_out: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        config: Mapping[str, int],
    ) -> None:
        # Same lever as the contiguous S3/S5 reuse: rows are addressed only
        # through src2dst, which here carries compact rows
        # seg_offsets[e] + slot.  src2dst is only READ, so the documented
        # in-place-src2dst-store hazard does not apply.
        self._down_b_scatter(
            down_rows=down_out.view(-1, self.hidden_size),
            src2dst=ws.src2dst,
            bridge=bridge,
            b_down=b_down,
            routing=routing,
            config=config,
        )

    def run_shared_rank_finalize(
        self,
        ws: ContiguousRowWorkspace,
        *,
        implementation: str,
        down_masked: torch.Tensor,
        bridge: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        config: Mapping[str, Mapping[str, int]],
    ) -> None:
        if set(config) != {"reduce", "tail"}:
            raise ValueError("shared-rank config must contain exactly reduce and tail")
        self.run_shared_rank_reduce(
            ws,
            implementation=implementation,
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            token_rank=token_rank,
            config=config["reduce"],
        )
        self.finish_shared_rank_finalize(
            ws,
            implementation=implementation,
            down_masked=down_masked,
            b_down=b_down,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            token_rank=token_rank,
            config=config["tail"],
        )

    def run_shared_rank_reduce(
        self,
        ws: ContiguousRowWorkspace,
        *,
        implementation: str,
        bridge: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        try:
            invoke = self._shared_reduce_impls[implementation]
        except KeyError as exc:
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} shared-rank reduction"
            ) from exc
        # `ws` is deliberately opaque and unused by this pair-domain launch;
        # retaining it in the provider ABI lets every scheduled stage be
        # invoked uniformly.
        del ws
        invoke(
            bridge=bridge,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            token_rank=token_rank,
            config=config,
        )

    def finish_shared_rank_finalize(
        self,
        ws: ContiguousRowWorkspace,
        *,
        implementation: str,
        down_masked: torch.Tensor,
        b_down: torch.Tensor,
        routing: RouteView,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        token_rank: torch.Tensor,
        config: Mapping[str, int],
    ) -> None:
        # ``down_masked`` keeps the provider-ABI parameter name; on this
        # domain it is the compact 2-D [m_pad_ceiling, hidden] down output,
        # and the verbatim tail kernel addresses its rows only through
        # src2dst (compact rows seg_offsets[e] + slot).
        try:
            invoke = self._shared_tail_impls[implementation]
        except KeyError as exc:
            raise NotImplementedError(
                f"{self.contract.key} has no {implementation!r} shared B tail"
            ) from exc
        invoke(
            down_masked=down_masked,
            src2dst=ws.src2dst,
            token_rank=token_rank,
            b_down=b_down,
            routing=routing,
            topk_weights=topk_weights,
            routed_scaling_factor=routed_scaling_factor,
            output=output,
            num_local_experts=self.num_local_experts,
            config=config,
        )

    def finalize(
        self,
        ws: ContiguousRowWorkspace,
        down_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        *,
        lora_delta: torch.Tensor | None = None,
    ) -> None:
        # Verbatim masked finalize: post_reorder_deepgemm gathers each
        # token's top-k rows through src2dst (now compact rows), applies the
        # router weight and routed scaling exactly once over base + pair
        # delta, and writes per-token in-register FP32 K-sums — identical
        # arithmetic order to the masked domain.
        num_tokens, hidden = output.shape
        self._post_reorder(
            down_out.view(-1, hidden),
            output,
            ws.src2dst,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            num_tokens,
            hidden,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            lora_delta=lora_delta,
        )

    def gateup_out_shape(self, ws: ContiguousRowWorkspace) -> tuple[int, ...]:
        return (
            ws.m_pad_ceiling,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, ws: ContiguousRowWorkspace) -> tuple[int, ...]:
        return (ws.m_pad_ceiling, self.quant_info.intermediate_size)

    def down_out_shape(self, ws: ContiguousRowWorkspace) -> tuple[int, ...]:
        return (ws.m_pad_ceiling, self.quant_info.hidden_size)
