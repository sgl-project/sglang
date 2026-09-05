"""Build both GEMM schedules from the dispatcher's row counts in one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Shared with scheduler.py: one signed int64 per tile, leaving bit 63 clear.

EXPERT_SHIFT = 0
EXPERT_BITS = 20
EXPERT_MASK = (1 << EXPERT_BITS) - 1

TOKEN_CLUSTER_SHIFT = EXPERT_SHIFT + EXPERT_BITS
TOKEN_CLUSTER_BITS = 22
TOKEN_CLUSTER_MASK = (1 << TOKEN_CLUSTER_BITS) - 1

OUTPUT_CLUSTER_SHIFT = TOKEN_CLUSTER_SHIFT + TOKEN_CLUSTER_BITS
OUTPUT_CLUSTER_BITS = 21
OUTPUT_CLUSTER_MASK = (1 << OUTPUT_CLUSTER_BITS) - 1

PACKED_BITS = OUTPUT_CLUSTER_SHIFT + OUTPUT_CLUSTER_BITS
assert PACKED_BITS == 63, PACKED_BITS

MAX_EXPERTS = 1 << EXPERT_BITS
MAX_TOKEN_CLUSTERS = 1 << TOKEN_CLUSTER_BITS
MAX_OUTPUT_CLUSTERS = 1 << OUTPUT_CLUSTER_BITS


@triton.jit
def _dual_schedule_kernel(
    masked_m_ptr,
    schedule1_ptr,
    tiles1_ptr,
    schedule2_ptr,
    tiles2_ptr,
    token_width,
    out_clusters1,
    out_clusters2,
    EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
    ENTRY_BLOCK: tl.constexpr,
    TOKEN_SHIFT: tl.constexpr,
    OUTPUT_SHIFT: tl.constexpr,
):
    expert = tl.program_id(0)
    stage = tl.program_id(1)
    out_clusters = tl.where(stage == 0, out_clusters1, out_clusters2)

    offs = tl.arange(0, BLOCK_EXPERTS)
    all_rows = tl.load(masked_m_ptr + offs, mask=offs < EXPERTS, other=0)
    all_entries = tl.cdiv(all_rows, token_width) * out_clusters
    begin = tl.sum(tl.where(offs < expert, all_entries, 0))
    if expert == 0:
        total = tl.sum(tl.where(offs < EXPERTS, all_entries, 0))
        if stage == 0:
            tl.store(tiles1_ptr, total)
        else:
            tl.store(tiles2_ptr, total)

    rows = tl.load(masked_m_ptr + expert)
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
        # Widen before shifting to preserve the upper fields.
        packed = (
            expert.to(tl.int64)
            | (tc_i.to(tl.int64) << TOKEN_SHIFT)
            | (oc_i.to(tl.int64) << OUTPUT_SHIFT)
        )
        if stage == 0:
            tl.store(schedule1_ptr + begin + local, packed, mask=valid)
        else:
            tl.store(schedule2_ptr + begin + local, packed, mask=valid)


def dual_stage_schedule_capacities_masked(
    *,
    num_experts: int,
    m_max: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[int, int]:
    """CTA-tile entries require 1x1 clusters and 1-CTA MMA."""
    if tuple(cluster_shape_mn) != (1, 1) or use_2cta_instrs:
        raise ValueError(
            "the packed direct schedule is emitted at CTA-tile granularity and "
            "the device expands it by cluster_shape_mn, so only a (1, 1) "
            "cluster with 1-CTA MMA is representable; got "
            f"cluster_shape_mn={tuple(cluster_shape_mn)}, "
            f"use_2cta_instrs={use_2cta_instrs}"
        )
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width
    max_token_clusters = (m_max + token_width - 1) // token_width
    if num_experts > MAX_EXPERTS:
        raise ValueError(
            f"direct schedule packs expert INDICES in "
            f"{MAX_EXPERTS.bit_length() - 1} bits (counts up to "
            f"{MAX_EXPERTS}); got {num_experts}"
        )
    if max_token_clusters > MAX_TOKEN_CLUSTERS:
        raise ValueError(
            f"m_max={m_max} at token width {token_width} needs "
            f"{max_token_clusters} token clusters; the packing holds "
            f"{MAX_TOKEN_CLUSTERS}. Admission must pick a wider tile."
        )
    if max(out_clusters1, out_clusters2) > MAX_OUTPUT_CLUSTERS:
        raise ValueError(
            f"{max(out_clusters1, out_clusters2)} output clusters exceed the "
            f"{MAX_OUTPUT_CLUSTERS} the non-negative-word config allows"
        )
    # Individually valid fields can still overflow the int32 prefix sum.
    capacity = num_experts * max_token_clusters * max(out_clusters1, out_clusters2)
    if capacity > 2**31 - 1:
        raise ValueError(
            f"worst-case schedule capacity {capacity} overflows the builder's "
            "int32 prefix arithmetic"
        )
    return (
        num_experts * max_token_clusters * out_clusters1,
        num_experts * max_token_clusters * out_clusters2,
    )


def validate_tile_geometry_contiguous(token_width: int, m_alignment: int) -> None:
    """Require exact segment-base division and keep partial tiles in bounds."""
    if not isinstance(token_width, int) or token_width < 1:
        raise ValueError(f"token_width must be a positive int, got {token_width!r}")
    if not isinstance(m_alignment, int) or m_alignment < 1:
        raise ValueError(f"m_alignment must be a positive int, got {m_alignment!r}")
    if m_alignment % token_width:
        raise ValueError(
            f"token tile {token_width} does not divide the segment alignment "
            f"{m_alignment}: a partial tile could cross into the next "
            "expert's rows"
        )


def dual_stage_schedule_capacities_contiguous(
    *,
    num_experts: int,
    m_pad_ceiling: int,
    max_expert_rows: int,
    m_alignment: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[int, int]:
    """Bound capacity by total padded rows instead of per-expert slabs."""
    dual_stage_schedule_capacities_masked(
        num_experts=num_experts,
        m_max=max_expert_rows,
        token_width=token_width,
        n_gemm1=n_gemm1,
        n_gemm2=n_gemm2,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    validate_tile_geometry_contiguous(token_width, m_alignment)
    if m_pad_ceiling < 0 or m_pad_ceiling % m_alignment:
        raise ValueError(
            f"m_pad_ceiling={m_pad_ceiling} must be a non-negative multiple "
            f"of the segment alignment {m_alignment}"
        )
    total_clusters = m_pad_ceiling // token_width
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width
    capacity = total_clusters * max(out_clusters1, out_clusters2)
    if capacity > 2**31 - 1:
        raise ValueError(
            f"worst-case contiguous schedule capacity {capacity} overflows "
            "the builder's int32 prefix arithmetic"
        )
    return (
        max(total_clusters * out_clusters1, 1),
        max(total_clusters * out_clusters2, 1),
    )


def build_dual_stage_schedules_contiguous(
    seg_counts: torch.Tensor,
    *,
    m_pad_ceiling: int,
    max_expert_rows: int,
    m_alignment: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
    schedule1_out: torch.Tensor | None = None,
    tiles1_out: torch.Tensor | None = None,
    schedule2_out: torch.Tensor | None = None,
    tiles2_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = seg_counts.numel()
    capacity1, capacity2 = dual_stage_schedule_capacities_contiguous(
        num_experts=num_experts,
        m_pad_ceiling=m_pad_ceiling,
        max_expert_rows=max_expert_rows,
        m_alignment=m_alignment,
        token_width=token_width,
        n_gemm1=n_gemm1,
        n_gemm2=n_gemm2,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width

    device = seg_counts.device
    schedule1 = _schedule_output(
        schedule1_out,
        elements=capacity1,
        device=device,
    )
    schedule2 = _schedule_output(
        schedule2_out,
        elements=capacity2,
        device=device,
    )
    tiles1 = _tile_count_output(
        tiles1_out,
        device=device,
    )
    tiles2 = _tile_count_output(
        tiles2_out,
        device=device,
    )
    _dual_schedule_kernel[(num_experts, 2)](
        seg_counts,
        schedule1,
        tiles1,
        schedule2,
        tiles2,
        token_width,
        out_clusters1,
        out_clusters2,
        EXPERTS=num_experts,
        BLOCK_EXPERTS=max(triton.next_power_of_2(num_experts), 2),
        ENTRY_BLOCK=128,
        TOKEN_SHIFT=TOKEN_CLUSTER_SHIFT,
        OUTPUT_SHIFT=OUTPUT_CLUSTER_SHIFT,
    )
    return schedule1, tiles1, schedule2, tiles2


def _schedule_output(
    output: torch.Tensor | None, *, elements: int, device: torch.device
) -> torch.Tensor:
    if output is None:
        return torch.empty(elements, dtype=torch.int64, device=device)
    return output


def _tile_count_output(
    output: torch.Tensor | None, *, device: torch.device
) -> torch.Tensor:
    # int32: the device compares it to an Int32 work index.
    if output is None:
        return torch.empty(1, dtype=torch.int32, device=device)
    return output


def build_dual_stage_schedules_masked(
    masked_m: torch.Tensor,
    *,
    m_max: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
    schedule1_out: torch.Tensor | None = None,
    tiles1_out: torch.Tensor | None = None,
    schedule2_out: torch.Tensor | None = None,
    tiles2_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = masked_m.numel()
    capacity1, capacity2 = dual_stage_schedule_capacities_masked(
        num_experts=num_experts,
        m_max=m_max,
        token_width=token_width,
        n_gemm1=n_gemm1,
        n_gemm2=n_gemm2,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width

    device = masked_m.device
    schedule1 = _schedule_output(
        schedule1_out,
        elements=capacity1,
        device=device,
    )
    schedule2 = _schedule_output(
        schedule2_out,
        elements=capacity2,
        device=device,
    )
    tiles1 = _tile_count_output(
        tiles1_out,
        device=device,
    )
    tiles2 = _tile_count_output(
        tiles2_out,
        device=device,
    )
    _dual_schedule_kernel[(num_experts, 2)](
        masked_m,
        schedule1,
        tiles1,
        schedule2,
        tiles2,
        token_width,
        out_clusters1,
        out_clusters2,
        EXPERTS=num_experts,
        BLOCK_EXPERTS=max(triton.next_power_of_2(num_experts), 2),
        ENTRY_BLOCK=128,
        TOKEN_SHIFT=TOKEN_CLUSTER_SHIFT,
        OUTPUT_SHIFT=OUTPUT_CLUSTER_SHIFT,
    )
    return schedule1, tiles1, schedule2, tiles2
