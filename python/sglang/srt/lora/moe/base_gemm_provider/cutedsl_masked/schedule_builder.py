"""One launch builds the schedules for both GEMM stages from one ``masked_m``
tensor.

The two GEMMs read that same ``masked_m``. A schedule built from any other row
count corrupts the output, and nothing reports the error. Each function here
therefore takes the row counts through one argument only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousSchedulePack,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_abi import (
    MAX_EXPERTS,
    MAX_OUTPUT_CLUSTERS,
    MAX_TOKEN_CLUSTERS,
    OUTPUT_CLUSTER_SHIFT,
    TOKEN_CLUSTER_SHIFT,
)


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
    SINGLE_STAGE: tl.constexpr,
):
    """A benchmark rejected one program with a serial loop over all experts.
    At prefill sizes that loop takes more time than the GEMM saves.
    """
    expert = tl.program_id(0)
    stage = tl.program_id(1)
    if SINGLE_STAGE:
        out_clusters = out_clusters1
    else:
        out_clusters = tl.where(stage == 0, out_clusters1, out_clusters2)

    offs = tl.arange(0, BLOCK_EXPERTS)
    all_rows = tl.load(masked_m_ptr + offs, mask=offs < EXPERTS, other=0)
    all_entries = tl.cdiv(all_rows, token_width) * out_clusters
    begin = tl.sum(tl.where(offs < expert, all_entries, 0))
    if expert == 0:
        total = tl.sum(tl.where(offs < EXPERTS, all_entries, 0))
        if SINGLE_STAGE:
            tl.store(tiles1_ptr, total)
        else:
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
        # Widen to int64 before the shift. An int32 shift drops the upper
        # fields, and nothing reports the loss.
        packed = (
            expert.to(tl.int64)
            | (tc_i.to(tl.int64) << TOKEN_SHIFT)
            | (oc_i.to(tl.int64) << OUTPUT_SHIFT)
        )
        if SINGLE_STAGE:
            tl.store(schedule1_ptr + begin + local, packed, mask=valid)
        else:
            if stage == 0:
                tl.store(schedule1_ptr + begin + local, packed, mask=valid)
            else:
                tl.store(schedule2_ptr + begin + local, packed, mask=valid)


def dual_stage_schedule_capacities(
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
    """The builder writes one index for each CTA tile. The device scheduler
    reads each index as a cluster index and multiplies it by the cluster shape.
    Only a (1, 1) cluster is therefore correct, and this function rejects every
    other value. The device has no assertion for it.
    """
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
    # The kernel adds the entry counts in int32. The field width limits above
    # do not bound that sum, so check the worst-case capacity here.
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


def single_stage_schedule_capacity(
    *,
    num_experts: int,
    m_max: int,
    token_width: int,
    n_gemm: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> int:
    capacity, _ = dual_stage_schedule_capacities(
        num_experts=num_experts,
        m_max=m_max,
        token_width=token_width,
        n_gemm1=n_gemm,
        n_gemm2=n_gemm,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    return capacity


def validate_contiguous_tile_geometry(token_width: int, m_alignment: int) -> None:
    """The contiguous kernel adds ``seg_offsets[e] // token_width`` to the flat
    tile index. The last tile of an expert can also write past the valid rows.
    Both are safe only if the token tile divides the segment alignment. Each
    segment base is then an exact multiple of the tile. Each overrun then stays
    inside the same expert's segment.
    """
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


def contiguous_dual_stage_schedule_capacities(
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
    """Only the capacity differs from the masked builder. The experts share one
    flat aligned buffer, and the token tile divides the alignment. Each stage
    therefore needs ``(m_pad_ceiling / token_width) * out_clusters`` entries.
    That count grows with the total rows. The masked builder instead needs the
    worst case for one expert times the expert count.
    """
    dual_stage_schedule_capacities(
        num_experts=num_experts,
        m_max=max_expert_rows,
        token_width=token_width,
        n_gemm1=n_gemm1,
        n_gemm2=n_gemm2,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    validate_contiguous_tile_geometry(token_width, m_alignment)
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
    """The entries match the masked builder's entries. Each entry holds the
    expert, the local token cluster, and the output cluster. The device kernel
    adds the aligned segment base itself. ``seg_counts`` takes the place of
    ``masked_m``, and the rule above applies to it.

    The serving path does not call this function. It packs the same entries
    inside its segment-layout launch, and it saves one launch for each forward
    pass. See :func:`contiguous_dual_stage_schedule_pack`. A test compares the
    entries of the two paths.
    """
    num_experts = seg_counts.numel()
    capacity1, capacity2 = contiguous_dual_stage_schedule_capacities(
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
        label="schedule1_out",
    )
    schedule2 = _schedule_output(
        schedule2_out,
        elements=capacity2,
        device=device,
        label="schedule2_out",
    )
    tiles1 = _tile_count_output(
        tiles1_out,
        device=device,
        label="tiles1_out",
    )
    tiles2 = _tile_count_output(
        tiles2_out,
        device=device,
        label="tiles2_out",
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
        SINGLE_STAGE=False,
    )
    return schedule1, tiles1, schedule2, tiles2


def contiguous_dual_stage_schedule_pack(
    *,
    num_experts: int,
    m_pad_ceiling: int,
    max_expert_rows: int,
    m_alignment: int,
    token_width: int,
    n_gemm1: int,
    n_gemm2: int,
    output_width: int,
    device: torch.device,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
    schedule1_out: torch.Tensor | None = None,
    tiles1_out: torch.Tensor | None = None,
    schedule2_out: torch.Tensor | None = None,
    tiles2_out: torch.Tensor | None = None,
) -> ContiguousSchedulePack:
    """This function is the only constructor of
    :class:`ContiguousSchedulePack`. The capacities and the field shifts come
    from this module, so they always match the standalone builder.
    """
    capacity1, capacity2 = contiguous_dual_stage_schedule_capacities(
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
    return ContiguousSchedulePack(
        schedule1=_schedule_output(
            schedule1_out,
            elements=capacity1,
            device=device,
            label="schedule1_out",
        ),
        tiles1=_tile_count_output(tiles1_out, device=device, label="tiles1_out"),
        schedule2=_schedule_output(
            schedule2_out,
            elements=capacity2,
            device=device,
            label="schedule2_out",
        ),
        tiles2=_tile_count_output(tiles2_out, device=device, label="tiles2_out"),
        token_width=token_width,
        out_clusters1=(n_gemm1 + output_width - 1) // output_width,
        out_clusters2=(n_gemm2 + output_width - 1) // output_width,
        token_cluster_shift=TOKEN_CLUSTER_SHIFT,
        output_cluster_shift=OUTPUT_CLUSTER_SHIFT,
    )


def _int_output(
    output: torch.Tensor | None,
    *,
    elements: int,
    device: torch.device,
    label: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if output is None:
        return torch.empty(elements, dtype=dtype, device=device)
    if (
        output.shape != (elements,)
        or output.dtype != dtype
        or output.device != device
        or not output.is_contiguous()
    ):
        name = str(dtype).removeprefix("torch.")
        raise ValueError(f"{label} must be contiguous {name} [{elements}] on {device}")
    return output


def _schedule_output(
    output: torch.Tensor | None,
    *,
    elements: int,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    return _int_output(
        output, elements=elements, device=device, label=label, dtype=torch.int64
    )


def _tile_count_output(
    output: torch.Tensor | None,
    *,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    """The tile count is int32, not int64. The device compares it to an Int32
    work index. The capacity check above keeps the count inside int32.
    """
    return _int_output(
        output, elements=1, device=device, label=label, dtype=torch.int32
    )


def build_single_stage_schedule(
    masked_m: torch.Tensor,
    *,
    m_max: int,
    token_width: int,
    n_gemm: int,
    output_width: int,
    cluster_shape_mn: tuple[int, int] = (1, 1),
    use_2cta_instrs: bool = False,
    schedule_out: torch.Tensor | None = None,
    tiles_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = masked_m.numel()
    capacity = single_stage_schedule_capacity(
        num_experts=num_experts,
        m_max=m_max,
        token_width=token_width,
        n_gemm=n_gemm,
        output_width=output_width,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    out_clusters = (n_gemm + output_width - 1) // output_width
    device = masked_m.device
    schedule = _schedule_output(
        schedule_out,
        elements=capacity,
        device=device,
        label="schedule_out",
    )
    tiles = _tile_count_output(
        tiles_out,
        device=device,
        label="tiles_out",
    )
    # The stage-2 arguments repeat the stage-1 buffers. The kernel never reads
    # or writes them, because SINGLE_STAGE is constexpr and the grid holds
    # stage 0 only.
    _dual_schedule_kernel[(num_experts, 1)](
        masked_m,
        schedule,
        tiles,
        schedule,
        tiles,
        token_width,
        out_clusters,
        out_clusters,
        EXPERTS=num_experts,
        BLOCK_EXPERTS=max(triton.next_power_of_2(num_experts), 2),
        ENTRY_BLOCK=128,
        TOKEN_SHIFT=TOKEN_CLUSTER_SHIFT,
        OUTPUT_SHIFT=OUTPUT_CLUSTER_SHIFT,
        SINGLE_STAGE=True,
    )
    return schedule, tiles


def build_dual_stage_schedules(
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
    """The capacity assumes that every expert holds ``m_max`` rows. The counts
    that the kernel computes on device therefore always fit in the buffers.
    """
    num_experts = masked_m.numel()
    capacity1, capacity2 = dual_stage_schedule_capacities(
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
        label="schedule1_out",
    )
    schedule2 = _schedule_output(
        schedule2_out,
        elements=capacity2,
        device=device,
        label="schedule2_out",
    )
    tiles1 = _tile_count_output(
        tiles1_out,
        device=device,
        label="tiles1_out",
    )
    tiles2 = _tile_count_output(
        tiles2_out,
        device=device,
        label="tiles2_out",
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
        SINGLE_STAGE=False,
    )
    return schedule1, tiles1, schedule2, tiles2
