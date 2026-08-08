"""Per-forward packed tile schedules for the direct-schedule CuTeDSL GEMMs.

The direct-schedule kernel consumes a packed int32 per tile —
``expert | token_cluster << 10 | output_cluster << 20`` — plus a device tile
count. Both GEMM stages' schedules derive from the SAME ``masked_m`` in ONE
launch, which is the section 45 dual-ownership rule: a schedule built from any
other row-count source than the masked_m the GEMMs will read is silent
corruption, so this module never accepts row counts through a second path.

Per-expert tile ordering follows the study's heuristic: when an expert has no
more token clusters than output clusters, tiles iterate token-fastest
(consecutive tiles share the output cluster); otherwise output-fastest. The
loop bound is a runtime scalar, so no per-m_max recompilation.

ABI bounds are enforced HERE, and the packing shifts are passed to the kernel
from the SAME constants (section 45's flagged risk: guard and builder must not
drift apart -- the first version hardcoded the shifts as literals while only
the guard read the constants, which is exactly the drift the comment warned
about): expert counts up to 1024 (indices 0..1023 fit the 10-bit field),
token clusters <= 1024 per expert, output clusters <= 2048 (a non-negative-
word policy; see the constant), and the worst-case capacity within int32.
The device decoder in `scheduler.py` carries the matching literals; changing
a field width is a coordinated edit across both files.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

TOKEN_CLUSTER_BITS = 10
OUTPUT_CLUSTER_SHIFT = 20
MAX_EXPERTS = 1 << TOKEN_CLUSTER_BITS  # expert field: bits [0, 10)
MAX_TOKEN_CLUSTERS = 1 << TOKEN_CLUSTER_BITS
# The output-cluster field occupies bits [20, 32). Bit 31 IS representable --
# a packed word with it set is simply a negative int32, and the device decoder
# (shift then mask) recovers the field exactly; a host tool just has to
# construct the signed equivalent rather than a >2^31-1 python int. Capping at
# 11 bits is therefore a POLICY, not an ABI limit: every packed word stays
# non-negative, which keeps host-side decoding and debugging free of
# sign-extension traps. It forfeits nothing real -- output clusters >= 2048
# would need N >= 262144 at the 128-wide output tile (reviewer-corrected
# justification; the earlier claim that bit 31 was "unrepresentable" was
# wrong).
MAX_OUTPUT_CLUSTERS = 1 << 11


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
    """One program per (expert, stage); entries written as vector blocks.

    The first version ran ONE program with a serial per-entry loop across all
    experts — 17 us at prefill scale, larger than the GEMM saving it enabled
    (the measured cause of the pipeline's 0.98x prefill). Each program now
    recomputes its own exclusive prefix from masked_m (O(E) vectorized loads,
    E <= 1024) and fills its entries ENTRY_BLOCK at a time.
    """
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
        packed = expert | (tc_i << TOKEN_SHIFT) | (oc_i << OUTPUT_SHIFT)
        if stage == 0:
            tl.store(schedule1_ptr + begin + local, packed, mask=valid)
        else:
            tl.store(schedule2_ptr + begin + local, packed, mask=valid)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(schedule1, tiles1, schedule2, tiles2)`` for one forward.

    Capacity is the worst case over this forward's ``m_max`` (every expert
    full), so the kernel's device-side counts can never overflow the buffers.

    ``cluster_shape_mn`` / ``use_2cta_instrs`` are the CALLER'S kernel config,
    taken only to be REJECTED unless trivial. This builder emits indices at
    CTA-tile granularity (derived from ``token_width`` / ``output_width``),
    while the device scheduler treats them as CLUSTER indices and expands them
    by ``cluster_shape_mn``. Those agree only at a 1x1 cluster with 1-CTA MMA;
    with a real cluster the schedule would over-enumerate and address tiles
    out of range, silently — no device assertion exists to catch it. Supporting
    clusters means dividing the counts here by the cluster extents, which is a
    deliberate change with its own boundary cases, not a default.
    """
    if tuple(cluster_shape_mn) != (1, 1) or use_2cta_instrs:
        raise ValueError(
            "the packed direct schedule is emitted at CTA-tile granularity and "
            "the device expands it by cluster_shape_mn, so only a (1, 1) "
            "cluster with 1-CTA MMA is representable; got "
            f"cluster_shape_mn={tuple(cluster_shape_mn)}, "
            f"use_2cta_instrs={use_2cta_instrs}"
        )
    num_experts = masked_m.numel()
    out_clusters1 = (n_gemm1 + output_width - 1) // output_width
    out_clusters2 = (n_gemm2 + output_width - 1) // output_width
    max_token_clusters = (m_max + token_width - 1) // token_width
    if num_experts > MAX_EXPERTS:
        raise ValueError(
            f"direct schedule packs expert INDICES in 10 bits (counts up to "
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
            f"{MAX_OUTPUT_CLUSTERS} the non-negative-word policy allows"
        )
    # The kernel's begin/total arithmetic is int32 over the worst-case
    # capacity; per-field width caps do not imply this bound, so check it
    # directly (reviewer finding).
    capacity = num_experts * max_token_clusters * max(out_clusters1, out_clusters2)
    if capacity > 2**31 - 1:
        raise ValueError(
            f"worst-case schedule capacity {capacity} overflows the builder's "
            "int32 prefix arithmetic"
        )

    device = masked_m.device
    schedule1 = torch.empty(
        num_experts * max_token_clusters * out_clusters1,
        dtype=torch.int32,
        device=device,
    )
    schedule2 = torch.empty(
        num_experts * max_token_clusters * out_clusters2,
        dtype=torch.int32,
        device=device,
    )
    tiles1 = torch.empty(1, dtype=torch.int32, device=device)
    tiles2 = torch.empty(1, dtype=torch.int32, device=device)
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
        TOKEN_SHIFT=TOKEN_CLUSTER_BITS,
        OUTPUT_SHIFT=OUTPUT_CLUSTER_SHIFT,
    )
    return schedule1, tiles1, schedule2, tiles2
