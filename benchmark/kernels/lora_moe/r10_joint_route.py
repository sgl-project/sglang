"""R10: one joint build for the shared-outer forward's two per-layer plans.

Re-scoped §63.11: after the dedup/T-plan-hoist rulings, the per-layer route
work on a shared-outer forward is exactly TWO aligned plans off the SAME
``topk_ids``/``token_slots`` — the per-expert plan (V = E_local × L_cap;
B consumes it at both sites) and the shared-outer PAIR plan (V = L_cap;
down-B's outer route).  Building them separately reads every pair twice
and pays two full kernel chains.  §43's joint form reads each pair ONCE —
two fused keys, two histogram atomics — and shares the launch overhead:

* K1 (joint hist): one pair pass computing BOTH keys inline
  (``virtual_expert_ids_inline`` at the per-expert and §60.5 shared-outer
  constexpr configurations; the second key's loads hit L1);
* K2 (dual scan): one launch, two programs — each runs the production
  padded scan (counts re-zeroed after read, the standing cache invariant)
  over its own bucket array;
* K3 (joint expand+scatter): one grid split four ways — both plans' block
  halves label blocks and fill padding tails, and ONE scatter pass claims
  a slot in BOTH plans' cursors per pair.

The plan CONTRACT is unchanged: consumers see two ordinary aligned
RouteViews.  Within-bucket pair order differs from separate builds (atomic
claim order), which is output-invisible to grouped A/B — admission pins
that at the consumer (bitwise bridge equality), plus block labels and
padded sizes exactly equal.

Bench-owned per §63.11 (production adoption is a gate-3 question); scratch
buffers are module-cached per (device, V_a, V_b) with the same
zero-between-calls invariant as production's — and the same §40.5 aliasing
caveat, safe under the serial bench.

Usage::

    python3 -m benchmark.kernels.lora_moe.r10_joint_route \
        --output r10_route.json --source-revision <sha>
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.bench_shared_dedup import _SharedFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    RouteView,
    _routing_capacity,
    virtual_expert_ids_inline,
)

HIST_BLOCK = 512
EXPAND_BLOCK = 128
SCAN_CHUNK = 2048


@triton.jit
def _joint_hist_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    counts_a_ptr,
    counts_b_ptr,
    num_pairs,
    shared_outer_local_expert_count,
    NUM_BUCKETS_A: tl.constexpr,
    E_LOCAL: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_BUCKETS_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    ids_a = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        0,
        LORA_EXPERTS_PER_ADAPTER=E_LOCAL,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=False,
    )
    # The second key's topk/slot loads hit L1 — this is the read the
    # separate builds pay twice from HBM.
    ids_b = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        shared_outer_local_expert_count,
        LORA_EXPERTS_PER_ADAPTER=1,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=True,
    )
    tl.atomic_add(
        counts_a_ptr + tl.where(ids_a < 0, NUM_BUCKETS_A - 1, ids_a),
        1,
        mask=pair_mask,
    )
    tl.atomic_add(
        counts_b_ptr + tl.where(ids_b < 0, NUM_BUCKETS_B - 1, ids_b),
        1,
        mask=pair_mask,
    )


@triton.jit
def _scan_one(
    counts_ptr,
    block_cum_ptr,
    cursor_ptr,
    bucket_end_ptr,
    npp_ptr,
    num_buckets,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """The production padded scan, incl. the counts-re-zero invariant."""
    running = 0
    for base in range(0, num_buckets, CHUNK):
        offs = base + tl.arange(0, CHUNK)
        mask = offs < num_buckets
        counts = tl.load(counts_ptr + offs, mask=mask, other=0)
        tl.store(counts_ptr + offs, 0, mask=mask)
        blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_start = running + tl.cumsum(blocks) - blocks
        tl.store(block_cum_ptr + offs, block_start, mask=mask)
        slot_start = block_start * BLOCK_SIZE_M
        tl.store(cursor_ptr + offs, slot_start, mask=mask)
        tl.store(bucket_end_ptr + offs, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cum_ptr + num_buckets, running)
    tl.store(npp_ptr, running * BLOCK_SIZE_M)


@triton.jit
def _dual_scan_kernel(
    counts_a_ptr,
    block_cum_a_ptr,
    cursor_a_ptr,
    bucket_end_a_ptr,
    npp_a_ptr,
    num_buckets_a,
    counts_b_ptr,
    block_cum_b_ptr,
    cursor_b_ptr,
    bucket_end_b_ptr,
    npp_b_ptr,
    num_buckets_b,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
):
    if tl.program_id(0) == 0:
        _scan_one(
            counts_a_ptr,
            block_cum_a_ptr,
            cursor_a_ptr,
            bucket_end_a_ptr,
            npp_a_ptr,
            num_buckets_a,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            CHUNK=CHUNK,
        )
    else:
        _scan_one(
            counts_b_ptr,
            block_cum_b_ptr,
            cursor_b_ptr,
            bucket_end_b_ptr,
            npp_b_ptr,
            num_buckets_b,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            CHUNK=CHUNK,
        )


@triton.jit
def _block_half(
    pid,
    block_cum_ptr,
    bucket_end_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_blocks,
    num_pairs,
    NUM_BUCKETS: tl.constexpr,
    NUM_VIRTUAL: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """The production block half: label owners, fill padding tails."""
    block_ids = pid * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_ids < num_blocks
    low = tl.zeros(block_ids.shape, dtype=tl.int32)
    high = tl.full(block_ids.shape, NUM_BUCKETS, dtype=tl.int32)
    for _ in range(SEARCH_STEPS):
        mid = (low + high) // 2
        bound = tl.load(
            block_cum_ptr + tl.minimum(mid + 1, NUM_BUCKETS),
            mask=block_mask,
            other=0,
        )
        take = block_ids >= bound
        low = tl.where(take & (low < high), mid + 1, low)
        high = tl.where(take | (low >= high), high, mid)
    owner = tl.minimum(low, NUM_BUCKETS - 1)
    total_blocks = tl.load(block_cum_ptr + NUM_BUCKETS)
    in_plan = block_mask & (block_ids < total_blocks)
    labelled = in_plan & (owner < NUM_VIRTUAL)
    tl.store(
        block_virtual_expert_ids_ptr + block_ids,
        tl.where(labelled, owner, -1),
        mask=block_mask,
    )
    real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
    slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
    tl.store(
        sorted_pair_ids_ptr + slots,
        num_pairs,
        mask=in_plan[:, None] & (slots >= real_end[:, None]),
    )


@triton.jit
def _joint_expand_scatter_kernel(
    topk_ids_ptr,
    token_slots_ptr,
    cursor_a_ptr,
    bucket_end_a_ptr,
    block_cum_a_ptr,
    sorted_a_ptr,
    block_ids_a_ptr,
    num_blocks_a,
    nprog_blocks_a,
    cursor_b_ptr,
    bucket_end_b_ptr,
    block_cum_b_ptr,
    sorted_b_ptr,
    block_ids_b_ptr,
    num_blocks_b,
    nprog_blocks_b,
    num_pairs,
    shared_outer_local_expert_count,
    NUM_BUCKETS_A: tl.constexpr,
    NUM_VIRTUAL_A: tl.constexpr,
    E_LOCAL: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_BUCKETS_B: tl.constexpr,
    NUM_VIRTUAL_B: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS_A: tl.constexpr,
    SEARCH_STEPS_B: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < nprog_blocks_a:
        _block_half(
            pid,
            block_cum_a_ptr,
            bucket_end_a_ptr,
            sorted_a_ptr,
            block_ids_a_ptr,
            num_blocks_a,
            num_pairs,
            NUM_BUCKETS=NUM_BUCKETS_A,
            NUM_VIRTUAL=NUM_VIRTUAL_A,
            BLOCK=BLOCK,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SEARCH_STEPS=SEARCH_STEPS_A,
        )
        return
    if pid < nprog_blocks_a + nprog_blocks_b:
        _block_half(
            pid - nprog_blocks_a,
            block_cum_b_ptr,
            bucket_end_b_ptr,
            sorted_b_ptr,
            block_ids_b_ptr,
            num_blocks_b,
            num_pairs,
            NUM_BUCKETS=NUM_BUCKETS_B,
            NUM_VIRTUAL=NUM_VIRTUAL_B,
            BLOCK=BLOCK,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SEARCH_STEPS=SEARCH_STEPS_B,
        )
        return
    pair_ids = (pid - nprog_blocks_a - nprog_blocks_b) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    ids_a = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        0,
        LORA_EXPERTS_PER_ADAPTER=E_LOCAL,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=False,
    )
    ids_b = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        topk_ids_ptr,
        pair_ids,
        pair_mask,
        shared_outer_local_expert_count,
        LORA_EXPERTS_PER_ADAPTER=1,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=False,
        SHARED_OUTER=True,
    )
    buckets_a = tl.where(ids_a < 0, NUM_BUCKETS_A - 1, ids_a)
    slots_a = tl.atomic_add(cursor_a_ptr + buckets_a, 1, mask=pair_mask)
    tl.store(sorted_a_ptr + slots_a, pair_ids, mask=pair_mask)
    buckets_b = tl.where(ids_b < 0, NUM_BUCKETS_B - 1, ids_b)
    slots_b = tl.atomic_add(cursor_b_ptr + buckets_b, 1, mask=pair_mask)
    tl.store(sorted_b_ptr + slots_b, pair_ids, mask=pair_mask)


class _JointScratch:
    """Zero-counts-invariant scratch per (device, buckets_a, buckets_b)."""

    def __init__(self, num_buckets_a: int, num_buckets_b: int, device) -> None:
        def per_plan(num_buckets):
            return {
                "counts": torch.zeros(num_buckets, dtype=torch.int32, device=device),
                "block_cum": torch.empty(
                    num_buckets + 1, dtype=torch.int32, device=device
                ),
                "cursor": torch.empty(num_buckets, dtype=torch.int32, device=device),
                "bucket_end": torch.empty(
                    num_buckets, dtype=torch.int32, device=device
                ),
                "npp": torch.empty(1, dtype=torch.int32, device=device),
            }

        self.a = per_plan(num_buckets_a)
        self.b = per_plan(num_buckets_b)


_SCRATCH: dict[tuple, _JointScratch] = {}


def build_joint_shared_routes(
    topk_ids_local: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    num_experts_local: int,
    max_loras: int,
    block_size: int,
) -> tuple[RouteView, RouteView]:
    """(per_expert_plan, shared_outer_pair_plan) from ONE pair pass."""
    device = topk_ids_local.device
    num_pairs = topk_ids_local.numel()
    top_k = topk_ids_local.shape[1]
    num_virtual_a = num_experts_local * max_loras
    num_virtual_b = max_loras
    num_buckets_a = num_virtual_a + 1
    num_buckets_b = num_virtual_b + 1
    capacity_a = _routing_capacity(num_pairs, block_size, num_virtual_a)
    capacity_b = _routing_capacity(num_pairs, block_size, num_virtual_b)
    num_blocks_a = capacity_a // block_size
    num_blocks_b = capacity_b // block_size

    key = (device.type, device.index, num_buckets_a, num_buckets_b)
    scratch = _SCRATCH.get(key)
    if scratch is None:
        scratch = _JointScratch(num_buckets_a, num_buckets_b, device)
        _SCRATCH[key] = scratch
    sorted_a = torch.empty(capacity_a, dtype=torch.int32, device=device)
    block_ids_a = torch.empty(num_blocks_a, dtype=torch.int32, device=device)
    sorted_b = torch.empty(capacity_b, dtype=torch.int32, device=device)
    block_ids_b = torch.empty(num_blocks_b, dtype=torch.int32, device=device)

    _joint_hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids_local,
        token_slots,
        scratch.a["counts"],
        scratch.b["counts"],
        num_pairs,
        num_experts_local,
        NUM_BUCKETS_A=num_buckets_a,
        E_LOCAL=num_experts_local,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        NUM_BUCKETS_B=num_buckets_b,
        BLOCK=HIST_BLOCK,
        num_warps=8,
    )
    _dual_scan_kernel[(2,)](
        scratch.a["counts"],
        scratch.a["block_cum"],
        scratch.a["cursor"],
        scratch.a["bucket_end"],
        scratch.a["npp"],
        num_buckets_a,
        scratch.b["counts"],
        scratch.b["block_cum"],
        scratch.b["cursor"],
        scratch.b["bucket_end"],
        scratch.b["npp"],
        num_buckets_b,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        num_warps=4,
    )
    nprog_blocks_a = triton.cdiv(max(num_blocks_a, 1), EXPAND_BLOCK)
    nprog_blocks_b = triton.cdiv(max(num_blocks_b, 1), EXPAND_BLOCK)
    nprog_pairs = triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK)
    _joint_expand_scatter_kernel[(nprog_blocks_a + nprog_blocks_b + nprog_pairs,)](
        topk_ids_local,
        token_slots,
        scratch.a["cursor"],
        scratch.a["bucket_end"],
        scratch.a["block_cum"],
        sorted_a,
        block_ids_a,
        num_blocks_a,
        nprog_blocks_a,
        scratch.b["cursor"],
        scratch.b["bucket_end"],
        scratch.b["block_cum"],
        sorted_b,
        block_ids_b,
        num_blocks_b,
        nprog_blocks_b,
        num_pairs,
        num_experts_local,
        NUM_BUCKETS_A=num_buckets_a,
        NUM_VIRTUAL_A=num_virtual_a,
        E_LOCAL=num_experts_local,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        NUM_BUCKETS_B=num_buckets_b,
        NUM_VIRTUAL_B=num_virtual_b,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        SEARCH_STEPS_A=max(1, (num_buckets_a - 1).bit_length()),
        SEARCH_STEPS_B=max(1, (num_buckets_b - 1).bit_length()),
        num_warps=4,
    )

    def view(
        num_virtual,
        lora_experts_per_adapter,
        shared_outer_local_expert_count,
        sorted_ids,
        block_ids,
        npp,
    ):
        return RouteView(
            view=ROUTE_ALIGNED,
            num_virtual_experts=num_virtual,
            block_size=block_size,
            topk_ids=topk_ids_local,
            token_slots=token_slots,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            shared_outer_local_expert_count=shared_outer_local_expert_count,
            maybe_sorted_pair_ids=sorted_ids,
            maybe_block_virtual_expert_ids=block_ids,
            maybe_num_pairs_post_padded=npp,
        )

    return (
        view(
            num_virtual_a,
            num_experts_local,
            None,
            sorted_a,
            block_ids_a,
            scratch.a["npp"],
        ),
        view(
            num_virtual_b, 1, num_experts_local, sorted_b, block_ids_b, scratch.b["npp"]
        ),
    )


def _separate_builds(fixture: _SharedFixture):
    """The PRODUCTION control: two build_virtual_expert_routing calls.

    Eighth S3 review: the first cut forced two fused_align_block_size
    calls without PDL, but production routes through a selector that
    picks the JIT align below 8192 virtual experts / 16384 pairs (all
    of this bench's decode cells) and fused WITH PDL above — so the
    archived speedups were measured against a control production does
    not run. The control is now the production entry point itself.
    """
    return fixture.per_expert_plan(), fixture.outer_pair_plan()


def _admit(fixture: _SharedFixture, label: str) -> None:
    """Joint plans must be consumer-equivalent to the separate builds."""
    case = fixture.case
    joint_pe, joint_outer = build_joint_shared_routes(
        fixture.topk_ids_local,
        fixture.base.token_slots,
        num_experts_local=case.num_experts_local,
        max_loras=case.slot_capacity,
        block_size=case.routing_block_size,
    )
    # Structural equality is checked against the FUSED align (the encoding
    # the joint kernel replicates). The production selector may pick the
    # JIT align at these sizes, which uses a different-but-valid block
    # ordering (sentinel bucket first — every label shifts), so the
    # production control is compared through CONSUMER outputs below, the
    # encoding-agnostic contract.
    from sglang.srt.lora.sgl_lora.fused_align import fused_align_block_size
    from sglang.srt.lora.sgl_lora.routing import _routing_capacity

    for name, joint, lora_experts_per_adapter, shared_outer_local_expert_count in (
        ("per_expert", joint_pe, case.num_experts_local, None),
        ("outer", joint_outer, 1, case.num_experts_local),
    ):
        num_virtual = lora_experts_per_adapter * case.slot_capacity
        _, fused_blocks, fused_npp = fused_align_block_size(
            fixture.topk_ids_local,
            fixture.base.token_slots,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=case.slot_capacity,
            block_size=case.routing_block_size,
            capacity=_routing_capacity(
                fixture.topk_ids_local.numel(),
                case.routing_block_size,
                num_virtual,
            ),
            shared_outer_local_expert_count=shared_outer_local_expert_count,
        )
        if int(joint.num_pairs_post_padded.item()) != int(fused_npp.item()):
            raise AssertionError(f"{label} {name}: padded sizes differ")
        if not torch.equal(joint.block_virtual_expert_ids, fused_blocks):
            raise AssertionError(f"{label} {name}: block labels differ")
    # Consumer-level: the grouped A bridge is pair-major canonical, so it is
    # bitwise identical regardless of within-bucket claim order. Each plan
    # runs against ITS OWN weight form — in a shared-outer case the gate A
    # factors exist only per adapter (the outer plan's domain), while the
    # per-expert plan's consumers are the down/B kernels over per-expert
    # factors (a first cut ran the per-expert plan against the L-factor
    # gate weights: OOB weight indexing that only the second seed faulted).
    config = PROVISIONAL_LAUNCH_CONFIG.lora_a
    consumers = (
        (
            "per_expert",
            joint_pe,
            fixture.per_expert_plan,
            fixture.base.act_pair,
            fixture.base.a_down,
            fixture.base.down_rank_out,
            True,
        ),
        (
            "outer",
            joint_outer,
            fixture.outer_pair_plan,
            fixture.base.hidden_states,
            fixture.base.a_gate_up,
            fixture.rank_out_pairs,
            False,
        ),
    )
    for name, route, separate_plan, inp, weight, out, pair_input in consumers:
        out.fill_(float("nan"))
        grouped_lora_a(inp, weight, out, route, config=config, pair_input=pair_input)
        joint_bridge = out.clone()
        out.fill_(float("nan"))
        grouped_lora_a(
            inp, weight, out, separate_plan(), config=config, pair_input=pair_input
        )
        # Compare where the separate-plan bridge is defined (valid pairs
        # write real values; sentinel rows stay NaN-filled in both).
        defined = ~torch.isnan(out).any(dim=1)
        if not torch.equal(joint_bridge[defined], out[defined]):
            raise AssertionError(f"{label} {name}: consumer bridge differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapters", default="4,8")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    suite = new_suite("r10_joint_route", source_revision=arguments.source_revision)
    seeds = (11, 137, 997)
    repeats = 2

    samples: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for active in (int(a) for a in arguments.adapters.split(",")):
        for num_tokens in (16, 64, 256, 2048):
            for seed in seeds:
                case = build_case(
                    device=str(device),
                    model_preset="qwen35_35b",
                    topology=Topology(tp_size=8, ep_size=8),
                    adapter_cell=AdapterCell(
                        active_adapters=active,
                        include_base_rows=True,
                        slot_capacity=8,
                    ),
                    route_generator="iid",
                    expert_id_domain="ep_local",
                    num_tokens=num_tokens,
                    active_rank=16,
                    shared_factor_signature="shared_gate_up_a",
                    seed=seed,
                    source_revision=suite.source_revision,
                )
                fixture = _SharedFixture(case, device)
                _admit(fixture, f"L{active} T={num_tokens} s{seed}")

                def separate():
                    _separate_builds(fixture)

                def joint():
                    build_joint_shared_routes(
                        fixture.topk_ids_local,
                        fixture.base.token_slots,
                        num_experts_local=case.num_experts_local,
                        max_loras=case.slot_capacity,
                        block_size=case.routing_block_size,
                    )

                arms = {"route_separate": separate, "route_joint": joint}
                cell = (active, num_tokens)
                for repeat in range(repeats):
                    names = tuple(arms) if repeat % 2 == 0 else tuple(arms)[::-1]
                    for name in names:
                        record = measure(
                            arms[name],
                            suite=suite,
                            candidate=name,
                            boundary=BOUNDARY_ISOLATED,
                            params={
                                "case_id": case.case_id,
                                "T": num_tokens,
                                "active_adapters": active,
                                "seed": seed,
                                "repeat": repeat,
                            },
                            graph_replay=True,
                        )
                        samples[cell][name].append(record.median_s)

    for cell in sorted(samples):
        decision = decide_cell(
            arm_a="route_separate",
            samples_a=samples[cell]["route_separate"],
            arm_b="route_joint",
            samples_b=samples[cell]["route_joint"],
        )
        sep_us = sorted(samples[cell]["route_separate"])[2] * 1e6
        joint_us = sorted(samples[cell]["route_joint"])[2] * 1e6
        print(
            f"L{cell[0]} T={cell[1]:<5d} separate {sep_us:7.2f}us "
            f"joint {joint_us:7.2f}us geo(s/j)={decision.geo_a_over_b:.3f} "
            f"-> {decision.winner or 'tied'}"
        )

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
