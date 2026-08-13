"""Token-plan hoisting study (reframed per the fourth S3 review).

Originally committed as an SGMV "upper bound" — it is NOT one: all three
arms execute the same padded grouped kernel, so the deltas bound (a) the
T-plan construction cost and (b) THAT kernel's locality sensitivity, not a
segmented kernel's different padding, launch topology, tiling, occupancy,
metadata reads, or reduction schedule. D6 stays OPEN until the real
in-tree arm (kernels/ops/gemm/chunked_sgmv_shrink.py,
``chunked_sgmv_lora_shrink_forward``) is benchmarked with representative
segment metadata.

What the study DOES establish, on both devices:

* ``tplan_prepared == ideal_contig`` to noise — the grouped kernel does
  not care about adapter-contiguity of the mapping;
* the whole ``tplan_full`` gap is the T-plan build (~6-8 us flat). The
  plan depends only on (token_slots, L_cap), so it is identical for every
  layer of a forward — a PER-BATCH HOIST recovers the entire gap
  (~0.4-0.5 ms/forward at 61 layers, decode scale).

Boundaries are declared honestly: only the full arm includes its A-plan
build (route-inclusive); prepared/ideal run at the prepared-input
boundary, and the per-expert plan is outside every arm.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_tplan_hoist \
        --output tplan_hoist_v1.json --source-revision <sha>
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.bench_shared_dedup import _SharedFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.lora_a_shared import (
    build_token_adapter_plan,
    shared_gate_up_a_token_dedup,
    shared_gate_up_delta_from_token_bridge,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_PREPARED_INPUT,
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG

T_GRID = (16, 64, 256, 2048, 8192)
SEEDS = (11, 137, 997)
LORA_A = PROVISIONAL_LAUNCH_CONFIG.lora_a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,64")
    parser.add_argument("--adapters", default="1,4,8")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    adapter_counts = tuple(int(count) for count in arguments.adapters.split(","))
    suite = new_suite("tplan_hoist", source_revision=arguments.source_revision)

    samples: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rank in ranks:
        for active in adapter_counts:
            for num_tokens in T_GRID:
                for seed in SEEDS:
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
                        num_tokens=num_tokens,
                        active_rank=rank,
                        shared_factor_signature="shared_gate_up_a",
                        seed=seed,
                        source_revision=suite.source_revision,
                    )
                    fixture = _SharedFixture(case, device)
                    per_expert = fixture.per_expert_plan()
                    prepared_plan = fixture.token_plan()
                    sorted_slots, _ = torch.sort(fixture.base.token_slots)
                    ideal_plan = build_token_adapter_plan(
                        sorted_slots,
                        max_loras=case.slot_capacity,
                        block_size=case.routing_block_size,
                    )

                    def a_and_b(plan) -> None:
                        shared_gate_up_a_token_dedup(
                            fixture.base.hidden_states,
                            fixture.base.a_gate_up,
                            plan,
                            fixture.rank_out_tokens,
                            config=LORA_A,
                        )
                        shared_gate_up_delta_from_token_bridge(
                            fixture.rank_out_tokens,
                            fixture.base.b_gate_up,
                            fixture.delta,
                            per_expert,
                            intermediate_size=fixture.base.intermediate,
                            config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
                        )

                    # Honest boundaries (fourth S3 review): only the full
                    # arm includes its A-plan build; prepared/ideal run at
                    # the prepared-input boundary, and the per-expert plan
                    # is outside every arm.
                    arms = {
                        "tplan_full": (
                            BOUNDARY_ROUTE_INCLUSIVE,
                            lambda: a_and_b(fixture.token_plan()),
                        ),
                        "tplan_prepared": (
                            BOUNDARY_PREPARED_INPUT,
                            lambda: a_and_b(prepared_plan),
                        ),
                        "ideal_contig": (
                            BOUNDARY_PREPARED_INPUT,
                            lambda: a_and_b(ideal_plan),
                        ),
                    }
                    cell = (rank, active, num_tokens)
                    for repeat in range(2):
                        names = tuple(arms) if repeat % 2 == 0 else tuple(arms)[::-1]
                        for name in names:
                            boundary, thunk = arms[name]
                            thunk()  # warm
                            record = measure(
                                thunk,
                                suite=suite,
                                candidate=f"tplan_hoist_{name}",
                                boundary=boundary,
                                params={
                                    "case_id": case.case_id,
                                    "T": num_tokens,
                                    "rank": rank,
                                    "active_adapters": active,
                                    "seed": seed,
                                    "repeat": repeat,
                                },
                                graph_replay=True,
                            )
                            samples[cell][name].append(record.median_s)

    worst = 0.0
    for cell in sorted(samples):
        full = statistics.median(samples[cell]["tplan_full"])
        prepared = statistics.median(samples[cell]["tplan_prepared"])
        ideal = statistics.median(samples[cell]["ideal_contig"])
        bound = full / ideal
        worst = max(worst, bound)
        rank, active, num_tokens = cell
        print(
            f"r{rank:<4d} L{active} T={num_tokens:<5d} "
            f"full {full * 1e6:7.2f}us prepared {prepared * 1e6:7.2f}us "
            f"ideal {ideal * 1e6:7.2f}us -> hoist gap {bound:.3f}x"
        )
    print(
        f"worst-cell full/ideal gap {worst:.3f}x — recovered by the "
        "per-batch T-plan hoist; D6's SGMV question needs the real "
        "chunked arm (chunked_sgmv_lora_shrink_forward), not this study"
    )

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
