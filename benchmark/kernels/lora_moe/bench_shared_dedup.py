"""The §41.1(3b) decision bench: repeated-pair control vs token-dedup.

Shared-outer gate/up-A computes the same (token, adapter) product K times in
the control form; the dedup form (lora_a_shared) computes it once into a
token-major bridge B reads directly (``intermediate_top_k=K``).  The two are
BITWISE equal through B (pinned by test_sgl_lora_shared_dedup), so this
bench is the entire remaining decision — measured at the boundary the forms
actually differ on:

* one thunk = the arm's OWN A-side plan build + the shared per-expert plan
  build (B needs it in BOTH arms) + A + B;
* control's A-side plan is the pair-domain adapter-keyed route (V = L_cap;
  the FUSED align path above the §37 policy thresholds — at substantial
  prefill this histograms all T*K pairs into L_cap+1 buckets);
* dedup's A-side plan is the T-domain token plan (always the JIT path —
  T "pairs" over L_cap+1 buckets) and its A GEMM runs T rows instead of
  T*K, writing a K-times smaller bridge.

Win model: tokens-per-adapter density and K-fold GEMM work, NOT the archive
candidate's request-span length — the T-plan sorts tokens globally by
adapter, so fragmented decode still groups.

Cells are (regime, rank, adapter-count) with the ratified seeded
methodology (3 seeds x 2 interleaved repeats, decide_cell unanimity +
margin); decided flips along T enter the ledger through the evidence-bound
``site_crossover``.  Bitwise arm equality is re-asserted per fixture before
any timing record (admission), so a route-form regression cannot produce a
plausible-looking timing suite.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_shared_dedup \
        --output shared_dedup_v1.json --source-revision <sha> \
        [--ranks 16,64] [--adapters 1,4,8]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture
from benchmark.kernels.lora_moe.cases import (
    AdapterCell,
    Topology,
    build_case,
)
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.lora_a_shared import (
    build_token_adapter_plan,
    masked_token_slots_for_plan,
    run_shared_gate_up,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ROUTE_INCLUSIVE,
    CACHE_L2_HOT_GRAPH,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    build_virtual_expert_routing,
)

T_GRID = (16, 64, 256, 2048, 8192)
SEEDS = (11, 137, 997)
REPEATS = 2
LORA_A = PROVISIONAL_LAUNCH_CONFIG.lora_a
LORA_B = PROVISIONAL_LAUNCH_CONFIG.lora_b
# Typed arm identity; the shared-outer WORKLOAD is the case's property
# (shared_factor_signature), the FORM is the spec's shared_handling.
SPEC_CONTROL = LoraAExecutionSpec(site="gate_up", ownership="grouped")
SPEC_DEDUP = LoraAExecutionSpec(
    site="gate_up", ownership="grouped", shared_handling="token_dedup"
)
ARM_KEYS = {"control": SPEC_CONTROL.key(), "dedup": SPEC_DEDUP.key()}


class _SharedFixture:
    """Shared-outer gate/up leg tensors and both arms' buffers.

    Global-domain cases are LOCALIZED at construction (production's
    convention: the dispatcher hands the runner local ids) so the density
    axis exercises sparse-local routing without teaching any form a second
    id domain.
    """

    def __init__(self, case, device: torch.device) -> None:
        base = _LegFixture(case, device)
        self.base = base
        self.case = case
        if base.lora_expert_map is not None:
            table = base.lora_expert_map
            in_map = (base.topk_ids >= 0) & (base.topk_ids < table.numel())
            localized = table[base.topk_ids.clamp(min=0, max=table.numel() - 1).long()]
            self.topk_ids_local = torch.where(
                in_map,
                localized.to(base.topk_ids.dtype),
                torch.full_like(base.topk_ids, -1),
            )
        else:
            self.topk_ids_local = base.topk_ids
        num_pairs = base.num_pairs
        r_phys = case.physical_rank
        self.rank_out_pairs = torch.empty(
            num_pairs, 2 * r_phys, dtype=torch.bfloat16, device=device
        )
        self.rank_out_tokens = torch.empty(
            case.num_tokens, 2 * r_phys, dtype=torch.bfloat16, device=device
        )
        self.delta = torch.empty(
            num_pairs,
            2 * case.intermediate_size_local,
            dtype=torch.bfloat16,
            device=device,
        )

    def per_expert_plan(self):
        return build_virtual_expert_routing(
            self.topk_ids_local,
            self.base.token_slots,
            lora_experts_per_adapter=self.case.num_experts_local,
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
            view=ROUTE_ALIGNED,
        )

    def outer_pair_plan(self):
        return build_virtual_expert_routing(
            self.topk_ids_local,
            self.base.token_slots,
            lora_experts_per_adapter=1,
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
            shared_outer_local_expert_count=self.case.num_experts_local,
            view=ROUTE_ALIGNED,
        )

    def token_plan(self):
        # Fourth S3 review: tokens with zero valid local pairs are masked
        # out — an unmasked T-plan schedules A work the control skips.
        return build_token_adapter_plan(
            masked_token_slots_for_plan(
                self.base.token_slots,
                self.topk_ids_local,
                num_local_experts=self.case.num_experts_local,
            ),
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
        )

    def control_leg(self, config_a=None) -> None:
        run_shared_gate_up(
            SPEC_CONTROL,
            hidden_states=self.base.hidden_states,
            gate_up_a=self.base.a_gate_up,
            gate_up_b=self.base.b_gate_up,
            rank_out=self.rank_out_pairs,
            gate_up_delta=self.delta,
            a_route=self.outer_pair_plan(),
            per_expert_route=self.per_expert_plan(),
            intermediate_size=self.base.intermediate,
            config_a=config_a or LORA_A,
            config_b=LORA_B,
        )

    def dedup_leg(self, config_a=None) -> None:
        run_shared_gate_up(
            SPEC_DEDUP,
            hidden_states=self.base.hidden_states,
            gate_up_a=self.base.a_gate_up,
            gate_up_b=self.base.b_gate_up,
            rank_out=self.rank_out_tokens,
            gate_up_delta=self.delta,
            a_route=self.token_plan(),
            per_expert_route=self.per_expert_plan(),
            intermediate_size=self.base.intermediate,
            config_a=config_a or LORA_A,
            config_b=LORA_B,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,64")
    parser.add_argument("--adapters", default="1,4,8")
    parser.add_argument(
        "--domains",
        default="ep_local,global",
        help="local-hit density axis (fourth S3 review): ep_local = every "
        "pair locally valid (dedup-favorable ceiling); global = EP8 "
        "sparse-local routing with no-local tokens",
    )
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    adapter_counts = tuple(int(count) for count in arguments.adapters.split(","))
    domains = tuple(arguments.domains.split(","))
    suite = new_suite("shared_dedup", source_revision=arguments.source_revision)

    samples: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    records: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for domain in domains:
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
                            expert_id_domain=domain,
                            num_tokens=num_tokens,
                            active_rank=rank,
                            shared_factor_signature="shared_gate_up_a",
                            seed=seed,
                            source_revision=suite.source_revision,
                        )
                        fixture = _SharedFixture(case, device)

                        # Admission: the forms must be BITWISE equal here, with
                        # both output buffers starting as different garbage.
                        fixture.delta.fill_(71.0)
                        fixture.control_leg()
                        control = fixture.delta.clone()
                        fixture.delta.fill_(-3.0)
                        fixture.dedup_leg()
                        if not torch.equal(control, fixture.delta):
                            raise AssertionError(
                                f"dedup != control bitwise at T={num_tokens} "
                                f"r{rank} L{active} seed {seed}"
                            )

                        arms = {
                            "control": fixture.control_leg,
                            "dedup": fixture.dedup_leg,
                        }
                        cell = (domain, rank, active, num_tokens)
                        for repeat in range(REPEATS):
                            names = (
                                tuple(arms) if repeat % 2 == 0 else tuple(arms)[::-1]
                            )
                            for name in names:
                                record = measure(
                                    arms[name],
                                    suite=suite,
                                    candidate=ARM_KEYS[name],
                                    boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                    params={
                                        "case_id": case.case_id,
                                        "T": num_tokens,
                                        "rank": rank,
                                        "active_adapters": active,
                                        "expert_id_domain": domain,
                                        "seed": seed,
                                        "repeat": repeat,
                                        "config_a": dict(LORA_A),
                                        "config_b": dict(LORA_B),
                                    },
                                    graph_replay=True,
                                )
                                samples[cell][name].append(record.median_s)
                                records[cell][name].append(record.record_id)

    decisions = {}
    for domain in domains:
        for rank in ranks:
            for active in adapter_counts:
                for num_tokens in T_GRID:
                    cell = (domain, rank, active, num_tokens)
                    decision = decide_cell(
                        arm_a="control",
                        samples_a=samples[cell]["control"],
                        arm_b="dedup",
                        samples_b=samples[cell]["dedup"],
                    )
                    decisions[cell] = decision
                    print(
                        f"{domain:8s} r{rank:<4d} L{active} T={num_tokens:<5d} "
                        f"geo(c/d)={decision.geo_a_over_b:.3f} -> "
                        f"{decision.winner or 'tied'}"
                    )

    sited = 0
    for domain in domains:
        for rank in ranks:
            for active in adapter_counts:
                for t_low, t_high in zip(T_GRID, T_GRID[1:]):
                    low = decisions[(domain, rank, active, t_low)]
                    high = decisions[(domain, rank, active, t_high)]
                    if (
                        low.winner is not None
                        and high.winner is not None
                        and low.winner != high.winner
                    ):
                        low_cell = (domain, rank, active, t_low)
                        high_cell = (domain, rank, active, t_high)
                        suite.site_crossover(
                            site="gate_up_a_shared",
                            boundary=BOUNDARY_ROUTE_INCLUSIVE,
                            candidates=(ARM_KEYS["control"], ARM_KEYS["dedup"]),
                            axis=(
                                f"num_tokens (K-fold dedup work at rank={rank}, "
                                f"L={active}, domain={domain})"
                            ),
                            crossover_location=f"T in ({t_low}, {t_high}]",
                            bracketing_low_record_ids=tuple(
                                records[low_cell]["control"]
                                + records[low_cell]["dedup"]
                            ),
                            bracketing_high_record_ids=tuple(
                                records[high_cell]["control"]
                                + records[high_cell]["dedup"]
                            ),
                            cache_state=CACHE_L2_HOT_GRAPH,
                            axis_param="T",
                            workload_params=(
                                "rank",
                                "active_adapters",
                                "expert_id_domain",
                                "config_a",
                                "config_b",
                            ),
                            notes=(
                                f"{low.winner} wins T={t_low} "
                                f"(margin {low.margin():.3f}), {high.winner} "
                                f"T={t_high} (margin {high.margin():.3f})"
                            ),
                        )
                        sited += 1

    digest = write_suite(suite, arguments.output)
    print(
        f"{len(suite.records)} records, {sited} crossovers sited -> "
        f"{arguments.output} sha256 {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
