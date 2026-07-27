"""Honest re-test of the CuTeDSL gate/up prefill frontier (eighth review).

The seeded pass (§63.24) left CuTeDSL one winning region: gate/up A at
rank 128 large prefill. The eighth review showed that comparison still
favored CuTeDSL three ways: the grouped config came from a decode-tuned
table (a better prefill config existed in the same sweep archive), the
timing was graph-replay only (production prefill runs eager), and the
CuTeDSL arm's per-layer metadata (dispatch, schedules, staging) was built
outside timing while grouped would carry its plan build in a
route-inclusive frame.

This bench closes all three at once, on the frontier cells only
(gate/up, ranks 64/128, T = 2048/8192):

* grouped runs a PER-CELL best config found by an in-bench sweep of 12
  candidate configs (block sizes 32/64/128 x 64/128, stages 2/3), timed
  at the same boundary as the challenger;
* both arms are ROUTE-INCLUSIVE: grouped builds its aligned plan in-thunk
  through the production selector; CuTeDSL re-runs dispatch + schedules
  (verified once per fixture, then verify=False in-thunk — dispatch
  counts are deterministic for a fixed route) + staging + GEMM + scatter;
* every cell is decided in BOTH eager and graph modes, seeded 3x2.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_p5_frontier \
        --output p5_frontier.json --source-revision <sha>
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture, site_baseline
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.lora_a_candidates import run_lora_a
from benchmark.kernels.lora_moe.lora_a_cutedsl import (
    CutedslAConfig,
    build_cutedsl_lora_a_plan,
    supported_token_widths,
)
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.signal_gates import require_delta_close
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, ROUTE_FUSED_IDS

SEEDS = (11, 137, 997)
REPEATS = 2
SPEC_GROUPED = LoraAExecutionSpec(site="gate_up", ownership="grouped")
SPEC_CUTEDSL = LoraAExecutionSpec(
    site="gate_up", ownership="grouped", implementation="cutedsl"
)
# The grouped candidate grid swept PER CELL (not a decode-tuned table).
GROUPED_CONFIGS = tuple(
    {
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": stages,
    }
    for bn in (32, 64, 128)
    for bk in (64, 128)
    for stages in (2, 3)
)


def _config_key(config: dict) -> str:
    return (
        f"bn{config['BLOCK_SIZE_N']}-bk{config['BLOCK_SIZE_K']}"
        f"-w{config['num_warps']}-s{config['num_stages']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="64,128")
    parser.add_argument("--t-grid", default="2048,8192")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    suite = new_suite("p5_frontier", source_revision=arguments.source_revision)
    widths = supported_token_widths(device)

    samples: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rank in (int(r) for r in arguments.ranks.split(",")):
        for num_tokens in (int(t) for t in arguments.t_grid.split(",")):
            for seed in SEEDS:
                case = build_case(
                    device=str(device),
                    model_preset="qwen35_35b",
                    topology=Topology(tp_size=8, ep_size=8),
                    adapter_cell=AdapterCell(
                        active_adapters=4,
                        include_base_rows=True,
                        slot_capacity=8,
                    ),
                    route_generator="iid",
                    num_tokens=num_tokens,
                    active_rank=rank,
                    seed=seed,
                    source_revision=suite.source_revision,
                )
                fixture = _LegFixture(case, device)
                aligned = fixture.route(ROUTE_ALIGNED)
                fused = fixture.route(ROUTE_FUSED_IDS)
                baseline = site_baseline(fixture, "gate_up", aligned).clone()
                valid = fixture.valid_pairs

                # Per-cell grouped sweep at the ROUTE-INCLUSIVE boundary
                # (plan built in-thunk through the production selector),
                # graph mode, quick single measures — the sweep picks the
                # config; the decided records rerun it seeded.
                def grouped_thunk(config):
                    def thunk():
                        plan = fixture.route(ROUTE_ALIGNED)
                        grouped_lora_a(
                            fixture.hidden_states,
                            fixture.a_gate_up,
                            fixture.gate_rank_out,
                            plan,
                            config=config,
                        )

                    return thunk

                best_config, best_time = None, float("inf")
                for config in GROUPED_CONFIGS:
                    fixture.gate_rank_out.fill_(float("nan"))
                    grouped_thunk(config)()
                    require_delta_close(
                        fixture.gate_rank_out[valid].float(),
                        baseline[valid].float(),
                        gate_dtype=torch.bfloat16,
                        label=f"grouped {_config_key(config)} r{rank} T={num_tokens}",
                    )
                    probe = measure(
                        grouped_thunk(config),
                        suite=suite,
                        candidate=f"sweep_{SPEC_GROUPED.key()}_{_config_key(config)}",
                        boundary=BOUNDARY_ROUTE_INCLUSIVE,
                        params={
                            "case_id": case.case_id,
                            "T": num_tokens,
                            "rank": rank,
                            "seed": seed,
                            "config": dict(config),
                            "role": "config_sweep_probe",
                        },
                        graph_replay=True,
                    )
                    if probe.median_s < best_time:
                        best_config, best_time = config, probe.median_s

                # CuTeDSL arm per width; per-layer work charged in-thunk.
                plans = {}
                for width in widths:
                    plan = build_cutedsl_lora_a_plan(
                        fused_route=fused,
                        gate_up_weight=fixture.a_gate_up,
                        down_weight=fixture.a_down,
                        config=CutedslAConfig(token_width=width),
                    )
                    plan.build_metadata(verify=True)  # once per fixture
                    fixture.gate_rank_out.fill_(float("nan"))
                    run_lora_a(
                        SPEC_CUTEDSL,
                        input=fixture.hidden_states,
                        weight=fixture.a_gate_up,
                        output=fixture.gate_rank_out,
                        routing=aligned,
                        config=best_config,
                        cutedsl_plan=plan,
                    )
                    require_delta_close(
                        fixture.gate_rank_out[valid].float(),
                        baseline[valid].float(),
                        gate_dtype=torch.bfloat16,
                        label=f"cutedsl tw{width} r{rank} T={num_tokens}",
                    )
                    plans[width] = plan

                def cutedsl_thunk(plan):
                    def thunk():
                        plan.build_metadata(verify=False)
                        run_lora_a(
                            SPEC_CUTEDSL,
                            input=fixture.hidden_states,
                            weight=fixture.a_gate_up,
                            output=fixture.gate_rank_out,
                            routing=aligned,
                            config=best_config,
                            cutedsl_plan=plan,
                        )

                    return thunk

                arms = {"grouped_best": grouped_thunk(best_config)}
                for width, plan in plans.items():
                    arms[f"cutedsl_tw{width}"] = cutedsl_thunk(plan)
                cell = (rank, num_tokens)
                for graph in (True, False):
                    mode = "graph" if graph else "eager"
                    for repeat in range(REPEATS):
                        names = tuple(arms) if repeat % 2 == 0 else tuple(arms)[::-1]
                        for name in names:
                            key = (
                                f"{SPEC_GROUPED.key()}_cellbest_"
                                f"{_config_key(best_config)}"
                                if name == "grouped_best"
                                else f"{SPEC_CUTEDSL.key()}_{name.split('_')[1]}"
                                "_charged"
                            )
                            record = measure(
                                arms[name],
                                suite=suite,
                                candidate=key,
                                boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                params={
                                    "case_id": case.case_id,
                                    "T": num_tokens,
                                    "rank": rank,
                                    "seed": seed,
                                    "repeat": repeat,
                                    "grouped_config": dict(best_config),
                                    "role": "decided_arm",
                                },
                                graph_replay=graph,
                            )
                            samples[(*cell, mode)][name].append(record.median_s)

    for (rank, num_tokens, mode), arms_samples in sorted(samples.items()):
        for name in arms_samples:
            if name == "grouped_best":
                continue
            decision = decide_cell(
                arm_a="grouped_best",
                samples_a=arms_samples["grouped_best"],
                arm_b=name,
                samples_b=arms_samples[name],
            )
            print(
                f"{mode:5s} r{rank:<4d} T={num_tokens:<5d} {name:14s} "
                f"geo(g/c)={decision.geo_a_over_b:.3f} -> "
                f"{decision.winner or 'tied'}"
            )

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
