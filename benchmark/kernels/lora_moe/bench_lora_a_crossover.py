"""Site LoRA-A schedule crossovers with the ratified seeded methodology.

The pooled `bench_lora_a` grids showed a split-K/grouped flip somewhere in
the decode band (split-K wins gate/up at T=16 rank-16, loses at T=64) — a
crossover may only enter the §31.7 ledger when it is SITED between two
DECIDED cells, and a cell is decided only by the align-boundary rules gate 2
ratified: three route seeds x two interleaved repeats per arm (arm order
alternates within a cell so clock drift cannot systematically favor one),
an arm wins only on unanimous sign across the six paired samples AND a
geometric-mean margin of at least MIN_MARGIN.

Cells are (site, rank, T) at BOUNDARY_ISOLATED in graph mode — the
crossover in question is kernel occupancy at decode, where graph replay is
the serving execution and the aligned plan exists for B either way (§33.1:
route cost is identical between these arms by construction, so the isolated
boundary IS the decision boundary for grouped-family reduction variants;
the indexed arm's records are for the per-T decode slices, with its route
economics settled separately).

Ledger entries are emitted through ``TimingSuite.site_crossover`` — the
evidence-bound path — for every adjacent decided pair that flips winner,
citing all 24 bracketing records of the two cells (six per arm per cell).

Arms run PER-FAMILY TUNED configs supplied as a JSON table
(``--config-table``, ``{site: {rank: {family: "bnX-bkY-wW-sS[-kK]"}}}``,
typically distilled from the tile-sweep archives) — the second S3 review
showed untuned-arm crossovers can move or vanish once each family is
tuned, so a run without a table is labeled provisional in its suite name.
Every (arm, config, shape) is correctness-admitted against the grouped
baseline before its first timing record.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_lora_a_crossover \
        --output lora_a_crossover_v1.json --source-revision <sha> \
        [--ranks 16,32,64,128] [--pairs splitk,indexed]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
import triton

from benchmark.kernels.lora_moe.bench_lora_a import (
    ADAPTER_CELL,
    SPEC_DOWN_CUTEDSL,
    SPEC_DOWN_GROUPED,
    SPEC_DOWN_INDEXED,
    SPEC_DOWN_SPLITK,
    SPEC_GATE_CUTEDSL,
    SPEC_GATE_GROUPED,
    SPEC_GATE_INDEXED,
    SPEC_GATE_SPLITK,
    _LegFixture,
    _parse_indexed_config,
    admit_candidate,
    run_site,
    site_baseline,
)
from benchmark.kernels.lora_moe.cases import Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.lora_a_candidates import (
    INDEXED_DEFAULT_CONFIG,
    SPLIT_K_DEFAULT_CONFIG,
    resolve_split_k,
    split_k_workspace,
)
from benchmark.kernels.lora_moe.lora_a_cutedsl import (
    CutedslAConfig,
    build_cutedsl_lora_a_plan,
    supported_token_widths,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    CACHE_L2_HOT_GRAPH,
    measure,
    new_suite,
    resolve_source_revision,
    write_suite,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, ROUTE_FUSED_IDS, ROUTE_RAW

# Decode band with bracketing cells around the observed flip (16 vs 64);
# --t-grid overrides (the P5 confirmation runs the prefill frontier).
DEFAULT_T_GRID = "8,16,32,64,128,256"
SEEDS = (11, 137, 997)
REPEATS = 2

SPECS = {
    ("gate_up", "grouped"): SPEC_GATE_GROUPED,
    ("gate_up", "splitk"): SPEC_GATE_SPLITK,
    ("gate_up", "indexed"): SPEC_GATE_INDEXED,
    ("down", "grouped"): SPEC_DOWN_GROUPED,
    ("down", "splitk"): SPEC_DOWN_SPLITK,
    ("down", "indexed"): SPEC_DOWN_INDEXED,
}


def _arm_spec(site: str, arm: str):
    """Arm name -> typed spec; ``cutedsl_twN`` names the P5 composite."""
    if arm.startswith("cutedsl_tw"):
        return SPEC_GATE_CUTEDSL if site == "gate_up" else SPEC_DOWN_CUTEDSL
    return SPECS[(site, arm)]


def _arm_key(site: str, arm: str) -> str:
    """Candidate string: the spec key, plus launch geometry for cutedsl."""
    spec = _arm_spec(site, arm)
    if arm.startswith("cutedsl_tw"):
        return f"{spec.key()}_{arm.removeprefix('cutedsl_')}"
    return spec.key()


def _cutedsl_width(arm: str) -> int:
    return int(arm.removeprefix("cutedsl_tw"))


def _parse_config(text: str) -> dict[str, int]:
    """Family config string; an optional ``-kN`` suffix sets SPLIT_K."""
    split_k = 0
    pieces = text.split("-")
    if pieces[-1].startswith("k"):
        split_k = int(pieces[-1][1:])
        text = "-".join(pieces[:-1])
    config = dict(_parse_indexed_config(text))
    config["GROUP_SIZE_M"] = 8
    if split_k:
        config["SPLIT_K"] = split_k
    return config


def _resolve_configs(table: dict | None, site: str, rank: int) -> dict[str, dict]:
    """Per-family configs for one cell.

    With no table every family runs its default and the suite is named
    provisional.  With a table, a missing (site, rank, family) entry is an
    ERROR — silently reverting one arm to a default would time a
    tuned-vs-untuned comparison under a tuned-vs-tuned label (third S3
    review: partial tables must fail closed).
    """
    if table is None:
        return {
            "grouped": dict(PROVISIONAL_LAUNCH_CONFIG.lora_a),
            "splitk": dict(SPLIT_K_DEFAULT_CONFIG),
            "indexed": dict(INDEXED_DEFAULT_CONFIG),
        }
    cell = table.get(site, {}).get(str(rank))
    if cell is None:
        raise ValueError(f"config table has no entry for {site} rank {rank}")
    configs = {}
    for family in ("grouped", "splitk", "indexed"):
        if family not in cell:
            raise ValueError(
                f"config table entry for {site} rank {rank} lacks "
                f"{family!r} — a partial table cannot fail open"
            )
        configs[family] = _parse_config(cell[family])
    return configs


def _cell_arms(site: str, fixture: _LegFixture, aligned, raw, configs, cutedsl_plans):
    """Arm name -> (spec, route, config, cutedsl_plan) for one fixture.

    Triton families carry their tuned config and a None plan; cutedsl arms
    carry their prepared plan and record only seed-invariant geometry as
    config (per-seed values like m_max would trip the ledger's
    per-candidate cell-consistency check, correctly).
    """
    arms = {}
    for name in ("grouped", "splitk", "indexed"):
        spec = SPECS[(site, name)]
        route = raw if spec.ownership == "indexed" else aligned
        arms[name] = (spec, route, configs[name], None)
    for width, plan in cutedsl_plans.items():
        arms[f"cutedsl_tw{width}"] = (
            _arm_spec(site, f"cutedsl_tw{width}"),
            aligned,
            {"token_width": width, "output_width": plan.config.output_width},
            plan,
        )
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument(
        "--pairs",
        default="splitk,indexed",
        help="challenger arms decided against grouped per cell "
        "(triton families and/or cutedsl_twN P5 composites)",
    )
    parser.add_argument(
        "--t-grid",
        default=DEFAULT_T_GRID,
        help="comma list of T cells (default = the decode band; the P5 "
        "confirmation runs 256,2048,8192)",
    )
    parser.add_argument(
        "--config-table",
        default=None,
        help="JSON file {site: {rank: {family: config}}} of tuned configs",
    )
    parser.add_argument(
        "--allow-unverified-revision",
        action="store_true",
        help="permit tuned-table consumption when the running tree's git "
        "state cannot confirm --source-revision (file-synced pod trees); "
        "the suite name is suffixed _unverified so provenance stays honest",
    )
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    challengers = tuple(arguments.pairs.split(","))
    config_table = None
    if arguments.config_table:
        with open(arguments.config_table) as handle:
            config_table = json.load(handle)
        if not arguments.source_revision:
            raise ValueError(
                "--source-revision is required when consuming a config "
                "table (fifth S3 review: provenance validation must not "
                "be skippable)"
            )
        # Sixth S3 review: prefix-matching the revision let an H200 table
        # tune a GB300 run, dirty trees pass, and Triton upgrades go
        # unnoticed. Every provenance field the producer records is now
        # matched EXACTLY against this run's environment.
        meta = config_table.get("_meta", {})
        expected = {
            "source_revision": arguments.source_revision,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__),
            "triton_version": triton.__version__,
        }
        for field, want in expected.items():
            got = meta.get(field)
            if got != want:
                raise ValueError(
                    f"config table _meta[{field!r}] is {got!r}; this run "
                    f"requires {want!r} — regenerate the table on THIS "
                    "device/toolchain/revision (sixth S3 review)"
                )
        if "-dirty" in expected["source_revision"]:
            raise ValueError(
                "refusing a config table pinned to a dirty tree — commit "
                "first so the provenance is reproducible"
            )
        # Seventh S3 review: matching the table against the CALLER-PROVIDED
        # string alone lets a dirty checkout claim a clean revision. Verify
        # the claim against the tree that is actually running.
        actual = resolve_source_revision()
        if actual != arguments.source_revision:
            if not arguments.allow_unverified_revision:
                raise ValueError(
                    f"the running tree resolves to {actual!r}, not the "
                    f"claimed {arguments.source_revision!r} — commit/clean "
                    "the tree, or pass --allow-unverified-revision on a "
                    "file-synced pod tree (suite is then marked unverified)"
                )
            print(
                f"WARNING: revision claim {arguments.source_revision!r} "
                f"unverified (tree resolves to {actual!r}); suite marked "
                "unverified"
            )
    suite_name = "lora_a_crossover" if config_table else "lora_a_crossover_provisional"
    if (
        config_table
        and arguments.allow_unverified_revision
        and resolve_source_revision() != arguments.source_revision
    ):
        suite_name += "_unverified"
    suite = new_suite(suite_name, source_revision=arguments.source_revision)

    # samples[(site, rank, T)][arm] -> six medians in (seed, repeat) order;
    # records[...] -> the record ids behind them (ledger evidence).
    samples: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    records: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    t_grid = tuple(int(t) for t in arguments.t_grid.split(","))
    cutedsl_widths = tuple(
        _cutedsl_width(arm) for arm in challengers if arm.startswith("cutedsl_tw")
    )
    if cutedsl_widths:
        unsupported = set(cutedsl_widths) - set(supported_token_widths(device))
        if unsupported:
            raise ValueError(
                f"cutedsl token widths {sorted(unsupported)} are not "
                "supported on this arch — fail closed rather than time a "
                "silently different arm"
            )
    arm_names = ("grouped",) + challengers
    for rank in ranks:
        for num_tokens in t_grid:
            for seed in SEEDS:
                case = build_case(
                    device=str(device),
                    model_preset="qwen35_35b",
                    topology=Topology(tp_size=8, ep_size=8),
                    adapter_cell=ADAPTER_CELL,
                    route_generator="iid",
                    num_tokens=num_tokens,
                    active_rank=rank,
                    seed=seed,
                    source_revision=suite.source_revision,
                )
                fixture = _LegFixture(case, device)
                aligned = fixture.route(ROUTE_ALIGNED)
                raw = fixture.route(ROUTE_RAW)
                fixture.gate_split_ws = split_k_workspace(
                    fixture.gate_rank_out,
                    split_k=resolve_split_k(
                        fixture.a_gate_up, aligned, SPLIT_K_DEFAULT_CONFIG
                    ),
                )
                fixture.down_split_ws = split_k_workspace(
                    fixture.down_rank_out,
                    split_k=resolve_split_k(
                        fixture.a_down, aligned, SPLIT_K_DEFAULT_CONFIG
                    ),
                )
                cutedsl_plans = {}
                if cutedsl_widths:
                    fused = fixture.route(ROUTE_FUSED_IDS)
                    cutedsl_plans = {
                        width: build_cutedsl_lora_a_plan(
                            fused_route=fused,
                            gate_up_weight=fixture.a_gate_up,
                            down_weight=fixture.a_down,
                            config=CutedslAConfig(token_width=width),
                        )
                        for width in cutedsl_widths
                    }
                for site in ("gate_up", "down"):
                    configs = _resolve_configs(config_table, site, rank)
                    arms = _cell_arms(
                        site, fixture, aligned, raw, configs, cutedsl_plans
                    )
                    # The tuned split-K config may resolve a different split
                    # factor than the default the fixture pre-sized for.
                    split_resolved = resolve_split_k(
                        fixture.site_buffers(site)[1], aligned, configs["splitk"]
                    )
                    site_ws = split_k_workspace(
                        fixture.site_buffers(site)[2], split_k=split_resolved
                    )
                    if site == "gate_up":
                        fixture.gate_split_ws = site_ws
                    else:
                        fixture.down_split_ws = site_ws
                    baseline = site_baseline(fixture, site, aligned)
                    for name in arm_names:
                        # Correctness admission + JIT warm, un-timed.
                        spec, route, config, plan = arms[name]
                        admit_candidate(
                            fixture,
                            spec,
                            route,
                            config,
                            baseline=baseline,
                            label=(
                                f"{_arm_key(site, name)} T={num_tokens} "
                                f"r{rank} s{seed}"
                            ),
                            cutedsl_plan=plan,
                        )
                    for repeat in range(REPEATS):
                        ordered = arm_names if repeat % 2 == 0 else arm_names[::-1]
                        for arm in ordered:
                            spec, route, config, plan = arms[arm]
                            arm_params = {
                                "case_id": case.case_id,
                                "T": num_tokens,
                                "rank": rank,
                                "seed": seed,
                                "repeat": repeat,
                                "config": {key: value for key, value in config.items()},
                            }
                            if arm == "splitk":
                                arm_params["split_k_resolved"] = split_resolved
                            record = measure(
                                lambda s=spec, r=route, c=config, p=plan: run_site(
                                    fixture, s, r, c, cutedsl_plan=p
                                ),
                                suite=suite,
                                candidate=_arm_key(site, arm),
                                boundary=BOUNDARY_ISOLATED,
                                params=arm_params,
                                graph_replay=True,
                            )
                            cell = (site, rank, num_tokens)
                            samples[cell][arm].append(record.median_s)
                            records[cell][arm].append(record.record_id)

    # Decide every cell once, then site crossovers between ADJACENT decided
    # cells along BOTH axes — the pooled grids suggested T, but the seeded
    # rank-16 slice shows the flip can sit on the RANK axis (N = 2R moves
    # the split-K occupancy tiers), and §31.7 requires naming the axis that
    # actually drives the reversal.
    decisions: dict[tuple, object] = {}
    for challenger in challengers:
        for site in ("gate_up", "down"):
            for rank in ranks:
                for num_tokens in t_grid:
                    cell = (site, rank, num_tokens)
                    decision = decide_cell(
                        arm_a="grouped",
                        samples_a=samples[cell]["grouped"],
                        arm_b=challenger,
                        samples_b=samples[cell][challenger],
                    )
                    decisions[(challenger, *cell)] = decision
                    print(
                        f"{site:8s} r{rank:<4d} T={num_tokens:<4d} "
                        f"{challenger:8s} geo(g/{challenger[:1]})="
                        f"{decision.geo_a_over_b:.3f} -> "
                        f"{decision.winner or 'tied'}"
                    )

    def emit(challenger, site, low_cell, high_cell, axis, location, axis_param) -> bool:
        low = decisions[(challenger, *low_cell)]
        high = decisions[(challenger, *high_cell)]
        if low.winner is None or high.winner is None or low.winner == high.winner:
            return False
        suite.site_crossover(
            site=f"{site}_a",
            boundary=BOUNDARY_ISOLATED,
            candidates=(
                _arm_key(site, "grouped"),
                _arm_key(site, challenger),
            ),
            axis=axis,
            crossover_location=location,
            bracketing_low_record_ids=tuple(
                records[low_cell]["grouped"] + records[low_cell][challenger]
            ),
            bracketing_high_record_ids=tuple(
                records[high_cell]["grouped"] + records[high_cell][challenger]
            ),
            cache_state=CACHE_L2_HOT_GRAPH,
            axis_param=axis_param,
            # The OTHER grid dimension is the workload; per-arm tuning
            # configs are candidate-specific and validated per candidate.
            workload_params=("rank",) if axis_param == "T" else ("T",),
            notes=(
                f"{low.winner} wins the low cell (margin {low.margin():.3f}),"
                f" {high.winner} the high (margin {high.margin():.3f});"
                " seeded 3x2 interleaved, graph mode"
            ),
        )
        return True

    sited = 0
    for challenger in challengers:
        for site in ("gate_up", "down"):
            for rank in ranks:
                for t_low, t_high in zip(t_grid, t_grid[1:]):
                    sited += emit(
                        challenger,
                        site,
                        (site, rank, t_low),
                        (site, rank, t_high),
                        axis=f"num_tokens (grid fill at fixed rank={rank})",
                        location=f"T in ({t_low}, {t_high}]",
                        axis_param="T",
                    )
            for num_tokens in t_grid:
                for r_low, r_high in zip(ranks, ranks[1:]):
                    sited += emit(
                        challenger,
                        site,
                        (site, r_low, num_tokens),
                        (site, r_high, num_tokens),
                        axis=f"rank (split-K N tiers at fixed T={num_tokens})",
                        location=f"rank in ({r_low}, {r_high}]",
                        axis_param="rank",
                    )

    digest = write_suite(suite, arguments.output)
    print(
        f"{len(suite.records)} records, {sited} crossovers sited -> "
        f"{arguments.output} sha256 {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
