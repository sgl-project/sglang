"""LoRA-A family tuning sweeps and the whole-site M-tile comparison.

Three sections (plan §63.1 P3 as amended by the second S3 review):

* ``grouped_grid`` / ``indexed_grid`` / ``splitk_grid`` — isolated
  per-family config sweeps at the production BLOCK_M=16 plan.  Each family
  is tuned INDEPENDENTLY so cross-family comparisons (the crossover
  harness) run tuned-vs-tuned, not provisional-vs-default; the winners can
  be distilled into a config table with ``--emit-config-table``.
* ``m_axis`` — the M-tile question at the boundary production actually
  faces (second S3 review: the earlier equal-charge form measured neither
  real alternative).  One whole-site thunk per arm:

      M16 arm:  plan16 = route(16); A(plan16); B(plan16)
      M!=16 arm: plan16 = route(16); planM = route(M); A(planM); B(plan16)

  — the M=16 arm shares one plan between A and B exactly as the runner
  does, and the M!=16 arm pays ITS OWN extra plan while B keeps the
  16-plan.  The FULL config grid runs at every M (a top-M16 seeding can
  miss configs whose optimum changes with M).

Every (family, config, shape, site) is correctness-admitted against the
production-config grouped baseline before its first timing record.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_lora_a_tiles \
        --output lora_a_tiles_v2.json --source-revision <sha> \
        [--ranks 16,32,64,128] [--sections grouped_grid,indexed_grid,splitk_grid,m_axis] \
        [--emit-config-table tuned_configs.json]
"""

from __future__ import annotations

import argparse
import json
from itertools import product

import msgspec
import torch
import triton

from benchmark.kernels.lora_moe.bench_lora_a import (
    ADAPTER_CELL,
    REGIMES,
    _LegFixture,
    admit_candidate,
    run_site,
    site_baseline,
)
from benchmark.kernels.lora_moe.cases import Topology, build_case
from benchmark.kernels.lora_moe.lora_a_candidates import (
    resolve_split_k,
    split_k_workspace,
)
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.bf16 import stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, ROUTE_RAW

SWEEP_REGIMES = (
    "decode_iid_16",
    "decode_iid_64",
    "decode_iid_256",
    "prefill_2048",
)
M_AXIS_REGIMES = ("decode_iid_64", "prefill_2048")
M_GRID = (16, 32, 64)

GROUPED_GRID = tuple(
    {
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 8,
        "num_warps": w,
        "num_stages": s,
    }
    for bn, bk, w, s in product((16, 32, 64), (32, 64, 128), (4, 8), (2, 3, 4))
)
INDEXED_GRID = tuple(
    {"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "num_warps": w, "num_stages": s}
    for bn, bk, w, s in product((8, 16, 32), (32, 64, 128), (2, 4, 8), (2, 3))
)
SPLITK_GRID = tuple(
    {
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 8,
        "num_warps": w,
        "num_stages": s,
        "SPLIT_K": k,
    }
    for bn, bk, w, s, k in product(
        (16, 32, 64), (32, 64, 128), (4, 8), (2, 3), (0, 2, 4, 8)
    )
)

FAMILY_SECTIONS = {
    "grouped_grid": ("grouped", GROUPED_GRID),
    "indexed_grid": ("indexed", INDEXED_GRID),
    "splitk_grid": ("splitk", SPLITK_GRID),
}


def _spec(site: str, family: str) -> LoraAExecutionSpec:
    if family == "splitk":
        return LoraAExecutionSpec(
            site=site, ownership="grouped", reduction="deterministic_split_k"
        )
    return LoraAExecutionSpec(site=site, ownership=family)


def _config_key(family: str, config: dict) -> str:
    key = (
        f"bn{config['BLOCK_SIZE_N']}-bk{config['BLOCK_SIZE_K']}"
        f"-w{config['num_warps']}-s{config['num_stages']}"
    )
    if family == "splitk":
        key += f"-k{config['SPLIT_K']}"
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument(
        "--sections", default="grouped_grid,indexed_grid,splitk_grid,m_axis"
    )
    parser.add_argument("--emit-config-table", default=None)
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    sections = set(arguments.sections.split(","))
    suite = new_suite("lora_a_tiles", source_revision=arguments.source_revision)

    # medians[(site, rank, family)][config_key][regime] -> median seconds.
    # The config table is a STATIC-config compromise across the decode band,
    # chosen by GEOMEAN NORMALIZED SLOWDOWN (config median / cell best, geo
    # mean over the decode cells) — the third S3 review showed a raw-latency
    # comparison across T cells degenerates to "the T=64 winner everywhere"
    # because small-T medians are absolutely smaller.
    medians: dict[tuple, dict[str, dict[str, float]]] = {}

    for regime in SWEEP_REGIMES:
        num_tokens, generator_name = REGIMES[regime]
        for rank in ranks:
            case = build_case(
                device=str(device),
                model_preset="qwen35_35b",
                topology=Topology(tp_size=8, ep_size=8),
                adapter_cell=ADAPTER_CELL,
                route_generator=generator_name,
                num_tokens=num_tokens,
                active_rank=rank,
                source_revision=suite.source_revision,
            )
            fixture = _LegFixture(case, device)
            aligned16 = fixture.route(ROUTE_ALIGNED)
            raw = fixture.route(ROUTE_RAW)
            base_params = {
                "case_id": case.case_id,
                "regime": regime,
                "T": num_tokens,
                "rank": rank,
            }
            baselines = {
                site: site_baseline(fixture, site, aligned16)
                for site in ("gate_up", "down")
            }

            for section in sections & set(FAMILY_SECTIONS):
                family, grid = FAMILY_SECTIONS[section]
                for site in ("gate_up", "down"):
                    spec = _spec(site, family)
                    route = raw if family == "indexed" else aligned16
                    best: tuple[float, str] | None = None
                    for config in grid:
                        if family == "splitk":
                            split = resolve_split_k(
                                fixture.site_buffers(site)[1], aligned16, config
                            )
                            workspace = split_k_workspace(
                                fixture.site_buffers(site)[2], split_k=split
                            )
                            if site == "gate_up":
                                fixture.gate_split_ws = workspace
                            else:
                                fixture.down_split_ws = workspace
                        key = _config_key(family, config)
                        label = f"{spec.key()}_{key} {regime} r{rank}"
                        admit_candidate(
                            fixture,
                            spec,
                            route,
                            config,
                            baseline=baselines[site],
                            label=label,
                        )
                        record = measure(
                            lambda s=spec, r=route, c=config: run_site(
                                fixture, s, r, c
                            ),
                            suite=suite,
                            candidate=f"{spec.key()}_{key}",
                            boundary=BOUNDARY_ISOLATED,
                            params={
                                **base_params,
                                "family": family,
                                "config": dict(config),
                            },
                            graph_replay=True,
                        )
                        if best is None or record.median_s < best[0]:
                            best = (record.median_s, key)
                        if "decode" in regime:
                            cell = medians.setdefault((site, rank, family), {})
                            cell.setdefault(key, {})[regime] = record.median_s
                    print(
                        f"{section} {regime} r{rank} {site}: best {best[1]} "
                        f"{best[0] * 1e6:7.2f} us"
                    )

            if "m_axis" in sections and regime in M_AXIS_REGIMES:
                m_axis_baselines: dict[tuple, torch.Tensor] = {}
                for site in ("gate_up", "down"):
                    spec = _spec(site, "grouped")
                    for config in GROUPED_GRID:
                        for block_m in M_GRID:

                            def whole_site(m=block_m, cfg=config, s=site):
                                plan16 = fixture.route(ROUTE_ALIGNED)
                                if m == 16:
                                    a_plan = plan16
                                else:
                                    # §40.5 hazard, observed as an OOB in B:
                                    # on the FUSED align path two same-bucket
                                    # plans share the cached
                                    # num_pairs_post_padded scalar, so the
                                    # sibling build overwrites the count
                                    # plan16 still aliases and B walks past
                                    # plan16's capacity. Production never
                                    # holds two same-bucket plans alive;
                                    # this driver must, so it snapshots the
                                    # scalar before the second build.
                                    plan16 = msgspec.structs.replace(
                                        plan16,
                                        maybe_num_pairs_post_padded=(
                                            plan16.num_pairs_post_padded.clone()
                                        ),
                                    )
                                    a_plan = build_virtual_expert_routing_at(fixture, m)
                                run_site(fixture, spec, a_plan, cfg)
                                output = fixture.site_buffers(s)[2]
                                if s == "gate_up":
                                    stock_grouped_lora_b(
                                        output,
                                        fixture.b_gate_up,
                                        fixture.gate_up_delta,
                                        plan16,
                                        destination_offsets=(
                                            0,
                                            fixture.intermediate,
                                        ),
                                        config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
                                    )
                                else:
                                    stock_grouped_lora_b(
                                        output,
                                        fixture.b_down,
                                        fixture.down_delta,
                                        plan16,
                                        destination_offsets=(0,),
                                        config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
                                    )

                            # M-axis correctness admission (third S3
                            # review): the A kernel's per-element K
                            # reduction is the same serial BLOCK_K loop at
                            # every M, so for a FIXED config the complete
                            # A+B leg output must be BITWISE M-invariant.
                            # The M16 leg of this config is the baseline.
                            delta_buffer = (
                                fixture.gate_up_delta
                                if site == "gate_up"
                                else fixture.down_delta
                            )
                            if block_m == 16:
                                delta_buffer.fill_(71.0)
                                whole_site()
                                m16_delta = delta_buffer.clone()
                                m_axis_baselines[
                                    (site, _config_key("grouped", config))
                                ] = m16_delta
                            else:
                                delta_buffer.fill_(-3.0)
                                whole_site()
                                baseline = m_axis_baselines[
                                    (site, _config_key("grouped", config))
                                ]
                                if not torch.equal(delta_buffer, baseline):
                                    raise AssertionError(
                                        f"M={block_m} leg output diverges "
                                        f"bitwise from the M16 baseline at "
                                        f"{site} {regime} r{rank} "
                                        f"{_config_key('grouped', config)}"
                                    )
                            measure(
                                whole_site,
                                suite=suite,
                                candidate=(
                                    f"{site}_wholesite_m{block_m}_"
                                    f"{_config_key('grouped', config)}"
                                ),
                                boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                params={
                                    **base_params,
                                    "block_m": block_m,
                                    "config": dict(config),
                                },
                                graph_replay=True,
                            )

    if "m_axis" in sections:
        by_cell: dict[tuple, list] = {}
        for record in suite.records:
            if record.boundary != BOUNDARY_ROUTE_INCLUSIVE:
                continue
            key = (
                record.candidate.split("_wholesite_")[0],
                record.params["regime"],
                record.params["rank"],
            )
            by_cell.setdefault(key, []).append(record)
        for (site, regime, rank), cell_records in sorted(by_cell.items()):
            best = min(cell_records, key=lambda record: record.median_s)
            m16 = [r for r in cell_records if r.params["block_m"] == 16]
            m16_best = min(m16, key=lambda record: record.median_s)
            print(
                f"m_axis {regime} r{rank} {site}: best {best.candidate} "
                f"{best.median_s * 1e6:7.2f} us vs best-M16 "
                f"{m16_best.median_s * 1e6:7.2f} us "
                f"({m16_best.median_s / best.median_s:.2f}x)"
            )

    if arguments.emit_config_table:
        table: dict = {
            "_meta": {
                "objective": "geomean normalized slowdown over decode cells "
                + str([r for r in SWEEP_REGIMES if "decode" in r]),
                "source_revision": suite.source_revision,
                "device_name": suite.device_name,
                "torch_version": suite.torch_version,
                "triton_version": triton.__version__,
            }
        }
        for (site, rank, family), per_config in sorted(medians.items()):
            regimes_here = sorted(
                {regime for values in per_config.values() for regime in values}
            )
            best_per_regime = {
                regime: min(values[regime] for values in per_config.values())
                for regime in regimes_here
            }
            scored = []
            for key, values in per_config.items():
                if set(values) != set(regimes_here):
                    continue  # config missing a cell cannot be scored fairly
                slowdown = 1.0
                for regime in regimes_here:
                    slowdown *= values[regime] / best_per_regime[regime]
                slowdown **= 1 / len(regimes_here)
                scored.append((slowdown, key))
            scored.sort()
            objective, key = scored[0]
            table.setdefault(site, {}).setdefault(str(rank), {})[family] = key
            print(
                f"table {site} r{rank} {family}: {key} "
                f"(geomean slowdown {objective:.4f}; runner-up "
                f"{scored[1][1]} at {scored[1][0]:.4f})"
            )
        with open(arguments.emit_config_table, "w") as handle:
            json.dump(table, handle, indent=2)
        print(f"decode-band config table -> {arguments.emit_config_table}")

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


def build_virtual_expert_routing_at(fixture: _LegFixture, block_m: int):
    from sglang.srt.lora.sgl_lora.routing import build_virtual_expert_routing

    return build_virtual_expert_routing(
        fixture.topk_ids,
        fixture.token_slots,
        lora_experts_per_adapter=fixture.case.num_experts_local,
        max_loras=fixture.case.slot_capacity,
        block_size=block_m,
        lora_expert_map=fixture.lora_expert_map,
        view=ROUTE_ALIGNED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
