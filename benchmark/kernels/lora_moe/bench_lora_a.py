"""Step-3 LoRA-A schedule cells: per-stage isolated and route-inclusive SET.

Two cell families (plan §63.1 P2, measurement rules §33.1):

* ``per_stage`` — BOUNDARY_ISOLATED, one A GEMM per thunk, routes prebuilt
  OUTSIDE the timed region.  The pure-kernel comparison between schedule
  arms at one site.
* ``site_set`` — BOUNDARY_ROUTE_INCLUSIVE, one thunk = every route build the
  SCHEDULE needs plus all four LoRA GEMMs of one MoE leg (gate/up A+B,
  down A+B).  B stays the stock grouped kernel at both sites (B row
  ownership is Step 4), so EVERY schedule builds the aligned plan for B —
  which is exactly the §33.1 coupling: an indexed A site's no-plan advantage
  collapses to zero while any co-site still needs the plan.  This cell
  family measures that collapse instead of assuming it.

Execution identity is TYPED end to end (second S3 review): per-stage arms
are `LoraAExecutionSpec`s executed through the one exhaustive
`run_lora_a` dispatcher, site-set arms are `LegScheduleSpec` composites —
a record's candidate string names exactly what ran, and unsupported
combinations raise instead of falling through.

Every (arm, config, shape) is CORRECTNESS-ADMITTED before its first timing
record: one un-timed run must be finite and signal-gate-close to the
S1-tested grouped baseline on the valid pairs (second S3 review: timed
compile variants previously received no same-config check).

Timing values are BF16-random per case seed; ``token_lora_mapping`` is
cast to the serving dtype (int32, ``base_backend.py``) so route and
indexed-arm measurements read the bytes production reads.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_lora_a \
        --output lora_a_v1.json --source-revision <sha> [--ranks 16,64] \
        [--regimes decode_iid_64,prefill_2048] [--indexed-config bn16-bk64-w4-s3]
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kernels.lora_moe.cases import (
    AdapterCell,
    MoeLoraBenchCase,
    Topology,
    build_case,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.lora_a_candidates import (
    INDEXED_DEFAULT_CONFIG,
    SPLIT_K_DEFAULT_CONFIG,
    resolve_split_k,
    run_lora_a,
    split_k_workspace,
)
from benchmark.kernels.lora_moe.lora_a_cutedsl import (
    CutedslAConfig,
    build_cutedsl_lora_a_plan,
    supported_token_widths,
)
from benchmark.kernels.lora_moe.lora_a_execution import (
    LegScheduleSpec,
    LoraAExecutionSpec,
)
from benchmark.kernels.lora_moe.signal_gates import (
    require_delta_close,
    require_finite,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.bf16 import stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    ROUTE_FUSED_IDS,
    ROUTE_RAW,
    build_virtual_expert_routing,
)

# (regime, num_tokens, route_generator): fragmented decode and substantial
# prefill per the §14 Step-3 evidence requirements, on the primary anchor
# geometry (qwen35_35b at the EP8 local shape).
REGIMES: dict[str, tuple[int, str]] = {
    "decode_iid_16": (16, "iid"),
    "decode_iid_64": (64, "iid"),
    "decode_hotset_64": (64, "hotset_80_20"),
    "decode_iid_256": (256, "iid"),
    "prefill_2048": (2048, "iid"),
    "prefill_8192": (8192, "iid"),
}
RANKS = (16, 32, 64, 128)
ADAPTER_CELL = AdapterCell(active_adapters=4, include_base_rows=True, slot_capacity=8)

# Typed candidate identities: timing records carry spec keys, so arm
# identity is structured rather than a naming convention.
SPEC_GATE_GROUPED = LoraAExecutionSpec(site="gate_up", ownership="grouped")
SPEC_GATE_INDEXED = LoraAExecutionSpec(site="gate_up", ownership="indexed")
SPEC_GATE_SPLITK = LoraAExecutionSpec(
    site="gate_up", ownership="grouped", reduction="deterministic_split_k"
)
SPEC_DOWN_GROUPED = LoraAExecutionSpec(site="down", ownership="grouped")
SPEC_DOWN_INDEXED = LoraAExecutionSpec(site="down", ownership="indexed")
# P5 §21.3 funnel arms: the masked-grouped-GEMM composite. Token width is
# launch geometry (candidate-key suffix + params), not spec identity.
SPEC_GATE_CUTEDSL = LoraAExecutionSpec(
    site="gate_up", ownership="grouped", implementation="cutedsl"
)
SPEC_DOWN_CUTEDSL = LoraAExecutionSpec(
    site="down", ownership="grouped", implementation="cutedsl"
)
SPEC_DOWN_SPLITK = LoraAExecutionSpec(
    site="down", ownership="grouped", reduction="deterministic_split_k"
)
STAGE_SPECS = (
    SPEC_GATE_GROUPED,
    SPEC_GATE_INDEXED,
    SPEC_GATE_SPLITK,
    SPEC_DOWN_GROUPED,
    SPEC_DOWN_INDEXED,
    SPEC_DOWN_SPLITK,
)
SET_SPECS = tuple(
    LegScheduleSpec(gate_up=gate, down=down)
    for gate, down in (
        (SPEC_GATE_GROUPED, SPEC_DOWN_GROUPED),
        (SPEC_GATE_INDEXED, SPEC_DOWN_GROUPED),
        (SPEC_GATE_GROUPED, SPEC_DOWN_INDEXED),
        (SPEC_GATE_INDEXED, SPEC_DOWN_INDEXED),
        # The evidence-relevant mixed arm (third S3 review): split-K's only
        # surviving niche is the gate site while down stays grouped.
        (SPEC_GATE_SPLITK, SPEC_DOWN_GROUPED),
        (SPEC_GATE_SPLITK, SPEC_DOWN_SPLITK),
    )
)


def _parse_indexed_config(text: str) -> dict[str, int]:
    """``bn16-bk64-w4-s3`` -> the kernel config mapping."""
    parts = dict(
        (piece[:2] if piece[:2] in ("bn", "bk") else piece[:1], piece)
        for piece in text.split("-")
    )
    try:
        return {
            "BLOCK_SIZE_N": int(parts["bn"][2:]),
            "BLOCK_SIZE_K": int(parts["bk"][2:]),
            "num_warps": int(parts["w"][1:]),
            "num_stages": int(parts["s"][1:]),
        }
    except (KeyError, ValueError) as error:
        raise ValueError(f"cannot parse indexed config {text!r}") from error


class _LegFixture:
    """Device tensors and buffers for one (case, rank) LoRA leg."""

    def __init__(self, case: MoeLoraBenchCase, device: torch.device) -> None:
        tensors = materialize_case_tensors(case)
        self.case = case
        self.top_k = case.top_k
        self.num_pairs = case.num_tokens * case.top_k
        inter = case.intermediate_size_local
        hidden = case.moe_hidden_size
        r_phys = case.physical_rank
        self.hidden_states = tensors.hidden_states.to(device)
        self.topk_ids = tensors.topk_ids.to(device)
        # Serving parity: the LoRA batch backend materializes int32 mapping
        # metadata (base_backend.py); int64 here would double the bytes the
        # route builds and the indexed arm read per pair.
        self.token_slots = tensors.token_lora_mapping.to(torch.int32).to(device)
        self.lora_expert_map = (
            None
            if tensors.lora_expert_map is None
            else tensors.lora_expert_map.to(device)
        )
        self.a_gate_up = tensors.lora_a_gate_up.flatten(0, 1).to(device)
        self.b_gate_up = tensors.lora_b_gate_up.flatten(0, 1).to(device)
        self.a_down = tensors.lora_a_down.flatten(0, 1).to(device)
        self.b_down = tensors.lora_b_down.flatten(0, 1).to(device)
        # Timing inputs/buffers, allocated once (the boundary is the route
        # builds + GEMMs, not the runner's per-forward allocation policy).
        generator = torch.Generator(device="cpu").manual_seed(case.data_seed + 7)
        self.act_pair = (
            (torch.randn((self.num_pairs, inter), generator=generator) * 0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.gate_rank_out = torch.empty(
            (self.num_pairs, 2 * r_phys), dtype=torch.bfloat16, device=device
        )
        self.gate_up_delta = torch.empty(
            (self.num_pairs, 2 * inter), dtype=torch.bfloat16, device=device
        )
        self.down_rank_out = torch.empty(
            (self.num_pairs, r_phys), dtype=torch.bfloat16, device=device
        )
        self.down_delta = torch.empty(
            (self.num_pairs, hidden), dtype=torch.bfloat16, device=device
        )
        self.intermediate = inter
        # Filled by the driver once the aligned route (capacity) exists.
        self.gate_split_ws: torch.Tensor | None = None
        self.down_split_ws: torch.Tensor | None = None
        # Host mirror of the pair validity domain (the reference.py
        # convention) — admission gates compare valid rows only, because
        # sentinel rows are contractually undefined in A outputs.
        ids = tensors.topk_ids.to(torch.int64).reshape(-1)
        adapters = (
            tensors.token_lora_mapping.to(torch.int64)
            .reshape(-1, 1)
            .expand(-1, case.top_k)
            .reshape(-1)
        )
        if tensors.lora_expert_map is not None:
            table = tensors.lora_expert_map.to(torch.int64)
            in_map = (ids >= 0) & (ids < table.numel())
            ids = torch.where(
                in_map, table[ids.clamp(0, table.numel() - 1)], torch.tensor(-1)
            )
        self.valid_pairs = (
            (ids >= 0)
            & (ids < case.num_experts_local)
            & (adapters >= 0)
            & (adapters < case.slot_capacity)
        ).to(device)

    def route(self, view: str):
        return build_virtual_expert_routing(
            self.topk_ids,
            self.token_slots,
            lora_experts_per_adapter=self.case.num_experts_local,
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
            lora_expert_map=self.lora_expert_map,
            view=view,
        )

    def site_buffers(self, site: str):
        """(input, weight, output) for one A site."""
        if site == "gate_up":
            return self.hidden_states, self.a_gate_up, self.gate_rank_out
        return self.act_pair, self.a_down, self.down_rank_out


def run_site(
    fixture: _LegFixture,
    spec: LoraAExecutionSpec,
    route,
    config,
    cutedsl_plan=None,
) -> None:
    """One A-site execution through the exhaustive spec dispatcher."""
    input, weight, output = fixture.site_buffers(spec.site)
    workspace = (
        (fixture.gate_split_ws if spec.site == "gate_up" else fixture.down_split_ws)
        if spec.reduction == "deterministic_split_k"
        else None
    )
    run_lora_a(
        spec,
        input=input,
        weight=weight,
        output=output,
        routing=route,
        config=config,
        workspace=workspace,
        cutedsl_plan=cutedsl_plan,
    )


def admit_candidate(
    fixture: _LegFixture,
    spec: LoraAExecutionSpec,
    route,
    config,
    *,
    baseline: torch.Tensor,
    label: str,
    cutedsl_plan=None,
) -> None:
    """Correctness gate: one un-timed run vs the S1-tested grouped baseline.

    Runs the candidate into the site buffer and requires finiteness plus
    signal-gate closeness ON THE VALID PAIRS against the baseline computed
    with the production config.  A (config, shape, site) combination that
    fails here never produces a timing record.
    """
    _, _, output = fixture.site_buffers(spec.site)
    output.fill_(float("nan"))
    run_site(fixture, spec, route, config, cutedsl_plan=cutedsl_plan)
    valid = fixture.valid_pairs
    require_finite(output[valid], label=f"{label} finite on valid pairs")
    require_delta_close(
        output[valid].float(),
        baseline[valid].float(),
        gate_dtype=torch.bfloat16,
        label=f"{label} vs grouped baseline",
    )


def site_baseline(fixture: _LegFixture, site: str, route) -> torch.Tensor:
    """Valid-pair reference output from the production-config grouped path."""
    spec = SPEC_GATE_GROUPED if site == "gate_up" else SPEC_DOWN_GROUPED
    run_site(fixture, spec, route, PROVISIONAL_LAUNCH_CONFIG.lora_a)
    _, _, output = fixture.site_buffers(site)
    return output.clone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,64")
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--sections", default="per_stage,site_set")
    parser.add_argument("--indexed-config", default="bn16-bk64-w4-s3")
    parser.add_argument(
        "--cutedsl-token-widths",
        default="auto",
        help="comma list of masked-GEMM token widths for the P5 arms, or "
        "'auto' (all widths this arch supports), or 'off'",
    )
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)

    indexed_config = _parse_indexed_config(arguments.indexed_config)
    if indexed_config != INDEXED_DEFAULT_CONFIG:
        print(f"indexed config override: {indexed_config}")

    def config_for(spec: LoraAExecutionSpec):
        if spec.reduction == "deterministic_split_k":
            return SPLIT_K_DEFAULT_CONFIG
        if spec.ownership == "indexed":
            return indexed_config
        return PROVISIONAL_LAUNCH_CONFIG.lora_a

    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    regimes = tuple(arguments.regimes.split(","))
    sections = set(arguments.sections.split(","))
    if arguments.cutedsl_token_widths == "off":
        cutedsl_widths: tuple[int, ...] = ()
    elif arguments.cutedsl_token_widths == "auto":
        cutedsl_widths = supported_token_widths(device)
    else:
        cutedsl_widths = tuple(
            int(width) for width in arguments.cutedsl_token_widths.split(",")
        )
    suite = new_suite("lora_a_schedules", source_revision=arguments.source_revision)

    for regime in regimes:
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
            baselines = {
                site: site_baseline(fixture, site, aligned)
                for site in ("gate_up", "down")
            }
            # Records self-contain the exact configs (third S3 review) —
            # the resolved split factor too, since SPLIT_K=0 means
            # "heuristic" and the heuristic reads capacity.
            split_resolved = {
                "gate_up": resolve_split_k(
                    fixture.a_gate_up, aligned, SPLIT_K_DEFAULT_CONFIG
                ),
                "down": resolve_split_k(
                    fixture.a_down, aligned, SPLIT_K_DEFAULT_CONFIG
                ),
            }
            params = {
                "case_id": case.case_id,
                "regime": regime,
                "T": num_tokens,
                "P": fixture.num_pairs,
                "rank": rank,
                "configs": {
                    "grouped": dict(PROVISIONAL_LAUNCH_CONFIG.lora_a),
                    "indexed": dict(indexed_config),
                    "splitk": dict(SPLIT_K_DEFAULT_CONFIG),
                    "splitk_resolved": split_resolved,
                    "lora_b": dict(PROVISIONAL_LAUNCH_CONFIG.lora_b),
                },
            }

            def spec_route(spec: LoraAExecutionSpec):
                return raw if spec.ownership == "indexed" else aligned

            if "per_stage" in sections:
                for spec in STAGE_SPECS:
                    admit_candidate(
                        fixture,
                        spec,
                        spec_route(spec),
                        config_for(spec),
                        baseline=baselines[spec.site],
                        label=f"{spec.key()} {regime} r{rank}",
                    )
                for graph in (True, False):
                    for spec in STAGE_SPECS:
                        thunk = lambda s=spec: run_site(
                            fixture, s, spec_route(s), config_for(s)
                        )
                        thunk()
                        record = measure(
                            thunk,
                            suite=suite,
                            candidate=spec.key(),
                            boundary=BOUNDARY_ISOLATED,
                            params=params,
                            graph_replay=graph,
                        )
                        print(
                            f"per_stage {regime} r{rank} {spec.key()} "
                            f"{'graph' if graph else 'eager'} "
                            f"{record.median_s * 1e6:8.2f} us"
                        )

                # P5 §21.3 funnel arms: the masked-grouped composite per
                # token width, admitted like every candidate, plus the
                # GEMM-only diagnostic (the fused-gather IDEAL bound — the
                # one-sided number a family rejection is safe against).
                fused = fixture.route(ROUTE_FUSED_IDS)
                for token_width in cutedsl_widths:
                    try:
                        plan = build_cutedsl_lora_a_plan(
                            fused_route=fused,
                            gate_up_weight=fixture.a_gate_up,
                            down_weight=fixture.a_down,
                            config=CutedslAConfig(token_width=token_width),
                        )
                    except Exception as error:
                        print(
                            f"per_stage {regime} r{rank} cutedsl tw{token_width} "
                            f"REJECTED at build: {error}"
                        )
                        continue
                    cutedsl_params = dict(params)
                    cutedsl_params["cutedsl"] = {
                        "token_width": token_width,
                        "output_width": plan.config.output_width,
                        "m_max": plan.m_max,
                        "num_groups": plan.num_groups,
                    }
                    for spec in (SPEC_GATE_CUTEDSL, SPEC_DOWN_CUTEDSL):
                        admit_candidate(
                            fixture,
                            spec,
                            aligned,
                            PROVISIONAL_LAUNCH_CONFIG.lora_a,
                            baseline=baselines[spec.site],
                            label=f"{spec.key()}_tw{token_width} {regime} r{rank}",
                            cutedsl_plan=plan,
                        )
                        for graph in (True, False):
                            thunk = lambda s=spec, p=plan: run_site(
                                fixture,
                                s,
                                aligned,
                                PROVISIONAL_LAUNCH_CONFIG.lora_a,
                                cutedsl_plan=p,
                            )
                            thunk()
                            record = measure(
                                thunk,
                                suite=suite,
                                candidate=f"{spec.key()}_tw{token_width}",
                                boundary=BOUNDARY_ISOLATED,
                                params=cutedsl_params,
                                graph_replay=graph,
                            )
                            print(
                                f"per_stage {regime} r{rank} "
                                f"{spec.key()}_tw{token_width} "
                                f"{'graph' if graph else 'eager'} "
                                f"{record.median_s * 1e6:8.2f} us"
                            )
                        bound = measure(
                            lambda s=spec.site, p=plan: p.gemm_only(s),
                            suite=suite,
                            candidate=f"{spec.key()}_tw{token_width}_gemmbound",
                            boundary=BOUNDARY_ISOLATED,
                            params=cutedsl_params,
                            graph_replay=True,
                        )
                        print(
                            f"per_stage {regime} r{rank} "
                            f"{spec.key()}_tw{token_width}_gemmbound graph "
                            f"{bound.median_s * 1e6:8.2f} us"
                        )

            if "site_set" in sections:

                def leg(set_spec: LegScheduleSpec) -> None:
                    # Route bill of the SCHEDULE: B needs the aligned plan
                    # at both sites regardless, so it is built in-thunk for
                    # every arm; only the indexed schedule reads raw sources
                    # (a host-only wrapper — its price is the honest zero).
                    plan = fixture.route(ROUTE_ALIGNED)
                    gate_route = (
                        fixture.route(ROUTE_RAW)
                        if set_spec.gate_up.ownership == "indexed"
                        else plan
                    )
                    down_route = (
                        fixture.route(ROUTE_RAW)
                        if set_spec.down.ownership == "indexed"
                        else plan
                    )
                    run_site(
                        fixture,
                        set_spec.gate_up,
                        gate_route,
                        config_for(set_spec.gate_up),
                    )
                    stock_grouped_lora_b(
                        fixture.gate_rank_out,
                        fixture.b_gate_up,
                        fixture.gate_up_delta,
                        plan,
                        destination_offsets=(0, fixture.intermediate),
                        config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
                    )
                    run_site(
                        fixture,
                        set_spec.down,
                        down_route,
                        config_for(set_spec.down),
                    )
                    stock_grouped_lora_b(
                        fixture.down_rank_out,
                        fixture.b_down,
                        fixture.down_delta,
                        plan,
                        destination_offsets=(0,),
                        config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
                    )

                # Admission: every set arm's LEG OUTPUTS must match the
                # all-grouped leg (B zero-overwrites sentinel destinations,
                # so the full buffers are comparable).
                leg(SET_SPECS[0])
                reference_gate = fixture.gate_up_delta.clone()
                reference_down = fixture.down_delta.clone()
                for set_spec in SET_SPECS[1:]:
                    leg(set_spec)
                    require_delta_close(
                        fixture.gate_up_delta.float(),
                        reference_gate.float(),
                        gate_dtype=torch.bfloat16,
                        label=f"{set_spec.key()} gate/up delta {regime} r{rank}",
                    )
                    require_delta_close(
                        fixture.down_delta.float(),
                        reference_down.float(),
                        gate_dtype=torch.bfloat16,
                        label=f"{set_spec.key()} down delta {regime} r{rank}",
                    )

                for graph in (True, False):
                    for set_spec in SET_SPECS:
                        thunk = lambda s=set_spec: leg(s)
                        thunk()
                        record = measure(
                            thunk,
                            suite=suite,
                            candidate=set_spec.key(),
                            boundary=BOUNDARY_ROUTE_INCLUSIVE,
                            params=params,
                            graph_replay=graph,
                        )
                        print(
                            f"site_set  {regime} r{rank} {set_spec.key()} "
                            f"{'graph' if graph else 'eager'} "
                            f"{record.median_s * 1e6:8.2f} us"
                        )

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
