"""Provider pipeline benchmark: DeepGEMM vs CuTeDSL, matched-base, archived.

Gate-2 review remediation (plan section 57, revised on the second pass): the
D2/D3 provider ruling's numbers previously existed only as prose; this driver
is the committed, content-addressed producer. Three sections:

* ``pipeline``: the production forward (dispatch built once, thunk =
  ``runner.run`` + ``dispatcher.combine``) at ``complete_local_moe``, per
  regime — decode/mid/prefill on the E=32 study geometry, big_prefill on
  E=128. Each provider is measured in TWO arms: LoRA-active and its own
  ``disable_lora`` base-only forward, and ``pair_with_base`` attaches the
  active record to ITS OWN base arm — the plan-section-14 matched-base
  denominator (``ratio_to_base`` = active/base = the provider's LoRA
  overhead). The CROSS-PROVIDER comparison is a different relationship and is
  deliberately NOT expressed through ``base_record_id``: derive it from the
  JSON by matching records on identical ``params`` with the other
  ``candidate`` (the driver prints these deep/cute ratios as it runs).
* ``skew``: iid / hotset_80_20 / one_hot x T route-family cells, same
  double-arm discipline.
* ``per_stage``: each provider's S2/S4 GEMM alone on a prepared workspace
  with balanced m-rows-per-expert routes, at the ``isolated`` boundary (the
  preparation is deliberately NOT timed here; the pipeline section carries
  it). Candidate axes reproduce the study's siting experiments from this
  committed producer: CuTeDSL at each compiled tile width (8/64/128, via the
  provider's ``force_token_width`` lab hook) sites the narrow/wide/xwide
  crossovers, and DeepGEMM under an ``expected_m_hint`` sweep reproduces the
  plan-section-53.3 retune experiment (runnable on H200 with
  ``--providers deepgemm``).

Pipeline arms construct through the production selector
(``SGLANG_LORA_MOE_BASE_PROVIDER``); per-stage arms construct the provider
classes directly because the candidate axes are lab hooks. Repeats interleave
arm order so clock drift cannot systematically favor one engine.

Usage:
    python3 -m benchmark.kernels.lora_moe.bench_provider_pipeline \
        --output provider_pipeline_v2.json --source-revision <sha>
    # H200 (no SM100 CuTeDSL): --providers deepgemm --sections per_stage
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.cases import (
    AdapterCell,
    build_case,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.production_runner import (
    ensure_single_rank_distributed,
    prepare_production_forward,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_COMPLETE_LOCAL_MOE,
    BOUNDARY_ISOLATED,
    measure,
    new_suite,
    pair_with_base,
    write_suite,
)

REPEATS = 2
TOP_K = 8

# The section 53/55 study regimes, on the presets added for this driver.
PIPELINE_REGIMES = (
    ("decode", "bench_provider_e32", 16),
    ("mid", "bench_provider_e32", 256),
    ("prefill", "bench_provider_e32", 2048),
    ("big_prefill", "bench_provider_e128", 8192),
    # E=128 small-m: measures the pipeline bound behind the one-width-per-
    # forward ruling (expected_m ~= 16 on the e128 geometry).
    ("e128_small", "bench_provider_e128", 256),
    # E=128 expected_m ~= 192: the regime where the H200 default heuristic
    # loses ~13% per stage to small hints (4th review pass).
    ("e128_m192", "bench_provider_e128", 3072),
)
SKEW_FAMILIES = ("iid", "hotset_80_20", "one_hot")
SKEW_TOKENS = (16, 256, 2048)
# Representative adapter occupancy; the LoRA legs are identical across
# providers, so any fixed cell is a fair pipeline constant.
ADAPTER_CELL = AdapterCell(active_adapters=4, include_base_rows=True, slot_capacity=32)
ACTIVE_RANK = 16

PER_STAGE_EXPERTS = (32, 128)
# 96/112/128/192 flank the provider's xwide threshold so the wide->xwide
# transition is LOCATED by this producer (112 = the [96, 128) interior point;
# the transition is non-monotonic, so interior evidence beats extrapolation).
PER_STAGE_M = (1, 16, 64, 96, 112, 128, 192, 256)
PER_STAGE_HIDDEN = 512
PER_STAGE_INTERMEDIATE = {32: 256, 128: 512}
CUTE_TILE_WIDTHS = (8, 64, 128)
DEFAULT_DEEP_HINTS = (None, 1, 16, 64, 256)


def _cutedsl_supported() -> bool:
    return torch.cuda.get_device_capability() >= (9, 0)


def _pipeline_case(device: str, preset: str, family: str, num_tokens: int, rev: str):
    return build_case(
        device=device,
        model_preset=preset,
        adapter_cell=ADAPTER_CELL,
        route_generator=family,
        num_tokens=num_tokens,
        active_rank=ACTIVE_RANK,
        base_provider="deep_gemm_masked",
        source_revision=rev,
        seed=11,
    )


def _production_forward_thunk(
    case, tensors, *, device, disable_lora, provider_kwargs=None
):
    runner, dispatcher, batch, dispatch_output = prepare_production_forward(
        case,
        tensors,
        device=device,
        disable_lora=disable_lora,
        provider_kwargs=provider_kwargs,
    )

    def forward():
        return dispatcher.combine(
            runner.run(dispatch_output, batch, output_dtype=torch.bfloat16)
        )

    return forward


def _measure_pipeline_cell(
    suite,
    *,
    providers: tuple[str, ...],
    label: str,
    case,
    tensors,
    device: torch.device,
    mode: str,
    params: dict,
) -> None:
    """One cell: (provider x {active, base}) arms, REPEATS interleaved.

    Pairs each provider's active record with ITS OWN base record (the
    matched-base denominator); prints the cross-provider active ratio.
    """
    from sglang.srt.environ import envs

    thunks = {}
    for provider in providers:
        base_name, provider_kwargs = _provider_spec(provider)
        with envs.SGLANG_LORA_MOE_BASE_PROVIDER.override(base_name):
            for arm, disable_lora in (("active", False), ("base", True)):
                thunks[(provider, arm)] = _production_forward_thunk(
                    case,
                    tensors,
                    device=device,
                    disable_lora=disable_lora,
                    provider_kwargs=provider_kwargs,
                )

    medians: dict[tuple[str, str], list[float]] = defaultdict(list)
    eager_kwargs = {} if mode == "graph" else {"warmup_iters": 20, "replay_iters": 200}
    for repeat in range(REPEATS):
        arm_order = list(thunks)
        if repeat % 2:
            arm_order.reverse()
        cell_records: dict[tuple[str, str], tuple[int, object]] = {}
        for provider, arm in arm_order:
            rec = measure(
                thunks[(provider, arm)],
                suite=suite,
                candidate=f"pipeline_{provider}_{arm}",
                boundary=BOUNDARY_COMPLETE_LOCAL_MOE,
                params={**params, "repeat": repeat, "mode": mode, "arm": arm},
                graph_replay=(mode == "graph"),
                **eager_kwargs,
            )
            cell_records[(provider, arm)] = (len(suite.records) - 1, rec)
            medians[(provider, arm)].append(rec.median_s)
        for provider in providers:
            active_index, active_rec = cell_records[(provider, "active")]
            suite.records[active_index] = pair_with_base(
                active_rec, cell_records[(provider, "base")][1]
            )

    def geo(top: tuple[str, str], bottom: tuple[str, str]) -> float:
        value = 1.0
        for a, b in zip(medians[top], medians[bottom]):
            value *= a / b
        return value ** (1 / REPEATS)

    lora_costs = "  ".join(
        f"{p} lora {geo((p, 'active'), (p, 'base')):.3f}x" for p in providers
    )
    cross = (
        f"  {providers[0]}/{providers[1]} "
        f"{geo((providers[0], 'active'), (providers[1], 'active')):.3f}x"
        if len(providers) == 2
        else ""
    )
    print(f"  {label:<28} {mode:<6} {lora_costs}{cross}", flush=True)


def _balanced_topk_ids(num_experts: int, rows_per_expert: int, device) -> torch.Tensor:
    """Exactly ``rows_per_expert`` pairs per expert, [T, TOP_K] int32."""
    pairs = torch.arange(num_experts * rows_per_expert, dtype=torch.int32) % num_experts
    return pairs.view(-1, TOP_K).to(device)


def _provider_spec(name: str) -> tuple[str, dict]:
    """``deepgemm_em<h>`` selects deepgemm with an expected_m_hint lab arm."""
    if name.startswith("deepgemm_em"):
        return "deepgemm", {"expected_m_hint": int(name[len("deepgemm_em") :])}
    return name, {}


def _per_stage_arms(
    providers: tuple[str, ...], quant_info, deep_hints: tuple[int | None, ...]
) -> dict[str, object]:
    """Candidate providers for the isolated-GEMM axes, keyed by arm name."""
    arms: dict[str, object] = {}
    if "deepgemm" in providers:
        from sglang.srt.lora.sgl_lora.base_gemm_provider.deep_gemm_bf16 import (
            DeepGemmBf16Provider,
        )

        for hint in deep_hints:
            name = "deepgemm" if hint is None else f"deepgemm_em{hint}"
            arms[name] = DeepGemmBf16Provider(quant_info, expected_m_hint=hint)
    if "cutedsl" in providers:
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_bf16 import (
            CuteDslBf16Provider,
        )

        widths = CUTE_TILE_WIDTHS
        if torch.cuda.get_device_capability() < (10, 0):
            # SM90 compiles only the wide/xwide pair (no N=8 WGMMA tile yet).
            widths = tuple(w for w in CUTE_TILE_WIDTHS if w >= 64)
        for width in widths:
            arms[f"cutedsl_tw{width}"] = CuteDslBf16Provider(
                quant_info, force_token_width=width
            )
    return arms


def _measure_per_stage(
    suite,
    *,
    providers: tuple[str, ...],
    device: torch.device,
    deep_hints: tuple[int | None, ...],
):
    """Isolated S2/S4 GEMMs across the tile-width and expected_m-hint axes."""
    from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

    generator = torch.Generator(device="cpu").manual_seed(11)

    for num_experts in PER_STAGE_EXPERTS:
        hidden = PER_STAGE_HIDDEN
        intermediate = PER_STAGE_INTERMEDIATE[num_experts]
        w13 = torch.randn(
            (num_experts, 2 * intermediate, hidden), generator=generator
        ).to(device=device, dtype=torch.bfloat16)
        w2 = torch.randn((num_experts, hidden, intermediate), generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        quant_info = SglLoraBf16QuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            num_local_experts=num_experts,
            intermediate_size=intermediate,
            hidden_size=hidden,
        )
        arms = _per_stage_arms(providers, quant_info, deep_hints)
        for rows in PER_STAGE_M:
            num_tokens = num_experts * rows // TOP_K
            hidden_states = torch.randn((num_tokens, hidden), generator=generator).to(
                device=device, dtype=torch.bfloat16
            )
            topk_ids = _balanced_topk_ids(num_experts, rows, device)
            base_params = {
                "E": num_experts,
                "H": hidden,
                "I": intermediate,
                "m_per_expert": rows,
                "route": "balanced",
            }
            thunks: dict[tuple[str, str], object] = {}
            for name, provider in arms.items():
                ws = provider.prepare(hidden_states, topk_ids, TOP_K)
                gateup_out = torch.empty(
                    provider.gateup_out_shape(ws),
                    dtype=torch.bfloat16,
                    device=device,
                )
                act_out = torch.randn(
                    provider.act_out_shape(ws), generator=generator
                ).to(device=device, dtype=torch.bfloat16)
                down_out = torch.empty(
                    provider.down_out_shape(ws),
                    dtype=torch.bfloat16,
                    device=device,
                )
                thunks[(name, "gateup")] = (
                    lambda p=provider, w=ws, o=gateup_out: p.gateup(w, o)
                )
                thunks[(name, "down")] = (
                    lambda p=provider, w=ws, a=act_out, o=down_out: p.down(w, a, o)
                )
            for stage in ("gateup", "down"):
                medians: dict[str, list[float]] = defaultdict(list)
                for repeat in range(REPEATS):
                    ordered = list(arms)
                    if repeat % 2:
                        ordered.reverse()
                    for name in ordered:
                        rec = measure(
                            thunks[(name, stage)],
                            suite=suite,
                            candidate=f"stage_{stage}_{name}",
                            boundary=BOUNDARY_ISOLATED,
                            params={**base_params, "stage": stage, "repeat": repeat},
                        )
                        medians[name].append(rec.median_s)
                row = "  ".join(
                    f"{name} {min(times) * 1e6:7.2f}us"
                    for name, times in medians.items()
                )
                print(f"  {stage:<6} E={num_experts:<4} m={rows:<4} {row}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--providers",
        default="deepgemm,cutedsl",
        help="comma list; drop cutedsl on pre-SM100 devices",
    )
    parser.add_argument(
        "--sections",
        default="pipeline,skew,per_stage",
        help="comma list of sections to run",
    )
    parser.add_argument(
        "--deep-hints",
        default="default,1,16,64,256",
        help="comma list of expected_m hints for the per-stage DeepGEMM axis; "
        "'default' means the kernel's own heuristic",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    providers = tuple(name.strip() for name in args.providers.split(",") if name)
    if (
        any(_provider_spec(name)[0] == "cutedsl" for name in providers)
        and not _cutedsl_supported()
    ):
        raise SystemExit("cutedsl requires SM90+; pass --providers deepgemm")
    deep_hints = tuple(
        None if token in ("default", "none") else int(token)
        for token in (t.strip() for t in args.deep_hints.split(","))
        if token
    )
    sections = {name.strip() for name in args.sections.split(",") if name}
    ensure_single_rank_distributed()
    suite = new_suite("provider_pipeline", source_revision=args.source_revision)
    rev = suite.source_revision

    if "pipeline" in sections:
        print("== complete pipeline regimes ==", flush=True)
        for label, preset, num_tokens in PIPELINE_REGIMES:
            case = _pipeline_case(str(device), preset, "iid", num_tokens, rev)
            tensors = materialize_case_tensors(case)
            params = {
                "regime": label,
                "preset": preset,
                "num_tokens": num_tokens,
                "family": "iid",
                "case_id": case.case_id,
            }
            for mode in ("graph", "eager"):
                _measure_pipeline_cell(
                    suite,
                    providers=providers,
                    label=f"{label} T={num_tokens}",
                    case=case,
                    tensors=tensors,
                    device=device,
                    mode=mode,
                    params=params,
                )

    if "skew" in sections:
        print("== route-family skew cells (E=32) ==", flush=True)
        for family in SKEW_FAMILIES:
            for num_tokens in SKEW_TOKENS:
                case = _pipeline_case(
                    str(device), "bench_provider_e32", family, num_tokens, rev
                )
                tensors = materialize_case_tensors(case)
                params = {
                    "regime": "skew",
                    "preset": "bench_provider_e32",
                    "num_tokens": num_tokens,
                    "family": family,
                    "case_id": case.case_id,
                }
                _measure_pipeline_cell(
                    suite,
                    providers=providers,
                    label=f"{family} T={num_tokens}",
                    case=case,
                    tensors=tensors,
                    device=device,
                    mode="graph",
                    params=params,
                )

    if "per_stage" in sections:
        print("== per-stage isolated GEMM axes ==", flush=True)
        _measure_per_stage(
            suite, providers=providers, device=device, deep_hints=deep_hints
        )

    digest = write_suite(suite, args.output)
    print(f"\n{len(suite.records)} records -> {args.output} (sha256 {digest[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
