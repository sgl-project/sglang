"""Tune MoE LoRA launch tiles for explicit resident, single-GPU workloads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lora_tuning import compare, measure


@dataclass(frozen=True)
class Case:
    name: str
    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    tokens: int
    rank: int
    top_k: int
    slots: int = 2
    tp_size: int = 1
    ep_size: int = 1
    quant: str = "bf16"
    vendor: str = "cutedsl"
    layout: str = "per_expert"
    phase: str = "decode"
    mode: str = "eager"
    routing: str = "balanced"
    traffic: str = "mixed"
    routed_scaling_factor: float = 0.75

    @classmethod
    def from_dict(cls, value):
        case = cls(**value)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_local_experts",
            "tokens",
            "rank",
            "top_k",
            "slots",
            "tp_size",
            "ep_size",
        ):
            number = getattr(case, name)
            if type(number) is not int or number < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(case.name, str) or not case.name:
            raise ValueError("case name must be nonempty")
        if case.top_k > case.num_local_experts:
            raise ValueError(
                "top_k exceeds resident experts; remote expert dispatch is unsupported"
            )
        if case.tp_size % case.ep_size:
            raise ValueError(
                "total TP must be divisible by EP (MoE DP is fixed at one)"
            )
        if case.rank > 256 or case.rank % 8:
            raise ValueError("physical rank must be a multiple of eight <= 256")
        for name, choices in {
            "quant": ("bf16", "fp8"),
            "vendor": ("cutedsl", "triton"),
            "layout": ("per_expert", "shared"),
            "phase": ("decode", "prefill"),
            "mode": ("eager", "graph"),
            "routing": ("balanced", "skewed"),
            "traffic": ("active", "mixed", "base_only"),
        }.items():
            if getattr(case, name) not in choices:
                raise ValueError(f"unsupported {name}: {getattr(case, name)}")
        if case.quant == "fp8" and (
            case.hidden_size % 128 or case.intermediate_size % 128
        ):
            raise ValueError("FP8 requires 128-aligned resident H and I")
        scale = case.routed_scaling_factor
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("routed_scaling_factor must be finite and positive")
        return case


def model_geometry(path: Path, tp_size: int, ep_size: int) -> dict:
    """Translate only conventional routed-MoE config geometry; never download."""
    cfg = json.loads(path.read_text())
    for key in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(key), dict):
            cfg = cfg[key]
    experts = cfg.get("num_experts", cfg.get("n_routed_experts"))
    intermediate = cfg.get("moe_intermediate_size")
    hidden = cfg.get("hidden_size")
    if any(
        type(x) is not int or x <= 0
        for x in (experts, intermediate, hidden, tp_size, ep_size)
    ):
        raise ValueError(
            "model needs explicit hidden_size, moe_intermediate_size and routed expert count; otherwise supply resident geometry"
        )
    if tp_size % ep_size or intermediate % (tp_size // ep_size) or experts % ep_size:
        raise ValueError("model geometry is not divisible by TP/EP")
    activation = cfg.get("hidden_act", cfg.get("hidden_activation"))
    if activation != "silu":
        raise ValueError("only explicit gated SiLU model configs are supported")
    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate // (tp_size // ep_size),
        "num_local_experts": experts // ep_size,
        "tp_size": tp_size,
        "ep_size": ep_size,
    }


def consumed_sites(plan):
    sites = [
        name
        for name in ("gate_up_a", "down_a")
        if getattr(plan, name).family != "token_dense"
    ]
    if plan.gate_up_b is not None:
        sites.append("gate_up_b")
    if plan.down_b is not None or plan.finalize.family == "shared_token_delta":
        sites.append("down_b")
    return sites


def candidate_specs(
    incumbent: dict, sites=("gate_up_a", "down_a", "gate_up_b", "down_b")
) -> list[dict]:
    """One-axis tile variations; preserve plan families, provider and split mode."""
    candidates = [copy.deepcopy(incumbent)]
    for site in sites:
        for key, values in (
            ("num_warps", (4, 8)),
            ("num_stages", (2, 3)),
            ("BLOCK_SIZE_N", (16, 32, 64)),
        ):
            for value in values:
                candidate = copy.deepcopy(incumbent)
                candidate[site][key] = value
                candidates.append(candidate)
    return list({json.dumps(c, sort_keys=True): c for c in candidates}.values())


def study_override(
    case: Case, identity: dict, tiles: dict, search: dict, validation: dict
) -> dict:
    if search.get("status") != "WIN" or validation.get("status") != "WIN":
        raise ValueError("an override requires search and independent validation WIN")
    return {
        "schema": "moe-lora-exact-study-v1",
        "case": asdict(case),
        "identity": identity,
        "tiles": tiles,
        "production_table": False,
        "scope": "Exact case and recorded source/device only. Not a production plan-table override.",
        "unmeasured_regions": "All other tokens, ranks, geometry, layouts, phases, routing, traffic, vendors and devices. Production rows cannot express these exact restrictions.",
    }


def paired(baseline, candidate, case, args):
    samples = {"baseline": [], "candidate": []}
    for repeat in range(args.repeats):
        order = (
            ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        )
        for name in order:
            fn = baseline if name == "baseline" else candidate
            samples[name].extend(
                measure(
                    fn,
                    mode=case.mode,
                    warmup=args.warmup,
                    repeats=1,
                    iterations=args.iterations,
                )
            )
    return {**compare(samples["baseline"], samples["candidate"]), "samples_us": samples}


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / "python/sglang/srt/lora").rglob("*.py"))
    files += sorted((root / "python/sglang/srt/lora/moe/configs").rglob("*.json"))
    files += [
        Path(__file__),
        Path(__file__).with_name("workload.py"),
        root / "benchmark/kernels/lora_tuning.py",
    ]
    for relative in (
        "python/sglang/srt/runtime_context.py",
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py",
    ):
        files.append(root / relative)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def verify_loaded_sources():
    expected = Path(__file__).resolve().parents[3] / "python"
    metadata = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name == "sglang._version" and path:
            artifact = Path(path).resolve()
            metadata[name] = {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "version": str(getattr(module, "__version__", "unavailable")),
            }
            continue
        if (
            name.startswith("sglang.")
            and path
            and not Path(path).resolve().is_relative_to(expected)
        ):
            raise RuntimeError(f"loaded {name} outside this checkout: {path}")
    return metadata


def fatal_cuda_error(exc):
    return any(
        message in str(exc).lower()
        for message in (
            "illegal memory access",
            "device-side assert",
            "unspecified launch failure",
            "context is destroyed",
        )
    )


def run_case(case, args, emit):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    from sglang.srt.runtime_context import get_context

    verify_loaded_sources()
    with get_context().override_server_args():
        return _run_case(case, args, emit)


def _run_case(case, args, emit):
    from workload import Workload

    work = Workload(case, seed=1729)
    metadata = verify_loaded_sources()
    identity = {
        **work.identity,
        "sources": source_identity(),
        "generated_metadata": metadata,
    }

    def check_identity():
        if source_identity() != identity["sources"]:
            raise RuntimeError("source changed during study")
        if verify_loaded_sources() != metadata:
            raise RuntimeError("generated version metadata changed during study")

    incumbent = work.incumbent
    baseline = work.bind(incumbent)
    work.check(baseline)
    identity["resolved_base_config"] = work.base_config()
    emit(
        {
            "case": case.name,
            "stage": "baseline",
            "status": "CORRECT",
            "identity": identity,
            "tiles": incumbent,
        }
    )
    results = []
    sites = consumed_sites(work.selected.plan)
    for index, spec in enumerate(
        candidate_specs(incumbent, sites)[1 : args.max_candidates]
    ):
        row = {"case": case.name, "stage": "search", "candidate": index, "tiles": spec}
        try:
            candidate = work.bind(spec)
            work.check(candidate)
            row.update(paired(baseline, candidate, case, args))
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            if fatal_cuda_error(exc):
                emit(row)
                raise
        finally:
            candidate = None
        emit(row)
        results.append(row)
    wins = [row for row in results if row["status"] == "WIN"]
    if not wins:
        check_identity()
        failed = all(row["status"] == "FAILED" for row in results)
        return {
            "case": case.name,
            "status": "FAILED" if failed else "RETAIN_BASELINE",
            "reason": "all candidates failed" if failed else "no search WIN",
            "override": None,
        }
    best = max(wins, key=lambda row: row["median_gain"])
    del baseline, work
    heldout = Workload(case, seed=104729)
    baseline, candidate = heldout.bind(heldout.incumbent), heldout.bind(best["tiles"])
    heldout.check(baseline)
    heldout.check(candidate)
    if heldout.base_config() != identity["resolved_base_config"]:
        raise RuntimeError("base configuration changed in heldout validation")
    validation = paired(baseline, candidate, case, args)
    emit(
        {
            "case": case.name,
            "stage": "validation",
            "candidate": best["candidate"],
            **validation,
        }
    )
    check_identity()
    if validation["status"] != "WIN":
        return {
            "case": case.name,
            "status": "RETAIN_BASELINE",
            "reason": "heldout comparison did not WIN",
            "override": None,
        }
    return {
        "case": case.name,
        "status": "VALIDATED_WIN",
        "override": study_override(case, identity, best["tiles"], best, validation),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, required=True, help="JSON list of exact resident cases"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="optional local config.json supplying conventional MoE geometry",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, required=True, help="new exclusive study directory"
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=32,
        help="includes incumbent; bounded at 64",
    )
    args = parser.parse_args()
    if args.model_config is None and (args.tp_size != 1 or args.ep_size != 1):
        parser.error(
            "CLI TP/EP flags require --model-config; put resident-case TP/EP identity in the cases JSON"
        )
    if (
        args.repeats < 3
        or args.warmup < 1
        or args.iterations < 1
        or not 2 <= args.max_candidates <= 64
    ):
        parser.error(
            "require repeats>=3, warmup/iterations>=1 and 2<=max-candidates<=64"
        )
    geometry = (
        model_geometry(args.model_config, args.tp_size, args.ep_size)
        if args.model_config
        else {}
    )
    raw = json.loads(args.cases.read_text())
    if not isinstance(raw, list) or not raw:
        parser.error("cases must be a nonempty JSON list")
    cases = []
    for entry in raw:
        if any(key in entry and entry[key] != value for key, value in geometry.items()):
            parser.error("case geometry contradicts model-derived local shape")
        cases.append(Case.from_dict({**geometry, **entry}))
    if len({c.name for c in cases}) != len(cases):
        parser.error("case names must be unique")
    if any(
        key in os.environ
        for key in ("SGLANG_LORA_MOE_CONFIG_DIR", "SGLANG_MOE_CONFIG_DIR")
    ):
        parser.error(
            "unset external configuration overrides; this tuner retains shipped base configs"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "study.json").write_text(
        json.dumps(
            {
                "cases": [asdict(c) for c in cases],
                "repeats": args.repeats,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "max_candidates": args.max_candidates,
                "search_seed": 1729,
                "validation_seed": 104729,
            },
            indent=2,
        )
    )
    results = []
    with (args.out / "trials.jsonl").open("x") as stream:

        def emit(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        for case in cases:
            try:
                result = run_case(case, args, emit)
            except Exception as exc:
                result = {
                    "case": case.name,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "override": None,
                }
                emit(result)
                if fatal_cuda_error(exc):
                    results.append(result)
                    break
            results.append(result)
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False)
    )
    print(
        json.dumps(
            [
                {k: v for k, v in result.items() if k != "override"}
                for result in results
            ],
            indent=2,
        )
    )
    return int(any(row["status"] == "FAILED" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
