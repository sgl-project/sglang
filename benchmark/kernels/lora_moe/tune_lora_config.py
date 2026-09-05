"""Check a model's MoE LoRA plan coverage, or emit a seed table for it.

--check reports which shipped row serves each (layout, phase), or that the
model falls to the serial fallback (exit code 1).
--emit-seed writes <out>/<arch>.plans.json with the domain widened to the
model. The seed only reuses the existing plans for the wider geometry:
validate provider admission and correctness on that geometry before serving
it, then benchmark it with the campaign protocol in README.md. The former
--sweep mode was retired: it had drifted from the plan schema and was never
re-validated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PACKAGED = os.path.join(REPO, "python/sglang/srt/lora/moe/configs")


def load_geometry(model_path: str, args) -> dict:
    """Read checkpoint geometry; gating comes from --gated, not hidden_act."""
    cfg = json.load(open(os.path.join(model_path, "config.json")))
    for sub in ("language_config", "text_config", "llm_config"):
        cfg = cfg.get(sub, cfg) if isinstance(cfg.get(sub, None), dict) else cfg
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts")
    activation = (
        args.activation or cfg.get("hidden_act") or cfg.get("hidden_activation")
    )
    if activation is None:
        raise SystemExit(
            f"{model_path}/config.json names no hidden_act; pass --activation "
            "(guessing it would tune the tables for the wrong kernel)"
        )
    return {
        "hidden_size": cfg["hidden_size"],
        "num_experts": experts,
        "intermediate": cfg.get("moe_intermediate_size")
        or cfg.get("intermediate_size"),
        "activation": activation,
        "is_gated": args.gated,
    }


def resolve_report(args, geometry) -> int:
    sys.path.insert(0, os.path.join(REPO, "python"))
    if args.config_dir:
        os.environ["SGLANG_LORA_MOE_CONFIG_DIR"] = args.config_dir
    from sglang.srt.lora.moe.activation import ActivationFn
    from sglang.srt.lora.moe.execution_plan import (
        architecture_for_capability,
        resolve_plans,
    )

    e_local = geometry["num_experts"] // args.ep_size
    fell_through = 0
    for is_shared_outer in (False, True):
        selected = resolve_plans(
            architecture=architecture_for_capability(args.capability_major, 0),
            is_shared_outer=is_shared_outer,
            physical_rank=args.max_rank,
            activation=ActivationFn.parse(geometry["activation"]),
            hidden_size=geometry["hidden_size"],
            num_local_experts=e_local,
        )
        layout_name = "shared" if is_shared_outer else "per_expert"
        for phase, sel in selected.items():
            tag = "FALLBACK" if "fallback" in sel.name else "tuned"
            if tag == "FALLBACK":
                fell_through += 1
            print(f"  {layout_name:10s} {phase.value:7s} -> [{tag}] {sel.key}")
    return 1 if fell_through else 0


def emit_seed(args, geometry) -> str:
    arch = "gb300" if args.capability_major >= 10 else "h200"
    src = os.path.join(args.config_dir or PACKAGED, f"{arch}.plans.json")
    table = json.load(open(src))
    e_local = geometry["num_experts"] // args.ep_size
    table["domain"]["max_hidden"] = max(
        table["domain"]["max_hidden"], geometry["hidden_size"]
    )
    table["domain"]["max_local_experts"] = max(
        table["domain"]["max_local_experts"], e_local
    )
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"{arch}.plans.json")
    json.dump(table, open(dst, "w"), indent=1)
    tiles_src = os.path.join(args.config_dir or PACKAGED, f"{arch}.tiles.json")
    if os.path.isfile(tiles_src):
        tiles_dst = os.path.join(args.out, f"{arch}.tiles.json")
        json.dump(json.load(open(tiles_src)), open(tiles_dst, "w"), indent=1)
    print(f"seed config written: {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--capability-major", type=int, default=10)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument(
        "--activation",
        default=None,
        help="override the checkpoint's hidden_act (silu, relu2)",
    )
    ap.add_argument(
        "--gated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether gate/up is 2x intermediate; not in any config.json",
    )
    ap.add_argument("--max-rank", type=int, default=32)
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--out", default="./tuned_config")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-seed", action="store_true")
    args = ap.parse_args()

    geometry = load_geometry(args.model_path, args)
    print(f"geometry: {geometry} (ep={args.ep_size})")
    rc = 0
    if args.check or not args.emit_seed:
        rc = resolve_report(args, geometry)
    if args.emit_seed:
        emit_seed(args, geometry)
    sys.exit(rc)


if __name__ == "__main__":
    main()
