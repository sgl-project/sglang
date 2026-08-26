"""Produce an MoE LoRA config JSON for a model the shipped files don't cover.

Three steps, each usable alone:

1. ``--check``: resolve the model's MoE geometry against the shipped (or
   ``--config-dir``) plan tables and report, per (layout, phase), which row
   would serve it — or that it falls to the serial fallback. Exit code 1
   when anything falls through, so CI can gate onboarding.
2. ``--emit-seed``: write ``<out>/<arch>.plans.json`` seeded for this
   geometry: the shipped rows with the ``domain`` widened to include the
   model (the shipped ``<arch>.tiles.json`` is copied through unchanged).
   This serves correctly immediately (all rows still pass plan validation);
   it is a starting point for step 3, not a certified optimum.
3. ``--sweep``: e2e-tune, one server relaunch per arm, the axes the
   2026-08 campaign found can move with geometry:
   - LoRA config: shared-decode overlap windows (variant config dirs served
     through SGLANG_LORA_MOE_CONFIG_DIR; winner written back into the seed);
   - LoRA config: the route block size, separately for decode and for
     prefill, since occupancy decides it (winner written into the tiles
     file; scored on decode tok/s and prefill tok/s respectively);
   - base GEMM: the masked-GEMM M-bucket tiles, by delegating to
     ``sweep_masked_gemm_configs.py`` on this device (output lands in
     ``<out>/base_gemm``, so SGLANG_LORA_MOE_CONFIG_DIR=<out> serves both).
   Axes proven geometry-insensitive in the campaign (route builder, PDL
   edges, B families, prefill serial shape) are not re-swept; evidence in
   the campaign's best-config tables document.

TODO(quant): this harness currently tunes the bf16 serving path only. When
fp8 / nvfp4 qlora support lands, the sweep must grow a quant dimension:
quant-specific providers as sweep arms, a ``quant`` key on the plan rows
(the pydantic loaders reject fields they do not understand, so old builds
fail closed on newer tables instead of mis-matching), and per-quant seed
emission.

The sweep reuses the campaign protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PACKAGED = os.path.join(REPO, "python/sglang/srt/lora/moe/configs")

ROUTING_BLOCK_CANDIDATES = [16, 32, 64, 128]

SHARED_WINDOW_CANDIDATES = [
    ("gate_up_a_b", "down_a_b"),  # shipped winner
    ("gate_up_a", "down_a_b"),
    ("gate_up_a_b", "down_a"),
    ("gate_up_a", "down_a"),
    ("none", "none"),  # serial reference
]


def load_geometry(model_path: str, args) -> dict:
    """Read what the checkpoint states; take the rest from flags.

    The activation is in the config for most models under ``hidden_act``.
    Gating is NOT -- sglang decides it per architecture in the model file
    (nemotron_h passes is_gated=False), so there is nothing to read and the
    flag is the only source. Both were hardcoded here before, which meant
    every tuned table claimed the same activation whatever the model was.
    """
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
    for row in table["scenarios"]:
        row.setdefault("provenance", "campaign-2026-08")
    table["seeded_for"] = {
        "model": args.model_path,
        "hidden": geometry["hidden_size"],
        "local_experts": e_local,
        # Recorded because neither is recoverable from the emitted table:
        # rows are activation- and gating-agnostic by design.
        "activation": geometry["activation"],
        "is_gated": geometry["is_gated"],
        "date": str(date.today()),
        "note": "domain widened; rows beyond the campaign geometries are "
        "seed:untuned until --sweep certifies them",
    }
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"{arch}.plans.json")
    json.dump(table, open(dst, "w"), indent=1)
    tiles_src = os.path.join(args.config_dir or PACKAGED, f"{arch}.tiles.json")
    if os.path.isfile(tiles_src):
        tiles_dst = os.path.join(args.out, f"{arch}.tiles.json")
        json.dump(json.load(open(tiles_src)), open(tiles_dst, "w"), indent=1)
    print(f"seed config written: {dst}")
    return dst


def bench_once(
    args, env_extra: dict, tag: str, metric: str = "output_throughput"
) -> dict[int, float]:
    """``output_throughput`` is decode, ``input_throughput`` is prefill. Score
    each axis on the side it moves: a prefill-only tile barely shifts decode
    tok/s, so scoring it on the default reads as noise and picks at random.
    """
    out = os.path.join(args.out, f"arm_{tag}")
    os.makedirs(out, exist_ok=True)
    env = os.environ.copy()
    env.update(env_extra)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model_path,
            "--port",
            str(args.port),
        ]
        + args.server_args.split(),
        env=env,
        stdout=open(f"{out}/server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        import urllib.request

        for _ in range(400):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/health", timeout=2
                )
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(f"server died; see {out}/server.log")
                time.sleep(10)
        bench = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.bench_one_batch_server",
                "--model",
                "None",
                "--base-url",
                f"http://127.0.0.1:{args.port}",
                "--batch-size",
            ]
            + [str(b) for b in (1, 8, 16, 32) * 4]
            + [
                "--input-len",
                "4096",
                "--output-len",
                "1024",
                "--result-filename",
                f"{out}/r.jsonl",
            ]
            + (args.bench_args.split() if args.bench_args else []),
            capture_output=True,
        )
        rows = [json.loads(l) for l in open(f"{out}/r.jsonl")]
    finally:
        server.terminate()
        server.wait(timeout=60)
        time.sleep(10)
    by_bs: dict[int, list] = {}
    for r in rows:
        by_bs.setdefault(r["batch_size"], []).append(r[metric])
    return {bs: statistics.median(v[1:] or v) for bs, v in sorted(by_bs.items())}


def _set_route_block(sites: dict, block: int) -> None:
    """Move one rule's route block; B-site tripwires follow the route."""
    sites["routing_block_size"] = block
    for name in ("gate_up_b", "down_b"):
        site = sites.get(name)
        if site is not None and "BLOCK_SIZE_M" in site:
            site["BLOCK_SIZE_M"] = block


def sweep_route_block(args, seed_path: str, arch: str, variant_dir: str) -> dict:
    """Occupancy decides it -- routed pairs per virtual expert, i.e.
    tokens x top_k / virtual experts -- so decode and prefill get their own
    sweep and their own metric. The shipped tables want 16 at decode and 32
    at prefill; a geometry far from the campaign's can land elsewhere, which
    is why it is an axis here at all.
    """
    tiles_path = os.path.join(args.out, f"{arch}.tiles.json")
    if not os.path.isfile(tiles_path):
        return {}
    plans = json.load(open(seed_path))
    rows = plans["scenarios"] + plans.get("fallback", [])
    phase_of = {row["name"]: row.get("phase") for row in rows}
    tiles = json.load(open(tiles_path))
    all_results = {}
    for phase, metric in (
        ("decode", "output_throughput"),
        ("prefill", "input_throughput"),
    ):
        names = [n for n in tiles["rules"] if phase_of.get(n) == phase]
        if not names:
            continue
        results = {}
        for block in ROUTING_BLOCK_CANDIDATES:
            candidate = copy.deepcopy(tiles)
            for name in names:
                for rule in candidate["rules"][name]:
                    _set_route_block(rule["sites"], block)
            os.makedirs(variant_dir, exist_ok=True)
            json.dump(
                candidate, open(os.path.join(variant_dir, f"{arch}.tiles.json"), "w")
            )
            json.dump(plans, open(os.path.join(variant_dir, f"{arch}.plans.json"), "w"))
            tag = f"route_{phase}_{block}"
            results[block] = bench_once(
                args, {"SGLANG_LORA_MOE_CONFIG_DIR": variant_dir}, tag, metric
            )
            print(tag, results[block])
        winner = max(results, key=lambda b: sum(results[b].values()))
        print(f"route block winner ({phase}, scored on {metric}): {winner}")
        for name in names:
            for rule in tiles["rules"][name]:
                _set_route_block(rule["sites"], winner)
        all_results.update({f"{phase}-{b}": v for b, v in results.items()})
    json.dump(tiles, open(tiles_path, "w"), indent=1)
    return all_results


def sweep(args, seed_path: str) -> None:
    table = json.load(open(seed_path))
    arch = table["arch"]
    variant_dir = os.path.join(args.out, "variants")
    results = {}
    # Axis: shared-layout decode overlap windows (skips cleanly if the
    # adapter is per-expert; the arm then just re-measures the same plan).
    for gate_up, down in SHARED_WINDOW_CANDIDATES:
        candidate = copy.deepcopy(table)
        for row in candidate["scenarios"]:
            if row.get("layout") == "shared" and row.get("phase") == "decode":
                row["plan"]["gate_up_overlap"] = gate_up
                row["plan"]["down_overlap"] = down
                for k in ("early_overlap", "late_overlap"):
                    if row["plan"][k] == "none":
                        del row["plan"][k]
                row["provenance"] = "sweep-candidate"
        os.makedirs(variant_dir, exist_ok=True)
        json.dump(candidate, open(os.path.join(variant_dir, f"{arch}.plans.json"), "w"))
        tiles_src = os.path.join(args.out, f"{arch}.tiles.json")
        if os.path.isfile(tiles_src):
            json.dump(
                json.load(open(tiles_src)),
                open(os.path.join(variant_dir, f"{arch}.tiles.json"), "w"),
            )
        tag = f"win_{gate_up}-{down}"
        results[(gate_up, down)] = bench_once(
            args, {"SGLANG_LORA_MOE_CONFIG_DIR": variant_dir}, tag
        )
        print(tag, results[(gate_up, down)])
    gate_up, down = max(results, key=lambda pair: sum(results[pair].values()))
    print(f"window winner: early={gate_up} late={down}")
    for row in table["scenarios"]:
        if row.get("layout") == "shared" and row.get("phase") == "decode":
            for k in ("early_overlap", "late_overlap"):
                row["plan"].pop(k, None)
            if gate_up != "none":
                row["plan"]["gate_up_overlap"] = gate_up
            if down != "none":
                row["plan"]["down_overlap"] = down
            row["provenance"] = f"swept:{date.today()} ({args.model_path})"
    json.dump(table, open(seed_path, "w"), indent=1)
    scored = {f"window-{e}-{l}": v for (e, l), v in results.items()}
    # Axis: route block size, one value per phase. Swept after the window
    # winner is in the seed, so it is measured against the plan that ships.
    scored.update(sweep_route_block(args, seed_path, arch, variant_dir))
    json.dump(
        scored,
        open(os.path.join(args.out, "sweep_results.json"), "w"),
        indent=1,
    )

    gemm_out = os.path.join(args.out, "base_gemm")
    os.makedirs(gemm_out, exist_ok=True)
    geometry = load_geometry(args.model_path, args)
    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "sweep_masked_gemm_configs.py"),
            "--provider",
            "both",
            "--num-local-experts",
            str(geometry["num_experts"] // args.ep_size),
            "--hidden-size",
            str(geometry["hidden_size"]),
            "--intermediate-size",
            str(geometry["intermediate"] // args.tp_size),
            "--gate-up-slices",
            str(2 if geometry["is_gated"] else 1),
            "--top-k",
            str(args.top_k),
            "--output-dir",
            gemm_out,
        ],
        check=False,
    )
    print(f"base-GEMM tile store (may be empty = heuristics win): {gemm_out}")


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
    ap.add_argument("--port", type=int, default=31043)
    ap.add_argument("--server-args", default="")
    ap.add_argument("--bench-args", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-seed", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    geometry = load_geometry(args.model_path, args)
    print(f"geometry: {geometry} (ep={args.ep_size})")
    rc = 0
    if args.check or not (args.emit_seed or args.sweep):
        rc = resolve_report(args, geometry)
    seed = None
    if args.emit_seed or args.sweep:
        seed = emit_seed(args, geometry)
    if args.sweep:
        sweep(args, seed)
    sys.exit(rc)


if __name__ == "__main__":
    main()
