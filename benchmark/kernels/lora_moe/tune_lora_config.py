"""Produce an MoE LoRA config JSON for a model the shipped files don't cover.

Three steps, each usable alone:

1. ``--check``: resolve the model's MoE geometry against the shipped (or
   ``--config-dir``) files and report, per (layout, phase), which scenario
   row would serve it — or that it falls to the serial fallback. Exit code 1
   when anything falls through, so CI can gate onboarding.
2. ``--emit-seed``: write ``<out>/<arch>.json`` seeded for this geometry:
   the shipped rows with the ``domain`` widened to include the model. This
   serves correctly immediately (all rows still pass plan validation); it is
   a starting point for step 3, not a certified optimum.
3. ``--sweep``: e2e-tune, one server relaunch per arm, the two axes the
   2026-08 campaign found can move with geometry:
   - LoRA config: shared-decode overlap windows (variant config dirs served
     through SGLANG_LORA_MOE_CONFIG_DIR; winner written back into the seed);
   - base GEMM: the masked-GEMM M-bucket tiles, by delegating to
     ``sweep_masked_gemm_configs.py`` on this device (output lands in
     ``<out>/gemm``, so SGLANG_LORA_MOE_CONFIG_DIR=<out> serves both).
   Axes proven geometry-insensitive in the campaign (route builder, PDL
   edges, B families, prefill serial shape) are not re-swept; evidence in
   the campaign's best-config tables document.

TODO(quant): this harness currently tunes the bf16 serving path only. When
fp8 / nvfp4 qlora support lands, the sweep must grow a quant dimension:
quant-specific providers as sweep arms, a ``quant`` key in the config
``when`` predicates (the resolver already fails closed on keys it does not
understand, so old builds reject newer configs instead of mis-matching),
and per-quant seed emission.

The sweep reuses the campaign protocol: bench_one_batch_server, input 4096 /
output 1024, batch sizes 1/8/16/32, medians after first-pass discard.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PACKAGED = os.path.join(REPO, "python/sglang/srt/lora/moe/configs")

SHARED_WINDOW_CANDIDATES = [
    ("gate_a_b", "down_a_b"),  # shipped winner
    ("gate_a", "down_a_b"),
    ("gate_a_b", "down_a"),
    ("gate_a", "down_a"),
    ("none", "none"),  # serial reference
]


def load_geometry(model_path: str) -> dict:
    cfg = json.load(open(os.path.join(model_path, "config.json")))
    for sub in ("language_config", "text_config", "llm_config"):
        cfg = cfg.get(sub, cfg) if isinstance(cfg.get(sub, None), dict) else cfg
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts")
    return {
        "hidden_size": cfg["hidden_size"],
        "num_experts": experts,
        "intermediate": cfg.get("moe_intermediate_size")
        or cfg.get("intermediate_size"),
        "activation": "swiglu",  # relu2 models must pass --activation relu2
    }


def resolve_report(args, geometry) -> int:
    sys.path.insert(0, os.path.join(REPO, "python"))
    if args.config_dir:
        os.environ["SGLANG_LORA_MOE_CONFIG_DIR"] = args.config_dir
    from sglang.srt.lora.moe import config as cm
    from sglang.srt.lora.moe.execution_plan import ActivationFamily

    e_local = geometry["num_experts"] // args.ep_size
    fell_through = 0
    for is_shared_outer, mode, tokens in itertools.product(
        (False, True),
        (cm.Phase.DECODE, cm.Phase.PREFILL),
        (1, 32, 4096),
    ):
        if (mode is cm.Phase.DECODE) == (tokens == 4096):
            continue
        choice = cm.select_config(
            cm.ConfigInput(
                capability_major=args.capability_major,
                capability_minor=0,
                is_shared_outer=is_shared_outer,
                activation=ActivationFamily(geometry["activation"]),
                mode=mode,
                num_tokens=tokens,
                active_rank=args.max_rank,
                hidden_size=geometry["hidden_size"],
                num_local_experts=e_local,
                has_active_lora=True,
                use_cuda_graph=False,
            )
        )
        tag = "FALLBACK" if "fallback" in choice.key else "tuned"
        if tag == "FALLBACK":
            fell_through += 1
        layout_name = "shared" if is_shared_outer else "per_expert"
        print(
            f"  {layout_name:10s} {mode.value:7s} tokens={tokens:<5d} -> "
            f"[{tag}] {choice.key}"
        )
    return 1 if fell_through else 0


def emit_seed(args, geometry) -> str:
    arch = "gb300" if args.capability_major >= 10 else "h200"
    src = os.path.join(args.config_dir or PACKAGED, f"{arch}.json")
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
        "date": str(date.today()),
        "note": "domain widened; rows beyond the campaign geometries are "
        "seed:untuned until --sweep certifies them",
    }
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"{arch}.json")
    json.dump(table, open(dst, "w"), indent=1)
    print(f"seed config written: {dst}")
    return dst


def bench_once(args, env_extra: dict, tag: str) -> dict[int, float]:
    """One server launch + protocol bench; returns bs -> decode tok/s."""
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
        by_bs.setdefault(r["batch_size"], []).append(r["output_throughput"])
    return {bs: statistics.median(v[1:] or v) for bs, v in sorted(by_bs.items())}


def sweep(args, seed_path: str) -> None:
    """Cross the geometry-sensitive axes; write winners into the seed file."""
    table = json.load(open(seed_path))
    arch = table["arch"]
    variant_dir = os.path.join(args.out, "variants")
    results = {}
    # Axis: shared-layout decode overlap windows (skips cleanly if the
    # adapter is per-expert; the arm then just re-measures the same plan).
    for early, late in SHARED_WINDOW_CANDIDATES:
        candidate = copy.deepcopy(table)
        for row in candidate["scenarios"]:
            if row["when"].get("layout") == "shared" and (
                row["when"].get("phase") == "decode"
            ):
                row["plan"]["early_overlap"] = early
                row["plan"]["late_overlap"] = late
                for k in ("early_overlap", "late_overlap"):
                    if row["plan"][k] == "none":
                        del row["plan"][k]
                row["provenance"] = "sweep-candidate"
        os.makedirs(variant_dir, exist_ok=True)
        json.dump(candidate, open(os.path.join(variant_dir, f"{arch}.json"), "w"))
        tag = f"win_{early}_{late}"
        results[tag] = bench_once(
            args, {"SGLANG_LORA_MOE_CONFIG_DIR": variant_dir}, tag
        )
        print(tag, results[tag])
    best = max(results, key=lambda t: sum(results[t].values()))
    early, late = best[len("win_") :].split("_", 1)
    print(f"window winner: early={early} late={late}")
    for row in table["scenarios"]:
        if row["when"].get("layout") == "shared" and (
            row["when"].get("phase") == "decode"
        ):
            for k in ("early_overlap", "late_overlap"):
                row["plan"].pop(k, None)
            if early != "none":
                row["plan"]["early_overlap"] = early
            if late != "none":
                row["plan"]["late_overlap"] = late
            row["provenance"] = f"swept:{date.today()} ({args.model_path})"
    json.dump(table, open(seed_path, "w"), indent=1)
    json.dump(
        results, open(os.path.join(args.out, "sweep_results.json"), "w"), indent=1
    )

    # Base-GEMM tiles: delegate to the masked-GEMM bucket sweep on this
    # device; an empty output directory means the built-in heuristics won
    # everywhere (the result at all three campaign geometries).
    gemm_out = os.path.join(args.out, "base_gemm")
    os.makedirs(gemm_out, exist_ok=True)
    geometry = load_geometry(args.model_path)
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
            "2",
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

    geometry = load_geometry(args.model_path)
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
