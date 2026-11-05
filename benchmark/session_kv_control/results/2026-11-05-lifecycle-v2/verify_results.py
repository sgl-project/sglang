#!/usr/bin/env python3
"""Replay archived protocol/output ledgers and recompute five-pair acceptance."""

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    check(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--verify-hashes", action="store_true")
    cli.add_argument(
        "--deep", action="store_true", help="also hash decompressed originals"
    )
    args = cli.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1]))
    from analyze_lifecycle import analyze, load_json
    from bench_statistics import bootstrap_mean_ci, quantile, stable_seed
    from session_kv_bench import digest

    expected = json.loads((here / "summary.json").read_text())
    design = json.loads((here / "formal-design.json").read_text())
    if args.verify_hashes or args.deep:
        manifest = json.loads((here / "evidence-manifest.json").read_text())
        for name, entry in manifest["files"].items():
            path = args.archive / name
            check(path.resolve().is_relative_to(args.archive.resolve()), "archive path")
            sha = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    sha.update(chunk)
            check(sha.hexdigest() == entry["sha256"], "compressed/file SHA: " + name)
            if args.deep and "original_sha256" in entry:
                sha = hashlib.sha256()
                size = 0
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rb") as source:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        sha.update(chunk)
                        size += len(chunk)
                check(
                    sha.hexdigest() == entry["original_sha256"], "original SHA: " + name
                )
                check(size == entry["original_bytes"], "original size: " + name)
        print("archive hashes verified:", len(manifest["files"]))

    seen, run_ids = set(), set()
    for name in (design["selected_preflight"], "longwait30-three-actions"):
        manifest = load_json(args.archive / "cells" / name / "lifecycle-manifest.json")
        seen.update(digest(tokens) for tokens in manifest["input_material"])
        run_ids.add(manifest["args"]["run_id"])
    arms = {name: [] for name in ("on_demand", "lifecycle")}
    ttft = {name: [] for name in arms}
    deltas, full_deltas = [], []
    for index, seed in enumerate(design["pair_seeds"]):
        name = f"formal-v2-pair-{index}"
        cell = args.archive / "cells" / name
        manifest = load_json(cell / "lifecycle-manifest.json")
        check(manifest["args"]["seed"] == seed, name + ": seed")
        check(manifest["args"]["passes"] == design["passes"], name + ": fixed work")
        check(manifest["args"]["run_id"] not in run_ids, name + ": reused salt/run ID")
        run_ids.add(manifest["args"]["run_id"])
        fingerprint = manifest["configuration"]["fingerprint_b"]
        check(fingerprint["mem_fraction_static"] == 0.8, name + ": memory fraction")
        check(fingerprint["max_total_tokens"] is None, name + ": manual token cap")
        check(fingerprint["hicache_storage_backend"] == "mooncake", name + ": backend")
        hashes = {digest(tokens) for tokens in manifest["input_material"]}
        check(len(hashes) == 64 and not hashes & seen, name + ": input overlap")
        seen.update(hashes)
        ready = load_json(
            args.archive / "proofs" / f"router-{name}-ready-capabilities.json"
        )
        check(
            ready["response"]["data"]["preparation_cost_groups"] == 0,
            name + ": model reset",
        )
        result = analyze(cell)
        check(
            result["verification"]["verified"] and result["acceptance_passed"],
            name + ": ledger",
        )
        group = {arm["arm"]: arm for arm in result["arms"]}
        check(set(group) == set(arms), name + ": arms")
        for key, arm in group.items():
            check(arm["completed"] == arm["valid_tasks"] == 1792, name + ": task count")
            check(
                arm["wall_s"] >= 600 and not arm["drain"]["errors"],
                name + ": duration/drain",
            )
            arms[key].append(arm)
            rows = load_json(cell / (arm["id"] + ".json"))["rows"]
            values = [row["generation"]["ttft_ms"] for row in rows]
            close(
                quantile(values, 0.95),
                arm["all_completed_ttft_ms"]["p95"],
                name + ": p95",
            )
            ttft[key].extend(values)
        base, life = group["on_demand"], group["lifecycle"]
        deltas.append(100 * (base["wall_s"] / life["wall_s"] - 1))
        full_deltas.append(
            100 * (base["complete_job_wall_s"] / life["complete_job_wall_s"] - 1)
        )
        print("verified", name, "throughput change %", round(deltas[-1], 6))

    totals = {}
    for key, group in arms.items():
        count = sum(arm["effective_completed"] for arm in group)
        seconds = sum(arm["wall_s"] for arm in group)
        full = sum(arm["complete_job_wall_s"] for arm in group)
        check(len(group) == 5 and count >= 5000 and seconds >= 3600, key + ": minimums")
        totals[key] = (count / seconds, count / full, quantile(ttft[key], 0.95))
        for field, value in (
            ("work_seconds", seconds),
            ("complete_job_seconds", full),
            ("resumes", count),
            ("p95_ms", totals[key][2]),
            (
                "repeated_prefill_tokens",
                sum(a["repeated_prefill_tokens"] for a in group),
            ),
        ):
            close(value, expected["totals"][key][field], key + ": " + field)
    changes = [
        100 * (totals["lifecycle"][i] / totals["on_demand"][i] - 1) for i in range(3)
    ]
    for value, key in zip(
        changes,
        (
            "work_throughput_change_percent",
            "complete_job_throughput_change_percent",
            "pooled_p95_change_percent",
        ),
    ):
        close(value, expected[key], key)
    for values, label, key in (
        (deltas, "work", "paired_work_ci"),
        (full_deltas, "complete-job", "paired_complete_job_ci"),
    ):
        ci = bootstrap_mean_ci(
            values, seed=stable_seed("lifecycle-independent-formal-" + label + "-v2")
        )
        for actual, wanted in zip(ci["ci"], expected[key]["ci"]):
            close(actual, wanted, key)
    check(changes[0] >= 5 and changes[2] <= 5, "performance targets")
    check(
        expected["paired_work_ci"]["ci"][0] > 0 and expected["performance_accepted"],
        "paired interval",
    )
    print(
        "verified: 17920 resumes; five paired runs; correctness, time and performance gates passed"
    )


if __name__ == "__main__":
    main()
