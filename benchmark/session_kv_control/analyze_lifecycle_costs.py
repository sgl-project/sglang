#!/usr/bin/env python3
"""Offline lifecycle cost/resource observations; correctness uses analyze_lifecycle."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from analyze_lifecycle import load_json, open_text
from analyze_stress import resource_peaks, resource_sample
from bench_report import distribution
from bench_statistics import bootstrap_mean_ci, stable_seed
from lifecycle_bench import write_json

GAUGES = {
    "native_active_jobs",
    "native_queued_jobs",
    "native_demand_waiters",
    "native_reserved_host_tokens",
    "native_inflight_tokens",
    "native_speculative_inflight_tokens",
    "native_host_capacity_tokens",
    "native_host_available_tokens",
    "native_host_bytes_per_token",
}


def records(paths, issues, ordered=False):
    previous = -math.inf
    for path in paths:
        path = Path(path)
        with (
            gzip.open(path, "rt") if path.suffix == ".gz" else open_text(path)
        ) as source:
            for number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("sample must be an object")
                    timestamp = row.get("monotonic")
                    if type(timestamp) not in (int, float) or not math.isfinite(
                        timestamp
                    ):
                        raise ValueError("monotonic must be a finite number")
                    if ordered and (timestamp <= previous):
                        raise ValueError("overlapping/unordered sample")
                    previous = timestamp
                    yield row
                except (ValueError, KeyError) as error:
                    issues.append(
                        {"file": str(path), "line": number, "error": str(error)}
                    )


def coverage(times, start, end, gap):
    seconds = sum(b - a for a, b in zip(times, times[1:]) if 0 < b - a <= gap)
    return {
        "observed_seconds": seconds,
        "fraction": seconds / (end - start),
        "boundary_unobserved_s": None
        if not times
        else [times[0] - start, end - times[-1]],
        "gaps_s": [b - a for a, b in zip(times, times[1:]) if b - a > gap],
    }


def storage(arm, replies, url):
    before = (arm.get("costs_before") or {}).get(url, {})
    after = (arm.get("costs_after") or {}).get(url, {})
    observations = replies.get(arm["id"], [])
    contexts = [r.get("request", {}).get("context", {}) for r in observations]
    identity = lambda c: tuple(
        c.get(k) for k in ("engine_id", "engine_epoch", "namespace")
    )
    valid = (
        len(contexts) == 2
        and all(identity(contexts[0]))
        and identity(contexts[0]) == identity(contexts[1])
        and [r.get("response") for r in observations] == [before, after]
        and all(
            isinstance(r, dict) and r.get("success") is True for r in (before, after)
        )
    )
    left, right = [
        r["data"].get("storage_reads")
        if isinstance(r, dict) and isinstance(r.get("data"), dict)
        else None
        for r in (before, after)
    ]
    if not all(isinstance(value, dict) and value for value in (left, right)):
        return dict(
            epoch_verified=valid,
            available=False,
            before=left,
            after=right,
            delta=None,
            gauges=None,
            wire_bytes=None,
            reason="storage_reads missing, null, empty or not an object",
        )
    gauges = {
        k
        for k in left.keys() | right.keys()
        if k in GAUGES or k.startswith("native_waiting_")
    }
    keys = (left.keys() | right.keys()) - gauges
    delta = {
        k: right[k] - left[k]
        if all(
            type(v) in (int, float) and math.isfinite(v)
            for v in (left.get(k), right.get(k))
        )
        else None
        for k in keys
    }
    reset = any(value is not None and value < 0 for value in delta.values())
    return {
        "epoch_verified": valid,
        "available": True,
        "counter_reset": reset,
        "before": left,
        "after": right,
        "delta": delta if valid and not reset else {k: None for k in delta},
        "gauges": {k: [left.get(k), right.get(k)] for k in gauges},
        "snapshot_rpc_bounds": [
            [r.get("started_monotonic"), r.get("monotonic")] for r in observations
        ],
        "wire_bytes": None,
        "bytes_definition": "addressed KV GET payload; wire traffic unknown",
        "time_definition": "summed job/query durations, including concurrent jobs; not wall stall time",
    }


def cpu(samples, start, end, gap, effective):
    def identities(role):
        try:
            return {
                (p["pid"], p["created"]): p["user_s"] + p["system_s"]
                for p in role.get("processes", [])
            }
        except (TypeError, KeyError):
            return {}

    result = {}
    for role in sorted({role for row in samples for role in row.get("roles", {})}):
        seconds, observed, invalid = 0, 0, []
        for left, right in zip(samples, samples[1:]):
            elapsed = right["monotonic"] - left["monotonic"]
            a, b = [r.get("roles", {}).get(role, {}) for r in (left, right)]
            x, y = identities(a), identities(b)
            changes = [y[k] - x[k] for k in x if k in y]
            if (
                not 0 < elapsed <= gap
                or a.get("error")
                or b.get("error")
                or not x
                or x.keys() != y.keys()
                or a.get("root_pid") != b.get("root_pid")
                or any(not math.isfinite(v) or v < 0 for v in changes)
            ):
                invalid.append([left["monotonic"], right["monotonic"]])
                continue
            seconds += sum(changes)
            observed += elapsed
        result[role] = {
            "state": "partial"
            if invalid or observed < 0.95 * (end - start)
            else "observed",
            "observed_core_seconds": seconds if observed else None,
            "observed_seconds": observed,
            "coverage_fraction": observed / (end - start),
            "average_cores_observed": seconds / observed if observed else None,
            "observed_core_seconds_per_effective_task": seconds / effective
            if observed and effective
            else None,
            "unknown_intervals": invalid,
            "full_arm_core_seconds": None,
        }
    return result


def gpu_capacity(server):
    if not isinstance(server, dict) or any(
        server.get(name, 1) != 1 for name in ("tp_size", "pp_size", "dp_size")
    ):
        return None
    states = server.get("internal_states")
    if states is not None:
        if (
            not isinstance(states, list)
            or len(states) != 1
            or not isinstance(states[0], dict)
        ):
            return None
        if any(
            states[0].get(name, 1) != 1 for name in ("tp_size", "pp_size", "dp_size")
        ):
            return None
        usage = states[0].get("memory_usage")
        value = usage.get("token_capacity") if isinstance(usage, dict) else None
    else:
        value = server.get("max_total_num_tokens")
        if value is None:
            usage = server.get("memory_usage")
            value = usage.get("token_capacity") if isinstance(usage, dict) else None
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and value > 0
        else None
    )


def analyze(roots, resource_paths=(), process_paths=(), gap=6, b_gpu=1):
    issues, arms, incomplete = [], [], []
    for directory in roots:
        root = Path(directory)
        manifest = load_json(root / "lifecycle-manifest.json")
        url, replies = manifest["args"]["engine_b"].rstrip("/"), {}
        for row in records([root / "lifecycle-raw.jsonl"], issues):
            if (
                row.get("kind") == "http"
                and row.get("url", "").rstrip("/") == url
                and (row.get("request") or {}).get("kind") == "capacity"
            ):
                replies.setdefault(row.get("trial"), []).append(row)
        for spec in manifest["schedule"]:
            path = root / (spec["id"] + ".json")
            if not path.exists() and not Path(str(path) + ".gz").exists():
                incomplete.append({"cell": str(root), "id": spec["id"]})
                continue
            arm = load_json(path)
            start, end = arm["started_monotonic"], arm["completed_monotonic"]
            if end <= start or abs(end - start - arm["wall_s"]) > 1e-6:
                raise ValueError("invalid arm interval: " + str(path))
            done = [r for r in arm["rows"] if r.get("generation")]
            ready = [r for r in done if r.get("ready_before_resume")]
            owner = [
                r for r in ready if r.get("selected_worker_url", "").rstrip("/") == url
            ]
            missed = sum(
                r.get("cache", {}).get("device", 0) + r.get("cache", {}).get("cpu", 0)
                < manifest["args"]["prefix_tokens"]
                for r in owner
            )
            boot = arm.get("server_before", {}).get("b", {})
            capacity = gpu_capacity(boot)
            arms.append(
                {
                    "cell": str(root),
                    **spec,
                    "start": start,
                    "end": end,
                    "effective": arm["effective_completed"],
                    "completed": len(done),
                    "effective_tasks_per_s": arm["effective_tasks_per_s"],
                    "ttft_ms": distribution([r["generation"]["ttft_ms"] for r in done]),
                    "latency_ms": distribution(
                        [r["generation"]["latency_ms"] for r in done]
                    ),
                    "ready_total": len(ready),
                    "ready_same_owner": len(owner),
                    "ready_restore_miss": missed,
                    "ready_miss_fraction": missed / len(owner) if owner else None,
                    "storage": storage(arm, replies, url),
                    "url": url,
                    "capacity": capacity,
                    "resource_samples": [],
                    "process_samples": [],
                }
            )
    for paths, field in (
        (resource_paths, "resource_samples"),
        (process_paths, "process_samples"),
    ):
        for row in records(paths, issues, ordered=True):
            for arm in arms:
                if not arm["start"] <= row["monotonic"] <= arm["end"]:
                    continue
                duration = row.get("sample_elapsed_s")
                if duration is not None and (
                    type(duration) not in (int, float)
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    issues.append(
                        {
                            "sample": row["monotonic"],
                            "error": "invalid capture duration",
                        }
                    )
                    continue
                if row["monotonic"] + (duration or 0) > arm["end"]:
                    continue
                if field == "resource_samples":
                    try:
                        sample = resource_sample(row, arm["url"], arm["capacity"])
                    except (KeyError, TypeError, ValueError) as error:
                        issues.append({"sample": row["monotonic"], "error": str(error)})
                        continue
                    sample["capture_duration_unknown"] = duration is None
                    metrics = sample["metrics"]
                    if (
                        metrics.get("b_host_total_tokens", 0) > 0
                        and "b_host_used_tokens" in metrics
                    ):
                        metrics["b_host_fraction"] = (
                            metrics["b_host_used_tokens"]
                            / metrics["b_host_total_tokens"]
                        )
                    arm[field].append(sample)
                else:
                    arm[field].append(row)
    for arm in arms:
        samples, processes = arm.pop("resource_samples"), arm.pop("process_samples")
        arm["resources"] = resource_peaks(samples)
        arm["resources"]["definition"] = (
            "capture starts inside exact arm bounds; known captures finish inside; unknown durations explicit; GPU capacity minus free includes reserved transfers; sampled peaks"
        )
        arm["resources"]["b_gpu"] = arm["resources"]["gpu"].get(str(b_gpu), {})
        arm["resource_coverage"] = coverage(
            [s["monotonic"] for s in samples], arm["start"], arm["end"], gap
        )
        arm["resource_coverage"]["capture_duration_unknown"] = not samples or any(
            s["capture_duration_unknown"] for s in samples
        )
        arm["cpu"] = cpu(processes, arm["start"], arm["end"], gap, arm["effective"])
        arm["cpu_sample_coverage"] = coverage(
            [s["monotonic"] for s in processes], arm["start"], arm["end"], gap
        )
        unknown = not processes or any(
            r.get("sample_elapsed_s") is None for r in processes
        )
        arm["cpu_sample_coverage"].update(
            capture_duration_unknown=unknown,
            capture_boundary_error_bounds_s=None
            if unknown
            else [processes[0]["sample_elapsed_s"], processes[-1]["sample_elapsed_s"]],
            clock_definition="monotonic is capture start; individual process read instants are unrecorded",
        )
    paired, intervals = [], {}
    bases = {(a["cell"], a["block"]): a for a in arms if a["arm"] == "on_demand"}
    for arm in arms:
        base = bases.get((arm["cell"], arm["block"]))
        if arm["arm"] == "on_demand" or base is None:
            continue
        pair = {"cell": arm["cell"], "block": arm["block"], "candidate": arm["arm"]}
        values = [
            ("throughput", base["effective_tasks_per_s"], arm["effective_tasks_per_s"])
        ]
        values += [
            (name + "_" + stat, base[name][stat], arm[name][stat])
            for name in ("ttft_ms", "latency_ms")
            for stat in ("mean", "p95")
        ]
        for metric, before, after in values:
            change = (
                100 * (after / before - 1) if before and after is not None else None
            )
            pair[metric + "_percent"] = change
            if change is not None:
                key = (arm["cell"], arm["arm"], metric)
                intervals.setdefault(key, []).append(change)
        paired.append(pair)
    ci = [
        {
            "cell": c,
            "candidate": n,
            "metric": m,
            **bootstrap_mean_ci(v, seed=stable_seed(n + ":" + m)),
        }
        for (c, n, m), v in intervals.items()
    ]
    return dict(
        arms=arms,
        paired=paired,
        incomplete=incomplete,
        issues=issues,
        paired_ci=ci,
        b_gpu_index=b_gpu,
        definitions=[
            "Explicit physical B GPU index must match launch CUDA_VISIBLE_DEVICES.",
            "CPU uses stable PID/create-time intervals and capture-start timestamps; unknown read times are not extrapolated.",
            "Ready observations with later GPU+host shortfall do not prove I/O waste.",
            "Run analyze_lifecycle for correctness; timing includes all completed rows.",
        ],
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results", action="append", required=True)
    cli.add_argument("--resources", action="append", default=[])
    cli.add_argument("--process-resources", action="append", default=[])
    cli.add_argument("--max-sample-gap-s", type=float, default=6)
    cli.add_argument("--b-gpu-index", type=int, choices=(0, 1), default=1)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if not math.isfinite(args.max_sample_gap_s) or args.max_sample_gap_s <= 0:
        cli.error("sample gap must be finite and positive")
    result = analyze(
        args.results,
        args.resources,
        args.process_resources,
        args.max_sample_gap_s,
        args.b_gpu_index,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "cost-resources.json", result)
    with (args.output / "paired.csv").open("w", newline="") as output:
        columns = sorted({key for pair in result["paired"] for key in pair})
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["paired"])
    return 2 if result["issues"] or result["incomplete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
