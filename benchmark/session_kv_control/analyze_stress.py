#!/usr/bin/env python3
"""Offline verification and all-request analysis of completed stress cells."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import urlparse

from bench_report import distribution
from session_kv_bench import (cache_details, ready_tokens, require_completed,
                             require_pin_released, require_pin_success, status)
from stress_bench import output_equal, summarize


class VerificationError(ValueError):
    pass


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def locate(directory, name):
    candidates = [p for p in (directory/name, directory/(name+".gz")) if p.is_file()]
    if len(candidates) > 1:
        raise VerificationError(f"ambiguous plain/compressed files: {name}")
    return candidates[0] if candidates else None


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def read_json(path):
    with open_text(path) as handle:
        return json.load(handle)


class Evidence:
    """Verify physical files before parsing; gzip packing requires new hashes."""
    def __init__(self, directory):
        self.directory, self.checked = directory, {}
        index = locate(directory, "checksums.json")
        self.complete = index is not None
        if index:
            checksums = read_json(index)
            for name, expected in checksums.items():
                path = directory/name
                if Path(name).name != name or not path.is_file() or path.resolve().parent != directory.resolve():
                    raise VerificationError(f"checksum references missing/unsafe file: {name}")
                actual = sha256(path)
                if actual != expected:
                    raise VerificationError(f"checksum mismatch: {name}")
                self.checked[name] = actual

    def path(self, name, required=True):
        path = locate(self.directory, name)
        if path is None and required:
            raise VerificationError(f"missing evidence: {name}")
        if path and self.complete and path.name not in self.checked:
            raise VerificationError(f"evidence absent from checksums: {path.name}")
        return path

    def load(self, name, required=True):
        path = self.path(name, required)
        return read_json(path) if path else None


def numeric_differences(expected, observed, path="summary"):
    """Ignore prose/schema additions, require every recomputed numeric fact."""
    errors = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [path+": missing object"]
        for key, value in expected.items():
            if key not in ("notes", "retention_mode"):
                errors += numeric_differences(value, observed.get(key), path+"."+str(key))
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return [path+": list length mismatch"]
        for index, value in enumerate(expected):
            errors += numeric_differences(value, observed[index], f"{path}[{index}]")
    elif type(expected) in (int, float, bool):
        if type(observed) not in (int, float, bool) or not math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-8):
            errors.append(f"{path}: recomputed {expected!r} != recorded {observed!r}")
    return errors


def aggregate(arms):
    rows = [r for a in arms for r in a["rows"]]
    completed = [r for r in rows if r.get("generation")]
    resumed = [r for r in completed if r.get("phase") == "resume"]
    pressure = [r for r in completed if r.get("phase") == "pressure"]
    wall_s = sum(a["wall_ms"] for a in arms)/1000
    foreground_s = sum(a.get("foreground_wall_ms", 0) for a in arms)/1000
    planned = sum(a["planned_requests"] for a in arms)
    count_tokens = lambda items: sum(r["generation"]["meta_info"].get("completion_tokens", 0) for r in items)
    return {"arms": len(arms), "valid_arms": sum(a["valid"] for a in arms),
            "planned_requests": planned, "recorded_requests": len(rows), "completed_requests": len(completed),
            "failed_requests": planned-len(completed), "invalid_requests": sum(not r["valid"] for r in rows),
            "unattempted_requests": planned-len(rows), "output_tokens": count_tokens(completed),
            "sum_arm_wall_s": wall_s, "completed_requests_per_s": len(completed)/wall_s if wall_s else None,
            "output_tokens_per_s": count_tokens(completed)/wall_s if wall_s else None,
            "resumed_requests": len(resumed), "resume_ttft_ms": distribution([r["generation"]["ttft_ms"] for r in resumed]),
            "resume_latency_ms": distribution([r["generation"]["latency_ms"] for r in resumed]),
            "client_cycle_ms": distribution([r["client_cycle_ms"] for r in rows if "client_cycle_ms" in r]),
            "sum_l2_foreground_wall_s": foreground_s if foreground_s else None,
            "l2_foreground_requests_per_s": len(resumed)/foreground_s if foreground_s else None,
            "pressure_requests": len(pressure), "pressure_ttft_ms": distribution([r["generation"]["ttft_ms"] for r in pressure]),
            "pressure_latency_ms": distribution([r["generation"]["latency_ms"] for r in pressure]),
            "sum_pressure_wall_s": sum(a.get("pressure_wall_ms", 0) for a in arms)/1000,
            "resume_tier_tokens": {k: sum(r.get("cache", {}).get(k, 0) for r in resumed) for k in ("device", "cpu", "storage")},
            "repeated_prefill_tokens": sum(r.get("repeated_prefill_tokens", 0) for r in resumed),
            "prefetch_ready_before_resume": sum(r.get("prefetch_ready_before_resume", False) for r in rows),
            "prefetch_late": sum(r.get("prefetch_late", False) for r in rows),
            "unused_prepared_tokens_lower_bound": sum(r.get("unused_prepared_tokens_lower_bound", 0) for r in rows),
            "prefetch_usage_ambiguous": sum(r.get("prefetch_usage_ambiguous", False) for r in rows),
            "failed_control_rows": sum(any(status(r[key].get("operation", r[key])) != "completed"
                for key in ("publication", "demote", "host_copy", "prefetch") if r.get(key))
                or any(any(word in str(error).lower() for word in ("prefetch", "demote", "control", "operation", "publication", "copy"))
                       for error in r.get("errors", [])) for r in rows),
            "cleanup_failures": sum(not c["valid"] for a in arms for c in a.get("cleanup", []))}


def verify_rows(arms, manifest):
    errors, expected = [], {}
    length = manifest["args"]["length"]
    proofs = {"transfer_receipts": 0, "ready_before_resume_checked": 0, "cleanup_tombstones": 0}
    for arm in arms:
        if arm["planned_requests"] != len(arm["rows"]):
            errors.append(arm["id"]+": planned != recorded requests; unattempted jobs retained")
        for row in arm["rows"]:
            label = row.get("id", arm["id"]+"-unknown")
            generation = row.get("generation")
            if row["valid"]:
                required = []
                mode = manifest["args"].get("retention_mode", "demote")
                if row.get("phase") == "prime":
                    required = ["publication", "demote"]
                elif row.get("phase") == "pressure":
                    required = ["host_copy" if mode == "natural" else "demote"]
                elif manifest["args"]["command"] == "l3" and row.get("phase") == "resume":
                    required = (["demote"] if mode == "demote" else [])+(["prefetch"] if arm["arm"] == "prefetch" else [])
                for key in required:
                    if key not in row:
                        errors.append(label+": valid row lacks "+key+" proof")
            if generation:
                try:
                    tiers = cache_details(generation["meta_info"])
                    if any(tiers[k] != row.get("cache", {}).get(k) for k in ("device", "cpu", "storage")):
                        raise VerificationError("row cache tier counts disagree with generation metadata")
                    for key in ("ttft_ms", "latency_ms"):
                        if not isinstance(generation[key], (int, float)) or not math.isfinite(generation[key]) or generation[key] < 0:
                            raise VerificationError("invalid "+key)
                    if generation["ttft_ms"] > generation["latency_ms"]:
                        raise VerificationError("TTFT exceeds complete latency")
                    count = generation["meta_info"]["completion_tokens"]
                    if generation.get("output_ids") is not None and len(generation["output_ids"]) != count:
                        raise VerificationError("output ID count differs from completed token count")
                    if row.get("phase") == "resume":
                        wanted = expected.setdefault(row["case"], generation)
                        if not output_equal(wanted, generation):
                            raise VerificationError("output IDs/text differ across arms/passes")
                        repeated = max(0, length-sum(tiers[k] for k in ("device", "cpu", "storage")))
                        if repeated != row.get("repeated_prefill_tokens"):
                            raise VerificationError("repeated prefill accounting disagrees with actual tiers")
                except Exception as exc:
                    errors.append(label+": "+str(exc))
            for key in ("publication", "demote", "host_copy", "prefetch"):
                transfer = row.get(key)
                if not transfer:
                    continue
                proofs["transfer_receipts"] += 1
                try:
                    require_completed(transfer, length, key)
                except Exception as exc:
                    if row["valid"]:
                        errors.append(label+": valid row has failed transfer proof: "+str(exc))
            if "prefetch" in row:
                prepared = ready_tokens(row["prefetch"]["operation"])
                if prepared != row.get("prepared_tokens"):
                    errors.append(label+": prepared token count differs from ready scope")
                if row.get("prefetch_ready_before_resume"):
                    proofs["ready_before_resume_checked"] += 1
                    completed_at = row["prefetch"].get("timing", {}).get("completion_observed_monotonic")
                    if (not generation or completed_at is None or completed_at > generation["started_monotonic"]
                            or prepared < length or status(row["prefetch"]["operation"]) != "completed"):
                        errors.append(label+": readiness does not precede actual generation send")
        for receipt in arm.get("cleanup", []):
            if receipt.get("valid"):
                claim = receipt.get("query", {})
                try:
                    require_pin_released(claim, receipt["claim_id"], claim.get("owner_epoch"))
                    proofs["cleanup_tombstones"] += 1
                except Exception as exc:
                    errors.append(arm["id"]+": invalid cleanup proof: "+str(exc))
    return errors, proofs


def raw_pin_proofs(evidence, arms, manifest):
    path = evidence.path("raw.jsonl")
    acquired, errors, generation_counts = {}, [], {"attempted": 0, "by_origin": {}}
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                errors.append(f"raw line {number}: {exc}")
                continue
            if row.get("kind") == "generate":
                generation_counts["attempted"] += 1
                origin = row.get("origin", "unknown")
                generation_counts["by_origin"][origin] = generation_counts["by_origin"].get(origin, 0)+1
            if row.get("kind") != "pin_acquire":
                continue
            op = row.get("operation", {})
            claim = (op.get("public_result") or {}).get("claim") or {}
            try:
                owner = op.get("ref", {}).get("context", {}).get("engine_epoch")
                if owner is None:
                    owner = manifest["capabilities"]["b"]["engine_epoch"]
                require_pin_success(op, manifest["args"]["length"], claim.get("claim_id"), owner)
                acquired.setdefault(row["trial"], set()).add(claim["claim_id"])
            except Exception as exc:
                errors.append(f"raw PIN proof line {number}: {exc}")
    for arm in arms:
        if arm["arm"] == "pin" and arm["valid"] and len(acquired.get(arm["id"], set())) != manifest["args"]["hot_sessions"]:
            errors.append(arm["id"]+": missing full active PIN acquisition proofs")
    return errors, {k: len(v) for k, v in acquired.items()}, generation_counts


def analyze_cell(directory):
    cell = {"name": directory.name, "path": str(directory.resolve()), "verification_errors": []}
    try:
        evidence = Evidence(directory)
        failure = evidence.load("failure.json", False)
        manifest = evidence.load("manifest.json", False)
        if failure:
            cell.update(state="failed_setup_or_run", failure=failure, args=(manifest or {}).get("args"))
            return cell
        if not manifest or not evidence.complete:
            cell.update(state="incomplete", reason="manifest/checksum closure missing; live partial cells are excluded")
            return cell
        args = manifest.get("args", {})
        cell["args"] = args
        if args.get("command") not in ("l2", "l3"):
            cell["state"] = "unrelated"
            return cell
        if args.get("allow_small_working_set") is not False:
            cell["state"] = "functional_only"
            return cell
        schedule = evidence.load("schedule.json")
        rounds = args.get("rounds") if args["command"] == "l2" else args.get("blocks")
        if len(schedule) != 2*rounds:
            raise VerificationError("schedule does not contain complete paired blocks")
        if len({spec["id"] for spec in schedule}) != len(schedule):
            raise VerificationError("duplicate arm IDs in schedule")
        wanted = {"on_demand", "pin" if args["command"] == "l2" else "prefetch"}
        for block in range(rounds):
            if {spec["arm"] for spec in schedule if spec["block"] == block} != wanted:
                raise VerificationError("incomplete paired block in schedule")
        arms = [evidence.load(spec["id"]+".json") for spec in schedule]
        observed = evidence.load("summary.json")
        # Preserve the input schedule order: summarize uses it for arm ordering.
        recomputed = summarize(arms)
        errors = numeric_differences(recomputed, observed)
        row_errors, proofs = verify_rows(arms, manifest)
        pin_errors, acquisitions, generation_counts = raw_pin_proofs(evidence, arms, manifest)
        errors += row_errors+pin_errors
        source = evidence.load("source-publication.json", False)
        source_rows = (source or {}).get("rows", [])
        for row in source_rows:
            if row.get("generation"):
                try:
                    actual = cache_details(row["generation"]["meta_info"])
                    if any(actual[k] != row.get("cache", {}).get(k) for k in ("device", "cpu", "storage")):
                        errors.append("source publication cache tier metadata mismatch")
                except Exception as exc:
                    errors.append("source publication cache metadata: "+str(exc))
            if row.get("valid"):
                try:
                    require_completed(row["publication"], args["length"], "source publication")
                except Exception as exc:
                    errors.append("source publication: "+str(exc))
        if args["command"] == "l3" and (not source or len(source_rows) != args["sessions"]):
            errors.append("full source publication dataset missing")
        cell.update(state="completed", verified=not errors, valid=all(a["valid"] for a in arms),
                    destination_boot_server={k: (manifest.get("metadata", {}).get("b", {}).get("server") or {}).get(k)
                                             for k in ("max_total_num_tokens", "memory_usage")},
                    verification_errors=errors, checked_files=evidence.checked,
                    groups={name: aggregate([a for a in arms if a["arm"] == name]) for name in sorted({a["arm"] for a in arms})},
                    paired=recomputed["paired"], recomputed_summary=recomputed, proofs={**proofs, "pin_acquisitions": acquisitions},
                    source_publication={"planned_requests": args.get("sessions", 0) if args["command"] == "l3" else 0,
                        "recorded_requests": len(source_rows), "completed_requests": sum(bool(r.get("generation")) for r in source_rows),
                        "invalid_requests": sum(not r["valid"] for r in source_rows),
                        "failed_generation_requests": args.get("sessions", 0)-sum(bool(r.get("generation")) for r in source_rows) if args["command"] == "l3" else 0,
                        "wall_ms": (source or {}).get("wall_ms", 0)},
                    raw_generation_counts=generation_counts,
                    total_generation_requests_including_source=generation_counts["attempted"],
                    arm_windows=[arm_window(a) for a in arms],
                    total_completed_generation_requests_including_source=aggregate(arms)["completed_requests"]+sum(bool(r.get("generation")) for r in source_rows),
                    total_planned_generation_requests_including_source=sum(a["planned_requests"] for a in arms)+(args.get("sessions", 0) if args["command"] == "l3" else 0))
    except Exception as exc:
        cell.update(state="verification_failed", verified=False)
        cell["verification_errors"].append(str(exc))
    return cell


def arm_window(arm):
    starts = [r.get("started_monotonic", r.get("client_cycle_started_monotonic", r.get("generation", {}).get("started_monotonic"))) for r in arm["rows"]]
    starts = [s for s in starts if s is not None]
    ends = [r.get("completed_monotonic", r.get("generation", {}).get("completed_monotonic")) for r in arm["rows"]]
    ends = [e for e in ends if e is not None]
    return {"id": arm["id"], "arm": arm["arm"], "block": arm["block"],
            "start": min(starts) if starts else None,
            "end": min(starts)+arm["wall_ms"]/1000 if starts else None,
            "last_row_end": max(ends) if ends else None,
            "bounds": "approximate: earliest row start plus full arm wall time; includes shifted flush/cleanup boundary; may include adjacent idle/work"}


def prometheus(text):
    out = {}
    if isinstance(text, str):
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.match(r"([^\s{]+)(?:\{[^\n]*\})?\s+([-+\d.eE]+)(?:\s|$)", line)
            if match:
                value = float(match[2])
                if math.isfinite(value):
                    out.setdefault(match[1], []).append(value)
    return out


def with_capacity(sample, capacity):
    sample = {**sample, "metrics": dict(sample["metrics"])}
    available = sample["metrics"].get("b_gpu_free_tokens")
    if isinstance(capacity, (int, float)) and capacity > 0:
        sample["metrics"]["b_gpu_capacity_tokens"] = capacity
        if available is not None and 0 <= available <= capacity:
            allocated = capacity-available
            sample["metrics"]["b_gpu_resident_or_reserved_tokens"] = allocated
            sample["metrics"]["b_gpu_resident_or_reserved_fraction"] = allocated/capacity
    return sample


def resource_sample(row, b_url, capacity=None):
    sample = {"monotonic": row["monotonic"], "gpu": {}, "metrics": {}}
    for values in csv.reader(io.StringIO(row.get("gpu_csv", ""))):
        if len(values) >= 7:
            try:
                index = int(values[0])
                if index in (0, 1):
                    sample["gpu"][str(index)] = {"memory_total_mib": float(values[3]), "memory_used_mib": float(values[4]),
                                                  "busy_percent": float(values[5]), "memory_busy_percent": float(values[6])}
            except ValueError:
                continue
    names = {"sglang:token_usage": "b_gpu_active_token_usage_fraction", "sglang:kv_used_tokens": "b_gpu_active_tokens",
             "sglang:kv_available_tokens": "b_gpu_free_tokens", "sglang:kv_evictable_tokens": "b_gpu_evictable_tokens",
             "sglang:hicache_host_used_tokens": "b_host_used_tokens", "sglang:hicache_host_total_tokens": "b_host_total_tokens",
             "master_allocated_bytes": "master_allocated_bytes", "master_total_capacity_bytes": "master_capacity_bytes"}
    for reply in row.get("http", []):
        url = reply.get("url", "")
        is_b = urlparse(url).netloc == urlparse(b_url).netloc
        parsed = prometheus(reply.get("body"))
        for raw, label in names.items():
            if raw in parsed and (is_b if label.startswith("b_") else True):
                # One TP1 owner is expected. Multiple series remain explicit.
                sample["metrics"][label] = max(parsed[raw])
    return with_capacity(sample, capacity)


def resource_peaks(samples):
    def describe(values):
        return {"samples": len(values), "min": min(values) if values else None, "mean": statistics.mean(values) if values else None,
                "sampled_peak": max(values) if values else None}
    times = [s["monotonic"] for s in samples]
    return {"samples": len(samples), "first_monotonic": min(times) if times else None, "last_monotonic": max(times) if times else None,
            "observed_interval_s": distribution([b-a for a, b in zip(sorted(times), sorted(times)[1:])]),
            "gpu": {gpu: {key: describe([s["gpu"][gpu][key] for s in samples if gpu in s["gpu"]])
                          for key in ("memory_used_mib", "busy_percent", "memory_busy_percent")}
                    for gpu in sorted({g for s in samples for g in s["gpu"]})},
            "metrics": {key: describe([s["metrics"][key] for s in samples if key in s["metrics"]])
                        for key in sorted({k for s in samples for k in s["metrics"]})},
            "definition": "sampled maxima, not exact memory peaks or physical I/O; arm windows approximate; active token metrics exclude evictable radix KV; capacity minus free counts resident/reserved allocations including transfer work, not purely cached tokens"}


def attach_resources(cells, path):
    if path is None:
        return {"available": False}
    b_urls = {c["args"]["engine_b"] for c in cells if c["state"] == "completed"}
    by_url = {url: [] for url in b_urls}
    parse_errors = []
    with open_text(path) as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                for url in b_urls:
                    by_url[url].append(resource_sample(row, url))
            except (ValueError, KeyError, TypeError) as exc:
                parse_errors.append({"line": number, "error": str(exc)})
    for cell in cells:
        if cell["state"] != "completed":
            continue
        server = cell.get("destination_boot_server", {})
        capacity = server.get("max_total_num_tokens")
        if capacity is None:
            capacity = (server.get("memory_usage") or {}).get("token_capacity")
        samples = [with_capacity(s, capacity) for s in by_url[cell["args"]["engine_b"]]]
        selected = []
        cell["resources_by_arm"] = {}
        for window in cell["arm_windows"]:
            subset = [s for s in samples if window["start"] is not None and window["start"] <= s["monotonic"] <= window["end"]]
            selected += subset
            cell["resources_by_arm"][window["id"]] = resource_peaks(subset)
        unique = {s["monotonic"]: s for s in selected}
        cell["resources"] = resource_peaks(list(unique.values()))
    return {"available": True, "path": str(path.resolve()), "sha256": sha256(path), "parse_errors": parse_errors,
            "note": "external observer data; sampled 2s nominal cadence, observed cadence recorded; no synchronization beyond local monotonic clock"}


def report_text(analysis):
    cells = analysis["cells"]
    lines = ["# Sustained KV stress measurements", "", "All completed requests remain in latency and throughput accounting, including invalid control paths. Failed requests remain in planned denominators. Source publication is separate. Negative deltas mean lower latency/cost; every direction is retained.", "",
             "| Cell | mode | group | verified / valid | planned / completed / invalid | TTFT mean / p95 ms | arm seconds | requests/s | output tokens/s | host / storage tokens |",
             "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    for cell in cells:
        if cell["state"] != "completed":
            continue
        for name, group in cell["groups"].items():
            lines.append(f"| {cell['name']} | {cell['args'].get('retention_mode', 'demote')} | {name} | {cell['verified']} / {cell['valid']} | {group['planned_requests']} / {group['completed_requests']} / {group['invalid_requests']} | {fmt(group['resume_ttft_ms']['mean'])} / {fmt(group['resume_ttft_ms']['p95'])} | {fmt(group['sum_arm_wall_s'])} | {fmt(group['completed_requests_per_s'])} | {fmt(group['output_tokens_per_s'])} | {group['resume_tier_tokens']['cpu']} / {group['resume_tier_tokens']['storage']} |")
    lines += ["", "| Cell / paired block | candidate | valid | mean TTFT delta ms | whole arm delta ms | requests/s delta |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for cell in cells:
        for pair in cell.get("paired", []):
            lines.append(f"| {cell['name']} / {pair['block']} | {pair['candidate']} | {pair['valid']} | {fmt(pair['mean_resume_ttft_delta_ms'])} | {fmt(pair['wall_delta_ms'])} | {fmt(pair['completed_requests_per_s_delta'])} |")
    lines += ["", "GPU and host peaks in analysis.json are sampled maxima. Arm attribution uses approximate bounds inferred from row starts and arm duration. GPU allocated memory does not establish occupied KV capacity. Active token_usage/kv_used exclude evictable radix KV; resident/reserved tokens use boot capacity minus kv_available and can include allocation/transfer work. Three blocks do not establish independent request-level confidence intervals or stable p99.", "", "L2 foreground, pressure, complete arm metrics and L3 tool-wait client cycles are separate in analysis.json. The natural profile preserves native GPU residency; the demote profile deliberately removes GPU prefixes. Compare profiles separately.", ""]
    for cell in cells:
        source = cell.get("source_publication")
        if source and source["planned_requests"]:
            lines.append(f"- {cell['name']} source publication: {source['completed_requests']}/{source['planned_requests']} generations; {fmt(source['wall_ms']/1000)} seconds, excluded from arm throughput.")
        if cell["state"] != "completed" or cell.get("verification_errors") or not cell.get("valid", True):
            lines.append(f"- {cell['name']}: {cell['state']}; {json.dumps(cell.get('verification_errors') or cell.get('failure') or cell.get('reason') or 'invalid measured arms', ensure_ascii=False)}")
    return "\n".join(lines)+"\n"


def export_plot_csv(output, cells):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"generated": False, "reason": "matplotlib unavailable; no packages installed"}
    records = []
    for cell in cells:
        for name, group in cell.get("groups", {}).items():
            records.append({"cell": cell["name"], "group": name, "retention_mode": cell["args"].get("retention_mode", "demote"),
                            "verified": cell["verified"], "valid": cell["valid"], "completed": group["completed_requests"],
                            "invalid": group["invalid_requests"], "ttft_mean_ms": group["resume_ttft_ms"]["mean"],
                            "requests_per_s": group["completed_requests_per_s"], "output_tokens_per_s": group["output_tokens_per_s"]})
    if not records:
        return {"generated": False, "reason": "no completed cells"}
    with (output/"groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(records)*.8), 5))
    labels = [r["cell"]+"\n"+r["group"] for r in records]
    axes[0].bar(labels, [r["ttft_mean_ms"] if r["ttft_mean_ms"] is not None else math.nan for r in records])
    axes[0].set_ylabel("All completed resume TTFT mean (ms)")
    axes[1].bar(labels, [r["requests_per_s"] if r["requests_per_s"] is not None else math.nan for r in records])
    axes[1].set_ylabel("Whole arm completed requests/s")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=60)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("KV stress: paired measurements; invalid rows retained, no inferred confidence intervals")
    figure.tight_layout()
    figure.savefig(output/"stress-comparison.png", dpi=180)
    plt.close(figure)
    return {"generated": True, "files": ["groups.csv", "stress-comparison.png"]}


def run(results, output):
    output.mkdir(parents=True, exist_ok=False)
    directories = {p.parent for pattern in ("manifest.json", "manifest.json.gz", "failure.json", "failure.json.gz", "raw.jsonl", "raw.jsonl.gz") for p in results.rglob(pattern)}
    cells = [analyze_cell(p) for p in sorted(directories)]
    resource_path = locate(results, "resources.jsonl")
    resources = attach_resources(cells, resource_path)
    completed = [c for c in cells if c["state"] == "completed"]
    analysis = {"results": str(results.resolve()), "cells": cells, "resources": resources,
                "counts": {"completed_formal_cells": len(completed), "verified_cells": sum(c["verified"] for c in completed),
                           "valid_cells": sum(c["valid"] for c in completed),
                           "total_generation_requests_including_source": sum(c["total_generation_requests_including_source"] for c in completed),
                           "total_planned_generation_requests_including_source": sum(c["total_planned_generation_requests_including_source"] for c in completed),
                           "total_completed_generation_requests_including_source": sum(c["total_completed_generation_requests_including_source"] for c in completed)},
                "physical_io_bytes": None, "exact_host_peak_bytes": None}
    analysis["exports"] = export_plot_csv(output, cells)
    (output/"analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False)+"\n")
    (output/"report.md").write_text(report_text(analysis))
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        analysis = run(args.results, args.output)
        print(json.dumps(analysis["counts"]))
        failed = any(c["state"] == "verification_failed" or (c["state"] == "completed" and (not c["verified"] or not c["valid"])) for c in analysis["cells"])
        return 2 if failed else 0
    except Exception as exc:
        print("STRESS ANALYSIS FAILED: "+str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
