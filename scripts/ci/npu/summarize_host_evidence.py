"""Summarize NPU pinned-host diagnostic lines from downloaded CI job logs.

Usage:
    python3 scripts/ci/npu/summarize_host_evidence.py job.log [other-job.log ...]

The output is JSON, one summary per input. This script needs only the Python
standard library and does not require NPU access or model weights.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

MARKERS = (
    "NPU host baseline: ",
    "NPU pinned host monitor: ",
    "NPU pinned host allocation failed: ",
)
AUTO_SIZE = re.compile(
    r"HiCache auto-sizing: ratio (?P<requested>[\d.]+) -> (?P<actual>[\d.]+); "
    r"(?P<budget>[\d.]+) GiB host memory per rank "
    r"\(fraction (?P<fraction>[\d.]+), (?P<ranks>\d+) ranks on this host\), "
    r"host pools (?P<pools>[\d.]+) GiB"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _records(lines):
    for raw_line in lines:
        line = ANSI.sub("", raw_line)
        for marker in MARKERS:
            if marker in line:
                try:
                    record = ast.literal_eval(line.split(marker, 1)[1].strip())
                except (SyntaxError, ValueError):
                    break
                if isinstance(record, dict):
                    yield marker, record
                break
        else:
            match = AUTO_SIZE.search(line)
            if match:
                yield (
                    "auto_size",
                    {key: float(value) for key, value in match.groupdict().items()},
                )


def _min_numeric(records, key):
    values = [
        record[key] for record in records if isinstance(record.get(key), (int, float))
    ]
    return min(values) if values else None


def _max_numeric(records, key):
    values = [
        record[key] for record in records if isinstance(record.get(key), (int, float))
    ]
    return max(values) if values else None


def summarize(lines) -> dict[str, object]:
    parsed = list(_records(lines))
    samples = [record for marker, record in parsed if marker != "auto_size"]
    timestamps = sorted(
        record["timestamp_utc"] for record in samples if record.get("timestamp_utc")
    )
    with_headroom = [
        record
        for record in samples
        if isinstance(record.get("effective_host_memory_headroom_bytes"), int)
    ]
    tightest = min(
        with_headroom,
        key=lambda record: record["effective_host_memory_headroom_bytes"],
        default=None,
    )
    numa_min_free = {}
    for record in samples:
        for node, memory in record.get("numa_node_memory", {}).items():
            free = memory.get("memfree_bytes")
            if isinstance(free, int):
                numa_min_free[node] = min(numa_min_free.get(node, free), free)

    events = [
        record.get("cgroup_memory_events", {})
        for record in samples
        if isinstance(record.get("cgroup_memory_events"), dict)
    ]
    event_delta = {}
    for key in ("high", "max", "oom", "oom_kill"):
        values = [event[key] for event in events if isinstance(event.get(key), int)]
        if len(values) > 1:
            event_delta[key] = max(values) - min(values)

    auto_size = [record for marker, record in parsed if marker == "auto_size"]
    for record in auto_size:
        record["inferred_effective_headroom_gib"] = round(
            record["budget"] * record["ranks"] / record["fraction"] + 10, 2
        )

    return {
        "diagnostic_records": len(samples),
        "first_timestamp_utc": timestamps[0] if timestamps else None,
        "last_timestamp_utc": timestamps[-1] if timestamps else None,
        "host_boot_id_hashes": sorted(
            {
                record["host_boot_id_hash"]
                for record in samples
                if record.get("host_boot_id_hash")
            }
        ),
        "ci_run_ids": sorted(
            {record["ci_run_id"] for record in samples if record.get("ci_run_id")}
        ),
        "runner_names": sorted(
            {record["runner_name"] for record in samples if record.get("runner_name")}
        ),
        "limiting_scopes_seen": sorted(
            {
                record["effective_host_memory_limiter"]
                for record in samples
                if record.get("effective_host_memory_limiter")
            }
        ),
        "cgroup_limit_kinds_seen": sorted(
            {
                record["cgroup_memory_limit_kind"]
                for record in samples
                if record.get("cgroup_memory_limit_kind")
            }
        ),
        "min_host_available_gib": _to_gib(
            _min_numeric(samples, "host_memory_available_bytes")
        ),
        "min_cgroup_headroom_gib": _to_gib(
            _min_numeric(samples, "cgroup_memory_headroom_bytes")
        ),
        "min_effective_headroom_gib": _to_gib(
            _min_numeric(samples, "effective_host_memory_headroom_bytes")
        ),
        "tightest_headroom_snapshot": (
            {
                key: tightest.get(key)
                for key in (
                    "timestamp_utc",
                    "pid",
                    "phase",
                    "effective_host_memory_limiter",
                    "host_memory_available_bytes",
                    "cgroup_memory_headroom_bytes",
                    "cgroup_memory_limit_kind",
                    "cgroup_memory_limit_path",
                    "cgroup_memory_events",
                    "numa_node_memory",
                    "numa_bind_policies",
                )
            }
            if tightest is not None
            else None
        ),
        "min_numa_free_gib": {
            node: _to_gib(value) for node, value in sorted(numa_min_free.items())
        },
        "max_pinned_reserved_gib_per_process": _to_gib(
            _max_numeric(samples, "pinned_reserved_bytes")
        ),
        "max_pinned_reserved_peak_gib_per_process": _to_gib(
            _max_numeric(samples, "pinned_reserved_peak_bytes")
        ),
        "pinned_peak_phases": sorted(
            {
                record["reserved_peak_phase"]
                for record in samples
                if record.get("reserved_peak_phase")
            }
        ),
        "cgroup_event_delta": event_delta,
        "pinned_allocation_failures": sum(
            marker == "NPU pinned host allocation failed: " for marker, _ in parsed
        ),
        "failure_sites": [
            {
                key: record.get(key)
                for key in ("timestamp_utc", "site", "requested_bytes", "pid")
            }
            for marker, record in parsed
            if marker == "NPU pinned host allocation failed: "
        ],
        "auto_size": auto_size,
    }


def _to_gib(value):
    return round(value / 1024**3, 2) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs", type=Path, nargs="+", help="Downloaded CI job log files"
    )
    args = parser.parse_args()
    for path in args.logs:
        try:
            with path.open(errors="replace") as file:
                summary = summarize(file)
        except OSError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"file": str(path), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
