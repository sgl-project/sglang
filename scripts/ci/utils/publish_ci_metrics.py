#!/usr/bin/env python3
"""Print a workflow's CI metrics as one JSON object per line.

The `ci-metrics` job runs this over the artifacts its test jobs uploaded
(.github/actions/ci-metrics-upload): one directory per test job, holding
job.json (which job it was) and the *.jsonl files that
sglang.test.ci.metrics.record wrote. Every valid record is printed between
two marker lines, with its job attached, so the job's log is the whole of
the run's metrics:

    SGLANG_CI_METRICS_BEGIN {"v": 1, "records": 2, "jobs": 1, "skipped": 0}
    {"v": 1, "name": "median_ttft_ms", "value": 81.7, ..., "job": {...}}
    {"v": 1, "name": "output_throughput", "value": 1534.0, ..., "job": {...}}
    SGLANG_CI_METRICS_END

A line that isn't a valid record is counted as skipped and reported on
stderr, never printed between the markers. Standard library only: the job
runs it without installing sglang.

    python3 scripts/ci/utils/publish_ci_metrics.py <artifacts dir>
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

SCHEMA_VERSION = 1
BEGIN = "SGLANG_CI_METRICS_BEGIN"
END = "SGLANG_CI_METRICS_END"
KINDS = {"perf", "accuracy", "spec", "resource"}
JOB_FIELDS = ("name", "partition", "runner", "gpu_config")


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def valid(rec):
    """Why `rec` isn't a v1 record, or None if it is."""
    if not isinstance(rec, dict):
        return "not a JSON object"
    if rec.get("v") != SCHEMA_VERSION:
        return f"v is {rec.get('v')!r}, not {SCHEMA_VERSION}"
    if not isinstance(rec.get("name"), str) or not rec["name"]:
        return "no name"
    if not _number(rec.get("value")):
        return "value is not a finite number"
    if rec.get("kind") not in KINDS:
        return f"kind {rec.get('kind')!r} is not one of {sorted(KINDS)}"
    if rec.get("bound") is not None and not _number(rec["bound"]):
        return "bound is not a finite number"
    if not isinstance(rec.get("labels", {}), dict):
        return "labels is not an object"
    return None


def read_job(directory):
    try:
        job = json.loads((directory / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    return {k: job.get(k) for k in JOB_FIELDS}


def collect(root):
    """(records, jobs, problems) from every job directory under root."""
    records, jobs, problems = [], 0, []
    for directory in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            continue
        jobs += 1
        job = read_job(directory)
        for path in files:
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    problems.append(f"{directory.name}/{path.name}:{n}: not JSON")
                    continue
                why = valid(rec)
                if why:
                    problems.append(f"{directory.name}/{path.name}:{n}: {why}")
                    continue
                records.append({**rec, "job": job})
    return records, jobs, problems


def summary(records):
    """A markdown table for the step summary: records per metric."""
    counts = {}
    for rec in records:
        counts[rec["name"]] = counts.get(rec["name"], 0) + 1
    rows = "".join(f"| {name} | {n} |\n" for name, n in sorted(counts.items()))
    return f"### CI metrics\n\n{len(records)} records\n\n| metric | records |\n|---|---|\n{rows}"


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument(
        "root", help="directory holding one subdirectory per test job's artifact"
    )
    args = a.parse_args(argv)
    root = Path(args.root)
    records, jobs, problems = collect(root) if root.is_dir() else ([], 0, [])
    for problem in problems:
        print(f"skipped {problem}", file=sys.stderr)
    header = {
        "v": SCHEMA_VERSION,
        "records": len(records),
        "jobs": jobs,
        "skipped": len(problems),
    }
    print(f"{BEGIN} {json.dumps(header)}")
    for rec in records:
        print(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    print(END)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary and records:
        with open(step_summary, "a") as f:
            f.write(summary(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
