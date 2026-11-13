import argparse
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath

SUITE = "stage-b-test-1-gpu-small-amd"
RUNNER = "linux-mi300-1gpu-sglang"
ROCM_VERSION = "rocm10"
REPOSITORY = "sgl-project/sglang"


def parse_test_timings(log):
    timings = {}
    for line in log.splitlines():
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        start = line.find("{")
        if start < 0:
            continue
        try:
            record = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(record, dict)
            or not {"file", "passed", "elapsed"}.issubset(record)
            or record.get("passed") is not True
        ):
            continue
        filename = record.get("file")
        elapsed = record.get("elapsed")
        if not isinstance(filename, str):
            continue
        filename = filename.removeprefix("/sglang-checkout/")
        path = PurePosixPath(filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not filename.startswith("test/registered/")
        ):
            raise ValueError(f"Unexpected test path in timing record: {filename!r}")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise ValueError(f"Invalid elapsed time for {filename!r}: {elapsed!r}")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Invalid elapsed time for {filename!r}: {elapsed!r}")
        timings[filename] = max(float(elapsed), 1.0)
    return timings


def build_model(traces, source_runs):
    if len(traces) != len(source_runs) or not traces:
        raise ValueError("Supply one non-empty timing trace per source run")
    if len(source_runs) != len(set(source_runs)):
        raise ValueError("Each source run must be unique")
    samples = defaultdict(list)
    for trace in traces:
        if not trace:
            raise ValueError("A source run contains no successful test timings")
        for filename, elapsed in trace.items():
            samples[filename].append(elapsed)
    estimates = {
        filename: round(statistics.median(values), 2)
        for filename, values in sorted(samples.items())
    }
    return {
        "version": 1,
        "metadata": {
            "backend": "amd",
            "runner": RUNNER,
            "rocm_version": ROCM_VERSION,
            "source_runs": source_runs,
            "samples": {
                filename: len(values) for filename, values in sorted(samples.items())
            },
        },
        "est": {SUITE: estimates},
    }


def github_api(endpoint, *, raw=False):
    command = ["gh", "api"]
    if raw:
        command.append("--allow-escape-sequences")
    command.append(f"repos/{REPOSITORY}/{endpoint}")
    response = subprocess.check_output(command, text=True)
    return response if raw else json.loads(response)


def collect_run_timings(run_id, *, expected_shards):
    pattern = re.compile(rf"{re.escape(SUITE)} \({ROCM_VERSION}, {RUNNER}, (\d+)\)")
    jobs = []
    page = 1
    while True:
        batch = github_api(f"actions/runs/{run_id}/jobs?per_page=100&page={page}")[
            "jobs"
        ]
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    selected = [job for job in jobs if pattern.fullmatch(job["name"])]
    parts = [int(pattern.fullmatch(job["name"]).group(1)) for job in selected]
    if sorted(parts) != list(range(expected_shards)) or any(
        job["conclusion"] != "success" for job in selected
    ):
        raise ValueError(
            f"Run {run_id} needs {expected_shards} successful {ROCM_VERSION}/{RUNNER} shards"
        )
    trace = {}
    for job in selected:
        timings = parse_test_timings(
            github_api(f"actions/jobs/{job['id']}/logs", raw=True)
        )
        if not timings:
            raise ValueError(f"Job {job['id']} contains no successful test timings")
        duplicate = trace.keys() & timings.keys()
        if duplicate:
            raise ValueError(
                f"Run {run_id} repeats files across shards: {sorted(duplicate)}"
            )
        trace.update(timings)
    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Build MI300/ROCm 10 AMD shard estimates from completed CI runs."
    )
    parser.add_argument("--run-id", type=int, action="append", required=True)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_shards <= 0 or any(run_id <= 0 for run_id in args.run_id):
        parser.error("run IDs and expected shard count must be positive")
    traces = [
        collect_run_timings(run_id, expected_shards=args.expected_shards)
        for run_id in args.run_id
    ]
    args.output.write_text(
        json.dumps(build_model(traces, args.run_id), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
