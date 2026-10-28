"""Merge run_suite per-file metrics jsonl into one CSV: file_name,suite,status,duration.

Usage:
  python3 scripts/ci/xpu/metrics_to_csv.py --out results.csv \
      a.jsonl:stage-a-test-1-gpu-xpu,stage-b-test-1-gpu-xpu \
      b.jsonl:nightly-xpu-kernel-wheel-1-gpu

Each input is ``<jsonl>:<comma-separated suites>``, the suites passed to the
run_suite.py call that wrote it. A file's suite is looked up in its
``register_xpu_ci`` calls, since one run_suite call can pool several suites.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

from sglang.test.ci.ci_register import HWBackend, collect_tests

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def xpu_suites_by_file() -> dict:
    files = [
        f
        for f in glob.glob(
            os.path.join(REPO_ROOT, "test", "registered", "**", "*.py"),
            recursive=True,
        )
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    ]
    suites = defaultdict(set)
    for t in collect_tests(files, sanity_check=False):
        if t.backend == HWBackend.XPU:
            suites[os.path.basename(t.filename)].add(t.effective_suite)
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+", metavar="JSONL:SUITES")
    args = parser.parse_args()

    registered = xpu_suites_by_file()
    rows = []
    for spec in args.inputs:
        path, _, suite_arg = spec.rpartition(":")
        run_suites = [s for s in suite_arg.split(",") if s]
        if not os.path.exists(path):
            print(f"Warning: {path} not found; skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("kind") != "file":
                    continue
                name = rec["test_file"]
                matches = [s for s in run_suites if s in registered.get(name, ())]
                rows.append(
                    [
                        name,
                        matches[0] if matches else ",".join(run_suites),
                        rec["status"],
                        rec["duration"],
                    ]
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "suite", "status", "duration"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
