#!/usr/bin/env python
"""Diff two benchmark run directories by cell_hash (regression check).

Usage:
    python benchmark/ascend_bench/compare.py RUN_A RUN_B \
        --metrics output_throughput p99_ttft_ms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asc_bench.diff_runs import DEFAULT_METRICS, diff_runs, render_compare_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="baseline run directory")
    parser.add_argument("run_b", help="candidate run directory")
    parser.add_argument(
        "--metrics", nargs="*", default=DEFAULT_METRICS, help="metrics to diff"
    )
    args = parser.parse_args(argv)

    diffs = diff_runs(Path(args.run_a), Path(args.run_b), args.metrics)
    if not diffs:
        print("no overlapping cell_hash between the two runs", file=sys.stderr)
        return 1
    print(render_compare_md(diffs, args.metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
