#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from hicache_policy_simulator import (  # noqa: E402
    SUPPORTED_POLICIES,
    load_trace_records,
    metrics_to_json_dict,
    simulate_policies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate L1/L2 HiCache policies on Mooncake-style traces."
    )
    parser.add_argument("--trace", required=True, help="Path to JSONL trace file.")
    parser.add_argument("--l1-pages", required=True, type=int, help="L1 capacity.")
    parser.add_argument("--l2-pages", required=True, type=int, help="L2 capacity.")
    parser.add_argument(
        "--policies",
        nargs="*",
        default=[",".join(SUPPORTED_POLICIES)],
        help=(
            "Policies to simulate. Accepts either comma-separated values or "
            f"separate arguments. Supported: {', '.join(SUPPORTED_POLICIES)}"
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Optional maximum number of trace records to read.",
    )
    parser.add_argument(
        "--write-through-threshold",
        type=int,
        default=1,
        help="Hit threshold for write-through backup policies.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for JSON metrics output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the interactive single-line progress display.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Refresh progress every N requests per policy, subject to rate limiting.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=0.5,
        help="Refresh progress at least every N seconds while requests are completing.",
    )
    parser.add_argument(
        "--sanity-check-interval",
        type=int,
        default=0,
        help=(
            "Run expensive full-tree invariant checks every N requests. "
            "Default 0 disables them for benchmark runs."
        ),
    )
    return parser.parse_args()


def _format_rate(value: float) -> str:
    return f"{100.0 * value:7.2f}%"


def print_summary(results) -> None:
    headers = (
        "policy",
        "requests",
        "pages",
        "l1_hit",
        "l1+l2_hit",
        "D",
        "H",
        "D+H",
        "unique",
        "failed_H",
    )
    rows = []
    for name, metrics in results.items():
        rows.append(
            (
                name,
                str(metrics.requests),
                str(metrics.total_input_pages),
                _format_rate(metrics.l1_hit_rate),
                _format_rate(metrics.l1_l2_hit_rate),
                str(metrics.d_pages),
                str(metrics.h_pages),
                str(metrics.dh_pages),
                str(metrics.unique_cached_pages),
                str(metrics.failed_h_allocations),
            )
        )

    widths = [
        max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * width for width in widths)))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    args = parse_args()
    policies = [
        policy
        for policy_arg in args.policies
        for policy in (p.strip() for p in policy_arg.split(","))
        if policy
    ]
    unsupported = [p for p in policies if p not in SUPPORTED_POLICIES]
    if unsupported:
        raise SystemExit(f"Unsupported policies: {unsupported}")

    records = load_trace_records(args.trace, max_requests=args.max_requests)
    results = simulate_policies(
        records=records,
        policies=policies,
        l1_pages=args.l1_pages,
        l2_pages=args.l2_pages,
        write_through_threshold=args.write_through_threshold,
        show_progress=not args.no_progress,
        progress_interval=args.progress_interval,
        progress_seconds=args.progress_seconds,
        sanity_check_interval=args.sanity_check_interval,
    )
    print_summary(results)

    if args.output_json:
        payload = {
            "trace": args.trace,
            "l1_pages": args.l1_pages,
            "l2_pages": args.l2_pages,
            "max_requests": args.max_requests,
            "write_through_threshold": args.write_through_threshold,
            "metrics": metrics_to_json_dict(results),
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
