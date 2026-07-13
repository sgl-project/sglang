#!/usr/bin/env python3

"""Run tail single-NIC RDMA small-flow sweeps."""

import argparse
from typing import List, Optional

try:
    from scripts.playground.disaggregation.kv_transfer_bench import (
        kv_auto_experiment as auto,
    )
except ModuleNotFoundError:  # pragma: no cover - used when run as a script in-place.
    import kv_auto_experiment as auto


def build_parser() -> argparse.ArgumentParser:
    parser = auto.build_parser()
    parser.description = "Run tail single-NIC RDMA small-flow sweeps."
    for action in parser._actions:
        if action.dest == "suite":
            action.choices = ("tail-rdma-small",)
            action.default = "tail-rdma-small"
            action.help = "Run tail RDMA 1x100 and 1x200 small-flow sweeps."
    parser.add_argument("--src-host", default="fd03:4514:80:6240::1")
    parser.add_argument("--tgt-host", default="fd03:4514:80:7b80::1")
    parser.add_argument("--ib-device", default="mlx5_0")
    parser.add_argument("--rates", default="100,200", help="Comma-separated Gbps rates.")
    parser.add_argument("--small-max-mb", type=int, default=32)
    parser.set_defaults(suite="tail-rdma-small")
    return parser


def parse_rates(raw: str) -> List[float]:
    rates = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not rates:
        raise ValueError("at least one rate is required")
    return rates


def small_sizes(max_mb: int) -> str:
    if max_mb <= 0:
        raise ValueError(f"small-max-mb must be positive: {max_mb}")
    return ",".join(f"{size}MB" for size in range(1, max_mb + 1))


def tail_rdma_small_runs(args: argparse.Namespace) -> List[auto.MatrixRun]:
    endpoint = auto.one_lane_endpoint(args.src_host, args.tgt_host, args.ib_device)
    sizes = small_sizes(args.small_max_mb)
    max_bytes = f"{args.small_max_mb}MB"
    return [
        auto.MatrixRun(
            run=f"tail_rdma_small_1x{int(rate)}",
            lanes=(endpoint,),
            shards=1,
            max_bytes=max_bytes,
            sizes=sizes,
            protocol="rdma",
            lane_cap_gbps=rate,
            bg_rate_gbps=0.0,
            fg_rate_gbps=rate,
            capfill_lanes=(),
            capfill_rate_gbps=0.0,
        )
        for rate in parse_rates(args.rates)
    ]


def selected_runs(args: argparse.Namespace) -> List[auto.MatrixRun]:
    runs = tail_rdma_small_runs(args)
    if args.only:
        requested = set(args.only)
        runs = [run for run in runs if run.run in requested]
    return runs


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runner = auto.Runner(args)
    for run in selected_runs(args):
        runner.run_matrix(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
