#!/usr/bin/env python3

"""Run a fixed-size TCP KV transfer sweep."""

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
    parser.description = "Run one fixed-size TCP KV transfer experiment."
    for action in parser._actions:
        if action.dest == "suite":
            action.choices = ("tcp-fixed-size",)
            action.default = "tcp-fixed-size"
            action.help = "Run one fixed-size TCP transfer profile."
    parser.add_argument("--run-name", default="TCP_200_16MB")
    parser.add_argument("--src-host", default="192.168.0.42")
    parser.add_argument("--tgt-host", default="192.168.0.40")
    parser.add_argument("--ib-device", default="mlx5_bond_0")
    parser.add_argument("--rate-gbps", type=float, default=200.0)
    parser.add_argument("--size", default="16MB")
    parser.set_defaults(suite="tcp-fixed-size")
    return parser


def tcp_fixed_size_run(args: argparse.Namespace) -> auto.MatrixRun:
    endpoint = auto.one_lane_endpoint(args.src_host, args.tgt_host, args.ib_device)
    return auto.MatrixRun(
        run=args.run_name,
        lanes=(endpoint,),
        shards=1,
        max_bytes=args.size,
        sizes=args.size,
        protocol="tcp",
        lane_cap_gbps=args.rate_gbps,
        bg_rate_gbps=0.0,
        fg_rate_gbps=args.rate_gbps,
        capfill_lanes=(),
        capfill_rate_gbps=0.0,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runner = auto.Runner(args)
    runner.run_matrix(tcp_fixed_size_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
