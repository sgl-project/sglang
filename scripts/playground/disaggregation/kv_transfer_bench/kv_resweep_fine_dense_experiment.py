#!/usr/bin/env python3

"""Rerun visible KV transfer experiments with dense 1..32MiB logical sizes.

This script reuses the run definitions and Runner from kv_auto_experiment.py,
but replaces each matrix sweep with:

  logical flow sizes 1MiB, 2MiB, ..., 32MiB, then the original >32MiB points.

For split runs, kv_auto_experiment.py passes per-shard sizes to each foreground
initiator and aggregates them afterward. This wrapper therefore converts the
1..32MiB logical schedule into per-shard sizes before launching the run.
"""

import argparse
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from scripts.playground.disaggregation.kv_transfer_bench import (
        kv_auto_experiment as base,
    )
except ModuleNotFoundError:  # pragma: no cover - used when run as a script in-place.
    import kv_auto_experiment as base


MIB = 1024**2
FINE_LOGICAL_MAX_BYTES = 32 * MIB
DEFAULT_SRC_IPS = dict(base.SRC_IPS)
DEFAULT_TGT_IPS = dict(base.TGT_IPS)

SUITE_CHOICES = (
    "singleflow",
    "fixed-missing",
    "head-tcp-sweep",
    "head-tcp",
    "multi-hca-bg",
    "multi-hca-portcap-bg",
    "legacy-multi-hca-unaverage",
    "multi-hca-unaverage",
    "ratelimit-empty",
    "small-flow",
    "competition",
    "all-visible",
    "all",
    "list",
)


def _format_size_for_cli(num_bytes: int) -> str:
    if num_bytes % (1024**3) == 0:
        return f"{num_bytes // (1024**3)}GB"
    if num_bytes % MIB == 0:
        return f"{num_bytes // MIB}MB"
    if num_bytes % 1024 == 0:
        return f"{num_bytes // 1024}KB"
    return f"{num_bytes}B"


def _csv_sizes(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_lane_hosts(raw: str) -> Tuple[str, str, str, str]:
    hosts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(hosts) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated lane hosts")
    return hosts


def apply_lane_host_overrides(args: argparse.Namespace) -> None:
    src_hosts = getattr(args, "src_rdma_hosts", None)
    tgt_hosts = getattr(args, "tgt_rdma_hosts", None)

    if src_hosts:
        for lane, host in enumerate(src_hosts):
            base.SRC_IPS[lane] = host
        if args.single_nic_src_host == DEFAULT_SRC_IPS[0]:
            args.single_nic_src_host = base.SRC_IPS[0]
        if args.head_rdma_src_host == DEFAULT_SRC_IPS[0]:
            args.head_rdma_src_host = base.SRC_IPS[0]

    if tgt_hosts:
        for lane, host in enumerate(tgt_hosts):
            base.TGT_IPS[lane] = host
        if args.single_nic_tgt_host == DEFAULT_TGT_IPS[0]:
            args.single_nic_tgt_host = base.TGT_IPS[0]
        if args.head_rdma_tgt_host == DEFAULT_TGT_IPS[0]:
            args.head_rdma_tgt_host = base.TGT_IPS[0]


def fine_dense_sizes_for_shards(shards: int, original_sizes: str) -> str:
    """Return per-shard sizes for a 1..32MiB logical sweep plus old >32MiB.

    The input and output strings are the sizes passed to each foreground
    initiator. For a 4-shard run, for example, logical 1MiB is 256KB per shard.
    """

    if shards <= 0:
        raise ValueError(f"shards must be positive: {shards}")

    sizes: List[str] = []
    seen_per_shard_bytes = set()

    for logical_mib in range(1, 33):
        logical_bytes = logical_mib * MIB
        if logical_bytes % shards != 0:
            raise ValueError(f"{logical_mib}MiB is not divisible across {shards} shards")
        per_shard_bytes = logical_bytes // shards
        sizes.append(_format_size_for_cli(per_shard_bytes))
        seen_per_shard_bytes.add(per_shard_bytes)

    for size in _csv_sizes(original_sizes):
        per_shard_bytes = base.parse_size(size)
        logical_bytes = per_shard_bytes * shards
        if logical_bytes <= FINE_LOGICAL_MAX_BYTES:
            continue
        if per_shard_bytes in seen_per_shard_bytes:
            continue
        sizes.append(size)
        seen_per_shard_bytes.add(per_shard_bytes)

    return ",".join(sizes)


def logical_sizes_bytes(run: base.MatrixRun) -> List[int]:
    return [base.parse_size(size) * run.shards for size in _csv_sizes(run.sizes)]


def with_fine_dense_sizes(run: base.MatrixRun) -> base.MatrixRun:
    return base.MatrixRun(
        run=run.run,
        lanes=run.lanes,
        shards=run.shards,
        max_bytes=run.max_bytes,
        sizes=fine_dense_sizes_for_shards(run.shards, run.sizes),
        protocol=run.protocol,
        lane_cap_gbps=run.lane_cap_gbps,
        bg_rate_gbps=run.bg_rate_gbps,
        fg_rate_gbps=run.fg_rate_gbps,
        capfill_lanes=run.capfill_lanes,
        capfill_rate_gbps=run.capfill_rate_gbps,
    )


def _fine_runs(runs: Iterable[base.MatrixRun]) -> List[base.MatrixRun]:
    return [with_fine_dense_sizes(run) for run in runs]


def singleflow_runs(args: Optional[argparse.Namespace] = None) -> List[base.MatrixRun]:
    return _fine_runs(base.matrix_runs(args) + base.ratelimit_empty_runs())


def fixed_missing_runs(args: Optional[argparse.Namespace] = None) -> List[base.MatrixRun]:
    return _fine_runs(base.matrix_runs(args))


def head_tcp_runs(args: Optional[argparse.Namespace] = None) -> List[base.MatrixRun]:
    return _fine_runs(base.head_tcp_sweep_runs(args))


def multi_hca_portcap_runs() -> List[base.MatrixRun]:
    return _fine_runs(base.multi_hca_matrix_runs())


def legacy_multi_hca_unaverage_runs() -> List[base.MatrixRun]:
    runs: List[base.MatrixRun] = []
    specs: Sequence[Tuple[str, Tuple[int, ...], float]] = (
        ("200_2x100", (0, 1), 100),
        ("200_4x50", (0, 1, 2, 3), 50),
        ("400_4x100", (0, 1, 2, 3), 100),
        ("400_2x200", (0, 1), 200),
    )
    for prefix, lanes, lane_cap_gbps in specs:
        for bg_percent in (1, 10, 50, 90):
            runs.append(
                base.multi_hca_run(
                    prefix=prefix,
                    lanes=lanes,
                    lane_cap_gbps=lane_cap_gbps,
                    bg_percent=bg_percent,
                    suffix="multi_hca_moonbg",
                )
            )
    for bg_percent in (1, 10, 50, 90):
        runs.append(base.split_4x100_run(bg_percent))
    return _fine_runs(runs)


def _small_head_split_run(
    *,
    run: str,
    lanes: Sequence[int],
    per_lane_rate_gbps: float,
    fg_rate_gbps: Optional[float],
) -> base.MatrixRun:
    shards = len(lanes)
    sizes, max_bytes = base.split_empty_sizes_and_max_bytes(shards)
    return base.MatrixRun(
        run=run,
        lanes=tuple(base.bond_endpoint(lane) for lane in lanes),
        shards=shards,
        max_bytes=max_bytes,
        sizes=sizes,
        protocol="rdma",
        lane_cap_gbps=per_lane_rate_gbps,
        bg_rate_gbps=0.0,
        fg_rate_gbps=fg_rate_gbps,
        capfill_lanes=(),
        capfill_rate_gbps=0.0,
    )


def _small_tail_single_nic_run(
    *,
    args: Optional[argparse.Namespace],
    run: str,
    lane_cap_gbps: float,
    fg_rate_gbps: Optional[float],
) -> base.MatrixRun:
    endpoint = base.one_lane_endpoint(
        getattr(args, "single_nic_src_host", base.SRC_IPS[0]),
        getattr(args, "single_nic_tgt_host", base.TGT_IPS[0]),
        getattr(args, "single_nic_ib_device", "mlx5_0"),
    )
    return base.MatrixRun(
        run=run,
        lanes=(endpoint,),
        shards=1,
        max_bytes="2GB",
        sizes=base.DENSE_SIZES_1,
        protocol="rdma",
        lane_cap_gbps=lane_cap_gbps,
        bg_rate_gbps=0.0,
        fg_rate_gbps=fg_rate_gbps,
        capfill_lanes=(),
        capfill_rate_gbps=0.0,
    )


def small_flow_runs(args: Optional[argparse.Namespace] = None) -> List[base.MatrixRun]:
    runs = [
        _small_head_split_run(
            run="head_rdma_small_uncapped_4x200_split",
            lanes=(0, 1, 2, 3),
            per_lane_rate_gbps=200.0,
            fg_rate_gbps=None,
        ),
        _small_head_split_run(
            run="head_rdma_small_cap4x100_split",
            lanes=(0, 1, 2, 3),
            per_lane_rate_gbps=100.0,
            fg_rate_gbps=100.0,
        ),
        _small_head_split_run(
            run="head_rdma_small_cap2x200_split",
            lanes=(0, 1),
            per_lane_rate_gbps=200.0,
            fg_rate_gbps=200.0,
        ),
        _small_tail_single_nic_run(
            args=args,
            run="tail_rdma_small_uncapped_single_nic",
            lane_cap_gbps=200.0,
            fg_rate_gbps=None,
        ),
        _small_tail_single_nic_run(
            args=args,
            run="tail_rdma_small_cap100_single_nic",
            lane_cap_gbps=100.0,
            fg_rate_gbps=100.0,
        ),
        _small_tail_single_nic_run(
            args=args,
            run="tail_rdma_small_cap200_single_nic",
            lane_cap_gbps=200.0,
            fg_rate_gbps=200.0,
        ),
    ]
    return _fine_runs(runs)


def all_visible_matrix_runs(args: Optional[argparse.Namespace] = None) -> List[base.MatrixRun]:
    return (
        singleflow_runs(args)
        + multi_hca_portcap_runs()
        + legacy_multi_hca_unaverage_runs()
        + small_flow_runs(args)
        + head_tcp_runs(args)
    )


def matrix_runs_for_suite(args: argparse.Namespace) -> List[base.MatrixRun]:
    suite = args.suite
    if suite in ("all", "all-visible", "list"):
        return all_visible_matrix_runs(args)
    if suite == "singleflow":
        return singleflow_runs(args)
    if suite == "fixed-missing":
        return fixed_missing_runs(args)
    if suite in ("head-tcp", "head-tcp-sweep"):
        return head_tcp_runs(args)
    if suite in ("multi-hca-bg", "multi-hca-portcap-bg"):
        return multi_hca_portcap_runs()
    if suite in ("legacy-multi-hca-unaverage", "multi-hca-unaverage"):
        return legacy_multi_hca_unaverage_runs()
    if suite == "ratelimit-empty":
        return _fine_runs(base.ratelimit_empty_runs())
    if suite == "small-flow":
        return small_flow_runs(args)
    if suite == "competition":
        return []
    raise ValueError(f"unsupported suite: {suite}")


def competition_cases_for_suite(args: argparse.Namespace) -> List[base.CompetitionCase]:
    if args.suite in ("all", "all-visible", "list", "competition"):
        return base.competition_cases()
    return []


def _filter_by_only(items, only: Optional[Sequence[str]]):
    if not only:
        return list(items)
    requested = set(only)
    return [item for item in items if item.run in requested]


def selected_matrix_runs(args: argparse.Namespace) -> List[base.MatrixRun]:
    return _filter_by_only(matrix_runs_for_suite(args), args.only)


def selected_competition_cases(args: argparse.Namespace) -> List[base.CompetitionCase]:
    return _filter_by_only(competition_cases_for_suite(args), args.only)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = (
        "Rerun visible KV transfer experiments with logical flow sizes "
        "1MiB..32MiB at 1MiB spacing, then original >32MiB points."
    )
    for action in parser._actions:
        if action.dest == "suite":
            action.choices = SUITE_CHOICES
            action.default = "list"
            action.help = (
                "Experiment group to run. all-visible/all cover the visible "
                "singleflow, multi-HCA, small-flow, head-TCP, and competition suites."
            )
    parser.add_argument(
        "--src-rdma-hosts",
        type=_parse_lane_hosts,
        default=None,
        help="Comma-separated source RDMA hosts for lanes 0..3.",
    )
    parser.add_argument(
        "--tgt-rdma-hosts",
        type=_parse_lane_hosts,
        default=None,
        help="Comma-separated target RDMA hosts for lanes 0..3.",
    )
    parser.set_defaults(suite="list")
    return parser


def _print_unique(names: Iterable[str]) -> None:
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        print(name)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_lane_host_overrides(args)
    matrix_runs = selected_matrix_runs(args)
    competition_cases = selected_competition_cases(args)

    if args.suite == "list":
        _print_unique(
            [run.run for run in matrix_runs]
            + [case.run for case in competition_cases]
        )
        return 0

    runner = base.Runner(args)
    for run in matrix_runs:
        runner.run_matrix(run)
    for case in competition_cases:
        runner.run_competition(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
