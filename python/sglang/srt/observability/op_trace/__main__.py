"""CLI: ``python -m sglang.srt.observability.op_trace {summarize,diff}``.

summarize TRACE [--op-map F ...] [--json OUT] [--records] [--top N] [--depth N]
    Join a torch trace with capture-time op maps (default: ``op_map-*.json``
    next to the trace) and print a per-op markdown table per step bucket.
diff BASE NEW [--bucket B] [--top N] [--depth N]
    Op-level delta of two summaries (TraceSummary JSON written by
    ``summarize --json``, or raw traces), sorted by absolute us/step change.
bench --model M [--workloads decode:1,prefill:4096] [--steps N] --out DIR [-- server args]
    Load the model once, trace fixed-shape workloads, write traces, op tables
    and benchmark marker records (bench-*.jsonl) into DIR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import msgspec

from sglang.srt.observability.op_trace.analyze import analyze_trace, discover_op_maps
from sglang.srt.observability.op_trace.bench import add_bench_args
from sglang.srt.observability.op_trace.report import render_diff, render_summary
from sglang.srt.observability.op_trace.schema import TraceSummary


def _load_summary(*, path: str, op_maps: list[str], depth) -> TraceSummary:
    if path.endswith(".summary.json"):
        return msgspec.json.decode(Path(path).read_bytes(), type=TraceSummary)
    return analyze_trace(
        trace_path=path, op_map_paths=op_maps or discover_op_maps(path), depth=depth
    )


def _summarize(args: argparse.Namespace) -> None:
    summary = analyze_trace(
        trace_path=args.trace,
        op_map_paths=args.op_map or discover_op_maps(args.trace),
        keep_records=args.records,
        depth=args.depth,
    )
    if args.json:
        Path(args.json).write_bytes(msgspec.json.encode(summary))
    print(render_summary(summary, top=args.top))


def _diff(args: argparse.Namespace) -> None:
    base = _load_summary(path=args.base, op_maps=args.op_map, depth=args.depth)
    new = _load_summary(path=args.new, op_maps=args.op_map, depth=args.depth)
    print(render_diff(base=base, new=new, bucket=args.bucket, top=args.top))


def _bench(args: argparse.Namespace) -> None:
    from sglang.srt.observability.op_trace.bench import run_bench

    print(f"records: {run_bench(args)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sglang.srt.observability.op_trace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("summarize", help="per-op table for one trace")
    p.add_argument("trace")
    p.add_argument("--op-map", action="append", default=[])
    p.add_argument("--json", help="write the TraceSummary here (*.summary.json)")
    p.add_argument(
        "--records", action="store_true", help="keep step/kernel records in --json"
    )
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--depth", type=int, help="truncate op kinds to N components")
    p.set_defaults(func=_summarize)

    p = sub.add_parser("diff", help="op-level delta of two summaries or traces")
    p.add_argument("base")
    p.add_argument("new")
    p.add_argument("--op-map", action="append", default=[])
    p.add_argument("--bucket")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--depth", type=int, help="truncate op kinds (raw traces only)")
    p.set_defaults(func=_diff)

    p = sub.add_parser(
        "bench", help="trace fixed-shape workloads into benchmark records"
    )
    add_bench_args(p)
    p.set_defaults(func=_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
