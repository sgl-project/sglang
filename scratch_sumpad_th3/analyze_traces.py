from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)

_GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation")

_BUCKET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("comm", re.compile(r"nccl|ncclDevKernel|AllReduce|AllGather|ReduceScatter", re.I)),
    ("attn", re.compile(r"flash|fmha|attention|paged|mha_|decode_attn", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|sm\d+_|matmul|nvjet|dot_kernel|s\d+gemm", re.I)),
    ("memcpy", re.compile(r"memcpy|memset|Memcpy|Memset", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|reduce_kernel|silu|rms|rope|norm|act|cast|copy", re.I)),
)


def _bucket_of(name: str) -> str:
    for bucket, pattern in _BUCKET_PATTERNS:
        if pattern.search(name):
            return bucket
    return "other"


def _merged_span(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    return total + (cur_end - cur_start)


def _analyze_one(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]

    gpu_events = [
        event
        for event in events
        if event.get("cat") in _GPU_CATS and "dur" in event and "ts" in event
    ]
    bucket_time: Counter[str] = Counter()
    bucket_calls: Counter[str] = Counter()
    kernel_time: Counter[str] = Counter()
    intervals: list[tuple[float, float]] = []
    for event in gpu_events:
        name = event.get("name", "?")
        bucket = _bucket_of(name)
        bucket_time[bucket] += event["dur"]
        bucket_calls[bucket] += 1
        kernel_time[name] += event["dur"]
        intervals.append((event["ts"], event["ts"] + event["dur"]))

    wall = 0.0
    if intervals:
        wall = max(end for _, end in intervals) - min(start for start, _ in intervals)

    return {
        "busy_us": _merged_span(intervals),
        "wall_us": wall,
        "bucket_time": bucket_time,
        "bucket_calls": bucket_calls,
        "kernel_time": kernel_time,
    }


@app.command()
def main(
    trace_dir: Annotated[Path, typer.Argument()],
    top_kernels: Annotated[int, typer.Option()] = 8,
) -> None:
    """Aggregate every *.trace.json.gz under one profile dir, per rank and in total."""
    traces = sorted(trace_dir.rglob("*.trace.json.gz"))
    if not traces:
        print(f"no traces under {trace_dir}")
        raise typer.Exit(code=1)

    totals: Counter[str] = Counter()
    total_calls: Counter[str] = Counter()
    kernel_totals: Counter[str] = Counter()
    print(f"dir: {trace_dir}  traces: {len(traces)}")
    print(f"{'rank':>6s} {'busy_ms':>9s} {'wall_ms':>9s} {'comm_ms':>9s} {'comm%':>7s} {'gemm_ms':>9s} {'attn_ms':>9s} {'other_ms':>9s}")
    for path in traces:
        match = re.search(r"-TP-(\d+)-DP-(\d+)", path.name)
        rank = match.group(2) if match else "?"
        result = _analyze_one(path)
        bucket_time = result["bucket_time"]
        busy = float(result["busy_us"]) / 1000
        comm = bucket_time["comm"] / 1000
        comm_pct = comm / busy * 100 if busy else 0.0
        other = sum(v for k, v in bucket_time.items() if k not in ("comm", "gemm", "attn")) / 1000
        print(
            f"{rank:>6s} {busy:9.2f} {float(result['wall_us']) / 1000:9.2f} {comm:9.2f} "
            f"{comm_pct:6.1f}% {bucket_time['gemm'] / 1000:9.2f} "
            f"{bucket_time['attn'] / 1000:9.2f} {other:9.2f}"
        )
        totals.update(bucket_time)
        total_calls.update(result["bucket_calls"])
        kernel_totals.update(result["kernel_time"])

    grand = sum(totals.values())
    print(f"--- bucket totals over {len(traces)} ranks (sum of kernel dur, ms) ---")
    for bucket, value in totals.most_common():
        share = value / grand * 100 if grand else 0.0
        print(f"  {bucket:<12s} {value / 1000:10.2f} ms {share:5.1f}%  calls={total_calls[bucket]}")
    print(f"--- top {top_kernels} kernels (ms, summed over ranks) ---")
    for name, value in kernel_totals.most_common(top_kernels):
        print(f"  {value / 1000:10.2f} ms  {name[:100]}")


if __name__ == "__main__":
    app()
