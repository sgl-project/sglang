"""Summarize every rank's kernel trace without treating sums as critical-path time."""

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def category(name):
    name = name.lower()
    if "dispatch" in name or "combine" in name:
        return "dispatch_combine_fused"
    if any(x in name for x in ("allreduce", "allgather", "reducescatter", "hccl")):
        return "collective"
    if "matmul" in name:
        return "matmul"
    return "routing_and_other"


def interval_union(intervals):
    result = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            result += stop - start
        elif stop > end:
            result += stop - end
        end = stop if end is None else max(end, stop)
    return result


def summarize(root, iters):
    results = []
    for path in sorted(root.rglob("kernel_details.csv")):
        rows = list(csv.DictReader(path.open()))
        totals = defaultdict(float)
        intervals = []
        collective_spans = []
        types = defaultdict(lambda: dict(count=0, total_us=0.0, shapes=set()))
        origin = min(Decimal(r["Start Time(us)"].strip()) for r in rows)
        for row in rows:
            duration = float(row["Duration(us)"])
            start = float(Decimal(row["Start Time(us)"].strip()) - origin)
            name = row["Type"]
            if name in ("", "N/A"):
                name = row["Name"]
            # HCCL operation spans can enclose the AICPU kernel below. Keep them
            # separately so the kernel duration sum does not count both levels.
            if name.startswith("hcom_"):
                collective_spans.append((start, start + duration))
                continue
            totals[category(name)] += duration
            intervals.append((start, start + duration))
            types[name]["count"] += 1
            types[name]["total_us"] += duration
            types[name]["shapes"].add(row.get("Input Shapes", ""))
        results.append(
            dict(
                file=str(path.relative_to(root)),
                device_ids=sorted(set(r["Device_id"] for r in rows)),
                profiled_iterations=iters,
                summed_kernel_ms_per_iter=sum(totals.values()) / 1000 / iters,
                union_kernel_ms_per_iter=interval_union(intervals) / 1000 / iters,
                hccl_operation_span_union_ms_per_iter=interval_union(collective_spans)
                / 1000
                / iters,
                categories_ms_per_iter={k: v / 1000 / iters for k, v in totals.items()},
                operators={
                    k: dict(
                        count=v["count"],
                        ms_per_iter=v["total_us"] / 1000 / iters,
                        input_shapes=sorted(v["shapes"]),
                    )
                    for k, v in types.items()
                },
            )
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(json.dumps(summarize(args.root, args.iters), indent=2) + "\n")
