#!/usr/bin/env python3
"""Compare DCP CUDA-graph replay traces from PyTorch profiler."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GPU_EVENT_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
RANK_PATTERN = re.compile(r"(?:TP-|rank)(\d+)")

# These labels intentionally describe kernel families, not inferred model stages.
KERNEL_FAMILIES = (
    ("c4_score", ("paged_mqa_logits",)),
    ("c4_topk", ("radixSortKVInPlace", "topk_combine_transform")),
    ("nccl_allgather", ("ncclDevKernel_AllGather",)),
    ("nccl_reducescatter", ("ncclDevKernel_ReduceScatter",)),
    ("nccl_sendrecv", ("ncclDevKernel_SendRecv",)),
    ("attn_allreduce_fp32", ("all_reduce_two_shot_kernel<float",)),
    ("allreduce_bf16", ("all_reduce_two_shot_kernel<__nv_bfloat16",)),
    ("flashmla_sparse", ("flash_fwd_splitkv_mla_fp8_sparse_kernel",)),
    ("flashmla_combine", ("flash_fwd_mla_combine_kernel",)),
    ("deepep_combine", ("deep_ep::internode_ll::combine",)),
    ("deepep_dispatch", ("deep_ep::internode_ll::dispatch",)),
    ("fp32_mul", ("MulFunctor<float>",)),
    ("nan_to_num", ("nan_to_num_kernel_cuda",)),
    ("fp32_reduce_128", ("reduce_kernel<128, 4",)),
    ("fp32_transpose", ("deep_gemm::transpose_fp32<512u",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--top-kernels", type=int, default=15)
    return parser.parse_args()


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def kernel_family(name: str) -> str:
    for label, patterns in KERNEL_FAMILIES:
        if any(pattern in name for pattern in patterns):
            return label
    return "other"


def get_rank(path: Path) -> int:
    match = RANK_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse TP rank from {path.name}")
    return int(match.group(1))


def load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as file:
        payload = json.load(file)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"{path} does not contain traceEvents")
    return events


def analyze_trace(path: Path) -> list[dict[str, Any]]:
    rank = get_rank(path)
    events = load_trace(path)
    graph_launches = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == "cudaGraphLaunch"
    ]
    if not graph_launches:
        raise ValueError(f"{path} contains no cudaGraphLaunch events")

    gpu_events_by_correlation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if (
            event.get("ph") != "X"
            or event.get("cat") not in GPU_EVENT_CATEGORIES
        ):
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            gpu_events_by_correlation[int(correlation)].append(event)

    samples = []
    for replay, launch in enumerate(graph_launches):
        correlation = int(launch.get("args", {}).get("correlation", -1))
        gpu_events = gpu_events_by_correlation.get(correlation, [])
        if not gpu_events:
            raise ValueError(
                f"{path} replay {replay} has no GPU children for correlation "
                f"{correlation}"
            )

        start_us = min(float(event["ts"]) for event in gpu_events)
        end_us = max(
            float(event["ts"]) + float(event.get("dur", 0))
            for event in gpu_events
        )
        family_duration_us: dict[str, float] = defaultdict(float)
        family_count: dict[str, int] = defaultdict(int)
        kernel_duration_us: dict[str, float] = defaultdict(float)
        kernel_count: dict[str, int] = defaultdict(int)

        for event in gpu_events:
            name = str(event.get("name", "unknown"))
            duration_us = float(event.get("dur", 0))
            family = kernel_family(name)
            family_duration_us[family] += duration_us
            family_count[family] += 1
            kernel_duration_us[name] += duration_us
            kernel_count[name] += 1

        samples.append(
            {
                "rank": rank,
                "replay": replay,
                "correlation": correlation,
                "launch_api_ms": float(launch.get("dur", 0)) / 1000,
                "graph_span_ms": (end_us - start_us) / 1000,
                "kernel_sum_ms": sum(
                    float(event.get("dur", 0)) for event in gpu_events
                )
                / 1000,
                "child_event_count": len(gpu_events),
                "family_duration_ms": {
                    name: duration / 1000
                    for name, duration in family_duration_us.items()
                },
                "family_count": dict(family_count),
                "kernel_duration_ms": {
                    name: duration / 1000
                    for name, duration in kernel_duration_us.items()
                },
                "kernel_count": dict(kernel_count),
            }
        )
    return samples


def summarize_variant(
    trace_dir: Path, label: str, top_kernels: int
) -> dict[str, Any]:
    paths = sorted(trace_dir.glob("*DECODE.trace.json.gz"))
    if not paths:
        paths = sorted(trace_dir.glob("*.trace.json.gz"))
    if not paths:
        paths = sorted(trace_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no CUDA graph traces found in {trace_dir}")

    rank_samples = {get_rank(path): analyze_trace(path) for path in paths}
    step_counts = {rank: len(samples) for rank, samples in rank_samples.items()}
    if len(set(step_counts.values())) != 1:
        raise ValueError(f"inconsistent replay counts: {step_counts}")

    samples = [
        sample
        for rank in sorted(rank_samples)
        for sample in rank_samples[rank]
    ]
    family_names = [name for name, _ in KERNEL_FAMILIES] + ["other"]
    metric_values = {
        "launch_api_ms": [sample["launch_api_ms"] for sample in samples],
        "graph_span_ms": [sample["graph_span_ms"] for sample in samples],
        "kernel_sum_ms": [sample["kernel_sum_ms"] for sample in samples],
        "child_event_count": [
            float(sample["child_event_count"]) for sample in samples
        ],
    }
    for family in family_names:
        metric_values[f"{family}_ms"] = [
            sample["family_duration_ms"].get(family, 0.0) for sample in samples
        ]
        metric_values[f"{family}_count"] = [
            float(sample["family_count"].get(family, 0)) for sample in samples
        ]

    kernel_names = {
        name
        for sample in samples
        for name in sample["kernel_duration_ms"]
    }
    top = []
    for name in kernel_names:
        durations = [
            sample["kernel_duration_ms"].get(name, 0.0) for sample in samples
        ]
        counts = [float(sample["kernel_count"].get(name, 0)) for sample in samples]
        top.append(
            {
                "name": name,
                "median_ms": median(durations),
                "median_count": median(counts),
            }
        )
    top.sort(key=lambda item: item["median_ms"], reverse=True)

    return {
        "label": label,
        "trace_dir": str(trace_dir),
        "rank_count": len(rank_samples),
        "ranks": sorted(rank_samples),
        "replays_per_rank": next(iter(step_counts.values())),
        "sample_count": len(samples),
        "metrics": {
            name: {
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in metric_values.items()
        },
        "top_kernels": top[:top_kernels],
    }


def compare_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, float | str | None]]:
    metric_order = [
        "graph_span_ms",
        "kernel_sum_ms",
        "launch_api_ms",
        "c4_score_ms",
        "c4_topk_ms",
        "nccl_allgather_ms",
        "nccl_reducescatter_ms",
        "nccl_sendrecv_ms",
        "attn_allreduce_fp32_ms",
        "allreduce_bf16_ms",
        "flashmla_sparse_ms",
        "deepep_combine_ms",
        "deepep_dispatch_ms",
        "fp32_mul_ms",
        "nan_to_num_ms",
        "fp32_reduce_128_ms",
        "fp32_transpose_ms",
    ]
    rows = []
    for name in metric_order:
        baseline_value = float(baseline["metrics"][name]["median"])
        candidate_value = float(candidate["metrics"][name]["median"])
        delta = candidate_value - baseline_value
        relative = 100 * delta / baseline_value if baseline_value else None
        rows.append(
            {
                "metric": name,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "relative_percent": relative,
            }
        )
    return rows


def compact_kernel_name(name: str, limit: int = 110) -> str:
    name = name.replace("|", r"\|")
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def render_markdown(
    baseline: dict[str, Any],
    candidate: dict[str, Any] | None,
    comparison: list[dict[str, float | str | None]],
) -> str:
    lines = [
        "# DCP CUDA Graph Trace Summary",
        "",
        (
            f"- {baseline['label']}: {baseline['rank_count']} ranks x "
            f"{baseline['replays_per_rank']} replays"
        ),
    ]
    if candidate is not None:
        lines.append(
            f"- {candidate['label']}: {candidate['rank_count']} ranks x "
            f"{candidate['replays_per_rank']} replays"
        )
        lines.extend(
            [
                "",
                "| Metric | Baseline ms | Candidate ms | Delta ms | Delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in comparison:
            relative = row["relative_percent"]
            relative_text = "n/a" if relative is None else f"{relative:+.2f}%"
            lines.append(
                f"| {row['metric']} | {row['baseline']:.3f} | "
                f"{row['candidate']:.3f} | {row['delta']:+.3f} | "
                f"{relative_text} |"
            )
    else:
        lines.extend(
            [
                "",
                "| Metric | Median | Min | Max |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in baseline["metrics"].items():
            if not name.endswith("_ms"):
                continue
            lines.append(
                f"| {name} | {values['median']:.3f} | "
                f"{values['min']:.3f} | {values['max']:.3f} |"
            )

    for variant in (baseline, candidate):
        if variant is None:
            continue
        lines.extend(
            [
                "",
                f"## Top Kernels: {variant['label']}",
                "",
                "| Median ms/replay | Median count/replay | Kernel |",
                "|---:|---:|---|",
            ]
        )
        for kernel in variant["top_kernels"]:
            lines.append(
                f"| {kernel['median_ms']:.3f} | {kernel['median_count']:.1f} | "
                f"{compact_kernel_name(kernel['name'])} |"
            )
    return "\n".join(lines) + "\n"


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    args = parse_args()
    if args.top_kernels <= 0:
        raise ValueError("--top-kernels must be positive")

    baseline = summarize_variant(
        args.baseline_dir, args.baseline_label, args.top_kernels
    )
    candidate = (
        summarize_variant(
            args.candidate_dir, args.candidate_label, args.top_kernels
        )
        if args.candidate_dir is not None
        else None
    )
    if candidate is not None and baseline["ranks"] != candidate["ranks"]:
        raise ValueError(
            f"rank mismatch: {baseline['ranks']} != {candidate['ranks']}"
        )
    comparison = (
        compare_metrics(baseline, candidate) if candidate is not None else []
    )
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }
    markdown = render_markdown(baseline, candidate, comparison)

    if args.json_out is not None:
        write_output(
            args.json_out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    if args.markdown_out is not None:
        write_output(args.markdown_out, markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
