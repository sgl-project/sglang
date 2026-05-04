#!/usr/bin/env python3
"""Render a small call/kernel tree for one module in a torch profiler trace."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
TREE_CATEGORIES = {"python_function", "cpu_op", "user_annotation"}
SGLANG_SOURCE_RE = re.compile(
    r"^/sgl-workspace/sglang/(?P<rel>python/sglang/[^()]+\.py)"
    r"\((?P<line>\d+)\): (?P<name>.+)$"
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def load_trace(path: Path) -> list[dict[str, Any]]:
    with open_text(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload["traceEvents"]
    return payload


def event_end(event: dict[str, Any]) -> float:
    return float(event["ts"]) + float(event.get("dur", 0.0))


def fmt_us(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.3f} ms"
    return f"{us:.3f} us"


def fmt_compact_us(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.3f}ms"
    return f"{us:.3f}us"


def fmt_gpu_stat(kernels: list[dict[str, Any]], denominator_us: float) -> str:
    if not kernels:
        return ""
    total_us = sum(float(kernel.get("dur", 0.0)) for kernel in kernels)
    percent = total_us / denominator_us * 100 if denominator_us > 0 else 0.0
    return f"[{len(kernels)}|{fmt_compact_us(total_us)}|{percent:.2f}%]"


def rounded_gpu_percent(
    kernels: list[dict[str, Any]], denominator_us: float
) -> float | None:
    if not kernels:
        return None
    total_us = sum(float(kernel.get("dur", 0.0)) for kernel in kernels)
    percent = total_us / denominator_us * 100 if denominator_us > 0 else 0.0
    return round(percent, 1)


def fmt_node_gpu_stat(
    kernels: list[dict[str, Any]],
    denominator_us: float,
    parent_percent: float | None = None,
) -> str:
    percent = rounded_gpu_percent(kernels, denominator_us)
    if percent is None or percent == parent_percent:
        return ""
    return f"[{percent:.1f}%]"


def short_name(name: str) -> str:
    replacements = {
        "/sgl-workspace/sglang/python/": "",
        "torch/nn/modules/module.py(1779): _call_impl": "_call_impl",
        "torch/nn/modules/module.py(1951): __getattr__": "__getattr__",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    if len(name) <= 120:
        return name
    return name[:96] + "..." + name[-20:]


def event_ref(name: str) -> str:
    match = SGLANG_SOURCE_RE.match(name)
    if match:
        rel_path = match.group("rel")
        line = match.group("line")
        label = match.group("name")
        return f"[`{label}`]({rel_path}:{line})"
    return f"`{short_name(name)}`"


def kernel_ref(name: str) -> str:
    label = short_name(name)
    target = quote(label, safe="/:@._-+")
    return f"[kernel]({target}) `{label}`"


def find_target(events: list[dict[str, Any]], target: str) -> tuple[int, dict[str, Any]]:
    exact: list[tuple[int, dict[str, Any]]] = []
    contains: list[tuple[int, dict[str, Any]]] = []
    for idx, event in enumerate(events):
        name = str(event.get("name", ""))
        if name == target:
            exact.append((idx, event))
        elif target in name:
            contains.append((idx, event))

    matches = exact or contains
    if not matches:
        raise SystemExit(f"target not found: {target}")
    if len(matches) > 1:
        print(f"warning: found {len(matches)} matches; using the first one")
    return matches[0]


def build_same_thread_tree(
    events: list[dict[str, Any]],
    target_idx: int,
    target_event: dict[str, Any],
) -> tuple[dict[int, list[int]], dict[int, int | None]]:
    start = float(target_event["ts"])
    stop = event_end(target_event)
    pid = target_event.get("pid")
    tid = target_event.get("tid")

    tree_events: list[tuple[int, dict[str, Any]]] = []
    for idx, event in enumerate(events):
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        if event.get("pid") != pid or event.get("tid") != tid:
            continue
        if event.get("cat") not in TREE_CATEGORIES:
            continue
        if start <= float(event["ts"]) and event_end(event) <= stop:
            tree_events.append((idx, event))

    tree_events.sort(key=lambda item: (float(item[1]["ts"]), -float(item[1].get("dur", 0.0))))

    stack: list[tuple[int, float]] = []
    children: dict[int, list[int]] = collections.defaultdict(list)
    parent: dict[int, int | None] = {}

    for idx, event in tree_events:
        start_ts = float(event["ts"])
        stop_ts = event_end(event)
        while stack and start_ts >= stack[-1][1] - 1e-6:
            stack.pop()

        parent_idx = stack[-1][0] if stack else None
        parent[idx] = parent_idx
        if parent_idx is not None:
            children[parent_idx].append(idx)
        stack.append((idx, stop_ts))

    if target_idx not in parent:
        parent[target_idx] = None
    return children, parent


def find_logical_root(
    events: list[dict[str, Any]],
    target_idx: int,
    children: dict[int, list[int]],
    prefer_forward: str | None,
) -> int:
    if prefer_forward:
        stack = list(reversed(children.get(target_idx, [])))
        while stack:
            idx = stack.pop()
            name = str(events[idx].get("name", ""))
            if prefer_forward in name:
                return idx
            stack.extend(reversed(children.get(idx, [])))

    return target_idx


def collect_gpu_events(
    events: list[dict[str, Any]],
    start: float,
    stop: float,
) -> list[dict[str, Any]]:
    gpu_events = []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        ts = float(event["ts"])
        if start <= ts and event_end(event) <= stop:
            gpu_events.append(event)
    gpu_events.sort(key=lambda event: float(event["ts"]))
    return gpu_events


def build_external_id_map(events: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    by_external_id: dict[Any, dict[str, Any]] = {}
    for event in events:
        args = event.get("args") or {}
        external_id = args.get("External id")
        if external_id is None:
            continue
        if event.get("cat") in {"cpu_op", "user_annotation"}:
            by_external_id.setdefault(external_id, event)
    return by_external_id


def build_correlation_external_id_map(events: list[dict[str, Any]]) -> dict[Any, Any]:
    by_correlation: dict[Any, Any] = {}
    for event in events:
        if event.get("cat") != "cuda_runtime":
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        external_id = args.get("External id")
        if correlation is not None and external_id is not None:
            by_correlation.setdefault(correlation, external_id)
    return by_correlation


def build_correlation_runtime_map(
    events: list[dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    by_correlation: dict[Any, dict[str, Any]] = {}
    for event in events:
        if event.get("cat") != "cuda_runtime":
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        if correlation is not None:
            by_correlation.setdefault(correlation, event)
    return by_correlation


def kernel_external_id(
    kernel: dict[str, Any],
    correlation_external_id_map: dict[Any, Any],
) -> Any:
    args = kernel.get("args") or {}
    external_id = args.get("External id")
    if external_id is not None:
        return external_id
    correlation = args.get("correlation")
    return correlation_external_id_map.get(correlation)


def group_kernels_by_external_id(
    kernels: list[dict[str, Any]],
    correlation_external_id_map: dict[Any, Any],
) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for kernel in kernels:
        external_id = kernel_external_id(kernel, correlation_external_id_map)
        if external_id is not None:
            grouped[external_id].append(kernel)
    return grouped


def find_enclosing_node(
    events: list[dict[str, Any]],
    children: dict[int, list[int]],
    idx: int,
    ts: float,
) -> int | None:
    event = events[idx]
    if not (float(event["ts"]) - 1e-6 <= ts <= event_end(event) + 1e-6):
        return None

    for child_idx in children.get(idx, []):
        found = find_enclosing_node(events, children, child_idx, ts)
        if found is not None:
            return found

    return idx


def group_kernels_by_launch_node(
    kernels: list[dict[str, Any]],
    correlation_runtime_map: dict[Any, dict[str, Any]],
    events: list[dict[str, Any]],
    children: dict[int, list[int]],
    root_idx: int,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for kernel in kernels:
        args = kernel.get("args") or {}
        correlation = args.get("correlation")
        if correlation is None:
            continue

        runtime_event = correlation_runtime_map.get(correlation)
        if runtime_event is None or "ts" not in runtime_event:
            continue

        launch_node = find_enclosing_node(
            events, children, root_idx, float(runtime_event["ts"])
        )
        if launch_node is not None:
            grouped[launch_node].append(kernel)

    return grouped


def kernels_in_window(
    kernels: list[dict[str, Any]],
    start: float,
    stop: float,
) -> list[dict[str, Any]]:
    return [
        kernel
        for kernel in kernels
        if start <= float(kernel["ts"]) and event_end(kernel) <= stop
    ]


def render_node_line(
    events: list[dict[str, Any]],
    idx: int,
    kernels: list[dict[str, Any]],
    kernels_by_external_id: dict[Any, list[dict[str, Any]]],
    depth: int,
    denominator_us: float,
    node_kernels: list[dict[str, Any]] | None = None,
    parent_percent: float | None = None,
) -> list[str]:
    event = events[idx]
    indent = "  " * depth
    if node_kernels is None:
        external_id = (event.get("args") or {}).get("External id")
        node_kernels = kernels_by_external_id.get(external_id)
        if not node_kernels:
            node_kernels = kernels_in_window(kernels, float(event["ts"]), event_end(event))
    gpu_stat = fmt_node_gpu_stat(node_kernels, denominator_us, parent_percent)
    suffix = f" {gpu_stat}" if gpu_stat else ""
    line = f"{indent}- {event_ref(str(event.get('name', '')))}{suffix}"
    return [line]


def render_kernel_lines(
    kernels: list[dict[str, Any]],
    depth: int,
    denominator_us: float,
) -> list[str]:
    indent = "  " * depth
    lines = []
    for kernel in kernels:
        gpu_stat = fmt_gpu_stat([kernel], denominator_us)
        lines.append(f"{indent}- {kernel_ref(str(kernel.get('name', '')))} {gpu_stat}")
    return lines


def render_tree(
    events: list[dict[str, Any]],
    root_idx: int,
    children: dict[int, list[int]],
    kernels: list[dict[str, Any]],
    kernels_by_external_id: dict[Any, list[dict[str, Any]]],
    kernels_by_launch_node: dict[int, list[dict[str, Any]]],
    max_depth: int,
    min_cpu_us: float,
    top_kernels: int,
    denominator_us: float,
) -> list[str]:
    subtree_has_kernel_cache: dict[int, bool] = {}

    def direct_kernels(idx: int) -> list[dict[str, Any]]:
        event = events[idx]
        external_id = (event.get("args") or {}).get("External id")
        out: dict[int, dict[str, Any]] = {}
        for kernel in kernels_by_external_id.get(external_id, []):
            out[id(kernel)] = kernel
        for kernel in kernels_by_launch_node.get(idx, []):
            out[id(kernel)] = kernel
        combined = list(out.values())
        combined.sort(key=lambda event: float(event["ts"]))
        return combined

    subtree_kernel_cache: dict[int, list[dict[str, Any]]] = {}

    def subtree_kernels(idx: int) -> list[dict[str, Any]]:
        if idx in subtree_kernel_cache:
            return subtree_kernel_cache[idx]
        out = list(direct_kernels(idx))
        for child_idx in children.get(idx, []):
            out.extend(subtree_kernels(child_idx))
        out.sort(key=lambda event: float(event["ts"]))
        subtree_kernel_cache[idx] = out
        return out

    def subtree_has_kernel(idx: int) -> bool:
        if idx in subtree_has_kernel_cache:
            return subtree_has_kernel_cache[idx]
        has_kernel = bool(combined_node_kernels(idx))
        subtree_has_kernel_cache[idx] = has_kernel
        return has_kernel

    def combined_node_kernels(idx: int) -> list[dict[str, Any]]:
        event = events[idx]
        out: dict[int, dict[str, Any]] = {}
        for kernel in subtree_kernels(idx):
            out[id(kernel)] = kernel
        for kernel in kernels_in_window(kernels, float(event["ts"]), event_end(event)):
            out[id(kernel)] = kernel
        combined = list(out.values())
        combined.sort(key=lambda event: float(event["ts"]))
        return combined

    lines = render_node_line(
        events,
        root_idx,
        kernels,
        kernels_by_external_id,
        0,
        denominator_us,
        combined_node_kernels(root_idx),
    )
    lines.extend(render_kernel_lines(direct_kernels(root_idx), 1, denominator_us))

    def walk(idx: int, depth: int) -> None:
        parent_percent = rounded_gpu_percent(combined_node_kernels(idx), denominator_us)
        for child_idx in children.get(idx, []):
            child = events[child_idx]
            dur = float(child.get("dur", 0.0))
            visible_by_depth = depth <= max_depth
            visible_by_filter = min_cpu_us < 0 or dur >= min_cpu_us
            if not (visible_by_depth and visible_by_filter) and not subtree_has_kernel(child_idx):
                continue

            child_kernels = combined_node_kernels(child_idx)
            lines.extend(
                render_node_line(
                    events,
                    child_idx,
                    kernels,
                    kernels_by_external_id,
                    depth,
                    denominator_us,
                    child_kernels,
                    parent_percent,
                )
            )
            lines.extend(
                render_kernel_lines(direct_kernels(child_idx), depth + 1, denominator_us)
            )
            walk(child_idx, depth + 1)

    walk(root_idx, 1)
    return lines


def render_kernel_timeline(
    kernels: list[dict[str, Any]],
    external_id_map: dict[Any, dict[str, Any]],
    correlation_external_id_map: dict[Any, Any],
    correlation_runtime_map: dict[Any, dict[str, Any]],
    root_start: float,
    limit: int,
) -> list[str]:
    lines = []
    for i, kernel in enumerate(kernels[:limit], start=1):
        args = kernel.get("args") or {}
        correlation = args.get("correlation")
        ext = kernel_external_id(kernel, correlation_external_id_map)
        id_part = f"ext={ext}" if ext is not None else f"corr={correlation or '-'}"
        launch_op = external_id_map.get(ext)
        launch_part = ""
        if launch_op is not None:
            launch_part = f" <- {event_ref(str(launch_op.get('name', '')))}"
        elif correlation is not None:
            runtime_event = correlation_runtime_map.get(correlation)
            if runtime_event is not None:
                launch_part = f" <- `{short_name(str(runtime_event.get('name', '')))}`"
        lines.append(
            f"{i}. +{fmt_us(float(kernel['ts']) - root_start)} "
            f"{kernel_ref(str(kernel.get('name', '')))} "
            f"{fmt_us(float(kernel.get('dur', 0.0)))} {id_part}{launch_part}"
        )
    if len(kernels) > limit:
        lines.append(f"... {len(kernels) - limit} more kernels omitted")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="torch profiler trace JSON or JSON.GZ")
    parser.add_argument(
        "--target",
        default="nn.Module: DeepseekV4DecoderLayer_39",
        help="event name or substring to render",
    )
    parser.add_argument(
        "--prefer-forward",
        default="deepseek_v4.py(1959): forward",
        help="descendant to use as the logical tree root; use empty string to disable",
    )
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path")
    parser.add_argument("--depth", type=int, default=2, help="tree depth below logical root")
    parser.add_argument(
        "--min-cpu-us",
        type=float,
        default=-1.0,
        help="include CPU-only nodes at or above this; negative hides all CPU-only nodes",
    )
    parser.add_argument("--top-kernels", type=int, default=3)
    parser.add_argument("--kernel-timeline-limit", type=int, default=80)
    args = parser.parse_args()

    events = load_trace(args.trace)
    target_idx, target_event = find_target(events, args.target)
    children, _ = build_same_thread_tree(events, target_idx, target_event)
    logical_root_idx = find_logical_root(
        events,
        target_idx,
        children,
        args.prefer_forward or None,
    )

    target_start = float(target_event["ts"])
    target_stop = event_end(target_event)
    logical_root = events[logical_root_idx]
    kernels = collect_gpu_events(events, target_start, target_stop)
    external_id_map = build_external_id_map(events)
    correlation_external_id_map = build_correlation_external_id_map(events)
    correlation_runtime_map = build_correlation_runtime_map(events)
    kernels_by_external_id = group_kernels_by_external_id(kernels, correlation_external_id_map)
    kernels_by_launch_node = group_kernels_by_launch_node(
        kernels,
        correlation_runtime_map,
        events,
        children,
        logical_root_idx,
    )
    root_kernels = kernels_in_window(kernels, float(logical_root["ts"]), event_end(logical_root))
    target_gpu_us = sum(float(kernel.get("dur", 0.0)) for kernel in kernels)

    target_name = str(target_event.get("name", ""))
    logical_name = str(logical_root.get("name", ""))
    md: list[str] = [
        f"# Profile Tree: `{target_name}`",
        "",
        f"- Trace: `{args.trace}`",
        f"- Logical root: {event_ref(logical_name)}",
        f"- Target-window GPU work: {fmt_gpu_stat(kernels, target_gpu_us)}",
        f"- Logical-root GPU work: {fmt_gpu_stat(root_kernels, target_gpu_us)}",
        "",
        "`[N|time|pct]` means N GPU events, total GPU time, and percent of target-window GPU time.",
        "In the tree, non-kernel nodes show `[pct]` only; child percentages identical to the parent are omitted.",
        "",
        "Note: GPU kernels are bucketed by GPU timestamp inside each CPU node's time window. "
        "Because launches are async, a kernel can appear slightly later than the CPU op that launched it.",
        "",
        "## Two-Level Call / Kernel Tree",
        "",
    ]
    md.extend(
        render_tree(
            events,
            logical_root_idx,
            children,
            kernels,
            kernels_by_external_id,
            kernels_by_launch_node,
            max_depth=args.depth,
            min_cpu_us=args.min_cpu_us,
            top_kernels=args.top_kernels,
            denominator_us=target_gpu_us,
        )
    )
    md.extend(["", "## Kernel Timeline Inside Target", ""])
    md.extend(
        render_kernel_timeline(
            kernels,
            external_id_map,
            correlation_external_id_map,
            correlation_runtime_map,
            target_start,
            args.kernel_timeline_limit,
        )
    )
    md.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(md))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
