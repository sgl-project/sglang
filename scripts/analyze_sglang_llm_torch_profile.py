#!/usr/bin/env python3
"""Analyze SGLang LLM torch profiler traces into kernel/category shares."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from profile_common import (
    coerce_optional_int,
    contains_any_keyword,
    discover_trace_targets,
    extract_trace_events,
    has_stream_marker,
    is_annotation_event,
    is_complete_duration_event,
    is_non_kernel_trace_category,
    is_trace_metadata_name,
    load_trace_json,
    looks_like_python_scope_name,
    normalize_repo_relative_path,
    normalize_text,
    parse_stage,
    run_profiler,
    select_heaviest_pid,
)

CATEGORY_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "hybrid_linear",
        (
            "gdn",
            "gated_delta",
            "mamba",
            "selective_scan",
            "ssd",
            "causal_conv",
            "ssm",
        ),
    ),
    (
        "attention",
        (
            "flash_attn",
            "flashattention",
            "flash_attention",
            "fmha",
            "attention",
            "mla",
            "paged_attention",
            "decode_attention",
        ),
    ),
    (
        "moe",
        (
            "fused_moe",
            "grouped_mm",
            "groupgemm",
            "group_gemm",
            "moe",
            "expert",
            "groupproblemshape",
        ),
    ),
    (
        "gemm",
        (
            "gemm",
            "gemv",
            "matmul",
            "cublas",
            "cutlass",
            "wgmma",
            "mma",
            "bmm",
            "nvjet",
        ),
    ),
    (
        "norm",
        (
            "rmsnorm",
            "layernorm",
            "_norm_",
            " norm",
            "normkernel",
        ),
    ),
    ("rope", ("rotary", "rope", "mrope")),
    ("softmax", ("softmax",)),
    ("activation", ("silu", "gelu", "relu", "act_and_mul", "sigmoid")),
    ("quantize", ("quant", "fp8", "mxfp", "nvfp4", "dequant", "cvt")),
    (
        "reduce_topk",
        ("topk", "reduce", "argmax", "argtopk", "sampling", "multinomial"),
    ),
    (
        "sampling_io",
        (
            "prepare_inputs",
            "write_req_to",
            "catarraybatched",
            "prepare_next",
            "copy_next",
        ),
    ),
    (
        "elementwise",
        (
            "elementwise",
            "vectorized_elementwise_kernel",
            "unrolled_elementwise_kernel",
            "gpu_kernel_impl",
            "binary_internal",
            "unaryfunctor",
            "add_kernel",
            "sub_kernel",
            "mul_kernel",
            "div_",
            "floor_kernel",
            "log_kernel",
            "neg_kernel",
        ),
    ),
]

COMMUNICATION_STRONG_KEYWORDS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "alltoall",
    "all_to_all",
    "cross_device_reduce",
    "deepep",
    "mooncake",
)

COMMUNICATION_WEAK_KEYWORDS = (
    "broadcast",
    "dispatch",
    "combine",
)

MEMORY_STRONG_KEYWORDS = (
    "memcpy",
    "memset",
    "dma",
    "prefetch",
)

MEMORY_WEAK_KEYWORDS = (
    "copy",
    "fill",
)

COMPUTE_HINT_KEYWORDS = (
    "gemm",
    "gemv",
    "matmul",
    "cublas",
    "cutlass",
    "wgmma",
    "mma",
    "bmm",
    "nvjet",
    "fmha",
    "attention",
    "flash_attn",
    "flashattention",
    "flash_attention",
    "grouped_mm",
    "groupgemm",
    "moe",
    "expert",
)

NOISE_FRAME_PREFIXES = (
    "threading.py(",
    "multiprocessing/",
    "contextlib.py(",
    "torch/utils/_contextlib.py(",
    "runpy.py(",
    "asyncio/",
    "selectors.py(",
    "queue.py(",
    "socket.py(",
    "tqdm/_monitor.py(",
    "<string>(",
    "<built-in method ",
)
ALLREDUCE_FUSION_PATH = (
    "python/sglang/srt/layers/layernorm.py:89 _forward_with_allreduce_fusion"
    "<br>python/sglang/srt/distributed/communication_op.py:21 "
    "tensor_model_parallel_fused_allreduce_rmsnorm"
)
QWEN3_QK_ROPE_FUSION_PATH = (
    "python/sglang/srt/models/qwen3.py:141 forward_prepare_native"
    "<br>python/sglang/srt/models/utils.py:230 apply_qk_norm"
    "<br>python/sglang/jit_kernel/fused_qknorm_rope.py:34 fused_qk_norm_rope_out"
    "<br>python/sglang/srt/models/qwen3_moe.py:592 apply_qk_norm_rope"
)


@dataclass
class KernelEvent:
    name: str
    canonical_name: str
    category: str
    pid: str
    tid: str
    ts: float
    dur: float
    external_id: Optional[int]
    correlation: Optional[int] = None


@dataclass
class CpuOpEvent:
    name: str
    pid: str
    tid: str
    ts: float
    dur: float
    external_id: int


@dataclass
class LaunchEvent:
    name: str
    pid: str
    tid: str
    ts: float
    dur: float
    correlation: int


@dataclass
class PythonFrame:
    name: str
    normalized_name: str
    pid: str
    tid: str
    ts: float
    dur: float
    python_id: Optional[int]
    parent_id: Optional[int]

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur


@dataclass
class Aggregate:
    total_us: float = 0.0
    count: int = 0
    max_us: float = 0.0

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


@dataclass
class MappingSiteAggregate:
    total_us: float = 0.0
    count: int = 0
    cpu_ops: Counter = field(default_factory=Counter)
    stacks: Counter = field(default_factory=Counter)


@dataclass
class KernelRow:
    name: str
    category: str
    aggregate: Aggregate
    location: str
    cpu_op: str
    entry: Optional[dict]

    @property
    def total_us(self) -> float:
        return self.aggregate.total_us


@dataclass
class FusionOpportunity:
    pattern: str
    confidence: str
    related_us: float
    evidence: str
    current_locations: str
    candidate_path: str
    rationale: str


def short_name(name: str, max_len: int = 96) -> str:
    text = normalize_text(name)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def canonicalize_name(name: str) -> str:
    text = normalize_text(name)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    if text.startswith("void ") and text.endswith(")"):
        depth = 0
        split_idx: Optional[int] = None
        for idx in range(len(text) - 1, -1, -1):
            char = text[idx]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    split_idx = idx
                    break
        if split_idx is not None:
            text = text[:split_idx]
    return text


def classify_kernel(name: str) -> str:
    # Keep the matching order explicit: strong communication/memory signals win
    # first, then we fall back to weaker category hints.
    lowered = name.lower()
    if contains_any_keyword(lowered, COMMUNICATION_STRONG_KEYWORDS):
        return "communication"
    if contains_any_keyword(lowered, MEMORY_STRONG_KEYWORDS):
        return "memory"
    looks_compute_like = contains_any_keyword(lowered, COMPUTE_HINT_KEYWORDS)
    if contains_any_keyword(lowered, MEMORY_WEAK_KEYWORDS) and not looks_compute_like:
        return "memory"
    for category, keywords in CATEGORY_PATTERNS:
        if contains_any_keyword(lowered, keywords):
            return category
    if (
        contains_any_keyword(lowered, COMMUNICATION_WEAK_KEYWORDS)
        and not looks_compute_like
    ):
        return "communication"
    return "other"


def normalize_source_location(name: str) -> str:
    text = normalize_text(name)
    match = re.match(r"(?P<path>.+?)\((?P<line>\d+)\): (?P<func>.+)$", text)
    if not match:
        return text
    path = normalize_repo_relative_path(match.group("path"))
    return f"{path}:{match.group('line')} {match.group('func')}"


def source_location_priority(location: str) -> int:
    text = str(location).strip()
    if not text or text == "unresolved":
        return -100
    if text.startswith("python/sglang/"):
        return 300
    if text.startswith("sglang/"):
        return 290
    if text.startswith("sgl_kernel/"):
        return 260
    if text.startswith("python/"):
        return 180
    if text.startswith("torch/") or "/torch/" in text:
        return 20
    if ".py:" in text:
        return 120
    return 0


def is_preferred_source_location(location: str) -> bool:
    text = str(location).strip()
    return (
        text.startswith("python/sglang/")
        or text.startswith("sglang/")
        or text.startswith("sgl_kernel/")
    )


def extract_preferred_stack_location(stack: Optional[str]) -> Optional[str]:
    if not stack:
        return None
    parts = [str(part).strip() for part in str(stack).split("->")]
    ranked: List[Tuple[int, int, str]] = []
    for index, part in enumerate(parts):
        normalized = normalize_source_location(part)
        priority = source_location_priority(normalized)
        if priority <= 0:
            continue
        ranked.append((priority, index, normalized))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def site_display_location(site: dict) -> str:
    location = str(site.get("location") or "unresolved").strip()
    if is_preferred_source_location(location):
        return location
    stack_location = extract_preferred_stack_location(site.get("stack"))
    if stack_location:
        return stack_location
    return location


def choose_best_location(locations: Dict[str, MappingSiteAggregate]) -> str:
    if not locations:
        return "unresolved"
    ranked = sorted(
        locations.items(),
        key=lambda pair: (
            source_location_priority(pair[0]),
            pair[1].total_us,
            pair[1].count,
        ),
        reverse=True,
    )
    return ranked[0][0]


def frame_priority(frame_name: str) -> int:
    text = str(frame_name).strip()
    if text.startswith(NOISE_FRAME_PREFIXES):
        return -20
    if text.startswith("/data/") or text.startswith("/Users/"):
        if "/sglang/" in text:
            return 120
        return 100
    if ".py(" in text and "/sglang/" in text:
        return 110
    if ".py(" in text and ("site-packages" in text or text.startswith("torch/")):
        return 45
    if ".py(" in text:
        return 35
    if text.startswith("<built-in method "):
        return -10
    return 0


def stage_label(stage: str) -> str:
    if stage == "extend":
        return "extend/prefill"
    return stage


def stage_aliases(stage: str) -> List[str]:
    if stage == "extend":
        return ["extend", "prefill", "all"]
    if stage == "prefill":
        return ["prefill", "extend", "all"]
    if stage == "decode":
        return ["decode", "all"]
    return [stage, "all"]


def escape_md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def format_ms(value_us: float) -> str:
    return f"{value_us / 1000.0:.2f} ms"


def is_cuda_launch_event(name: str, cat: str) -> bool:
    lowered_name = normalize_text(name).lower()
    lowered_cat = normalize_text(cat).lower()
    if lowered_cat not in {"cuda_runtime", "cuda_driver"}:
        return False
    return "launch" in lowered_name


def is_gpu_kernel_event(event: dict) -> bool:
    # Be conservative here: first drop trace metadata / Python scopes /
    # annotations, then only accept entries with clear GPU-kernel markers.
    if not is_complete_duration_event(event):
        return False
    name = normalize_text(event.get("name", ""))
    if is_trace_metadata_name(name):
        return False
    cat = normalize_text(event.get("cat", "")).lower()
    args = event.get("args") or {}
    if is_non_kernel_trace_category(cat):
        return False
    if is_annotation_event(name, cat):
        return False
    if "kernel" in cat or cat.startswith("gpu_"):
        return True
    if looks_like_python_scope_name(name):
        return False
    return has_stream_marker(args)


def extract_trace_data(
    trace: dict,
) -> Tuple[
    List[KernelEvent],
    List[CpuOpEvent],
    Dict[Tuple[str, str], List[PythonFrame]],
    List[LaunchEvent],
    Optional[str],
    float,
]:
    # Build the basic trace views in one pass so later stages can stay simple:
    # GPU kernels for ranking, CPU ops for External-id mapping, Python frames for
    # source attribution, and CUDA launch calls for correlation-based fallback.
    raw_events = extract_trace_events(trace)
    correlation_external = build_correlation_external_lookup(raw_events)
    chosen_pid = select_heaviest_pid(
        raw_events,
        is_gpu_kernel_event,
        preferred_substrings=("TP00", "TP-0"),
    )

    kernels: List[KernelEvent] = []
    cpu_ops: List[CpuOpEvent] = []
    launches: List[LaunchEvent] = []
    python_frames: DefaultDict[Tuple[str, str], List[PythonFrame]] = defaultdict(list)
    min_ts = None
    max_end = None

    for event in raw_events:
        if event.get("ph") != "X":
            continue

        pid = str(event.get("pid"))
        tid = str(event.get("tid"))
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        cat = str(event.get("cat", ""))
        args = event.get("args") or {}
        name = str(event.get("name", ""))

        if cat == "python_function":
            python_frames[(pid, tid)].append(
                PythonFrame(
                    name=name,
                    normalized_name=normalize_source_location(name),
                    pid=pid,
                    tid=tid,
                    ts=ts,
                    dur=dur,
                    python_id=coerce_optional_int(args.get("Python id")),
                    parent_id=coerce_optional_int(args.get("Python parent id")),
                )
            )

        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if external_id is None and correlation is not None:
            external_id = correlation_external.get(correlation)
        if cat == "cpu_op" and external_id is not None:
            cpu_ops.append(
                CpuOpEvent(
                    name=name,
                    pid=pid,
                    tid=tid,
                    ts=ts,
                    dur=dur,
                    external_id=external_id,
                )
            )
        if is_cuda_launch_event(name, cat) and correlation is not None:
            launches.append(
                LaunchEvent(
                    name=name,
                    pid=pid,
                    tid=tid,
                    ts=ts,
                    dur=dur,
                    correlation=correlation,
                )
            )

        if chosen_pid is None or not is_gpu_kernel_event(event) or pid != chosen_pid:
            continue

        min_ts = ts if min_ts is None else min(min_ts, ts)
        max_end = ts + dur if max_end is None else max(max_end, ts + dur)
        kernels.append(
            KernelEvent(
                name=name,
                canonical_name=canonicalize_name(name),
                category=classify_kernel(name),
                pid=pid,
                tid=tid,
                ts=ts,
                dur=dur,
                external_id=external_id,
                correlation=correlation,
            )
        )

    for frames in python_frames.values():
        frames.sort(key=lambda item: (item.ts, item.end_ts))

    window_us = 0.0 if min_ts is None or max_end is None else max_end - min_ts
    return kernels, cpu_ops, dict(python_frames), launches, chosen_pid, window_us


def build_correlation_external_lookup(raw_events: Sequence[dict]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for event in raw_events:
        args = event.get("args", {}) or {}
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if correlation is not None and external_id is not None:
            lookup[correlation] = external_id
    return lookup


def build_cpu_op_index(cpu_ops: Sequence[CpuOpEvent]) -> Dict[int, List[CpuOpEvent]]:
    output: DefaultDict[int, List[CpuOpEvent]] = defaultdict(list)
    for cpu_op in cpu_ops:
        output[cpu_op.external_id].append(cpu_op)
    for items in output.values():
        items.sort(key=lambda item: item.ts)
    return dict(output)


def match_cpu_op(
    kernel: KernelEvent, cpu_ops_by_external_id: Dict[int, List[CpuOpEvent]]
) -> Optional[CpuOpEvent]:
    if kernel.external_id is None:
        return None
    return match_timed_event(
        cpu_ops_by_external_id.get(kernel.external_id, []), kernel.ts
    )


def build_launch_index(
    launch_events: Sequence[LaunchEvent],
) -> Dict[int, List[LaunchEvent]]:
    output: DefaultDict[int, List[LaunchEvent]] = defaultdict(list)
    for launch in launch_events:
        output[launch.correlation].append(launch)
    for items in output.values():
        items.sort(key=lambda item: item.ts)
    return dict(output)


def match_launch_event(
    kernel: KernelEvent, launches_by_correlation: Dict[int, List[LaunchEvent]]
) -> Optional[LaunchEvent]:
    if kernel.correlation is None:
        return None
    return match_timed_event(
        launches_by_correlation.get(kernel.correlation, []), kernel.ts
    )


def match_timed_event(events: Sequence, probe_ts: float):
    if not events:
        return None
    earlier = [item for item in events if item.ts <= probe_ts + 1e-3]
    if earlier:
        return min(earlier, key=lambda item: abs((item.ts + item.dur) - probe_ts))
    return min(events, key=lambda item: abs(item.ts - probe_ts))


def find_active_python_frames(
    cpu_op: CpuOpEvent,
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
) -> List[PythonFrame]:
    frames = python_frames.get((cpu_op.pid, cpu_op.tid), [])
    if not frames:
        return []
    probe_ts = cpu_op.ts + min(cpu_op.dur * 0.5, 1.0)
    active = [item for item in frames if item.ts <= probe_ts <= item.end_ts]
    active.sort(key=lambda item: (item.ts, item.end_ts))
    return active


def find_active_python_frames_at_ts(
    *,
    pid: str,
    tid: str,
    ts: float,
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
) -> List[PythonFrame]:
    frames = python_frames.get((pid, tid), [])
    if not frames:
        return []
    active = [item for item in frames if item.ts <= ts <= item.end_ts]
    active.sort(key=lambda item: (item.ts, item.end_ts))
    return active


def render_kernel_site(
    active_frames: Sequence[PythonFrame], cpu_op_name: str
) -> Tuple[str, str, str]:
    chosen_frame = choose_mapping_frame(active_frames)
    if chosen_frame is None:
        return "unresolved", "", cpu_op_name
    return chosen_frame.normalized_name, build_stack_display(active_frames), cpu_op_name


def resolve_kernel_site_context(
    kernel: KernelEvent,
    cpu_ops_by_external_id: Dict[int, List[CpuOpEvent]],
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    launches_by_correlation: Dict[int, List[LaunchEvent]],
) -> Tuple[str, str, str]:
    # Prefer the normal External-id path first. If the kernel dropped that link,
    # fall back to the correlated CUDA launch and reuse the Python frames that
    # were active when the launch happened.
    cpu_op = match_cpu_op(kernel, cpu_ops_by_external_id)
    if cpu_op is not None:
        active_frames = find_active_python_frames(cpu_op, python_frames)
        if active_frames:
            return render_kernel_site(active_frames, cpu_op.name)

    launch_event = match_launch_event(kernel, launches_by_correlation)
    if launch_event is not None:
        active_frames = find_active_python_frames_at_ts(
            pid=launch_event.pid,
            tid=launch_event.tid,
            ts=launch_event.ts,
            python_frames=python_frames,
        )
        if active_frames:
            cpu_op_name = cpu_op.name if cpu_op is not None else launch_event.name
            return render_kernel_site(active_frames, cpu_op_name)
        return "unresolved", "", launch_event.name

    cpu_op_name = cpu_op.name if cpu_op is not None else ""
    return "unresolved", "", cpu_op_name


def choose_mapping_frame(active_frames: Sequence[PythonFrame]) -> Optional[PythonFrame]:
    if not active_frames:
        return None
    ranked = sorted(
        active_frames,
        key=lambda item: (frame_priority(item.name), item.ts, -item.dur),
    )
    return ranked[-1]


def build_stack_display(active_frames: Sequence[PythonFrame]) -> str:
    if not active_frames:
        return ""
    filtered = [
        item.normalized_name for item in active_frames if frame_priority(item.name) > 0
    ]
    if not filtered:
        filtered = [active_frames[-1].normalized_name]
    return " -> ".join(filtered[-4:])


def aggregate(events: Iterable[KernelEvent], key_fn) -> Dict[str, Aggregate]:
    output: Dict[str, Aggregate] = defaultdict(Aggregate)
    for event in events:
        key = key_fn(event)
        item = output[key]
        item.total_us += event.dur
        item.count += 1
        item.max_us = max(item.max_us, event.dur)
    return output


def aggregate_kernel_sites(
    kernels: Sequence[KernelEvent],
    cpu_ops_by_external_id: Dict[int, List[CpuOpEvent]],
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    launches_by_correlation: Optional[Dict[int, List[LaunchEvent]]] = None,
) -> Dict[str, Dict[str, MappingSiteAggregate]]:
    # Each kernel is mapped independently so the fallback behavior stays easy to
    # reason about and easy to regression-test.
    output: DefaultDict[str, DefaultDict[str, MappingSiteAggregate]] = defaultdict(
        lambda: defaultdict(MappingSiteAggregate)
    )
    launch_index = launches_by_correlation or {}
    for kernel in kernels:
        location, stack, cpu_op_name = resolve_kernel_site_context(
            kernel,
            cpu_ops_by_external_id,
            python_frames,
            launch_index,
        )

        item = output[kernel.canonical_name][location]
        item.total_us += kernel.dur
        item.count += 1
        if cpu_op_name:
            item.cpu_ops[cpu_op_name] += 1
        if stack:
            item.stacks[stack] += 1
    return {kernel_name: dict(locations) for kernel_name, locations in output.items()}


def merge_site_stats(
    destination: DefaultDict[str, DefaultDict[str, MappingSiteAggregate]],
    source: Dict[str, Dict[str, MappingSiteAggregate]],
) -> None:
    for kernel_name, locations in source.items():
        for location, aggregate_item in locations.items():
            target = destination[kernel_name][location]
            target.total_us += aggregate_item.total_us
            target.count += aggregate_item.count
            target.cpu_ops.update(aggregate_item.cpu_ops)
            target.stacks.update(aggregate_item.stacks)


def build_stage_payload(
    site_stats: Dict[str, Dict[str, MappingSiteAggregate]],
    kernel_categories: Dict[str, str],
) -> Dict[str, dict]:
    kernels_payload: Dict[str, dict] = {}
    for kernel_name, locations in sorted(site_stats.items()):
        total_us = sum(item.total_us for item in locations.values())
        sites = []
        for location, aggregate_item in sorted(
            locations.items(),
            key=lambda pair: pair[1].total_us,
            reverse=True,
        ):
            sites.append(
                {
                    "location": location,
                    "display_location": extract_preferred_stack_location(
                        aggregate_item.stacks.most_common(1)[0][0]
                        if aggregate_item.stacks
                        else None
                    )
                    or location,
                    "launches": aggregate_item.count,
                    "total_us": round(aggregate_item.total_us, 3),
                    "share_pct_within_kernel": round(
                        pct(aggregate_item.total_us, total_us), 3
                    ),
                    "top_cpu_op": (
                        aggregate_item.cpu_ops.most_common(1)[0][0]
                        if aggregate_item.cpu_ops
                        else None
                    ),
                    "stack": (
                        aggregate_item.stacks.most_common(1)[0][0]
                        if aggregate_item.stacks
                        else None
                    ),
                }
            )
        sites.sort(
            key=lambda site: (
                source_location_priority(site_display_location(site)),
                float(site.get("total_us", 0.0)),
                int(site.get("launches", 0)),
            ),
            reverse=True,
        )
        kernels_payload[kernel_name] = {
            "category": kernel_categories.get(kernel_name, "other"),
            "sites": sites,
            "best_location": (
                site_display_location(sites[0])
                if sites
                else choose_best_location(locations)
            ),
        }
    return {"kernels": kernels_payload}


def load_kernel_map(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def relaxed_kernel_entry_lookup(
    kernels: Dict[str, dict], kernel_name: str
) -> Optional[dict]:
    if kernel_name in kernels:
        return kernels[kernel_name]
    lowered = kernel_name.lower()
    best_key = None
    best_score = -1
    for candidate_key in kernels:
        candidate_lowered = candidate_key.lower()
        if candidate_lowered.startswith(lowered) or lowered.startswith(
            candidate_lowered
        ):
            score = min(len(candidate_lowered), len(lowered))
        elif candidate_lowered in lowered or lowered in candidate_lowered:
            score = min(len(candidate_lowered), len(lowered)) // 2
        else:
            continue
        if score > best_score:
            best_key = candidate_key
            best_score = score
    return kernels.get(best_key) if best_key else None


def lookup_kernel_map_entry(
    kernel_map: dict, stage: str, kernel_name: str
) -> Optional[dict]:
    stage_map = kernel_map.get("stages", {})
    for candidate_stage in stage_aliases(stage):
        entry = relaxed_kernel_entry_lookup(
            stage_map.get(candidate_stage, {}).get("kernels", {}),
            kernel_name,
        )
        if entry:
            return entry
    return relaxed_kernel_entry_lookup(
        kernel_map.get("global", {}).get("kernels", {}), kernel_name
    )


def best_site_summary(kernel_entry: Optional[dict]) -> Tuple[str, str]:
    if not kernel_entry:
        return "unresolved", "-"
    sites = kernel_entry.get("sites") or []
    if not sites:
        return kernel_entry.get("best_location", "unresolved"), "-"
    preferred_sites = [
        site
        for site in sites
        if is_preferred_source_location(site_display_location(site))
    ]
    candidate_sites = preferred_sites or sites
    rendered_locations = []
    rendered_cpu_ops = []
    for site in candidate_sites[:2]:
        location = site_display_location(site)
        share = site.get("share_pct_within_kernel")
        if len(candidate_sites) > 1 and share is not None:
            rendered_locations.append(f"{location} (site share {share:.0f}%)")
        else:
            rendered_locations.append(location)
        cpu_op = site.get("top_cpu_op")
        if cpu_op:
            rendered_cpu_ops.append(cpu_op)
    return "<br>".join(rendered_locations), (
        "<br>".join(rendered_cpu_ops) if rendered_cpu_ops else "-"
    )


def resolve_kernel_entry(
    stage: str,
    kernel_name: str,
    local_stage_payload: dict,
    external_kernel_map: Optional[dict],
) -> Optional[dict]:
    if external_kernel_map:
        kernel_entry = lookup_kernel_map_entry(external_kernel_map, stage, kernel_name)
        if kernel_entry:
            return kernel_entry
    return local_stage_payload.get("kernels", {}).get(kernel_name)


def build_kernel_rows(
    stage: str,
    kernel_stats: Dict[str, Aggregate],
    kernel_categories: Dict[str, str],
    local_stage_payload: dict,
    external_kernel_map: Optional[dict],
) -> List[KernelRow]:
    rows: List[KernelRow] = []
    for kernel_name, aggregate_item in sorted(
        kernel_stats.items(),
        key=lambda pair: pair[1].total_us,
        reverse=True,
    ):
        kernel_entry = resolve_kernel_entry(
            stage, kernel_name, local_stage_payload, external_kernel_map
        )
        location, cpu_op = best_site_summary(kernel_entry)
        rows.append(
            KernelRow(
                name=kernel_name,
                category=kernel_categories.get(kernel_name, "other"),
                aggregate=aggregate_item,
                location=location,
                cpu_op=cpu_op,
                entry=kernel_entry,
            )
        )
    return rows


def limit_kernel_rows(rows: Sequence[KernelRow], table_limit: int) -> List[KernelRow]:
    if table_limit <= 0:
        return list(rows)
    return list(rows[:table_limit])


def entry_sites(kernel_entry: Optional[dict]) -> List[dict]:
    if not kernel_entry:
        return []
    sites = kernel_entry.get("sites") or []
    return [site for site in sites if site.get("location")]


def ordered_unique(values: Iterable[str], limit: int = 4) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def kernel_row_locations(row: KernelRow, limit: int = 4) -> List[str]:
    values = [site_display_location(site) for site in entry_sites(row.entry)]
    if not values and row.location and row.location != "unresolved":
        values = [fragment.strip() for fragment in row.location.split("<br>")]
    return ordered_unique(values, limit=limit)


def format_location_for_fusion_display(location: str) -> str:
    text = normalize_text(location)
    match = re.match(r"(?P<path>.+?):(?P<line>\d+)\s+(?P<func>.+)$", text)
    if not match:
        return text
    return f"{match.group('func')} @ {match.group('path')}:{match.group('line')}"


def row_matches(row: KernelRow, *needles: str) -> bool:
    lowered = " ".join([row.name, row.location, row.cpu_op]).lower()
    return any(needle in lowered for needle in needles)


def summarize_text(values: Iterable[str], limit: int = 4) -> str:
    items = ordered_unique(values, limit=limit)
    return "<br>".join(items) if items else "-"


def summarize_locations(values: Iterable[str], limit: int = 4) -> str:
    items = ordered_unique(
        (format_location_for_fusion_display(value) for value in values),
        limit=limit,
    )
    return "<br>".join(items) if items else "-"


def summarize_evidence(
    rows: Sequence[KernelRow], total_us: float, limit: int = 3
) -> str:
    items = []
    for row in rows[:limit]:
        items.append(f"{row.name} ({pct(row.total_us, total_us):.1f}%)")
    return "<br>".join(items) if items else "-"


def model_path_from_server_args(server_args: Optional[dict]) -> str:
    if not isinstance(server_args, dict):
        return ""
    return str(server_args.get("model_path") or server_args.get("model") or "")


def detect_fusion_opportunities(
    stage: str,
    kernel_rows: Sequence[KernelRow],
    total_us: float,
    server_args: Optional[dict],
) -> List[FusionOpportunity]:
    opportunities: List[FusionOpportunity] = []
    if total_us <= 0:
        return opportunities

    model_path = model_path_from_server_args(server_args).lower()
    tp_size = 1
    if isinstance(server_args, dict):
        tp_size = int(server_args.get("tp_size") or 1)

    comm_rows = [
        row
        for row in kernel_rows
        if row.category == "communication"
        and (
            row_matches(
                row,
                "cross_device_reduce",
                "allreduce",
                "all_reduce",
                "custom_all_reduce_ops.py",
            )
        )
    ]
    comm_us = sum(row.total_us for row in comm_rows)
    if comm_rows and (
        tp_size > 1
        or any(row_matches(row, "custom_all_reduce_ops.py") for row in comm_rows)
    ):
        opportunities.append(
            FusionOpportunity(
                pattern="TP all-reduce + residual/RMSNorm",
                confidence="Likely" if pct(comm_us, total_us) >= 4.0 else "Conditional",
                related_us=comm_us,
                evidence=summarize_evidence(comm_rows, total_us),
                current_locations=summarize_locations(
                    location
                    for row in comm_rows
                    for location in kernel_row_locations(row)
                ),
                candidate_path=ALLREDUCE_FUSION_PATH,
                rationale=(
                    f"TP communication already consumes {pct(comm_us, total_us):.1f}% of cumulative GPU kernel "
                    "time, and SGLang already exposes a fused allreduce+RMSNorm path for the residual/norm "
                    "boundary."
                ),
            )
        )

    is_qwen3_dense = (
        "qwen3" in model_path
        and "moe" not in model_path
        and "qwen3-next" not in model_path
    )
    qwen3_qk_rows = [
        row
        for row in kernel_rows
        if row_matches(
            row,
            "apply_qk_norm",
            "fused_inplace_qknorm",
            "qknorm",
        )
    ]
    qwen3_rope_rows = [
        row
        for row in kernel_rows
        if row_matches(
            row,
            "apply_rope",
            "rope.py",
            "rope_inplace",
            "rotary",
        )
    ]
    qwen3_rows: List[KernelRow] = []
    seen_qwen3_keys = set()
    for row in qwen3_qk_rows + qwen3_rope_rows:
        row_key = (row.name, row.location, row.cpu_op)
        if row_key in seen_qwen3_keys:
            continue
        seen_qwen3_keys.add(row_key)
        qwen3_rows.append(row)
    qwen3_related_us = sum(row.total_us for row in qwen3_rows)
    if (
        is_qwen3_dense
        and qwen3_qk_rows
        and qwen3_rope_rows
        and pct(qwen3_related_us, total_us) >= 1.0
    ):
        opportunities.append(
            FusionOpportunity(
                pattern="Q/K RMSNorm + RoPE before attention",
                confidence="Conditional",
                related_us=qwen3_related_us,
                evidence=summarize_evidence(qwen3_rows, total_us),
                current_locations=summarize_locations(
                    location
                    for row in qwen3_rows
                    for location in kernel_row_locations(row)
                ),
                candidate_path=QWEN3_QK_ROPE_FUSION_PATH,
                rationale=(
                    "Dense Qwen3 still prepares Q/K with `apply_qk_norm` and then `rotary_emb` in separate source "
                    "steps, while SGLang already ships a fused QK-norm+RoPE kernel path in its JIT/MoE stack."
                ),
            )
        )

    opportunities.sort(key=lambda item: item.related_us, reverse=True)
    return opportunities


def generate_takeaways(
    stage: str,
    total_us: float,
    window_us: float,
    category_stats: Dict[str, Aggregate],
    resolved_us: float,
    server_args: Optional[dict],
    fusion_opportunities: Sequence[FusionOpportunity],
) -> List[str]:
    items = sorted(
        category_stats.items(), key=lambda pair: pair[1].total_us, reverse=True
    )
    if not items:
        return ["No GPU kernel events were found in the selected trace."]

    takeaways: List[str] = []
    top_name, top_agg = items[0]
    takeaways.append(
        f"{stage_label(stage)} is dominated by `{top_name}` at {pct(top_agg.total_us, total_us):.1f}% of cumulative GPU kernel time."
    )
    if len(items) > 1:
        second_name, second_agg = items[1]
        combined = pct(top_agg.total_us + second_agg.total_us, total_us)
        takeaways.append(
            f"The top two categories are `{top_name}` + `{second_name}` at {combined:.1f}% combined."
        )

    comm_share = pct(
        category_stats.get("communication", Aggregate()).total_us, total_us
    )
    if comm_share >= 10.0:
        tp = server_args.get("tp_size") if isinstance(server_args, dict) else None
        if tp and tp > 1:
            takeaways.append(
                f"`communication` already accounts for {comm_share:.1f}% of cumulative GPU time in this TP={tp} run."
            )
        else:
            takeaways.append(
                f"`communication` shows up at {comm_share:.1f}% even without an obvious large-TP context."
            )

    if pct(resolved_us, total_us) >= 70.0:
        takeaways.append(
            f"Kernel-to-Python mapping covers {pct(resolved_us, total_us):.1f}% of cumulative GPU time, so the table is representative enough for code triage."
        )

    if window_us > 0:
        parallelism = total_us / window_us
        if parallelism >= 1.15:
            takeaways.append(
                f"Summed kernel time is {parallelism:.2f}x the GPU time window, so these percentages are cumulative launch share rather than wall time share."
            )
    if fusion_opportunities:
        top_pattern = fusion_opportunities[0]
        takeaways.append(
            f"The strongest source-backed fuse candidate is `{top_pattern.pattern}` with {pct(top_pattern.related_us, total_us):.1f}% related GPU time in this stage."
        )
    return takeaways


def print_mapping_table(
    kernel_rows: Sequence[KernelRow],
    total_us: float,
    table_limit: int,
) -> float:
    resolved_us = 0.0
    rendered_rows = limit_kernel_rows(kernel_rows, table_limit)
    label = "all kernels" if table_limit <= 0 else f"first {len(rendered_rows)} kernels"
    print(f"\nKernel-to-Python mapping (Markdown, {label}):")
    print(
        "| Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |"
    )
    print("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for row in rendered_rows:
        if row.location != "unresolved":
            resolved_us += row.total_us
        print(
            "| {kernel} | {category} | {gpu_time} | {share:.1f}% | {launches} | {location} | {cpu_op} |".format(
                kernel=escape_md_cell(row.name),
                category=escape_md_cell(row.category),
                gpu_time=format_ms(row.total_us),
                share=pct(row.total_us, total_us),
                launches=row.aggregate.count,
                location=escape_md_cell(row.location),
                cpu_op=escape_md_cell(row.cpu_op),
            )
        )
    return resolved_us


def print_fusion_opportunity_table(
    opportunities: Sequence[FusionOpportunity],
    total_us: float,
) -> None:
    print("\nKernel fuse opportunities (Markdown):")
    print(
        "| Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |"
    )
    print("| --- | --- | ---: | ---: | --- | --- | --- | --- |")
    if not opportunities:
        print(
            "| No medium-confidence source-backed fusion opportunity matched this trace. | - | - | - | - | - | - | - |"
        )
        return
    for item in opportunities:
        print(
            "| {pattern} | {confidence} | {gpu_time} | {share:.1f}% | {evidence} | {current_locations} | {candidate_path} | {rationale} |".format(
                pattern=escape_md_cell(item.pattern),
                confidence=escape_md_cell(item.confidence),
                gpu_time=format_ms(item.related_us),
                share=pct(item.related_us, total_us),
                evidence=escape_md_cell(item.evidence),
                current_locations=escape_md_cell(item.current_locations),
                candidate_path=escape_md_cell(item.candidate_path),
                rationale=escape_md_cell(item.rationale),
            )
        )


def print_report(
    trace_path: Path,
    server_args: Optional[dict],
    kernels: List[KernelEvent],
    chosen_pid: Optional[str],
    window_us: float,
    local_stage_payload: dict,
    external_kernel_map: Optional[dict],
    top_k: int,
    kernel_table_limit: int,
    table_only: bool,
) -> None:
    stage = parse_stage(trace_path)
    total_us = sum(kernel.dur for kernel in kernels)
    print(f"Trace: {trace_path}")
    print(f"Stage: {stage_label(stage)}")
    if chosen_pid:
        print(f"Selected PID: {chosen_pid}")

    if server_args:
        model_path = server_args.get("model_path") or server_args.get("model")
        tp_size = server_args.get("tp_size")
        dp_size = server_args.get("dp_size")
        print(f"Model: {model_path}")
        if tp_size or dp_size:
            print(f"Parallelism: tp={tp_size or 1} dp={dp_size or 1}")

    if not kernels:
        print("No GPU kernel events found.\n")
        return

    print(
        f"GPU kernels: {len(kernels)} | cumulative kernel time: {format_ms(total_us)} | "
        f"GPU window: {format_ms(window_us)} | avg parallelism: {total_us / window_us:.2f}x"
        if window_us
        else f"GPU kernels: {len(kernels)} | cumulative kernel time: {format_ms(total_us)}"
    )

    category_stats = aggregate(kernels, key_fn=lambda item: item.category)
    kernel_stats = aggregate(kernels, key_fn=lambda item: item.canonical_name)
    kernel_categories = {kernel.canonical_name: kernel.category for kernel in kernels}
    kernel_rows = build_kernel_rows(
        stage=stage,
        kernel_stats=kernel_stats,
        kernel_categories=kernel_categories,
        local_stage_payload=local_stage_payload,
        external_kernel_map=external_kernel_map,
    )
    fusion_opportunities = detect_fusion_opportunities(
        stage=stage,
        kernel_rows=kernel_rows,
        total_us=total_us,
        server_args=server_args,
    )

    if not table_only:
        print("\nTop categories by cumulative GPU kernel time:")
        for idx, (name, aggregate_item) in enumerate(
            sorted(
                category_stats.items(), key=lambda pair: pair[1].total_us, reverse=True
            )[:8],
            start=1,
        ):
            print(
                f"  {idx}. {name:<16} {format_ms(aggregate_item.total_us):>10}  "
                f"{pct(aggregate_item.total_us, total_us):>5.1f}%  launches={aggregate_item.count}"
            )

        print("\nTop kernels by cumulative GPU kernel time:")
        for idx, (name, aggregate_item) in enumerate(
            sorted(
                kernel_stats.items(), key=lambda pair: pair[1].total_us, reverse=True
            )[:top_k],
            start=1,
        ):
            print(
                f"  {idx}. {short_name(name, 76):<76} {format_ms(aggregate_item.total_us):>10}  "
                f"{pct(aggregate_item.total_us, total_us):>5.1f}%  launches={aggregate_item.count}  avg={format_ms(aggregate_item.avg_us)}"
            )

    resolved_us = print_mapping_table(
        kernel_rows=kernel_rows,
        total_us=total_us,
        table_limit=kernel_table_limit,
    )
    print_fusion_opportunity_table(fusion_opportunities, total_us)

    if not table_only:
        print("\nTakeaways:")
        for takeaway in generate_takeaways(
            stage,
            total_us,
            window_us,
            category_stats,
            resolved_us,
            server_args,
            fusion_opportunities,
        ):
            print(f"  - {takeaway}")
        print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SGLang LLM torch profiler traces into kernel/category shares."
    )
    parser.add_argument("--input", type=str, help="Trace file or profile directory.")
    parser.add_argument("--url", type=str, help="Running SGLang server URL.")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output root for live profiling."
    )
    parser.add_argument(
        "--num-steps", type=int, default=5, help="Profiler steps for live mode."
    )
    parser.add_argument(
        "--profile-by-stage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--merge-profiles", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--profile-prefix", type=str, default=None)
    parser.add_argument("--probe-requests", type=int, default=1)
    parser.add_argument(
        "--probe-prompt",
        type=str,
        default="Explain what tensor parallelism is in one short paragraph.",
    )
    parser.add_argument("--probe-max-new-tokens", type=int, default=96)
    parser.add_argument("--probe-delay", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many top kernels to summarize above the tables.",
    )
    parser.add_argument(
        "--kernel-table-limit",
        type=int,
        default=0,
        help="How many kernels to include in the Markdown kernel table. Use 0 for all kernels.",
    )
    parser.add_argument(
        "--kernel-map",
        type=str,
        default=None,
        help="Existing kernel_map.json from a no-CUDA-graph torch pre-pass.",
    )
    parser.add_argument(
        "--export-kernel-map",
        type=str,
        default=None,
        help="Write kernel-to-Python mapping JSON to this file.",
    )
    parser.add_argument(
        "--all-traces",
        action="store_true",
        help="Analyze every matching trace in the selected run directory.",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Print the trace header plus the kernel and fuse-opportunity tables only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input and not args.url:
        raise SystemExit("Provide either --input or --url.")

    input_path = Path(args.input).resolve() if args.input else None
    if args.url:
        profile_dir = run_profiler(
            url=args.url,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            profile_by_stage=args.profile_by_stage,
            merge_profiles=args.merge_profiles,
            profile_prefix=args.profile_prefix,
            probe_requests=args.probe_requests,
            probe_prompt=args.probe_prompt,
            probe_max_new_tokens=args.probe_max_new_tokens,
            probe_delay=args.probe_delay,
            start_step=None,
        )
        print(f"Generated profile directory: {profile_dir}\n")
        input_path = profile_dir

    external_kernel_map = (
        load_kernel_map(Path(args.kernel_map).resolve()) if args.kernel_map else None
    )
    traces, server_args = discover_trace_targets(input_path, all_traces=args.all_traces)

    stage_site_stats: DefaultDict[
        str, DefaultDict[str, DefaultDict[str, MappingSiteAggregate]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(MappingSiteAggregate)))
    stage_kernel_categories: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
    global_site_stats: DefaultDict[str, DefaultDict[str, MappingSiteAggregate]] = (
        defaultdict(lambda: defaultdict(MappingSiteAggregate))
    )
    global_kernel_categories: Dict[str, str] = {}
    reports = []

    for trace_path in traces:
        trace = load_trace_json(trace_path)
        kernels, cpu_ops, python_frames, launch_events, chosen_pid, window_us = (
            extract_trace_data(trace)
        )
        cpu_ops_by_external_id = build_cpu_op_index(cpu_ops)
        launches_by_correlation = build_launch_index(launch_events)
        local_site_stats = aggregate_kernel_sites(
            kernels,
            cpu_ops_by_external_id,
            python_frames,
            launches_by_correlation=launches_by_correlation,
        )
        stage = parse_stage(trace_path)
        kernel_categories = {
            kernel.canonical_name: kernel.category for kernel in kernels
        }

        merge_site_stats(stage_site_stats[stage], local_site_stats)
        merge_site_stats(global_site_stats, local_site_stats)
        stage_kernel_categories[stage].update(kernel_categories)
        global_kernel_categories.update(kernel_categories)
        reports.append(
            (trace_path, kernels, chosen_pid, window_us, stage, kernel_categories)
        )

    stage_payloads = {
        stage: build_stage_payload(
            dict(site_stats), stage_kernel_categories.get(stage, {})
        )
        for stage, site_stats in stage_site_stats.items()
    }
    global_payload = build_stage_payload(
        dict(global_site_stats), global_kernel_categories
    )

    if args.export_kernel_map:
        export_path = Path(args.export_kernel_map).resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "generated_from": str(input_path),
            "notes": "Kernel-to-Python mapping extracted from torch profiler traces. Prefer generating this from a --disable-cuda-graph --disable-piecewise-cuda-graph pre-pass.",
            "server_args": server_args,
            "stages": stage_payloads,
            "global": global_payload,
        }
        with open(export_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Exported kernel map: {export_path}\n")

    for trace_path, kernels, chosen_pid, window_us, stage, _ in reports:
        print_report(
            trace_path=trace_path,
            server_args=server_args,
            kernels=kernels,
            chosen_pid=chosen_pid,
            window_us=window_us,
            local_stage_payload=stage_payloads.get(stage, {"kernels": {}}),
            external_kernel_map=external_kernel_map,
            top_k=args.top_k,
            kernel_table_limit=args.kernel_table_limit,
            table_only=args.table_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
