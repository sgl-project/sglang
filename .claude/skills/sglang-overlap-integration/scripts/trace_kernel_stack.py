"""Trace kernel stack analyzer for overlap integration.

Self-contained script that extracts full Python call stacks for specific GPU
kernels from a torch profiler trace.  Designed for Pattern 0 of the
sglang-overlap-integration skill: identify which source code is responsible
for launching the kernels you want to overlap.

Compared to the general-purpose triage script this tool:
  * Accepts ``--kernel-filter`` to restrict output to kernels whose name
    contains any of the given substrings (case-insensitive).
  * Provides ``--list-kernels`` mode to discover actual kernel names before
    running a full stack analysis — essential when user-provided names
    don't exactly match the canonical kernel names in the trace.
  * Shows **unlimited** call-stack depth instead of the default 4-frame
    cap, controlled by ``--stack-depth``.
  * Outputs a compact, goal-oriented table keyed by (kernel,
    python-location, stack) instead of a full triage report.

Typical two-step workflow:
    # Step 1: Discover actual kernel names matching your keywords
    python3 trace_kernel_stack.py --input <trace.json> \\
        --list-kernels --kernel-filter <keyword1> <keyword2> ...

    # Step 2: Use the discovered names for full stack analysis
    python3 trace_kernel_stack.py --input <trace.json> \\
        --kernel-filter <discovered_substring1> <discovered_substring2> ... \\
        --stack-depth 0 --format chain
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Text / path helpers (inlined from profile_common to stay self-contained)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=65536)
def _normalize_text_cached(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    for token in (" ", "\t", "\n", "\r", "\v", "\f"):
        if token in text:
            return " ".join(text.split())
    return text


def normalize_text(value: object) -> str:
    return _normalize_text_cached(value if isinstance(value, str) else str(value))


@lru_cache(maxsize=65536)
def _normalize_repo_relative_path_cached(text: str) -> str:
    text = text.replace("\\", "/")
    lowered = text.lower()
    for marker, normalized_marker in (
        ("python/sglang/", "python/sglang/"),
        ("sgl_kernel/", "sgl_kernel/"),
        ("vllm/", "vllm/"),
        ("tensorrt_llm/", "tensorrt_llm/"),
        ("tensorrt-llm/", "tensorrt_llm/"),
    ):
        idx = lowered.find(marker)
        if idx != -1:
            suffix = text[idx + len(marker) :].lstrip("/")
            return f"{normalized_marker}{suffix}".lstrip("/")
    idx = lowered.find("sglang/")
    if idx != -1:
        return ("python/" + text[idx:]).lstrip("/")
    return text.lstrip("/")


def normalize_repo_relative_path(path: object) -> str:
    return _normalize_repo_relative_path_cached(normalize_text(path))


def coerce_optional_int(value: object) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def extract_trace_events(trace: object) -> Sequence[dict]:
    if isinstance(trace, dict):
        events = trace.get("traceEvents", [])
        return events if isinstance(events, list) else []
    if isinstance(trace, list):
        return trace
    return []


def contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalize_for_match(text: str) -> str:
    """Lowercase and strip underscores so that snake_case and camelCase names
    can match each other.  For example, ``reduce_scatter`` and ``ReduceScatter``
    both become ``reducescatter``."""
    return text.lower().replace("_", "")


def _keyword_matches_text(keyword: str, text: str) -> bool:
    """Check whether *keyword* appears in *text*, trying both the raw
    case-insensitive match and the underscore-normalised match.

    This bridges the naming gap between user-facing snake_case names
    (``reduce_scatter``, ``all_reduce``) and camelCase / PascalCase kernel
    names (``ReduceScatter``, ``AllReduce``) commonly found in NCCL / CUDA
    profiler events.
    """
    kw_lower = keyword.lower()
    text_lower = text.lower()
    if kw_lower in text_lower:
        return True
    # Strip underscores from both sides so reduce_scatter matches ReduceScatter
    kw_norm = _normalize_for_match(keyword)
    text_norm = _normalize_for_match(text)
    return kw_norm in text_norm


TRACE_METADATA_NAMES = {
    "process_name",
    "thread_name",
    "process_sort_index",
    "thread_sort_index",
}

NON_KERNEL_TRACE_CATEGORIES = ("python_function", "cpu_op", "trace")

PYTHON_SCOPE_NAME_PREFIXES = ("python/", "nn.module:")


def is_trace_metadata_name(name: object) -> bool:
    return str(name) in TRACE_METADATA_NAMES


def is_complete_duration_event(event: dict) -> bool:
    if event.get("ph") != "X":
        return False
    dur = event.get("dur")
    ts = event.get("ts")
    if dur is None or ts is None:
        return False
    try:
        return float(dur) > 0
    except (TypeError, ValueError):
        return False


def is_annotation_event(name: object, category: object) -> bool:
    lowered_name = normalize_text(name).lower()
    lowered_category = normalize_text(category).lower()
    return "annotation" in lowered_category or lowered_name.startswith("## call ")


def is_non_kernel_trace_category(category: object) -> bool:
    lowered_category = normalize_text(category).lower()
    return any(token in lowered_category for token in NON_KERNEL_TRACE_CATEGORIES)


def looks_like_python_scope_name(name: object) -> bool:
    lowered_name = normalize_text(name).lower()
    return ".py(" in lowered_name or lowered_name.startswith(PYTHON_SCOPE_NAME_PREFIXES)


def has_stream_marker(args: Optional[dict]) -> bool:
    trace_args = args or {}
    return "stream" in trace_args or "cuda_stream" in trace_args


def is_gpu_kernel_event(event: dict) -> bool:
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


def is_cuda_launch_event(name: str, cat: str) -> bool:
    lowered_name = normalize_text(name).lower()
    lowered_cat = normalize_text(cat).lower()
    if lowered_cat not in {"cuda_runtime", "cuda_driver"}:
        return False
    return "launch" in lowered_name


def select_heaviest_pid(
    events: Sequence[dict],
    event_filter,
    preferred_substrings: Sequence[str] = (),
) -> Optional[str]:
    durations: Counter = Counter()
    for event in events:
        if not event_filter(event):
            continue
        pid = str(event.get("pid"))
        durations[pid] += float(event["dur"])
    if not durations:
        return None
    for substring in preferred_substrings:
        preferred = [pid for pid in durations if substring in pid]
        if preferred:
            return max(preferred, key=lambda pid: durations[pid])
    return max(durations, key=lambda pid: durations[pid])


def load_trace_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Kernel name canonicalization & classification
# ---------------------------------------------------------------------------

@lru_cache(maxsize=65536)
def normalize_source_location(name: str) -> str:
    text = normalize_text(name)
    match = re.match(r"(?P<path>.+?)\((?P<line>\d+)\): (?P<func>.+)$", text)
    if not match:
        return text
    path = normalize_repo_relative_path(match.group("path"))
    return f"{path}:{match.group('line')} {match.group('func')}"


@lru_cache(maxsize=65536)
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

CATEGORY_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("hybrid_linear", ("gdn", "gated_delta", "mamba", "selective_scan", "ssd", "causal_conv", "ssm")),
    ("attention", ("flash_attn", "flashattention", "flash_attention", "fmha", "attention", "mla", "paged_attention", "decode_attention")),
    ("moe", ("fused_moe", "grouped_mm", "groupgemm", "group_gemm", "moe", "expert", "groupproblemshape")),
    ("gemm", ("gemm", "gemv", "matmul", "cublas", "cutlass", "wgmma", "mma", "bmm", "nvjet")),
    ("norm", ("rmsnorm", "layernorm", "_norm_", " norm", "normkernel")),
    ("rope", ("rotary", "rope", "mrope")),
    ("softmax", ("softmax",)),
    ("activation", ("silu", "gelu", "relu", "act_and_mul", "sigmoid")),
    ("quantize", ("quant", "fp8", "mxfp", "nvfp4", "dequant", "cvt")),
    ("reduce_topk", ("topk", "reduce", "argmax", "argtopk", "sampling", "multinomial")),
    ("communication", ("broadcast", "dispatch", "combine")),
    ("memory", ("memcpy", "memset", "dma", "prefetch", "copy", "fill")),
]


@lru_cache(maxsize=65536)
def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if contains_any_keyword(lowered, COMMUNICATION_STRONG_KEYWORDS):
        return "communication"
    if contains_any_keyword(lowered, ("memcpy", "memset", "dma", "prefetch")):
        return "memory"
    looks_compute_like = contains_any_keyword(
        lowered, ("gemm", "gemv", "matmul", "cublas", "cutlass", "wgmma", "mma", "bmm", "nvjet", "fmha", "attention", "flash_attn", "flashattention", "flash_attention", "grouped_mm", "groupgemm", "moe", "expert")
    )
    if contains_any_keyword(lowered, ("copy", "fill")) and not looks_compute_like:
        return "memory"
    for category, keywords in CATEGORY_PATTERNS:
        if contains_any_keyword(lowered, keywords):
            return category
    if contains_any_keyword(lowered, ("broadcast", "dispatch", "combine")) and not looks_compute_like:
        return "communication"
    return "other"


# ---------------------------------------------------------------------------
# Python frame priority & stack display
# ---------------------------------------------------------------------------

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

LOW_LEVEL_FRAME_PREFIXES = (
    "triton/runtime/",
    "triton/backends/",
    "torch/_ops.py",
    "torch/nn/modules/module.py",
)

LOW_SIGNAL_FUNCTION_TOKENS = (
    "__torch_function__",
    "__torch_dispatch__",
    "__call__",
    "_call_impl",
    "_wrapped_call_impl",
)

LOW_SIGNAL_PATH_TOKENS = (
    "model_executor/parameter.py:",
    "model_executor/cuda_graph_runner.py:",
    "compilation/cuda_graph.py:",
    "pyexecutor/cuda_graph_runner.py:",
    "pyexecutor/py_executor.py:",
    "_torch/utils.py:",
    "torch/fx/graph_module.py:",
)


def is_low_signal_source_location(location: str) -> bool:
    lowered = str(location).strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in LOW_SIGNAL_FUNCTION_TOKENS) or any(
        token in lowered for token in LOW_SIGNAL_PATH_TOKENS
    )


@lru_cache(maxsize=65536)
def frame_priority(frame_name: str) -> int:
    raw_text = str(frame_name).strip()
    normalized_text = normalize_source_location(raw_text)
    penalty = 80 if is_low_signal_source_location(normalized_text) else 0
    if raw_text.startswith(NOISE_FRAME_PREFIXES):
        return -20
    if normalized_text.startswith("python/sglang/"):
        return 300 - penalty
    if normalized_text.startswith("sglang/"):
        return 290 - penalty
    if normalized_text.startswith("vllm/"):
        return 285 - penalty
    if normalized_text.startswith("tensorrt_llm/"):
        return 280 - penalty
    if normalized_text.startswith("sgl_kernel/"):
        return 260 - penalty
    if normalized_text.startswith("triton_kernels/"):
        return 220 - penalty
    if normalized_text.startswith(LOW_LEVEL_FRAME_PREFIXES):
        return 0
    if raw_text.startswith("/data/") or raw_text.startswith("/Users/"):
        if "/sglang/" in raw_text:
            return 120
        if "/vllm/" in raw_text:
            return 118
        if "/TensorRT-LLM/" in raw_text or "/tensorrt_llm/" in raw_text:
            return 116
        return 100
    if ".py(" in raw_text and "/sglang/" in raw_text:
        return 110
    if ".py(" in raw_text and "/vllm/" in raw_text:
        return 108
    if ".py(" in raw_text and ("/TensorRT-LLM/" in raw_text or "/tensorrt_llm/" in raw_text):
        return 106
    if ".py:" in normalized_text and ("site-packages" in raw_text or normalized_text.startswith("torch/")):
        return 45
    if ".py:" in normalized_text:
        return 35
    if raw_text.startswith("<built-in method "):
        return -10
    return 0


def build_stack_display(active_frames: Sequence["PythonFrame"], max_depth: int = 0) -> str:
    """Build a multi-layer stack string from active Python frames.

    Args:
        active_frames: Frames sorted by (ts, end_ts).
        max_depth: Maximum number of frames to show.  0 means unlimited.
    """
    if not active_frames:
        return ""
    filtered = [item.normalized_name for item in active_frames if item.priority > 0]
    if not filtered:
        filtered = [active_frames[-1].normalized_name]
    if max_depth > 0:
        filtered = filtered[-max_depth:]
    return " -> ".join(filtered)


def choose_mapping_frame(active_frames: Sequence["PythonFrame"]) -> Optional["PythonFrame"]:
    if not active_frames:
        return None
    best = active_frames[0]
    best_key = (best.priority, best.ts, -best.dur)
    for item in active_frames[1:]:
        key = (item.priority, item.ts, -item.dur)
        if key > best_key:
            best = item
            best_key = key
    return best


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    end_ts: float
    priority: int


@dataclass
class TimedEventIndex:
    events: List[object]
    start_ts: List[float]


@dataclass
class StackInfo:
    """Per-kernel stack attribution result."""
    kernel_name: str
    canonical_name: str
    category: str
    gpu_time_us: float
    location: str
    stack: str
    cpu_op: str
    stream: Optional[int] = None


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------

def build_correlation_external_lookup(raw_events: Sequence[dict]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for event in raw_events:
        args = event.get("args", {}) or {}
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if correlation is not None and external_id is not None:
            lookup[correlation] = external_id
    return lookup


def build_timed_event_index(events: Sequence[object]) -> TimedEventIndex:
    ordered = list(events)
    ordered.sort(key=lambda item: item.ts)
    return TimedEventIndex(
        events=ordered,
        start_ts=[float(item.ts) for item in ordered],
    )


def build_cpu_op_index(cpu_ops: Sequence[CpuOpEvent]) -> Dict[int, TimedEventIndex]:
    output: Dict[int, List[CpuOpEvent]] = defaultdict(list)
    for cpu_op in cpu_ops:
        output[cpu_op.external_id].append(cpu_op)
    return {
        external_id: build_timed_event_index(items)
        for external_id, items in output.items()
    }


def build_launch_index(launch_events: Sequence[LaunchEvent]) -> Dict[int, TimedEventIndex]:
    output: Dict[int, List[LaunchEvent]] = defaultdict(list)
    for launch in launch_events:
        output[launch.correlation].append(launch)
    return {
        correlation: build_timed_event_index(items)
        for correlation, items in output.items()
    }


def match_timed_event(index, probe_ts: float):
    if not index:
        return None
    if isinstance(index, TimedEventIndex):
        events = index.events
        if not events:
            return None
        right = bisect_right(index.start_ts, probe_ts + 1e-3)
        candidates: List[object] = []
        if right > 0:
            candidates.extend(events[max(0, right - 4) : right])
        if right < len(events):
            candidates.extend(events[right : min(len(events), right + 2)])
        if not candidates:
            return None
        earlier = [item for item in candidates if item.ts <= probe_ts + 1e-3]
        if earlier:
            return min(earlier, key=lambda item: abs((item.ts + item.dur) - probe_ts))
        return min(candidates, key=lambda item: abs(item.ts - probe_ts))
    events = list(index)
    if not events:
        return None
    earlier = [item for item in events if item.ts <= probe_ts + 1e-3]
    if earlier:
        return min(earlier, key=lambda item: abs((item.ts + item.dur) - probe_ts))
    return min(events, key=lambda item: abs(item.ts - probe_ts))


def match_cpu_op(kernel: KernelEvent, cpu_ops_by_external_id: Dict[int, TimedEventIndex]) -> Optional[CpuOpEvent]:
    if kernel.external_id is None:
        return None
    return match_timed_event(cpu_ops_by_external_id.get(kernel.external_id, []), kernel.ts)


def match_launch_event(kernel: KernelEvent, launches_by_correlation: Dict[int, TimedEventIndex]) -> Optional[LaunchEvent]:
    if kernel.correlation is None:
        return None
    return match_timed_event(launches_by_correlation.get(kernel.correlation, []), kernel.ts)


# ---------------------------------------------------------------------------
# Frame resolution
# ---------------------------------------------------------------------------

def resolve_active_frames_linear(frames: Sequence[PythonFrame], probe_ts: float) -> List[PythonFrame]:
    active = [item for item in frames if item.ts <= probe_ts <= item.end_ts]
    active.sort(key=lambda item: (item.ts, item.end_ts))
    return active


def find_active_python_frames(cpu_op: CpuOpEvent, python_frames: Dict[Tuple[str, str], List[PythonFrame]]) -> List[PythonFrame]:
    frames = python_frames.get((cpu_op.pid, cpu_op.tid), [])
    if not frames:
        return []
    probe_ts = cpu_op.ts + min(cpu_op.dur * 0.5, 1.0)
    return resolve_active_frames_linear(frames, probe_ts)


def find_active_python_frames_at_ts(*, pid: str, tid: str, ts: float, python_frames: Dict[Tuple[str, str], List[PythonFrame]]) -> List[PythonFrame]:
    frames = python_frames.get((pid, tid), [])
    if not frames:
        return []
    return resolve_active_frames_linear(frames, ts)


def resolve_kernel_site_context(
    kernel: KernelEvent,
    cpu_ops_by_external_id: Dict[int, TimedEventIndex],
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    launches_by_correlation: Dict[int, TimedEventIndex],
    max_stack_depth: int = 0,
) -> Tuple[str, str, str]:
    """Resolve the Python source location, full stack, and CPU op for a kernel.

    Args:
        max_stack_depth: Maximum stack frames to show. 0 = unlimited.
    """
    cpu_op = match_cpu_op(kernel, cpu_ops_by_external_id)
    if cpu_op is not None:
        active_frames = find_active_python_frames(cpu_op, python_frames)
        if active_frames:
            chosen = choose_mapping_frame(active_frames)
            location = chosen.normalized_name if chosen else "unresolved"
            stack = build_stack_display(active_frames, max_depth=max_stack_depth)
            return location, stack, cpu_op.name

    launch_event = match_launch_event(kernel, launches_by_correlation)
    if launch_event is not None:
        active_frames = find_active_python_frames_at_ts(
            pid=launch_event.pid, tid=launch_event.tid, ts=launch_event.ts,
            python_frames=python_frames,
        )
        if active_frames:
            chosen = choose_mapping_frame(active_frames)
            location = chosen.normalized_name if chosen else "unresolved"
            stack = build_stack_display(active_frames, max_depth=max_stack_depth)
            cpu_op_name = cpu_op.name if cpu_op is not None else launch_event.name
            return location, stack, cpu_op_name
        return "unresolved", "", launch_event.name

    cpu_op_name = cpu_op.name if cpu_op is not None else ""
    return "unresolved", "", cpu_op_name


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_trace_data(trace: dict):
    """Extract kernel events, CPU ops, Python frames, and launch events from a trace."""
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
    python_frames: Dict[Tuple[str, str], List[PythonFrame]] = defaultdict(list)

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
                    end_ts=ts + dur,
                    priority=frame_priority(name),
                )
            )

        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if external_id is None and correlation is not None:
            external_id = correlation_external.get(correlation)
        if cat == "cpu_op" and external_id is not None:
            cpu_ops.append(
                CpuOpEvent(name=name, pid=pid, tid=tid, ts=ts, dur=dur, external_id=external_id)
            )
        if is_cuda_launch_event(name, cat) and correlation is not None:
            launches.append(
                LaunchEvent(name=name, pid=pid, tid=tid, ts=ts, dur=dur, correlation=correlation)
            )

        if chosen_pid is None or not is_gpu_kernel_event(event) or pid != chosen_pid:
            continue

        stream = None
        if args:
            raw_stream = args.get("stream") or args.get("cuda_stream")
            if raw_stream is not None:
                try:
                    stream = int(str(raw_stream), 0)
                except (ValueError, TypeError):
                    pass

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

    return kernels, cpu_ops, dict(python_frames), launches, chosen_pid


def filter_kernels(
    kernels: Sequence[KernelEvent],
    cpu_ops: Sequence["CpuOpEvent"],
    kernel_filters: Sequence[str],
) -> List[KernelEvent]:
    """Keep kernels whose canonical name or associated CPU op name contains any
    filter substring.

    Matching is case-insensitive and also **underscore-normalised**: a filter
    like ``reduce_scatter`` will match kernel names containing ``ReduceScatter``
    because both sides are lowercased *and* stripped of underscores before
    comparison.

    This also matches kernels via their CPU op names: if a kernel's GPU name
    doesn't match but its associated CPU op name does, the kernel is included.
    This is important because some ops like ``aten::add_`` appear as CPU op
    names rather than GPU kernel names.
    """
    if not kernel_filters:
        return list(kernels)

    # Direct GPU kernel name match
    direct_matches = set()
    for k in kernels:
        if any(_keyword_matches_text(f, k.canonical_name) for f in kernel_filters):
            direct_matches.add(id(k))

    # CPU-op-name-based match: find CPU ops that match, then find their kernels
    cpu_op_names_matching: Dict[int, bool] = {}  # external_id -> True
    for op in cpu_ops:
        if any(_keyword_matches_text(f, op.name) for f in kernel_filters):
            cpu_op_names_matching[op.external_id] = True

    if cpu_op_names_matching:
        cpu_ops_by_eid = build_cpu_op_index(cpu_ops)
        for k in kernels:
            if id(k) in direct_matches:
                continue
            cpu_op = match_cpu_op(k, cpu_ops_by_eid)
            if cpu_op is not None and cpu_op_names_matching.get(cpu_op.external_id):
                direct_matches.add(id(k))

    return [k for k in kernels if id(k) in direct_matches]


def resolve_stacks(
    kernels: Sequence[KernelEvent],
    cpu_ops: Sequence[CpuOpEvent],
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    launches: Sequence[LaunchEvent],
    max_stack_depth: int = 0,
) -> List[StackInfo]:
    """Resolve Python call stacks for each kernel event."""
    cpu_ops_by_external_id = build_cpu_op_index(cpu_ops)
    launches_by_correlation = build_launch_index(launches)

    results: List[StackInfo] = []

    # Batch-resolve query times for efficiency
    query_times_by_thread: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    query_kernel_map: Dict[Tuple[str, str, float], KernelEvent] = {}

    for kernel in kernels:
        cpu_op = match_cpu_op(kernel, cpu_ops_by_external_id)
        if cpu_op is not None:
            qts = cpu_op.ts + min(cpu_op.dur * 0.5, 1.0)
            query_times_by_thread[(cpu_op.pid, cpu_op.tid)].append(qts)
            query_kernel_map[(cpu_op.pid, cpu_op.tid, qts)] = kernel
        launch_event = match_launch_event(kernel, launches_by_correlation)
        if launch_event is not None:
            qts = launch_event.ts
            if (launch_event.pid, launch_event.tid, qts) not in query_kernel_map:
                query_times_by_thread[(launch_event.pid, launch_event.tid)].append(qts)
                query_kernel_map[(launch_event.pid, launch_event.tid, qts)] = kernel

    # Resolve each kernel individually for full control over stack depth
    for kernel in kernels:
        location, stack, cpu_op_name = resolve_kernel_site_context(
            kernel, cpu_ops_by_external_id, python_frames, launches_by_correlation,
            max_stack_depth=max_stack_depth,
        )
        results.append(
            StackInfo(
                kernel_name=kernel.name,
                canonical_name=kernel.canonical_name,
                category=kernel.category,
                gpu_time_us=kernel.dur,
                location=location,
                stack=stack,
                cpu_op=cpu_op_name,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_stack_table(stacks: Sequence[StackInfo]) -> str:
    """Format stack results as a human-readable table."""
    if not stacks:
        return "No matching kernels found."

    lines = []
    lines.append("=" * 80)
    lines.append("KERNEL STACK TRACE REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Aggregate by (canonical_name, location, stack)
    agg: Dict[Tuple[str, str, str, str], Dict] = {}
    for s in stacks:
        key = (s.canonical_name, s.category, s.location, s.stack)
        if key not in agg:
            agg[key] = {"count": 0, "total_us": 0.0, "cpu_ops": Counter(), "kernel_name": s.kernel_name}
        agg[key]["count"] += 1
        agg[key]["total_us"] += s.gpu_time_us
        if s.cpu_op:
            agg[key]["cpu_ops"][s.cpu_op] += 1

    # Sort by total GPU time descending
    sorted_items = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)

    for idx, (key, info) in enumerate(sorted_items, 1):
        canonical_name, category, location, stack = key
        lines.append(f"--- [{idx}] ---")
        lines.append(f"  Kernel:     {canonical_name}")
        lines.append(f"  Category:   {category}")
        lines.append(f"  GPU time:   {info['total_us'] / 1000:.2f} ms  ({info['count']} launches)")
        top_cpu_op = info["cpu_ops"].most_common(1)
        if top_cpu_op:
            lines.append(f"  CPU op:     {top_cpu_op[0][0]}  (x{top_cpu_op[0][1]})")
        lines.append(f"  Location:   {location}")
        if stack:
            lines.append(f"  Call stack:")
            for frame_idx, frame in enumerate(stack.split(" -> ")):
                indent = "    " + "  " * frame_idx
                lines.append(f"{indent}-> {frame}")
        else:
            lines.append(f"  Call stack: (unresolved)")
        lines.append("")

    return "\n".join(lines)


def format_operator_chain(stacks: Sequence[StackInfo]) -> str:
    """Format results as an operator chain for Pattern 0c of the overlap integration skill."""
    if not stacks:
        return "No matching kernels found."

    lines = []
    lines.append("=== Operator Chain to Replace ===")

    # Group by location, preserve temporal-like ordering by GPU time
    by_location: Dict[str, List[StackInfo]] = defaultdict(list)
    for s in stacks:
        by_location[s.location].append(s)

    # Sort locations by total GPU time
    location_order = sorted(
        by_location.keys(),
        key=lambda loc: sum(s.gpu_time_us for s in by_location[loc]),
        reverse=True,
    )

    total_gpu_us = sum(s.gpu_time_us for s in stacks)

    for location in location_order:
        group = by_location[location]
        total_us = sum(s.gpu_time_us for s in group)
        kernels_seen: Dict[str, int] = {}
        for s in group:
            kernels_seen[s.canonical_name] = kernels_seen.get(s.canonical_name, 0) + 1

        kernel_summary = ", ".join(
            f"{name} (x{count})" for name, count in kernels_seen.items()
        )
        lines.append(f"  {location}")
        lines.append(f"    GPU time: {total_us / 1000:.2f} ms")
        lines.append(f"    Kernels:  {kernel_summary}")
        if group[0].stack:
            lines.append(f"    Stack:    {group[0].stack}")

    lines.append(f"")
    lines.append(f"Total GPU time: {total_gpu_us / 1000:.2f} ms")
    lines.append(f"Semantics: PENDING VALIDATION (see Pattern 0d)")
    return "\n".join(lines)


def list_kernels(
    kernels: Sequence[KernelEvent],
    cpu_ops: Sequence["CpuOpEvent"],
    kernel_filters: Sequence[str],
) -> str:
    """List all GPU kernel names (and matching CPU op names) in the trace.

    When ``kernel_filters`` is non-empty, kernels/CPU ops whose name contains
    any filter substring (case-insensitive) are marked with ``[MATCH]``; all
    others are still listed but unmarked so the user can discover nearby names.

    Output is sorted by total GPU time descending.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("KERNEL NAME DISCOVERY")
    lines.append("=" * 80)
    lines.append("")

    # --- GPU kernels ---
    gpu_agg: Dict[str, Dict] = {}  # canonical_name -> {count, total_us, category, sample_name}
    for k in kernels:
        if k.canonical_name not in gpu_agg:
            gpu_agg[k.canonical_name] = {
                "count": 0, "total_us": 0.0,
                "category": k.category,
                "sample_name": k.name,
            }
        gpu_agg[k.canonical_name]["count"] += 1
        gpu_agg[k.canonical_name]["total_us"] += k.dur

    lowered_filters = [f.lower() for f in kernel_filters] if kernel_filters else []

    def _matches(text: str) -> bool:
        if not lowered_filters:
            return True
        return any(_keyword_matches_text(f, text) for f in kernel_filters)

    gpu_sorted = sorted(gpu_agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    matched_gpu = [(name, info) for name, info in gpu_sorted if _matches(name)]
    unmatched_gpu = [(name, info) for name, info in gpu_sorted if not _matches(name)]

    if kernel_filters:
        lines.append(f"GPU kernels matching filter {list(kernel_filters)}:")
        lines.append("")
        if matched_gpu:
            for name, info in matched_gpu:
                tag = "[MATCH]"
                sample = info["sample_name"]
                display = sample if len(sample) <= 96 else sample[:93] + "..."
                lines.append(
                    f"  {tag} {info['total_us']/1000:10.2f} ms  "
                    f"({info['count']:5d} launches)  [{info['category']}]  {display}"
                )
                # Also show canonical if different from sample
                if info["sample_name"] != name:
                    lines.append(f"       canonical: {name}")
        else:
            lines.append("  (no match)")
        lines.append("")
        lines.append(f"Other GPU kernels ({len(unmatched_gpu)}):")
        lines.append("")
        for name, info in unmatched_gpu:
            sample = info["sample_name"]
            display = sample if len(sample) <= 96 else sample[:93] + "..."
            lines.append(
                f"        {info['total_us']/1000:10.2f} ms  "
                f"({info['count']:5d} launches)  [{info['category']}]  {display}"
            )
    else:
        lines.append(f"All GPU kernels ({len(gpu_sorted)}):")
        lines.append("")
        for name, info in gpu_sorted:
            sample = info["sample_name"]
            display = sample if len(sample) <= 96 else sample[:93] + "..."
            lines.append(
                f"  {info['total_us']/1000:10.2f} ms  "
                f"({info['count']:5d} launches)  [{info['category']}]  {display}"
            )

    # --- CPU ops (only show matches when filters are given) ---
    if kernel_filters:
        cpu_agg: Dict[str, Dict] = {}
        for op in cpu_ops:
            if op.name not in cpu_agg:
                cpu_agg[op.name] = {"count": 0, "total_us": 0.0}
            cpu_agg[op.name]["count"] += 1
            cpu_agg[op.name]["total_us"] += op.dur

        matched_cpu = [(n, i) for n, i in cpu_agg.items() if _matches(n)]
        if matched_cpu:
            matched_cpu.sort(key=lambda x: x[1]["total_us"], reverse=True)
            lines.append("")
            lines.append(f"CPU ops matching filter {list(kernel_filters)}:")
            lines.append("")
            for name, info in matched_cpu:
                lines.append(
                    f"  [MATCH] {info['total_us']/1000:10.2f} ms  "
                    f"({info['count']:5d} launches)  {name}"
                )

    lines.append("")
    lines.append(
        "TIP: Use the matched kernel name substrings with --kernel-filter "
        "to run stack analysis."
    )
    if matched_gpu and kernel_filters:
        # Generate short, distinctive filter suggestions from matched canonical names.
        # For each matched kernel, extract a short keyword that is likely unique.
        suggested = []
        for name, _ in matched_gpu:
            lowered = name.lower()
            # Try each user filter as base, extend to next word boundary
            for f in lowered_filters:
                idx = lowered.find(f)
                if idx != -1:
                    # Extend forward to next non-alnum/underscore char or +20 chars
                    end = idx + len(f)
                    while end < len(name) and (name[end].isalnum() or name[end] == '_'):
                        end += 1
                    end = min(end, idx + len(f) + 25)
                    seg = name[idx:end].rstrip("_(").strip()
                    if seg and len(seg) >= len(f) and seg not in suggested:
                        suggested.append(seg)
                    break
        if suggested:
            lines.append(
                f"Suggested --kernel-filter values: "
                + " ".join(repr(s) for s in suggested[:8])
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace_kernel_stack.py",
        description=(
            "Extract full Python call stacks for specific GPU kernels from a "
            "torch profiler trace. Designed for overlap-kernel integration: "
            "identify which source code launches the kernels you want to overlap."
        ),
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the torch profiler trace file (.json or .json.gz).",
    )
    parser.add_argument(
        "--kernel-filter", nargs="+", default=[],
        help=(
            "One or more case-insensitive substrings. Only kernels whose "
            "canonical name contains any of these substrings will be included. "
            "Omit to include all kernels.  Use --list-kernels first to discover "
            "the actual kernel names in the trace."
        ),
    )
    parser.add_argument(
        "--list-kernels", action="store_true", default=False,
        help=(
            "List all GPU kernel names (and matching CPU op names) found in the "
            "trace, sorted by total GPU time.  When combined with --kernel-filter, "
            "matching entries are marked with [MATCH].  Use this to discover exact "
            "kernel names before running a full stack analysis."
        ),
    )
    parser.add_argument(
        "--stack-depth", type=int, default=0,
        help=(
            "Maximum number of Python stack frames to display per kernel. "
            "0 means unlimited (show the full call stack). Default: 0."
        ),
    )
    parser.add_argument(
        "--format", choices=["table", "chain", "json"], default="table",
        help=(
            "Output format. 'table': human-readable per-kernel stack report; "
            "'chain': operator-chain summary for Pattern 0c; "
            "'json': machine-readable JSON. Default: table."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path. Omit to write to stdout.",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trace_path = Path(args.input)
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}", file=sys.stderr)
        return 1

    print(f"Loading trace: {trace_path} ...", file=sys.stderr)
    trace = load_trace_json(trace_path)
    print(f"Extracting events ...", file=sys.stderr)
    kernels, cpu_ops, python_frames, launches, chosen_pid = extract_trace_data(trace)
    print(f"Found {len(kernels)} GPU kernels on PID {chosen_pid}, "
          f"{len(cpu_ops)} CPU ops, {sum(len(v) for v in python_frames.values())} Python frames",
          file=sys.stderr)

    # --- List-kernels mode: just print kernel names and exit ---
    if args.list_kernels:
        output = list_kernels(kernels, cpu_ops, args.kernel_filter)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
        return 0

    # Filter kernels if requested
    original_count = len(kernels)
    filtered = filter_kernels(kernels, cpu_ops, args.kernel_filter)
    if args.kernel_filter:
        print(f"Kernel filter {args.kernel_filter}: {original_count} -> {len(filtered)} kernels",
              file=sys.stderr)

    if not filtered:
        print("No matching kernels found after filtering.", file=sys.stderr)
        return 0

    # Resolve stacks
    print(f"Resolving Python call stacks (max_depth={args.stack_depth}) ...", file=sys.stderr)
    stacks = resolve_stacks(filtered, cpu_ops, python_frames, launches, max_stack_depth=args.stack_depth)

    # Format output
    if args.format == "table":
        output = format_stack_table(stacks)
    elif args.format == "chain":
        output = format_operator_chain(stacks)
    elif args.format == "json":
        output = json.dumps(
            [{"kernel": s.canonical_name,
              "category": s.category,
              "gpu_time_us": s.gpu_time_us,
              "location": s.location,
              "stack": s.stack.split(" -> ") if s.stack else [],
              "cpu_op": s.cpu_op}
             for s in stacks],
            indent=2,
        )
    else:
        output = format_stack_table(stacks)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())