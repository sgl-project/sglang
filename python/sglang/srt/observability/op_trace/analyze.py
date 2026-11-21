"""Torch/kineto trace + capture-time op maps -> per-step, per-op records.

Graph kernels join to ops through CUPTI's ``graph node id`` (high 32 bits:
instantiated graph id, low 32 bits: node creation index). Eager kernels join
through the ``op::<path>`` spans the op-trace hooks emit, which needs CPU
activity in the trace; without it they stay ``(eager)``.
"""

from __future__ import annotations

import bisect
import difflib
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import msgspec

from sglang.srt.observability.op_trace.capture import EAGER_SPAN_PREFIX, ROOT_PATH
from sglang.srt.observability.op_trace.schema import (
    BucketStat,
    CapturedGraph,
    EagerReference,
    KernelRecord,
    OpMapFile,
    OpStat,
    StepRecord,
    TraceSummary,
)
from sglang.srt.observability.op_trace.symbols import demangle

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
GRAPH_GLUE = "(graph-glue)"
UNMAPPED_GRAPH = "(unmapped-graph)"
EAGER = "(eager)"
UNATTRIBUTED_KINDS = (UNMAPPED_GRAPH, EAGER)
_STEP_SPAN = re.compile(r"^step\[(\w+) bs=(\d+)(?: toks=(\d+))?")
_NODE_INDEX_MASK = (1 << 32) - 1


class _DeviceEvent(msgspec.Struct):
    start_ns: int
    dur_ns: int
    name: str
    stream: int
    graph_id: int
    node_index: int
    grid: tuple[int, int, int]
    correlation: int
    op_path: str = EAGER
    mismatch: bool = False


class _ResolvedGraph(msgspec.Struct, frozen=True):
    op_map: OpMapFile
    graph: CapturedGraph
    mismatches: int
    join: str


class _StepWindow(msgspec.Struct, frozen=True):
    start_ns: int
    end_ns: int
    forward_mode: str
    batch_size: Optional[int]
    num_tokens: Optional[int]
    is_graph: bool


def load_trace(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def load_trace_events(path: str) -> list[dict]:
    return load_trace(path)["traceEvents"]


def load_op_maps(paths: Iterable[str]) -> list[OpMapFile]:
    return [msgspec.json.decode(Path(p).read_bytes(), type=OpMapFile) for p in paths]


def discover_op_maps(trace_path: str) -> list[str]:
    """Op maps next to the trace written by the traced process (graph ids and
    node ids are per process)."""
    pids = _trace_pids(load_trace(trace_path))
    out = []
    for p in sorted(Path(trace_path).parent.glob("op_map-*.json")):
        pid = re.search(r"-pid(\d+)\.json$", p.name)
        if pid is not None and int(pid.group(1)) in pids:
            out.append(str(p))
    return out


def _trace_pids(trace: dict) -> set[int]:
    pids = {trace["opTracePid"]} if "opTracePid" in trace else set()
    for e in trace["traceEvents"]:
        if e.get("cat") in ("cuda_runtime", "cuda_driver", "cpu_op", "user_annotation"):
            pids.add(e["pid"])
    return pids


def analyze_trace(
    trace_path: str,
    op_map_paths: list[str],
    *,
    keep_records: bool = False,
    depth: Optional[int] = None,
    eager_reference: Optional[EagerReference] = None,
    eager_steps: Optional[int] = None,
) -> TraceSummary:
    """``depth`` truncates op kinds (``mlp.gate_up_proj`` -> ``mlp`` at 1), so
    ops a fused kernel moved between parent and child still line up in a diff.
    ``eager_reference`` with ``eager_steps`` splits a GPU-only trace of that
    many repeated eager steps at its largest idle gaps and takes each kernel's
    op from the reference by sequence alignment."""
    trace = load_trace(trace_path)
    events = trace["traceEvents"]
    # Written by the native collector: (exec id, graph handle) in creation order.
    graph_execs = [tuple(e) for e in trace.get("opTraceExecs", [])]
    op_maps = load_op_maps(op_map_paths)
    device = _device_events(events)
    resolved = _resolve_graphs(device=device, op_maps=op_maps, graph_execs=graph_execs)
    launch_at = _launch_times(events)
    _attribute_graph_kernels(device=device, resolved=resolved)
    _attribute_eager_kernels(device=device, events=events, launch_at=launch_at)
    if eager_reference is not None:
        steps, step_of = _ordinal_steps(
            device=device, ref=eager_reference, num_steps=eager_steps
        )
    else:
        steps, step_of = _build_steps(
            device=device, events=events, resolved=resolved, launch_at=launch_at
        )
    kernels = _kernel_records(device=device, step_of=step_of, depth=depth)
    joins = {r.join for r in resolved.values() if r is not None}
    kind_class = {
        _truncate(kind=op_kind_of(scope.path)[0], depth=depth): scope.module_class
        for op_map in op_maps
        for scope in op_map.ops
    }
    buckets, ops = _aggregate(
        steps=steps, kernels=kernels, kind_class=kind_class, joins=joins
    )
    return TraceSummary(
        trace=trace_path,
        op_maps=op_map_paths,
        buckets=buckets,
        ops=ops,
        steps=steps if keep_records else [],
        kernels=kernels if keep_records else [],
    )


def _device_events(events: list[dict]) -> list[_DeviceEvent]:
    out = []
    for e in events:
        if e.get("cat") not in DEVICE_CATEGORIES:
            continue
        args = e["args"]
        node_id = args.get("graph node id", 0)
        grid = args.get("grid", (0, 0, 0))
        out.append(
            _DeviceEvent(
                start_ns=int(e["ts"] * 1000),
                dur_ns=int(e["dur"] * 1000),
                name=e["name"],
                stream=args.get("stream", -1),
                graph_id=args.get("graph id", 0),
                node_index=node_id & _NODE_INDEX_MASK,
                grid=tuple(grid),
                correlation=args.get("correlation", -1),
            )
        )
    out.sort(key=lambda d: d.start_ns)
    return out


# ----------------------------------------------------------------- graph join


def _resolve_graphs(
    *,
    device: list[_DeviceEvent],
    op_maps: list[OpMapFile],
    graph_execs: list[tuple[int, int]],
) -> dict[int, Optional[_ResolvedGraph]]:
    """Exec graph id -> captured graph: by exec creation order when the trace
    carries it (native collector), else by node fingerprint."""
    observed: dict[int, dict[int, _DeviceEvent]] = defaultdict(dict)
    for d in device:
        if d.graph_id:
            observed[d.graph_id].setdefault(d.node_index, d)
    readable = demangle(sym for m in op_maps for sym in m.symbols)
    paired = _pair_execs(op_maps=op_maps, graph_execs=graph_execs)
    out = {}
    for graph_id, nodes in observed.items():
        out[graph_id] = _exact_match(
            pair=paired.get(graph_id), nodes=nodes, readable=readable
        ) or _fingerprint_match(nodes=nodes, op_maps=op_maps, readable=readable)
    return out


def _pair_execs(
    *, op_maps: list[OpMapFile], graph_execs: list[tuple[int, int]]
) -> dict[int, tuple[OpMapFile, CapturedGraph]]:
    """Walk execs in creation order and captures in capture order; an exec
    whose source handle equals the next capture's handle is that capture's
    (handles are reused after torch frees a captured graph, order is not)."""
    paired = {}
    for op_map in op_maps:
        graphs = sorted(op_map.graphs, key=lambda g: g.capture_seq)
        k = 0
        for exec_id, handle in graph_execs:
            if k < len(graphs) and handle == graphs[k].graph_handle:
                paired[exec_id] = (op_map, graphs[k])
                k += 1
    return paired


def _exact_match(*, pair, nodes, readable) -> Optional[_ResolvedGraph]:
    if pair is None:
        return None
    op_map, graph = pair
    mismatches = _fingerprint_mismatches(
        graph=graph, symbols=op_map.symbols, nodes=nodes, readable=readable
    )
    if mismatches is None:
        return None  # the pairing is contradicted by the grids; fall back
    return _ResolvedGraph(
        op_map=op_map, graph=graph, mismatches=mismatches, join="exact"
    )


def _fingerprint_match(*, nodes, op_maps, readable) -> Optional[_ResolvedGraph]:
    candidates = []
    for op_map in op_maps:
        for graph in op_map.graphs:
            mismatches = _fingerprint_mismatches(
                graph=graph, symbols=op_map.symbols, nodes=nodes, readable=readable
            )
            if mismatches is not None:
                candidates.append((mismatches, graph.capture_seq, op_map, graph))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    best = candidates[0]
    tied = [c for c in candidates if c[0] == best[0]]
    join = "fingerprint" if len(tied) == 1 else "fingerprint-ambiguous"
    return _ResolvedGraph(op_map=best[2], graph=best[3], mismatches=best[0], join=join)


def _fingerprint_mismatches(
    *,
    graph: CapturedGraph,
    symbols: list[str],
    nodes: dict[int, _DeviceEvent],
    readable: dict[str, str],
) -> Optional[int]:
    """None when the grids rule the graph out; else the symbol mismatch count."""
    num_nodes = len(graph.node_op)
    mismatches = 0
    for idx, d in nodes.items():
        if idx >= num_nodes:
            return None
        sym = graph.node_symbol[idx]
        if sym < 0:
            continue
        if tuple(graph.node_grid[idx]) != d.grid:
            return None
        if (
            symbol_matches(captured=symbols[sym], observed=d.name, readable=readable)
            is False
        ):
            mismatches += 1
    return mismatches


def symbol_matches(
    *, captured: str, observed: str, readable: dict[str, str]
) -> Optional[bool]:
    """Captured symbols are mangled (cuFuncGetName); kineto's are demangled,
    CUPTI's mangled. Compare exactly after demangling; None when neither side
    can be compared (no demangler and no parsable base name)."""
    if not captured or captured == observed:
        return True
    demangled = readable.get(captured, captured)
    if demangled != captured:
        return demangled == observed
    if not captured.startswith("_Z"):
        return False
    base = _mangled_base_name(captured)
    if base is None:
        return None
    # Without a demangler, require the base name as a whole qualified-name
    # component directly followed by its template or argument list.
    return re.search(rf"(^|[\s:]){re.escape(base)}[<(]", observed) is not None


def _mangled_base_name(name: str) -> Optional[str]:
    """Last <length><identifier> of the leading name of an Itanium mangled
    symbol (``_ZN6sglang14rmsnorm_kernelI...`` -> rmsnorm_kernel), skipping
    cv-qualifiers (``_ZNK``) and the ``St`` (std::) prefix."""
    i = 2
    if name.startswith("_ZN"):
        i = 3
        while i < len(name) and name[i] in "rVKRO":
            i += 1
    if name.startswith("St", i):
        i += 2
    base = None
    while i < len(name) and name[i].isdigit():
        j = i
        while j < len(name) and name[j].isdigit():
            j += 1
        n = int(name[i:j])
        base = name[j : j + n]
        i = j + n
    return base


def _attribute_graph_kernels(
    *, device: list[_DeviceEvent], resolved: dict[int, Optional[_ResolvedGraph]]
) -> None:
    readable = demangle(
        sym for r in resolved.values() if r is not None for sym in r.op_map.symbols
    )
    verdict: dict[tuple[int, int], bool] = {}
    for d in device:
        if not d.graph_id:
            continue
        r = resolved.get(d.graph_id)
        if r is None:
            d.op_path = UNMAPPED_GRAPH
            continue
        key = (d.graph_id, d.node_index)
        if key not in verdict:
            sym = r.graph.node_symbol[d.node_index]
            verdict[key] = sym >= 0 and (
                symbol_matches(
                    captured=r.op_map.symbols[sym], observed=d.name, readable=readable
                )
                is False
            )
        d.mismatch = verdict[key]
        op = r.graph.node_op[d.node_index]
        if op < 0:
            d.op_path = GRAPH_GLUE
        else:
            d.op_path = r.op_map.ops[op].path


# ----------------------------------------------------------------- eager join


def _launch_times(events: list[dict]) -> dict[int, tuple[int, float]]:
    """Correlation id -> (thread, timestamp) of the launching runtime call."""
    return {
        e["args"]["correlation"]: (e["tid"], e["ts"])
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }


def _attribute_eager_kernels(
    *, device: list[_DeviceEvent], events: list[dict], launch_at: dict
) -> None:
    """Kernel -> launching runtime call (correlation) -> innermost op:: span."""
    spans = _op_spans_by_thread(events)
    if not spans:
        return
    for d in device:
        if d.graph_id or d.correlation not in launch_at:
            continue
        tid, ts = launch_at[d.correlation]
        path = _innermost_span(index=spans.get(tid), ts=ts)
        if path is not None:
            d.op_path = path


def _op_spans_by_thread(events: list[dict]) -> dict:
    by_tid = defaultdict(list)
    for e in events:
        # RecordFunctionFast spans are exported as cpu_op, not user_annotation.
        if e.get("cat") in ("cpu_op", "user_annotation") and e["name"].startswith(
            EAGER_SPAN_PREFIX
        ):
            by_tid[e["tid"]].append(
                (e["ts"], e["ts"] + e["dur"], e["name"][len(EAGER_SPAN_PREFIX) :])
            )
    for spans in by_tid.values():
        spans.sort()
    return {tid: ([s[0] for s in spans], spans) for tid, spans in by_tid.items()}


def _innermost_span(*, index, ts: float) -> Optional[str]:
    if index is None:
        return None
    starts, spans = index
    # Spans nest, so the latest-starting span that still covers ts is innermost.
    i = bisect.bisect_right(starts, ts) - 1
    while i >= 0:
        start, end, path = spans[i]
        if end >= ts:
            return path
        i -= 1
    return None


# ----------------------------------------------------------------- steps


def eager_reference_from_trace(trace_path: str) -> EagerReference:
    """Every device event of a CPU+GPU trace of exactly one eager step, in
    order, with the op from the ``op::`` spans (including launches outside
    the model forward, which a GPU-only pass of the same call also records)."""
    events = load_trace_events(trace_path)
    device = _device_events(events)
    _attribute_eager_kernels(
        device=device, events=events, launch_at=_launch_times(events)
    )
    windows = _annotated_step_windows(events)
    w = windows[0] if windows else None
    return EagerReference(
        symbols=[d.name for d in device],
        op_paths=[d.op_path for d in device],
        forward_mode=w.forward_mode if w is not None else "UNKNOWN",
        batch_size=w.batch_size if w is not None else None,
        num_tokens=w.num_tokens if w is not None else None,
    )


def _ordinal_steps(
    *, device: list[_DeviceEvent], ref: EagerReference, num_steps: Optional[int]
) -> tuple[list[StepRecord], list[int]]:
    """Split the GPU-only events into ``num_steps`` steps at the largest idle
    gaps (each step is followed by a host synchronize), then align each step's
    symbol sequence to the reference. Unmatched events inside the step stay
    ``(eager)``; they and reference kernels missing from the step count as
    step mismatches."""
    segments = _split_at_gaps(device=device, num_segments=num_steps or 1)
    step_of = [-1] * len(device)
    steps = []
    for lo, hi in segments:
        names = [d.name for d in device[lo:hi]]
        matcher = difflib.SequenceMatcher(a=ref.symbols, b=names, autojunk=False)
        blocks = [m for m in matcher.get_matching_blocks() if m.size]
        if not blocks:
            continue
        # Unmatched events before the first / after the last match belong to
        # host work between steps (e.g. pool clears), not to this step.
        first = lo + blocks[0].b
        last = lo + blocks[-1].b + blocks[-1].size
        matched = 0
        for block in blocks:
            for k in range(block.size):
                device[lo + block.b + k].op_path = ref.op_paths[block.a + k]
            matched += block.size
        for i in range(first, last):
            step_of[i] = len(steps)
        ks = device[first:last]
        steps.append(
            StepRecord(
                seq=len(steps),
                forward_mode=ref.forward_mode,
                batch_size=ref.batch_size,
                num_tokens=ref.num_tokens,
                is_graph=False,
                start_ns=ks[0].start_ns,
                end_ns=max(k.start_ns + k.dur_ns for k in ks),
                kernel_sum_ns=sum(k.dur_ns for k in ks),
                num_kernels=len(ks),
                mismatches=(len(ks) - matched) + (len(ref.symbols) - matched),
            )
        )
    return steps, step_of


def _split_at_gaps(
    *, device: list[_DeviceEvent], num_segments: int
) -> list[tuple[int, int]]:
    if not device:
        return []
    gaps = sorted(
        range(1, len(device)),
        key=lambda i: (
            device[i].start_ns
            - max(d.start_ns + d.dur_ns for d in device[max(0, i - 8) : i])
        ),
        reverse=True,
    )
    cuts = sorted(gaps[: num_segments - 1])
    bounds = [0, *cuts, len(device)]
    return list(zip(bounds[:-1], bounds[1:]))


def _build_steps(
    *,
    device: list[_DeviceEvent],
    events: list[dict],
    resolved: dict[int, Optional[_ResolvedGraph]],
    launch_at: dict,
) -> tuple[list[StepRecord], list[int]]:
    """Steps come from the CPU step spans (kernels joined by launch time) when
    the trace has CPU activity, else from graph replays on the GPU timeline."""
    windows = _annotated_step_windows(events)
    by_launch = bool(windows)
    if not by_launch:
        windows = _replay_step_windows(device=device, resolved=resolved)
    starts = [w.start_ns for w in windows]
    step_of = []
    members: list[list[_DeviceEvent]] = [[] for _ in windows]
    for d in device:
        if by_launch:
            launch = launch_at.get(d.correlation)
            t = int(launch[1] * 1000) if launch is not None else -1
        else:
            t = d.start_ns
        i = bisect.bisect_right(starts, t) - 1
        if t < 0 or i < 0 or t > windows[i].end_ns:
            step_of.append(-1)
            continue
        step_of.append(i)
        members[i].append(d)
    steps = []
    for seq, (w, ks) in enumerate(zip(windows, members)):
        steps.append(
            StepRecord(
                seq=seq,
                forward_mode=w.forward_mode,
                batch_size=w.batch_size,
                num_tokens=w.num_tokens,
                is_graph=w.is_graph,
                start_ns=min((k.start_ns for k in ks), default=w.start_ns),
                end_ns=max((k.start_ns + k.dur_ns for k in ks), default=w.end_ns),
                kernel_sum_ns=sum(k.dur_ns for k in ks),
                num_kernels=len(ks),
                mismatches=sum(k.mismatch for k in ks),
            )
        )
    return steps, step_of


def _annotated_step_windows(events: list[dict]) -> list[_StepWindow]:
    """CPU-side ModelRunner step spans (needs CPU activity)."""
    windows = []
    for e in events:
        if e.get("cat") != "user_annotation":
            continue
        m = _STEP_SPAN.match(e["name"])
        if m is None:
            continue
        start = int(e["ts"] * 1000)
        windows.append(
            _StepWindow(
                start_ns=start,
                end_ns=start + int(e["dur"] * 1000),
                forward_mode=m.group(1),
                batch_size=int(m.group(2)),
                num_tokens=int(m.group(3)) if m.group(3) else None,
                is_graph=False,
            )
        )
    windows.sort(key=lambda w: w.start_ns)
    return windows


def _replay_step_windows(
    *, device: list[_DeviceEvent], resolved: dict[int, Optional[_ResolvedGraph]]
) -> list[_StepWindow]:
    """One step per graph replay; eager kernels up to the next replay join it."""
    replays = []  # (start_ns, graph_id)
    seen: dict[int, set] = {}
    for d in device:
        if not d.graph_id:
            continue
        nodes = seen.get(d.graph_id)
        if nodes is None or d.node_index in nodes or replays[-1][1] != d.graph_id:
            seen[d.graph_id] = nodes = set()
            replays.append((d.start_ns, d.graph_id))
        nodes.add(d.node_index)
    if not replays:
        end = max((d.start_ns + d.dur_ns for d in device), default=0)
        start = device[0].start_ns if device else 0
        return [
            _StepWindow(
                start_ns=start,
                end_ns=end,
                forward_mode="UNKNOWN",
                batch_size=None,
                num_tokens=None,
                is_graph=False,
            )
        ]
    windows = []
    for i, (start, graph_id) in enumerate(replays):
        r = resolved.get(graph_id)
        windows.append(
            _StepWindow(
                start_ns=start,
                end_ns=replays[i + 1][0] - 1 if i + 1 < len(replays) else 1 << 62,
                forward_mode=r.graph.forward_mode if r is not None else "GRAPH",
                batch_size=r.graph.size if r is not None else None,
                num_tokens=None,
                is_graph=True,
            )
        )
    return windows


def _kernel_records(
    *, device: list[_DeviceEvent], step_of: list[int], depth: Optional[int]
) -> list[KernelRecord]:
    out = []
    for d, step in zip(device, step_of):
        if step < 0:
            continue
        kind, layer = op_kind_of(d.op_path)
        kind = _truncate(kind=kind, depth=depth)
        out.append(
            KernelRecord(
                step_seq=step,
                start_ns=d.start_ns,
                dur_ns=d.dur_ns,
                symbol=d.name,
                op_path=d.op_path,
                op_kind=kind,
                layer=layer,
                stream=d.stream,
            )
        )
    return out


def op_kind_of(path: str) -> tuple[str, Optional[int]]:
    """``model.layers.3.self_attn.qkv_proj`` -> (``self_attn.qkv_proj``, 3)."""
    if path.startswith("(") or path == ROOT_PATH:
        return path, None
    parts = path.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            rest = parts[i + 1 :]
            return (".".join(rest) if rest else parts[i - 1]), int(part)
    return (
        ".".join(parts[1:]) if len(parts) > 1 and parts[0] == "model" else path
    ), None


def _truncate(*, kind: str, depth: Optional[int]) -> str:
    if depth is None or kind.startswith("("):
        return kind
    return ".".join(kind.split(".")[:depth])


# ----------------------------------------------------------------- aggregate


def _bucket_name(step: StepRecord) -> str:
    name = f"{step.forward_mode} {'graph' if step.is_graph else 'step'} bs={step.batch_size}"
    return name + (f" toks={step.num_tokens}" if step.num_tokens else "")


def _aggregate(
    *,
    steps: list[StepRecord],
    kernels: list[KernelRecord],
    kind_class: dict[str, str],
    joins: set[str],
) -> tuple[list[BucketStat], list[OpStat]]:
    bucket_steps: dict[str, list[StepRecord]] = defaultdict(list)
    for s in steps:
        if s.num_kernels:
            bucket_steps[_bucket_name(s)].append(s)
    by_step: dict[int, list[KernelRecord]] = defaultdict(list)
    for k in kernels:
        by_step[k.step_seq].append(k)
    buckets, ops = [], []
    for name, members in bucket_steps.items():
        bucket_kernels = [k for s in members for k in by_step[s.seq]]
        join = ",".join(sorted(joins)) if members[0].is_graph else "eager"
        buckets.append(
            _bucket_stat(
                name=name, members=members, bucket_kernels=bucket_kernels, join=join
            )
        )
        ops.extend(
            _op_stats(
                name=name,
                members=members,
                by_step=by_step,
                bucket_kernels=bucket_kernels,
                kind_class=kind_class,
            )
        )
    readable = demangle(o.top_symbol for o in ops)
    ops = [
        msgspec.structs.replace(o, top_symbol=readable.get(o.top_symbol, o.top_symbol))
        for o in ops
    ]
    return buckets, ops


def _bucket_stat(*, name, members, bucket_kernels, join) -> BucketStat:
    total = sum(k.dur_ns for k in bucket_kernels) or 1
    unattributed = sum(
        k.dur_ns for k in bucket_kernels if k.op_kind in UNATTRIBUTED_KINDS
    )
    return BucketStat(
        bucket=name,
        num_steps=len(members),
        step_kernel_sum_us_p50=statistics.median(s.kernel_sum_ns for s in members)
        / 1e3,
        step_span_us_p50=statistics.median(s.end_ns - s.start_ns for s in members)
        / 1e3,
        attributed_share=1.0 - unattributed / total,
        symbol_mismatches=sum(s.mismatches for s in members),
        graph_join=join,
        step_kernel_sum_samples_us=[s.kernel_sum_ns / 1e3 for s in members],
    )


def _op_stats(*, name, members, by_step, bucket_kernels, kind_class) -> list[OpStat]:
    total = sum(k.dur_ns for k in bucket_kernels) or 1
    per_step: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    per_layer: dict[str, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    symbols: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in members:
        for k in by_step[s.seq]:
            acc = per_step[k.op_kind][s.seq]
            acc[0] += k.dur_ns
            acc[1] += 1
            if k.layer is not None:
                per_layer[k.op_kind][(s.seq, k.layer)] += k.dur_ns
            symbols[k.op_kind][k.symbol] += k.dur_ns
    out = []
    for kind, steps_acc in per_step.items():
        sums = [steps_acc[s.seq][0] if s.seq in steps_acc else 0 for s in members]
        counts = [steps_acc[s.seq][1] if s.seq in steps_acc else 0 for s in members]
        layer_sums = list(per_layer[kind].values())
        layers = {layer for _, layer in per_layer[kind]}
        out.append(
            OpStat(
                bucket=name,
                op_kind=kind,
                module_class=kind_class.get(kind, ""),
                num_layers=len(layers),
                kernels_per_step=sum(counts) / len(members),
                us_per_step_p50=statistics.median(sums) / 1e3,
                share_of_step=sum(sums) / total,
                us_per_layer_p50=(
                    statistics.median(layer_sums) / 1e3 if layer_sums else None
                ),
                top_symbol=max(symbols[kind].items(), key=lambda kv: kv[1])[0],
                step_samples_us=[v / 1e3 for v in sums],
            )
        )
    out.sort(key=lambda o: -o.us_per_step_p50)
    return out
