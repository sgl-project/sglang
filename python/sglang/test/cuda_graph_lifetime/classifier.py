# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classify the device pointers a captured CUDA graph bakes in against the
PyTorch caching allocator's allocation lifetimes.

Pure helpers shared by the lifetime tracer: read every kernel/memcpy/memset
operand of a raw CUDA graph through cuda-python, rebuild allocation lifetimes
from the allocator trace, and classify each operand against a graph's capture
window and the pools its graphs own. Nothing here touches production code.

Operand words are read from the recorded launch arguments only: pointers held
in device-side tables are not followed, memory outside the PyTorch allocator
is not tracked, and a buffer that stays allocated but is no longer refreshed by
replay is invisible. Words that match no tracked allocation are counted as
``untracked`` (scalars or non-PyTorch addresses), which is a scope note and not
an inspection failure.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:  # pragma: no cover - the tracer refuses to start without it
    cuda_drv = None

_MARKER_BYTES = 512
_MAX_TRACE_ENTRIES = 2_000_000
_REPORT_FRAMES = 5
_REPORT_LIMIT = 16
# CU_LAUNCH_PARAM_* tags of the ``extra`` launch-argument array.
_LAUNCH_PARAM_END = 0
_LAUNCH_PARAM_BUFFER_POINTER = 1
_LAUNCH_PARAM_BUFFER_SIZE = 2
_CUDA_ERROR_INVALID_VALUE = 1


class InspectionError(RuntimeError):
    """A captured graph could not be inspected completely."""


def _check(result):
    err = result[0]
    if int(err) != 0:
        raise InspectionError(f"CUDA driver call failed: {err}")
    return result[1] if len(result) == 2 else result[1:]


def _kernel_name(func) -> str:
    try:
        return _check(cuda_drv.cuFuncGetName(func)).decode(errors="replace")
    except Exception:
        return f"func@{int(func):#x}"


def _words(raw: bytes) -> Iterator[Tuple[int, int]]:
    for offset in range(0, len(raw) - 7, 8):
        yield offset, int.from_bytes(raw[offset : offset + 8], "little")


def _kernel_node_operands(node) -> Tuple[str, List[Tuple[str, int]]]:
    """Kernel name and every 8-byte word of its launch arguments."""
    params = _check(cuda_drv.cuGraphKernelNodeGetParams(node))
    func = params.func
    if int(func) == 0 and int(params.kern) != 0:
        func = _check(cuda_drv.cuKernelGetFunction(params.kern))
    if int(func) == 0:
        raise InspectionError("kernel node without a function handle")
    words: List[Tuple[str, int]] = []
    kernel_params = int(params.kernelParams)
    if kernel_params:
        index = 0
        while True:
            info = cuda_drv.cuFuncGetParamInfo(func, index)
            if int(info[0]) == _CUDA_ERROR_INVALID_VALUE:
                break  # past the last parameter
            if int(info[0]) != 0:
                raise InspectionError(
                    f"cuFuncGetParamInfo({_kernel_name(func)}, {index}) failed: "
                    f"{info[0]}"
                )
            size = int(info[2])
            arg = ctypes.c_void_p.from_address(kernel_params + 8 * index).value
            if arg and size >= 8:
                for offset, word in _words(ctypes.string_at(arg, size)):
                    words.append((f"arg{index}+{offset}", word))
            index += 1
    elif int(params.extra):
        extra = int(params.extra)
        buffer = size = 0
        for slot in range(16):
            tag = ctypes.c_void_p.from_address(extra + 16 * slot).value or 0
            value = ctypes.c_void_p.from_address(extra + 16 * slot + 8).value or 0
            if tag == _LAUNCH_PARAM_END:
                break
            if tag == _LAUNCH_PARAM_BUFFER_POINTER:
                buffer = value
            elif tag == _LAUNCH_PARAM_BUFFER_SIZE and value:
                size = ctypes.c_size_t.from_address(value).value
        if not (buffer and size):
            raise InspectionError(
                f"kernel {_kernel_name(func)} uses an 'extra' argument buffer "
                "the audit could not locate"
            )
        for offset, word in _words(ctypes.string_at(buffer, size)):
            words.append((f"extra+{offset}", word))
    return _kernel_name(func), words


def _is_device_memory(memory_type) -> bool:
    return memory_type in (
        cuda_drv.CUmemorytype.CU_MEMORYTYPE_DEVICE,
        cuda_drv.CUmemorytype.CU_MEMORYTYPE_UNIFIED,
    )


def _graph_operands(raw_graph: int, operands: Dict[int, List[str]]) -> int:
    """Collect operands recursively and count nodes; reject uninspectable types."""
    _, count = _check(cuda_drv.cuGraphGetNodes(raw_graph, 0))
    nodes, _ = _check(cuda_drv.cuGraphGetNodes(raw_graph, count))
    node_type = cuda_drv.CUgraphNodeType
    total = len(nodes)
    for node in nodes:
        kind = _check(cuda_drv.cuGraphNodeGetType(node))
        if kind == node_type.CU_GRAPH_NODE_TYPE_KERNEL:
            name, words = _kernel_node_operands(node)
            for slot, word in words:
                operands.setdefault(word, []).append(f"{name} {slot}")
        elif kind == node_type.CU_GRAPH_NODE_TYPE_MEMCPY:
            params = _check(cuda_drv.cuGraphMemcpyNodeGetParams(node))
            if _is_device_memory(params.srcMemoryType):
                operands.setdefault(int(params.srcDevice), []).append("memcpy src")
            if _is_device_memory(params.dstMemoryType):
                operands.setdefault(int(params.dstDevice), []).append("memcpy dst")
        elif kind == node_type.CU_GRAPH_NODE_TYPE_MEMSET:
            params = _check(cuda_drv.cuGraphMemsetNodeGetParams(node))
            operands.setdefault(int(params.dst), []).append("memset dst")
        elif kind == node_type.CU_GRAPH_NODE_TYPE_GRAPH:
            child = _check(cuda_drv.cuGraphChildGraphNodeGetGraph(node))
            total += _graph_operands(int(child), operands)
        elif kind in (
            node_type.CU_GRAPH_NODE_TYPE_EMPTY,
            node_type.CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            node_type.CU_GRAPH_NODE_TYPE_EVENT_RECORD,
        ):
            continue
        else:
            raise InspectionError(f"node type {kind} is not inspected")
    return total


def _pool_of(entry: dict) -> Optional[tuple]:
    pool = entry.get("pool_id")
    return None if pool is None else tuple(pool)


class _Lifetimes:
    """Per-block trace intervals: alloc=-1 predates tracing; free=None is live."""

    def __init__(self, traces: List[dict]):
        self.last_alloc: Dict[int, int] = {}
        open_blocks: Dict[int, int] = {}
        lifetimes: Dict[int, List[Tuple[int, Optional[int], int, Optional[tuple]]]] = {}
        for index, entry in enumerate(traces):
            action = entry["action"]
            if action == "alloc":
                open_blocks[entry["addr"]] = index
                self.last_alloc[entry["addr"]] = index
            elif action == "free_requested":
                addr = entry["addr"]
                start = open_blocks.pop(addr, -1)
                lifetimes.setdefault(addr, []).append(
                    (start, index, entry["size"], _pool_of(entry))
                )
        for addr, start in open_blocks.items():
            entry = traces[start]
            lifetimes.setdefault(addr, []).append(
                (start, None, entry["size"], _pool_of(entry))
            )
        self.lifetimes = lifetimes
        self.bases = np.array(sorted(lifetimes), dtype=np.uint64)
        self.max_end = np.array(
            [
                int(base) + max(size for _, _, size, _ in lifetimes[int(base)])
                for base in self.bases
            ],
            dtype=np.uint64,
        )
        self.prefix_max_end = (
            np.maximum.accumulate(self.max_end) if len(self.max_end) else self.max_end
        )

    def covering(
        self, pointer: int
    ) -> List[Tuple[int, int, Optional[int], int, Optional[tuple]]]:
        """Return (base, alloc, free, size, pool) intervals containing pointer."""
        if not len(self.bases):
            return []
        key = np.uint64(pointer)
        last = int(np.searchsorted(self.bases, key, side="right")) - 1
        if last < 0:
            return []
        first = int(np.searchsorted(self.prefix_max_end, key, side="right"))
        if first > last:
            return []
        candidates = np.nonzero(self.max_end[first : last + 1] > key)[0] + first
        out = []
        for position in candidates:
            base = int(self.bases[position])
            for start, end, size, pool in self.lifetimes[base]:
                if base + size > pointer:
                    out.append((base, start, end, size, pool))
        return out


@dataclass
class Finding:
    graph: str
    kind: str
    pointer: int
    base: int
    size: int
    alloc_idx: int
    free_idx: int
    capture_end: int
    slots: List[str]


@dataclass
class Findings:
    graphs: int = 0
    nodes: int = 0
    pointers: int = 0
    static: int = 0
    temporaries: int = 0
    intermediates: int = 0
    outlived: int = 0
    untracked: int = 0
    incomplete: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    dangling: List[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        if self.dangling:
            return "dangling"
        if self.incomplete:
            return "incomplete"
        return "clean"


def _format_frames(entry: Optional[dict]) -> str:
    frames = (entry or {}).get("frames") or []
    shown = []
    for frame in frames[:_REPORT_FRAMES]:
        filename = str(frame.get("filename", "")).split("/")
        shown.append(
            f"{'/'.join(filename[-2:])}:{frame.get('line')} {frame.get('name')}"
        )
    return " <- ".join(shown) if shown else "(no stack recorded)"


def classify_record(
    record, lifetimes: _Lifetimes, findings: Findings, seen: set, unknown_pools: set
) -> None:
    """Classify every operand of one captured graph. ``record`` provides
    ``label``, marker tensors ``forward_begin``/``begin``/``end`` (their alloc
    events bound the windows), ``pools`` (the pools its graphs own) and the
    scanned ``operands``/``nodes``."""
    forward_begin = lifetimes.last_alloc.get(record.forward_begin.data_ptr())
    begin = lifetimes.last_alloc.get(record.begin.data_ptr())
    end = lifetimes.last_alloc.get(record.end.data_ptr())
    if forward_begin is None or begin is None or end is None:
        findings.incomplete.append(
            f"{record.label}: capture markers are no longer in the allocator trace"
        )
        return
    findings.graphs += 1
    findings.nodes += record.nodes
    for pointer, slots in record.operands.items():
        findings.pointers += 1
        overlapping = 0
        for base, start, free, size, pool in lifetimes.covering(pointer):
            if start > end or (free is not None and free < begin):
                continue  # not alive during this graph's capture
            overlapping += 1
            if start >= forward_begin:
                # Allocated during the captured forward: only the pools the
                # captured graphs own hold graph-owned memory.
                if pool is None or record.pools is None:
                    unknown_pools.add((record.label, base))
                    continue
                if pool in record.pools:
                    if start < begin:
                        findings.intermediates += 1
                    elif free is None or free <= end:
                        findings.temporaries += 1
                    else:
                        findings.outlived += 1
                    continue
                kind = (
                    "allocated outside the graph pool during capture"
                    if start >= begin
                    else "eager bridge allocated during the captured forward"
                )
            else:
                kind = "allocated before capture"
            if free is None:
                findings.static += 1
                continue
            key = (record.label, base, start, free)
            if key in seen:
                continue
            seen.add(key)
            findings.dangling.append(
                Finding(
                    record.label, kind, pointer, base, size, start, free, end, slots
                )
            )
        if not overlapping:
            findings.untracked += 1


def _finding_dict(finding: Finding, traces: List[dict]) -> dict:
    alloc = traces[finding.alloc_idx] if finding.alloc_idx >= 0 else None
    free = traces[finding.free_idx]
    return {
        "graph": finding.graph,
        "kind": finding.kind,
        "pointer": f"{finding.pointer:#x}",
        "block": f"{finding.base:#x}",
        "size": finding.size,
        "alloc_trace_index": finding.alloc_idx,
        "free_trace_index": finding.free_idx,
        "freed": "after capture"
        if finding.free_idx > finding.capture_end
        else "during capture",
        "pool": None if alloc is None else alloc.get("pool_id"),
        "operands": len(finding.slots),
        "example_operand": finding.slots[0][:160],
        "allocated_at": _format_frames(alloc),
        "freed_at": _format_frames(free),
    }


def findings_dict(findings: Findings, traces: List[dict]) -> dict:
    return {
        "status": findings.status,
        "graphs": findings.graphs,
        "nodes": findings.nodes,
        "pointers": findings.pointers,
        "static": findings.static,
        "temporaries": findings.temporaries,
        "intermediates": findings.intermediates,
        "outlived": findings.outlived,
        "untracked": findings.untracked,
        "incomplete": list(findings.incomplete),
        "failed": list(findings.failed),
        "dangling": [
            _finding_dict(f, traces) for f in findings.dangling[:_REPORT_LIMIT]
        ],
        "dangling_total": len(findings.dangling),
    }


def format_report(findings: Findings, traces: List[dict]) -> str:
    lines = [
        f"CUDA graph lifetime {findings.status}: {len(findings.dangling)} captured "
        "graph operand(s) point at memory freed while the graph is alive, "
        f"{len(findings.incomplete)} incomplete and {len(findings.failed)} failed "
        f"inspection(s) (graphs={findings.graphs}, nodes={findings.nodes}, "
        f"pointers={findings.pointers}, untracked words={findings.untracked})."
    ]
    for finding in findings.dangling[:_REPORT_LIMIT]:
        d = _finding_dict(finding, traces)
        lines.append(
            f"- graph {finding.graph!r}: {d['pointer']} in block {d['block']} "
            f"({finding.size} B), {finding.kind} (trace #{finding.alloc_idx}, pool "
            f"{d['pool']}), freed {d['freed']} (trace #{finding.free_idx}); read by "
            f"{len(finding.slots)} operand(s), e.g. {d['example_operand']}"
        )
        lines.append(f"    allocated at: {d['allocated_at']}")
        lines.append(f"    freed at: {d['freed_at']}")
    if len(findings.dangling) > _REPORT_LIMIT:
        lines.append(f"- ... {len(findings.dangling) - _REPORT_LIMIT} more")
    for reason in findings.incomplete[:_REPORT_LIMIT]:
        lines.append(f"- incomplete: {reason}")
    for reason in findings.failed[:_REPORT_LIMIT]:
        lines.append(f"- failed: {reason}")
    return "\n".join(lines)
