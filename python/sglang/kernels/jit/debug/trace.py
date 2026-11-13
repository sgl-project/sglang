"""Record and summarize in-kernel events marked with ``sgl_kernel/debug/trace.cuh``.

Two backends, chosen by ``SGLANG_JIT_DEBUG_TRACE`` when the module is built:

- ``iket`` / ``iket-extended``: NVIDIA IKET (In-Kernel Event Tracing; closed source, ships in
  ``nvidia-cutlass-dsl``, Hopper or newer). This module drives IKET's in-process profiling
  session through the C ABI of ``libiket_injection.so``; no ``run-iket`` wrapper is needed.
  ``%globaltimer`` nanoseconds (32 ns tick), warp lifetimes and locations come from IKET.
- ``naive`` / ``naive-debug``: ``%clock`` cycles written by the kernel itself into a buffer this
  module hands to every traced module (any architecture). Kernels add ``SGL_TRACE_KERNEL()`` and
  ``SGL_TRACE_SYNC()``; cycles are converted to nanoseconds with per-warp anchors. One traced
  launch per capture: a second launch writes the same slots.

Usage is the same for both::

    from sglang.kernels.jit.debug import trace

    op(x)                                   # build and load the module first
    with trace.capture() as cap:
        op(x)
        torch.cuda.synchronize()
    print(cap.trace.summary())
    cap.trace.save_chrome_trace("trace.json")  # open in ui.perfetto.dev

Every event of a warp is recorded once. Check the perturbation: with IKET, the span of
``capture(lifetime_only=True)`` on a build without tracing against the traced span. Do not run
IKET under Nsight Compute / Systems, and capture CUDA graphs before its session starts.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import importlib.util
import json
import os
import pathlib
import tempfile
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple

from sglang.srt.environ import envs

_LIBRARY = "libiket_injection.so"

# iket_profiling_session.h, C ABI v1.0
_FLAG_ENABLE_CLUSTER = 1 << 0
_FLAG_WARP_LIFETIME_ONLY = 1 << 2


class _Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("max_ts_per_warp", ctypes.c_uint64),
        ("device_buffer_size", ctypes.c_uint64),
        ("profiled_cuda_contexts", ctypes.c_void_p),
        ("num_profiled_cuda_contexts", ctypes.c_uint64),
        ("cluster_x", ctypes.c_uint32),
        ("cluster_y", ctypes.c_uint16),
        ("cluster_z", ctypes.c_uint16),
        ("log_level", ctypes.c_uint32),
    ]


assert ctypes.sizeof(_Options) == 56


def find_iket_library(path: Optional[str] = None) -> str:
    """Locate ``libiket_injection.so``: *path*, else the ``iket`` package of nvidia-cutlass-dsl."""
    if path is not None:
        return path
    spec = importlib.util.find_spec("iket")
    if spec is not None and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            candidate = pathlib.Path(root) / "profiler" / _LIBRARY
            if candidate.exists():
                return str(candidate)
    raise FileNotFoundError(
        f"{_LIBRARY} not found: pip install nvidia-cutlass-dsl (IKET ships with it), "
        "or pass the library path explicitly"
    )


class _Session:
    def __init__(self, library: str):
        self._lib = ctypes.CDLL(library)
        self._handle = ctypes.c_void_p()

    def create(self, options: _Options) -> None:
        self._check("Create", ctypes.byref(self._handle))
        self._check("SetOptions", self._handle, ctypes.byref(options))

    def default_options(self) -> _Options:
        options = _Options(struct_size=ctypes.sizeof(_Options))
        self._check("OptionsInit", ctypes.byref(options))
        return options

    def _check(self, name: str, *args) -> None:
        fn = getattr(self._lib, f"IketProfilingSession{name}")
        fn.restype = ctypes.c_uint32
        status = fn(*args)
        if status != 0:
            raise RuntimeError(
                f"IketProfilingSession{name} failed with status {status}"
            )

    def start(self) -> None:
        self._check("Start", self._handle)

    def stop(self) -> None:
        self._check("Stop", self._handle)

    def save(self, path: str) -> None:
        self._check("SaveTrace", self._handle, path.encode())

    def destroy(self) -> None:
        self._check("Destroy", self._handle)


@dataclasses.dataclass
class Capture:
    """Filled when the ``capture`` block exits."""

    trace: Optional[Trace] = None


def capture(
    *,
    backend: Optional[str] = None,
    max_records_per_warp: int = 256,
    cluster: Optional[Tuple[int, int, int]] = None,
    ctas: Tuple[int, int] = (0, 1024),
    lifetime_only: bool = False,
    library: Optional[str] = None,
    save_to: Optional[str] = None,
):
    """Trace the traced kernels launched inside the ``with`` block.

    :param backend: "iket" or "naive"; by default the one ``SGLANG_JIT_DEBUG_TRACE`` names
                    (IKET when it is unset, for ``lifetime_only``).
    :param max_records_per_warp: events one warp may record; extra events are counted, not stored.
    :param cluster: IKET: record only this (x, y, z) cluster (a CTA for non-cluster launches).
    :param ctas: naive: record only linear CTA IDs in [begin, end); sizes the buffer.
    :param lifetime_only: IKET: record only warp entry and exit, leaving kernel bodies untouched;
                          works on builds without tracing and is the baseline for the overhead check.
    :param save_to: IKET: keep its JSON at this path (a temporary file otherwise).
    """
    if backend is None:
        backend = (
            "naive"
            if (envs.SGLANG_JIT_DEBUG_TRACE.get() or "").startswith("naive")
            else "iket"
        )
    if backend == "naive":
        return _capture_naive(max_records_per_warp, ctas)
    if backend != "iket":
        raise ValueError(f"unknown backend {backend!r}")
    return _capture_iket(max_records_per_warp, cluster, lifetime_only, library, save_to)


@contextlib.contextmanager
def _capture_iket(
    max_records_per_warp: int,
    cluster: Optional[Tuple[int, int, int]],
    lifetime_only: bool,
    library: Optional[str],
    save_to: Optional[str],
) -> Generator[Capture, None, None]:
    session = _Session(find_iket_library(library))
    options = session.default_options()
    options.max_ts_per_warp = max_records_per_warp
    if lifetime_only:
        options.flags |= _FLAG_WARP_LIFETIME_ONLY
    if cluster is not None:
        options.flags |= _FLAG_ENABLE_CLUSTER
        options.cluster_x, options.cluster_y, options.cluster_z = cluster
    session.create(options)
    result = Capture()
    session.start()
    try:
        yield result
    finally:
        session.stop()
        path = save_to or tempfile.mkstemp(suffix=".iket.json")[1]
        try:
            session.save(path)
            with open(path) as f:
                result.trace = Trace(json.load(f))
        finally:
            session.destroy()
            if save_to is None:
                os.remove(path)


# Naive backend: modules built with SGLANG_JIT_DEBUG_TRACE=naive register here when they load.
_NAIVE_MODULES: List[Tuple[str, Any]] = []
_KINDS = ("mark", "push", "pop", "start", "end")
_HEADER_BYTES = 32
_MAX_WARPS_PER_CTA = 32


def register_naive_module(name: str, module: Any) -> None:
    _NAIVE_MODULES.append((name, module))


@contextlib.contextmanager
def _capture_naive(
    records_per_warp: int, ctas: Tuple[int, int]
) -> Generator[Capture, None, None]:
    import torch

    if not _NAIVE_MODULES:
        raise RuntimeError(
            "no traced module is loaded: build with SGLANG_JIT_DEBUG_TRACE=naive and launch the "
            "op once before capture()"
        )
    slot_bytes = _HEADER_BYTES + 8 * records_per_warp
    buffers = []
    for name, module in _NAIVE_MODULES:
        size = (ctas[1] - ctas[0]) * _MAX_WARPS_PER_CTA * slot_bytes
        buffer = torch.zeros(size, dtype=torch.uint8, device="cuda")
        module.sgl_trace_set_config(
            buffer.data_ptr(), records_per_warp, ctas[0], ctas[1]
        )
        buffers.append((name, module, buffer))
    result = Capture()
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        decoder = _NaiveDecoder(records_per_warp)
        for name, module, buffer in buffers:
            module.sgl_trace_set_config(0, 0, 0, 0)
            decoder.decode(name, str(module.sgl_trace_events()), buffer.cpu().numpy())
        result.trace = Trace(decoder.data)


class _NaiveDecoder:
    """Turns naive slots into the IKET JSON layout, adding cycle counts and a problem list."""

    def __init__(self, records_per_warp: int):
        self.capacity = records_per_warp
        self.data: Dict[str, Any] = {
            "backend": "naive",
            "launches": [],
            "stringTable": [],
            "locationTable": [],
            "problems": [],
        }
        self._strings: Dict[str, int] = {}

    def _string(self, name: str) -> int:
        if name not in self._strings:
            self._strings[name] = len(self.data["stringTable"])
            self.data["stringTable"].append(name)
        return self._strings[name]

    def decode(self, module_name: str, events: str, raw) -> None:
        import numpy as np

        names = {}
        for entry in filter(None, events.split(";")):
            event_id, _, name = entry.split(":", 2)
            names[int(event_id)] = name
        slots = raw.reshape(-1, _HEADER_BYTES + 8 * self.capacity)
        header = slots[:, :_HEADER_BYTES].view(
            np.dtype(
                [
                    ("count", "<u4"),
                    ("location", "<u4"),
                    ("cta", "<u4"),
                    ("clock_begin", "<u4"),
                    ("ns_begin", "<u8"),
                    ("clock_end", "<u4"),
                    ("ns_end_delta", "<u4"),
                ]
            )
        )[:, 0]
        records = slots[:, _HEADER_BYTES:].view("<u8")
        used = np.nonzero(header["ns_begin"])[0]
        launch = {
            "kernelName": module_name,
            "warpLifetimes": [],
            "ranges": [],
            "markers": [],
        }
        self.data["launches"].append(launch)
        if len(used) == 0:
            return
        cycles = (
            header["clock_end"][used].astype(np.int64) - header["clock_begin"][used]
        ) % (1 << 32)
        spans = header["ns_end_delta"][used].astype(np.int64)
        # cycles per ns over all recording warps: one 32 ns globaltimer tick is too coarse per warp
        ghz = cycles.sum() / max(int(spans.sum()), 1)
        launch["clockGHz"] = float(ghz)
        problems = self.data["problems"]
        tick = int(np.gcd.reduce(spans[spans > 0])) if (spans > 0).any() else 0
        if tick > 64:
            problems.append(
                f"{module_name}: %globaltimer ticks every {tick} ns on this GPU, so the SM clock estimate "
                f"({ghz * 1e3:.0f} MHz) and every ns value are approximate; cycle counts are exact"
            )
        for slot in used:
            h = header[slot]
            loc = len(self.data["locationTable"])
            self.data["locationTable"].append(
                {
                    "smId": int(h["location"] >> 16),
                    "ctaId": [int(h["cta"]), 0, 0],
                    "warpId": int(h["location"] & 0xFFFF),
                }
            )
            begin = int(h["ns_begin"])
            clock0 = int(h["clock_begin"])
            # integer nanoseconds: %globaltimer counts from the epoch, past float precision
            to_ns = lambda c: begin + round(((c - clock0) % (1 << 32)) / ghz)
            launch["warpLifetimes"].append(
                {
                    "locIdx": loc,
                    "startTs": begin,
                    "endTs": begin + int(h["ns_end_delta"]),
                    "cycles": (int(h["clock_end"]) - clock0) % (1 << 32),
                }
            )
            where = (
                f"{module_name} CTA {int(h['cta'])} warp {int(h['location'] & 0xFFFF)}"
            )
            count = int(h["count"])
            if count > self.capacity:
                problems.append(
                    f"{where}: {count - self.capacity} records dropped (raise max_records_per_warp)"
                )
            stack, open_spans = [], {}
            for index, value in enumerate(
                records[slot, : min(count, self.capacity)].tolist()
            ):
                clock, tag = value & 0xFFFFFFFF, value >> 32
                writers, kind, event_id = (
                    tag >> 26,
                    _KINDS[tag & 7] if tag & 7 < 5 else "?",
                    (tag & 0xFFFFFF) >> 3,
                )
                if writers != 1:
                    problems.append(
                        f"{where}: record {index} has {writers} writers (missing SGL_TRACE_SYNC, marks on both "
                        "sides of a divergent branch, or a second launch in this capture)"
                    )
                    continue
                if kind == "mark":
                    launch["markers"].append(
                        {
                            "locIdx": loc,
                            "markerNameIdx": self._string(
                                names.get(event_id, f"#{event_id}")
                            ),
                            "timestamp": to_ns(clock),
                        }
                    )
                elif kind in ("push", "start"):
                    if kind == "push":
                        stack.append((event_id, clock))
                    else:
                        open_spans[event_id] = clock
                elif (
                    kind == "pop" and stack or kind == "end" and event_id in open_spans
                ):
                    event_id, start = (
                        stack.pop()
                        if kind == "pop"
                        else (event_id, open_spans.pop(event_id))
                    )
                    launch["ranges"].append(
                        {
                            "rangeNameIdx": self._string(
                                names.get(event_id, f"#{event_id}")
                            ),
                            "startTs": to_ns(start),
                            "endTs": to_ns(clock),
                            "cycles": (clock - start) % (1 << 32),
                            "warpLocIdxs": [loc, loc],
                        }
                    )
                else:
                    problems.append(
                        f"{where}: record {index} {kind} without a matching start"
                    )
            for event_id, _ in stack + list(open_spans.items()):
                problems.append(
                    f"{where}: {names.get(event_id, event_id)} never closed"
                )


def _pct(sorted_values: List[int], q: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * q))]


class Trace:
    """IKET's decoded-results JSON (``launches``, ``stringTable``, ``locationTable``)."""

    def __init__(self, data: Dict):
        self.data = data
        self.names: List[str] = data.get("stringTable", [])
        self.locations: List[Dict] = data.get("locationTable", [])
        self.launches: List[Dict] = data.get("launches", [])

    def summary(self, kernel: Optional[str] = None) -> str:
        """One block per launch whose mangled kernel name contains *kernel*.

        Per launch: grid, warps, span (first warp entry to last warp exit), warp and CTA lifetimes,
        resident warps per SM averaged over the span, CTAs per SM, and per range the duration
        percentiles and the share of all warp time spent inside it.
        """
        lines = []
        launches = [
            l for l in self.launches if kernel is None or kernel in l["kernelName"]
        ]
        if not launches:
            return "no launch recorded (was the kernel launched inside capture()?)"
        problems = self.data.get("problems", [])
        if problems:
            lines.append(f"PROBLEMS ({len(problems)}; the numbers below may be wrong):")
            lines += [f"  {p}" for p in problems[:20]]
        for launch in launches:
            lines += self._summarize(launch)
        return "\n".join(lines)

    def _summarize(self, launch: Dict) -> List[str]:
        warps = launch["warpLifetimes"]
        lines = [f"kernel {launch['kernelName']}"]
        if "gridDimX" in launch:
            grid = (launch["gridDimX"], launch["gridDimY"], launch["gridDimZ"])
            block = (launch["blockDimX"], launch["blockDimY"], launch["blockDimZ"])
            lines.append(f"  grid {grid} block {block}, {len(warps)} warps recorded")
        else:
            lines.append(
                f"  {len(warps)} warps recorded; cycles converted at {launch.get('clockGHz', 0) * 1e3:.0f} MHz "
                "(mean SM clock of the recorded warps)"
            )
        if not warps:
            return lines
        begin = min(w["startTs"] for w in warps)
        end = max(w["endTs"] for w in warps)
        span = max(end - begin, 1)
        lifetimes = sorted(w["endTs"] - w["startTs"] for w in warps)
        lines.append(
            f"  span {span / 1e3:.2f} us; warp lifetime p10 / p50 / p90 "
            f"{_pct(lifetimes, 0.1):.0f} / {_pct(lifetimes, 0.5):.0f} / {_pct(lifetimes, 0.9):.0f} ns"
        )
        busy_by_sm: Dict[int, int] = defaultdict(int)
        ctas: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        for w in warps:
            loc = self.locations[w["locIdx"]]
            busy_by_sm[loc["smId"]] += w["endTs"] - w["startTs"]
            key = (loc["smId"], tuple(loc["ctaId"]))
            first, last = ctas.get(key, (w["startTs"], w["endTs"]))
            ctas[key] = [min(first, w["startTs"]), max(last, w["endTs"])]
        ctas_by_sm: Dict[int, int] = defaultdict(int)
        for sm, _ in ctas:
            ctas_by_sm[sm] += 1
        cta_lifetimes = sorted(e - s for s, e in ctas.values())
        per_sm = sorted(ctas_by_sm.values())
        lines.append(
            f"  {len(busy_by_sm)} SMs; resident warps per SM over the span {sum(busy_by_sm.values()) / span / len(busy_by_sm):.1f}; "
            f"CTAs per SM p50 {_pct(per_sm, 0.5)}, CTA lifetime p50 / p90 {_pct(cta_lifetimes, 0.5):.0f} / {_pct(cta_lifetimes, 0.9):.0f} ns"
        )
        durations: Dict[str, List[float]] = defaultdict(list)
        cycles: Dict[str, List[int]] = defaultdict(list)
        for r in launch["ranges"]:
            name = self.names[r["rangeNameIdx"]]
            durations[name].append(r["endTs"] - r["startTs"])
            if "cycles" in r:
                cycles[name].append(r["cycles"])
        warp_time = sum(lifetimes)
        for name, values in durations.items():
            values.sort()
            line = (
                f"  range {name:16s} n={len(values):8d}  p50 {_pct(values, 0.5):7.0f}  p90 {_pct(values, 0.9):7.0f}  "
                f"max {values[-1]:7.0f} ns  {sum(values) / warp_time:6.1%} of warp time"
            )
            if cycles[name]:
                c = sorted(cycles[name])
                line += f"  | cycles p10 / p50 / p90 {_pct(c, 0.1)} / {_pct(c, 0.5)} / {_pct(c, 0.9)}"
            lines.append(line)
        markers: Dict[str, int] = defaultdict(int)
        for m in launch["markers"]:
            markers[self.names[m["markerNameIdx"]]] += 1
        for name, count in markers.items():
            lines.append(f"  mark  {name:16s} n={count:8d}")
        return lines

    def save_chrome_trace(self, path: str, max_ctas: int = 64) -> None:
        """Write a Chrome trace (ui.perfetto.dev): one process per SM, one track per warp.

        Only the first *max_ctas* CTAs of each launch are kept; full grids produce files that the
        viewer cannot load.
        """
        events = []
        for index, launch in enumerate(self.launches):
            if not launch["warpLifetimes"]:
                continue
            begin = min(w["startTs"] for w in launch["warpLifetimes"])
            kept_ctas = set()
            kept_locs = set()
            for w in launch["warpLifetimes"]:
                loc = self.locations[w["locIdx"]]
                cta = tuple(loc["ctaId"])
                if cta not in kept_ctas and len(kept_ctas) >= max_ctas:
                    continue
                kept_ctas.add(cta)
                kept_locs.add(w["locIdx"])
                events.append(
                    self._event(
                        f"launch {index} warp", w["startTs"], w["endTs"], begin, loc
                    )
                )
            for r in launch["ranges"]:
                loc_idx = r["warpLocIdxs"][0]
                if loc_idx in kept_locs:
                    events.append(
                        self._event(
                            self.names[r["rangeNameIdx"]],
                            r["startTs"],
                            r["endTs"],
                            begin,
                            self.locations[loc_idx],
                        )
                    )
            for m in launch["markers"]:
                if m["locIdx"] in kept_locs:
                    events.append(
                        self._event(
                            self.names[m["markerNameIdx"]],
                            m["timestamp"],
                            None,
                            begin,
                            self.locations[m["locIdx"]],
                        )
                    )
        with open(path, "w") as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ns"}, f)

    @staticmethod
    def _event(
        name: str, start: int, end: Optional[int], origin: int, loc: Dict
    ) -> Dict:
        event = {
            "name": name,
            "ts": (start - origin) / 1e3,
            "pid": f"SM {loc['smId']}",
            "tid": f"CTA {tuple(loc['ctaId'])} warp {loc['warpId']}",
        }
        if end is None:
            event.update(ph="i", s="t")
        else:
            event.update(ph="X", dur=(end - start) / 1e3)
        return event
