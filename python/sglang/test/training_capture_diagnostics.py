"""Opt-in benchmark pause observations; no serving or GC policy changes."""

import asyncio
import gc
import json
import logging
import math
import os
import re
import socket
import threading
import time


def clock_anchor():
    before = time.perf_counter()
    wall = time.time_ns() / 1e9
    after = time.perf_counter()
    return {
        "perf_counter": (before + after) / 2,
        "unix_seconds": wall,
        "uncertainty_seconds": (after - before) / 2,
    }


class ClientDiagnostics:
    def __init__(self, *, interval=0.01, threshold=0.02, max_events=10000):
        if (
            not math.isfinite(interval)
            or not math.isfinite(threshold)
            or min(interval, threshold, max_events) <= 0
        ):
            raise ValueError("diagnostic bounds must be finite and positive")
        self.interval = interval
        self.threshold = threshold
        self.max_events = max_events
        self.events = []
        self.dropped = 0
        self.pending_gc = {}
        self.anchors = []
        self.loop = None
        self.handle = None
        self.callback = self._gc
        self.active = False

    def start(self):
        if self.active or self.anchors:
            raise RuntimeError("diagnostics can only be started once")
        self.anchors.append(clock_anchor())
        gc.callbacks.append(self.callback)
        self.active = True

    def _append(self, event):
        if len(self.events) < self.max_events:
            self.events.append(event)
        else:
            self.dropped += 1

    def _gc(self, phase, info):
        now = time.perf_counter()
        generation = info["generation"]
        if phase == "start":
            self.pending_gc[generation] = now
        elif phase == "stop":
            start = self.pending_gc.pop(generation, None)
            if start is not None:
                self._append(
                    {
                        "kind": "client_gc",
                        "start": start,
                        "end": now,
                        "generation": generation,
                        "thread_id": threading.get_ident(),
                        "collected": info["collected"],
                    }
                )

    def watch_loop(self):
        if not self.active:
            raise RuntimeError("diagnostics are not active")
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
            self._schedule()
        elif self.loop is not loop:
            raise RuntimeError("request diagnostics require one event loop")

    def _schedule(self):
        deadline = time.perf_counter() + self.interval
        self.handle = self.loop.call_later(self.interval, self._tick, deadline)

    def _tick(self, deadline):
        now = time.perf_counter()
        if now - deadline >= self.threshold:
            self._append(
                {
                    "kind": "client_loop_lag",
                    "start": deadline,
                    "end": now,
                    "thread_id": threading.get_ident(),
                }
            )
        if self.active:
            self._schedule()

    def close(self):
        if not self.active:
            raise RuntimeError("diagnostics are not active")
        self.active = False
        gc.callbacks.remove(self.callback)
        if self.handle is not None:
            self.handle.cancel()
        self.anchors.append(clock_anchor())
        return {
            "schema_version": 1,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "clock_anchors": self.anchors,
            "loop_interval_seconds": self.interval,
            "threshold_seconds": self.threshold,
            "max_events": self.max_events,
            "dropped_events": self.dropped,
            "incomplete_gc": len(self.pending_gc),
            "events": self.events,
        }


def scheduler_gc_events(log):
    """Read the existing single-rank SGLANG_LOG_GC start/end messages."""
    pattern = re.compile(r"GC (start|end): Time ([0-9.]+) \| Generation ([0-9]+)")
    pending = {}
    events = []
    unmatched = 0
    for match in pattern.finditer(log):
        phase, stamp, generation = match.groups()
        stamp, generation = float(stamp), int(generation)
        if phase == "start":
            if generation in pending:
                raise ValueError("overlapping scheduler GC starts; require one rank")
            pending[generation] = stamp
        elif generation in pending:
            start = pending.pop(generation)
            if not math.isfinite(stamp) or stamp < start:
                raise ValueError("invalid scheduler GC clock interval")
            events.append(
                {
                    "kind": "scheduler_gc",
                    "start_unix_seconds": start,
                    "end_unix_seconds": stamp,
                    "generation": generation,
                }
            )
        else:
            unmatched += 1
    return {
        "events": events,
        "unmatched_starts": len(pending),
        "unmatched_ends": unmatched,
    }


def install_tokenizer_gc_observer(threshold):
    """Used only by the diagnostic server entrypoint, in each tokenizer owner."""
    pending = {}
    logger = logging.getLogger(__name__)

    def callback(phase, info):
        now, wall = time.perf_counter(), time.time()
        generation = info["generation"]
        if phase == "start":
            pending[generation] = now, wall
        elif phase == "stop":
            start = pending.pop(generation, None)
            if start is not None and now - start[0] >= threshold:
                logger.warning(
                    "training_capture.tokenizer_gc %s",
                    json.dumps(
                        {
                            "kind": "tokenizer_gc",
                            "pid": os.getpid(),
                            "hostname": socket.gethostname(),
                            "generation": generation,
                            "start_unix_seconds": start[1],
                            "end_unix_seconds": wall,
                            "duration_seconds": now - start[0],
                            "collected": info["collected"],
                        }
                    ),
                )

    gc.callbacks.append(callback)
    logger.warning(
        "training_capture.tokenizer_gc_observer %s",
        json.dumps({"pid": os.getpid(), "threshold_seconds": threshold}),
    )
    return callback


def tokenizer_gc_events(log):
    events = []
    observers = []
    for line in log.splitlines():
        for name, target in (
            ("tokenizer_gc", events),
            ("tokenizer_gc_observer", observers),
        ):
            _, marker, payload = line.partition(f"training_capture.{name} ")
            if marker:
                target.append(json.loads(payload))
    if not observers:
        raise ValueError("tokenizer GC observer did not announce installation")
    if any(event["kind"] != "tokenizer_gc" for event in events):
        raise ValueError("invalid tokenizer GC event")
    return {"observers": observers, "events": events}


def correlate_pauses(records, diagnostics, server_gc, published_trace_ids):
    """Report temporal overlap, never a causal or additive latency estimate."""
    if (
        diagnostics["schema_version"] != 1
        or diagnostics["dropped_events"]
        or diagnostics["incomplete_gc"]
    ):
        raise ValueError("incomplete or unsupported client diagnostics")
    anchors = diagnostics["clock_anchors"]
    if len(anchors) != 2 or anchors[1]["perf_counter"] <= anchors[0]["perf_counter"]:
        raise ValueError("diagnostics need ordered start/end clock anchors")
    if any(
        not math.isfinite(anchor[field])
        for anchor in anchors
        for field in ("perf_counter", "unix_seconds", "uncertainty_seconds")
    ) or any(anchor["uncertainty_seconds"] < 0 for anchor in anchors):
        raise ValueError("invalid clock anchor")
    offsets = [a["unix_seconds"] - a["perf_counter"] for a in anchors]
    error = abs(offsets[1] - offsets[0]) + sum(
        a["uncertainty_seconds"] for a in anchors
    )
    stable = error <= 0.005
    events = list(diagnostics["events"])
    if stable:
        events += [
            dict(
                event,
                start=event["start_unix_seconds"] - offsets[0],
                end=event["end_unix_seconds"] - offsets[0],
            )
            for event in server_gc["events"]
        ]
    threshold = diagnostics["threshold_seconds"]
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid pause threshold")
    long_events = []
    for event in events:
        if (
            not all(math.isfinite(event[key]) for key in ("start", "end"))
            or event["end"] < event["start"]
        ):
            raise ValueError("invalid pause observation")
        if event["end"] - event["start"] >= threshold:
            long_events.append(
                dict(
                    event,
                    event_id=len(long_events),
                    duration_ms=(event["end"] - event["start"]) * 1000,
                    overlapping_requests=[],
                )
            )
    joined = []
    for request in records:
        start = request["start_time"]
        first = start + request["ttft"]
        end = start + request["latency"]
        if not anchors[0]["perf_counter"] <= start < end <= anchors[1]["perf_counter"]:
            raise ValueError("request timing is outside the diagnostic clock anchors")
        overlaps = []
        for event in long_events:
            ttft = max(0, min(first, event["end"]) - max(start, event["start"]))
            after = max(0, min(end, event["end"]) - max(first, event["start"]))
            if ttft or after:
                event["overlapping_requests"].append(request["request_index"])
                overlaps.append(
                    {
                        "event_id": event["event_id"],
                        "ttft_overlap_ms": ttft * 1000,
                        "after_first_token_overlap_ms": after * 1000,
                    }
                )
        joined.append(
            {
                "request_index": request["request_index"],
                "published": request["trace_id"] in published_trace_ids,
                "start_unix_seconds": start + offsets[0] if stable else None,
                "ttft_ms": request["ttft"] * 1000,
                "tpot_ms": (request["latency"] - request["ttft"])
                * 1000
                / (request["output_len"] - 1),
                "overlaps": overlaps,
            }
        )
    return {
        "clock_mapping_stable": stable,
        "clock_offset_change_bound_seconds": error,
        "server_events_mapped": len(server_gc["events"]) if stable else 0,
        "client_event_count": len(diagnostics["events"]),
        "max_ttft_ms": max(row["ttft_ms"] for row in joined),
        "ttft_over_500ms_requests": sum(row["ttft_ms"] > 500 for row in joined),
        "long_events": long_events,
        "worst_ttft": sorted(joined, key=lambda r: r["ttft_ms"], reverse=True)[:10],
        "worst_tpot": sorted(joined, key=lambda r: r["tpot_ms"], reverse=True)[:10],
        "scope": "Client GC uses the request perf_counter clock. Loop lag is a missed timer deadline, not an exact blocking interval. Server GC uses same-host wall time mapped from two client anchors; stability between anchors is not proven. Overlaps are temporal associations, may overlap each other, and must not be summed as causal latency. Detokenizer and OS scheduling pauses are not observed.",
    }
