# Copyright 2023-2024 SGLang Team
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
"""Async tracing — decouples OTel span creation from the inference hot path via ZMQ.

When SGLANG_TRACE_ASYNC=1, trace operations are buffered per-request and flushed
to a dedicated exporter process over ZMQ PUSH/PULL.  The exporter replays
operations on real TraceReqContext instances, keeping all OTel overhead out of
scheduler/tokenizer threads.

Architecture overview
---------------------

  Caller process                         Exporter process (daemon)
  ┌─────────────────────────┐            ┌──────────────────────────────┐
  │ TraceReqContextAsync    │   ZMQ      │ TraceReqContext (real OTel)  │
  │  • root span (real)     │──PUSH/PULL─▶  • thread span              │
  │  • pre-generate span IDs│            │  • slice spans (preset IDs)  │
  │  • buffer ops in list   │            │  • export via BatchProcessor │
  └─────────────────────────┘            └──────────────────────────────┘

Key design decisions:

1. Root span in caller process — only one per request, cheap.  Provides
   root_span_context for cross-process propagation (traceparent).

2. Span IDs pre-generated in caller — each trace_slice_start / trace_slice
   call generates a random 64-bit span ID and sends it to the exporter.
   The exporter calls TraceCustomIdGenerator.preset_next_span_id(id) right
   before tracer.start_span() so the span gets the pre-determined ID.
   This makes the exported span tree identical to sync mode.

3. Thread info via callback — trace_set_thread_info() fires a callback that
   pushes {pid, thread_label, ranks} to the exporter once.  The exporter
   stores it in its threads_info dict.  Batch messages only carry caller_pid
   (an int); the exporter looks up the full info from the dict.

4. Cross-process propagation — __getstate__ carries root_span_carrier
   (traceparent) and last_span_context (trace_id + span_id of the most
   recently completed first-level slice).  The destination exporter uses
   last_span_context as a link on its first span, reconstructing the
   execution flow across processes.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import random
import signal
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sglang.srt.environ import envs

if TYPE_CHECKING:
    import zmq as _zmq_type

    from sglang.srt.observability.trace import (
        TraceSliceContext,
        TraceThreadInfo,
    )

logger = logging.getLogger(__name__)

try:
    import zmq

    _zmq_available = True
except ImportError:
    _zmq_available = False
    logger.debug("pyzmq not installed — async tracing unavailable")


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_exporter_process: Optional[_TraceExporterProcess] = None  # type: ignore[name-defined]
_zmq_endpoint: Optional[str] = None
_zmq_context = None
_init_pid: Optional[int] = None
_thread_local = threading.local()
_socket_lock = threading.Lock()
_exporter_lock = threading.Lock()


def _get_zmq_socket() -> Optional[_zmq_type.Socket]:
    """Return a thread-local ZMQ PUSH socket, re-creating after fork."""
    global _zmq_context, _init_pid

    if not _zmq_available or _zmq_endpoint is None:
        return None

    cur_pid = os.getpid()
    if _init_pid != cur_pid:
        with _socket_lock:
            if _init_pid != cur_pid:
                _zmq_context = zmq.Context()
                _init_pid = cur_pid
                if hasattr(_thread_local, "socket"):
                    _thread_local.socket = None

    if not hasattr(_thread_local, "socket") or _thread_local.socket is None:
        from sglang.srt.utils.network import get_zmq_socket

        _thread_local.socket = get_zmq_socket(
            _zmq_context, zmq.PUSH, _zmq_endpoint, bind=False
        )
        _thread_local.socket.setsockopt(zmq.LINGER, 5000)
        _thread_local.socket.setsockopt(zmq.SNDHWM, 10000)

    return _thread_local.socket


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def start_trace_exporter(
    otlp_endpoint: str,
    server_name: str,
    trace_modules: Optional[str] = None,
) -> bool:
    """Start the exporter daemon.  Safe to call multiple times (no-op after first)."""
    global _exporter_process, _zmq_endpoint, _zmq_context, _init_pid

    if not _zmq_available:
        logger.warning("pyzmq not installed — cannot start async trace exporter")
        return False

    with _exporter_lock:
        if _exporter_process is not None and _exporter_process.is_alive():
            return True

        zmq_endpoint = f"ipc:///tmp/sglang_trace_{os.getpid()}.sock"

        _zmq_endpoint = zmq_endpoint
        _zmq_context = zmq.Context()
        _init_pid = os.getpid()

        _exporter_process = _TraceExporterProcess(
            otlp_endpoint=otlp_endpoint,
            server_name=server_name,
            zmq_endpoint=zmq_endpoint,
            trace_modules=trace_modules,
        )
        _exporter_process.start()

    logger.info(
        "Async trace exporter started on %s (pid %d)",
        zmq_endpoint,
        _exporter_process.pid,
    )

    # Register callback so that future trace_set_thread_info() calls in this
    # process automatically forward thread info to the exporter.
    import sglang.srt.observability.trace as _tm

    _tm._on_thread_info_set = _forward_thread_info_to_exporter

    return True


def stop_trace_exporter():
    """Gracefully stop the exporter process."""
    global _exporter_process
    if _exporter_process is None:
        return
    if _exporter_process.is_alive():
        try:
            sock = _get_zmq_socket()
            if sock:
                sock.send_pyobj({"action": "shutdown"}, zmq.NOBLOCK)
            _exporter_process.join(timeout=5)
            if _exporter_process.is_alive():
                _exporter_process.terminate()
                _exporter_process.join(timeout=2)
        except Exception:
            _exporter_process.terminate()
    _exporter_process = None
    import sglang.srt.observability.trace as _tm

    _tm._on_thread_info_set = None


def _forward_thread_info_to_exporter(info: TraceThreadInfo) -> None:
    """Callback for trace_set_thread_info — one-shot push to exporter."""
    sock = _get_zmq_socket()
    if sock is None:
        return
    try:
        sock.send_pyobj(
            {
                "action": "register_thread_info",
                "host_id": info.host_id,
                "pid": info.pid,
                "thread_label": info.thread_label,
                "tp_rank": info.tp_rank,
                "dp_rank": info.dp_rank,
                "pp_rank": info.pp_rank,
            },
            zmq.NOBLOCK,
        )
    except Exception:
        pass


def is_async_tracing_available() -> bool:
    return (
        _zmq_available
        and _exporter_process is not None
        and _exporter_process.is_alive()
    )


def maybe_start_trace_exporter(
    otlp_endpoint: str, server_name: str, trace_modules: Optional[str] = None
):
    """Conditionally start the async exporter based on SGLANG_TRACE_ASYNC env var."""
    if envs.SGLANG_TRACE_ASYNC.get():
        start_trace_exporter(otlp_endpoint, server_name, trace_modules=trace_modules)


# ---------------------------------------------------------------------------
# Exporter process
# ---------------------------------------------------------------------------


class _TraceExporterProcess(multiprocessing.Process):
    """Daemon that receives trace operation batches via ZMQ and exports OTel spans.

    The main loop dispatches two message types:
      • register_thread_info — stores caller thread metadata so thread spans
        carry the correct labels (e.g. "Scheduler [TP 0]").
      • batch — replays a list of trace operations on a real TraceReqContext.
        For slice_start / slice operations that carry a preset_span_id, the
        exporter calls TraceCustomIdGenerator.preset_next_span_id() right
        before tracer.start_span() so the span gets the caller's pre-generated
        ID.  This keeps the span tree identical to sync mode.
    """

    _CLEANUP_INTERVAL = 60
    _CONTEXT_TTL = 300

    def __init__(
        self,
        otlp_endpoint: str,
        server_name: str,
        zmq_endpoint: str,
        trace_modules: Optional[str] = None,
    ):
        super().__init__(daemon=True, name="sglang-trace-exporter")
        self.otlp_endpoint = otlp_endpoint
        self.server_name = server_name
        self.zmq_endpoint = zmq_endpoint
        self.trace_modules = trace_modules

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.environ.pop("SGLANG_TRACE_ASYNC", None)

        from sglang.srt.observability.trace import (
            TraceCustomIdGenerator,
            TraceEvent,
            TraceReqContext,
            TraceSliceContext,
            TraceThreadInfo,
            process_tracing_init,
            threads_info,
        )

        process_tracing_init(self.otlp_endpoint, self.server_name, self.trace_modules)

        from sglang.srt.utils.network import get_zmq_socket

        ctx = zmq.Context()
        socket = get_zmq_socket(ctx, zmq.PULL, self.zmq_endpoint, bind=True)
        socket.setsockopt(zmq.RCVHWM, 10000)

        contexts: Dict[str, tuple] = {}
        last_cleanup = time.perf_counter()

        logger.info("Trace exporter process running")

        try:
            while True:
                try:
                    if socket.poll(timeout=1000):
                        msg = socket.recv_pyobj()
                    else:
                        now = time.perf_counter()
                        if now - last_cleanup > self._CLEANUP_INTERVAL:
                            self._cleanup_stale(contexts, now)
                            last_cleanup = now
                        continue
                except Exception as e:
                    logger.error("Trace exporter recv error: %s", e)
                    continue

                if not isinstance(msg, dict):
                    logger.warning(
                        "Trace exporter received invalid message type: %s", type(msg)
                    )
                    continue

                action = msg.get("action")
                if action == "shutdown":
                    break
                elif action == "register_thread_info":
                    pid = msg["pid"]
                    if pid not in threads_info:
                        threads_info[pid] = TraceThreadInfo(
                            host_id=msg["host_id"],
                            pid=pid,
                            thread_label=msg["thread_label"],
                            tp_rank=msg.get("tp_rank"),
                            dp_rank=msg.get("dp_rank"),
                            pp_rank=msg.get("pp_rank"),
                        )
                elif action == "batch":
                    try:
                        self._replay_batch(
                            msg,
                            contexts,
                            threads_info,
                            TraceCustomIdGenerator,
                            TraceReqContext,
                            TraceSliceContext,
                            TraceEvent,
                        )
                    except Exception as e:
                        logger.error(
                            "Trace exporter replay error for rid=%s: %s",
                            msg.get("rid"),
                            e,
                        )

                now = time.perf_counter()
                if now - last_cleanup > self._CLEANUP_INTERVAL:
                    self._cleanup_stale(contexts, now)
                    last_cleanup = now
        finally:
            for rid, (trace_ctx, _) in contexts.items():
                try:
                    trace_ctx.abort()
                except Exception:
                    pass
            contexts.clear()
            socket.close()
            ctx.term()
            logger.info("Trace exporter process stopped")

    def _replay_batch(
        self,
        msg: Dict[str, Any],
        contexts: Dict[str, tuple],
        threads_info: Dict[int, TraceThreadInfo],
        TraceCustomIdGenerator: type,
        TraceReqContext: type,
        TraceSliceContext: type,
        TraceEvent: type,
    ) -> None:
        rid = msg["rid"]
        context_id = msg["context_id"]
        operations = msg["operations"]
        now = time.perf_counter()

        if context_id in contexts:
            trace_ctx, _ = contexts[context_id]
            contexts[context_id] = (trace_ctx, now)
        else:
            trace_ctx = None

        for op in operations:
            op_type = op["type"]
            try:
                if op_type == "init":
                    # First op for a new rid — create the TraceReqContext.
                    if trace_ctx is not None:
                        continue
                    init_args = op["init_args"]
                    trace_ctx = TraceReqContext(
                        rid=init_args["rid"],
                        bootstrap_room=init_args.get("bootstrap_room"),
                        role=init_args.get("role", "unified"),
                        module_name=init_args.get("module_name", ""),
                        external_trace_header=init_args.get("external_trace_header"),
                        trace_level=init_args.get("trace_level"),
                    )
                    if not trace_ctx.tracing_enable:
                        return
                    contexts[context_id] = (trace_ctx, now)
                    continue

                if trace_ctx is None:
                    return

                if op_type == "req_start":
                    carrier = op.get("root_span_carrier")
                    caller_pid = op.get("caller_pid")

                    if carrier:
                        from opentelemetry import propagate as otel_propagate

                        trace_ctx.root_span_context = otel_propagate.extract(carrier)
                        trace_ctx.start_time_ns = op.get("ts")
                        trace_ctx.is_copy = True
                        trace_ctx.rebuild_thread_context(op.get("ts"), pid=caller_pid)
                    else:
                        trace_ctx.trace_req_start(op.get("ts"))

                elif op_type == "req_finish":
                    # Root span is ended by the caller; we just close thread/slices.
                    trace_ctx.abort(op.get("ts"))
                    contexts.pop(context_id, None)

                elif op_type == "slice_start":
                    preset_id = op.get("preset_span_id")
                    if preset_id is not None:
                        TraceCustomIdGenerator.preset_next_span_id(preset_id)
                    trace_ctx.trace_slice_start(op["name"], op["level"], op.get("ts"))

                elif op_type == "slice_end":
                    trace_ctx.trace_slice_end(
                        op["name"],
                        op["level"],
                        op.get("ts"),
                        op.get("attrs"),
                        op.get("thread_finish_flag", False),
                    )

                elif op_type == "slice":
                    preset_id = op.get("preset_span_id")
                    if preset_id is not None:
                        TraceCustomIdGenerator.preset_next_span_id(preset_id)
                    events = None
                    if op.get("events"):
                        events = [
                            TraceEvent(
                                event_name=e["event_name"],
                                ts=e["ts"],
                                attrs=e.get("attrs", {}),
                            )
                            for e in op["events"]
                        ]
                    slice_ctx = TraceSliceContext(
                        slice_name=op["slice_name"],
                        start_time_ns=op["start_time_ns"],
                        end_time_ns=op.get("end_time_ns"),
                        level=op.get("level", 1),
                        attrs=op.get("attrs"),
                        events=events,
                    )
                    trace_ctx.trace_slice(
                        slice_ctx, op.get("thread_finish_flag", False)
                    )

                elif op_type == "event":
                    trace_ctx.trace_event(
                        op["name"], op["level"], op.get("ts"), op.get("attrs")
                    )

                elif op_type == "set_root_attrs":
                    trace_ctx.trace_set_root_attrs(op["attrs"])

                elif op_type == "set_thread_attrs":
                    trace_ctx.trace_set_thread_attrs(op["attrs"])

                elif op_type == "rebuild_thread_context":
                    caller_pid = op.get("caller_pid")
                    # Deserialized copy carries root_span_carrier here
                    # (no req_start op was sent for this rid in this exporter).
                    carrier = op.get("root_span_carrier")
                    if carrier:
                        from opentelemetry import propagate as otel_propagate

                        trace_ctx.root_span_context = otel_propagate.extract(carrier)
                        trace_ctx.is_copy = True
                    lsc = op.get("init_last_span_context")
                    if lsc:
                        from opentelemetry import trace

                        trace_ctx.last_span_context = trace.span.SpanContext(
                            trace_id=lsc["trace_id"],
                            span_id=lsc["span_id"],
                            is_remote=True,
                        )
                    # Pass caller_pid explicitly so rebuild uses it instead of
                    # the exporter's own native thread id.
                    trace_ctx.rebuild_thread_context(op.get("ts"), pid=caller_pid)

                elif op_type == "abort":
                    trace_ctx.abort(op.get("ts"), op.get("abort_info"))
                    contexts.pop(context_id, None)

            except Exception as e:
                logger.error("Replay op %s for rid %s failed: %s", op_type, rid, e)

    def _cleanup_stale(self, contexts: Dict[str, tuple], now: float) -> None:
        expired = [
            cid for cid, (_, ts) in contexts.items() if now - ts > self._CONTEXT_TTL
        ]
        for cid in expired:
            trace_ctx, _ = contexts.pop(cid)
            try:
                trace_ctx.abort()
            except Exception:
                pass
        if expired:
            logger.debug("Cleaned up %d stale trace contexts", len(expired))


# ---------------------------------------------------------------------------
# Async trace context (drop-in replacement for TraceReqContext)
# ---------------------------------------------------------------------------

import sglang.srt.observability.trace as _trace_mod


class TraceReqContextAsync:
    """Buffers trace operations and flushes them to the exporter process via ZMQ.

    Root span is created in this process (one per request — negligible cost).
    Thread and slice spans are deferred to the exporter.

    Span ID pre-generation:
      Each trace_slice_start / trace_slice pre-generates a random 64-bit span
      ID and records it in the operation dict.  The exporter injects it via
      TraceCustomIdGenerator.preset_next_span_id() before calling
      tracer.start_span(), so the exported span gets the exact same ID.
      This allows the caller to track last_span_context (trace_id + span_id)
      for cross-process propagation without waiting for actual span creation.
    """

    def __init__(
        self,
        rid: str,
        bootstrap_room: Optional[int] = None,
        role: str = "unified",
        module_name: str = "",
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        self.rid: str = str(rid)
        self.trace_level = _trace_mod.global_trace_level
        self.tracing_enable: bool = (
            is_async_tracing_available()
            and _trace_mod.opentelemetry_initialized
            and self.trace_level > 0
        )

        if (
            module_name
            and _trace_mod.global_trace_modules is not None
            and module_name not in _trace_mod.global_trace_modules
        ):
            self.tracing_enable = False

        if not self.tracing_enable:
            return

        self._init_args: Dict[str, Any] = {
            "rid": self.rid,
            "bootstrap_room": bootstrap_room,
            "role": role,
            "module_name": module_name,
            "external_trace_header": external_trace_header,
            "trace_level": self.trace_level,
        }
        self._operations: List[Dict[str, Any]] = []
        self._flush_threshold: int = envs.SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD.get()

        # Root span (created in this process for cross-process propagation)
        self.root_span = None
        self.root_span_context = None
        self._root_span_carrier: Optional[Dict[str, str]] = None

        # Pre-generated span ID tracking.
        # _id_rng generates IDs locally; _span_id_stack mirrors the slice
        # nesting depth so we know when a first-level slice completes.
        self._id_rng = random.Random(time.time_ns() ^ os.getpid() ^ id(self))
        self._span_id_stack: List[int] = []
        self._last_span_id: int = 0

        # Whether this context was deserialized (root span lives elsewhere)
        self.is_copy: bool = False

        # Caller's native thread ID for exporter thread-info lookup
        self._caller_pid: int = threading.get_native_id()

        # Unique ID for this context instance — the exporter uses this (not rid)
        # as its dispatch key, so ops from different threads (copy_for_thread)
        # sharing the same rid are routed to separate TraceReqContext instances.
        # Use a random 64-bit integer to avoid memory address reuse collisions of id(self).
        self._context_id: int = random.getrandbits(64)

        # For deserialized copies: one-shot data to pass to exporter on first rebuild
        self._init_last_span_context: Optional[Dict[str, int]] = None

        # Send init info to exporter once — subsequent flushes only carry ops.
        self._append_op(
            {
                "type": "init",
                "init_args": self._init_args,
            }
        )

    # -- helpers --------------------------------------------------------

    def is_tracing_enabled(self) -> bool:
        return self.tracing_enable

    def _check_fast_return(self, level: Optional[int] = None) -> bool:
        if not self.tracing_enable:
            return True
        if level is not None and level > self.trace_level:
            return True
        return False

    def _gen_span_id(self) -> int:
        return self._id_rng.getrandbits(64)

    # -- buffer / flush --------------------------------------------------

    def _append_op(self, op: Dict[str, Any]) -> None:
        self._operations.append(op)
        if len(self._operations) >= self._flush_threshold:
            self._flush()

    def flush(self) -> None:
        """Proactively flush buffered ops to the exporter.

        Called at natural overlap points (e.g., right before GPU forward)
        so the ZMQ send overlaps with GPU compute.
        """
        self._flush()

    def _flush(self):
        if not self._operations:
            return

        ops = self._operations
        self._operations = []

        sock = _get_zmq_socket()
        if sock is None:
            return

        try:
            sock.send_pyobj(
                {
                    "action": "batch",
                    "rid": self.rid,
                    "context_id": self._context_id,
                    "operations": ops,
                },
                zmq.NOBLOCK,
            )
        except zmq.Again:
            logger.warning(
                "ZMQ send buffer full, dropping %d trace ops for %s",
                len(ops),
                self.rid,
            )
        except Exception as e:
            logger.error("Failed to flush trace for %s: %s", self.rid, e)

    # -- trace interface (mirrors TraceReqContext) -----------------------

    def trace_req_start(self, ts: Optional[int] = None) -> None:
        if not self.tracing_enable:
            return

        from opentelemetry import propagate, trace

        ts = ts or _trace_mod.get_cur_time_ns()

        external_trace_context = _trace_mod._trace_context_propagator.extract(
            self._init_args.get("external_trace_header") or {}
        )

        orig_rid = self.rid.split("_")[-1]
        role = self._init_args.get("role", "unified")
        role_prefix = "" if role == "unified" else role
        attrs = {
            "rid": orig_rid,
            "module": f"sglang::{self._init_args.get('module_name', '')}",
        }
        bootstrap_room = self._init_args.get("bootstrap_room")
        if bootstrap_room:
            attrs["bootstrap_room"] = str(hex(bootstrap_room))

        # Create root span in THIS process — cheap (one per request).
        self.root_span = _trace_mod.tracer.start_span(
            name=f"{role_prefix} Req {orig_rid[:8]}",
            start_time=ts,
            context=external_trace_context,
            attributes=attrs,
        )
        self.root_span_context = trace.set_span_in_context(self.root_span)

        # Serialize root span context so the exporter can parent child spans.
        carrier: Dict[str, str] = {}
        propagate.inject(carrier, self.root_span_context)
        self._root_span_carrier = carrier

        self._append_op(
            {
                "type": "req_start",
                "ts": ts,
                "root_span_carrier": carrier,
                "caller_pid": self._caller_pid,
            }
        )

    def trace_req_finish(
        self, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.tracing_enable:
            return
        ts = ts or _trace_mod.get_cur_time_ns()
        self._append_op({"type": "req_finish", "ts": ts, "attrs": attrs})
        self._flush()

        # End root span in this process.
        if self.root_span:
            if attrs:
                self.root_span.set_attributes(attrs)
            self.root_span.end(end_time=ts)
            self.root_span = None

    def trace_slice_start(
        self, name: str, level: int, ts: Optional[int] = None
    ) -> None:
        if self._check_fast_return(level):
            return
        ts = ts or _trace_mod.get_cur_time_ns()

        # Pre-generate the span ID that the exporter will use.
        span_id = self._gen_span_id()
        self._span_id_stack.append(span_id)

        self._append_op(
            {
                "type": "slice_start",
                "name": name,
                "level": level,
                "ts": ts,
                "preset_span_id": span_id,
            }
        )

    def trace_slice_end(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        thread_finish_flag: bool = False,
    ) -> None:
        if self._check_fast_return(level):
            return
        ts = ts or _trace_mod.get_cur_time_ns()

        # Pop the span ID we generated in the matching trace_slice_start.
        if self._span_id_stack:
            completed_id = self._span_id_stack.pop()
            # First-level slice completed → update last_span_id for linking.
            if not self._span_id_stack:
                self._last_span_id = completed_id

        self._append_op(
            {
                "type": "slice_end",
                "name": name,
                "level": level,
                "ts": ts,
                "attrs": attrs,
                "thread_finish_flag": thread_finish_flag,
            }
        )

    def trace_slice(
        self, slice_ctx: TraceSliceContext, thread_finish_flag: bool = False
    ) -> None:
        if self._check_fast_return(slice_ctx.level):
            return

        # Pre-generate span ID for this complete slice.
        span_id = self._gen_span_id()
        # A complete slice at depth 0 is first-level.
        if not self._span_id_stack:
            self._last_span_id = span_id

        op: Dict[str, Any] = {
            "type": "slice",
            "slice_name": slice_ctx.slice_name,
            "start_time_ns": slice_ctx.start_time_ns,
            "end_time_ns": slice_ctx.end_time_ns,
            "level": slice_ctx.level,
            "attrs": slice_ctx.attrs,
            "preset_span_id": span_id,
            "thread_finish_flag": thread_finish_flag,
        }
        if slice_ctx.events:
            op["events"] = [
                {"event_name": e.event_name, "ts": e.ts, "attrs": e.attrs}
                for e in slice_ctx.events
            ]
        self._append_op(op)

    def trace_event(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._check_fast_return(level):
            return
        ts = ts or _trace_mod.get_cur_time_ns()
        self._append_op(
            {
                "type": "event",
                "name": name,
                "level": level,
                "ts": ts,
                "attrs": attrs,
            }
        )

    def trace_set_root_attrs(self, attrs: Dict[str, Any]):
        if not self.tracing_enable:
            return
        if self.root_span:
            self.root_span.set_attributes(attrs)
        self._append_op({"type": "set_root_attrs", "attrs": attrs})

    def trace_set_thread_attrs(self, attrs: Dict[str, Any]):
        if not self.tracing_enable:
            return
        self._append_op({"type": "set_thread_attrs", "attrs": attrs})

    def rebuild_thread_context(self, ts: Optional[int] = None) -> None:
        if not self.tracing_enable:
            return
        ts = ts or _trace_mod.get_cur_time_ns()
        self._caller_pid = threading.get_native_id()
        op: Dict[str, Any] = {
            "type": "rebuild_thread_context",
            "ts": ts,
            "caller_pid": self._caller_pid,
        }
        # Deserialized copy: exporter never saw a req_start op for this rid,
        # so it needs the carrier to set up root_span_context.
        if self.is_copy and self._root_span_carrier:
            op["root_span_carrier"] = self._root_span_carrier
        if self._init_last_span_context:
            op["init_last_span_context"] = self._init_last_span_context
            self._init_last_span_context = None
        self._append_op(op)

    def abort(
        self, ts: Optional[int] = None, abort_info: Optional[Dict] = None
    ) -> None:
        if not self.tracing_enable:
            return
        ts = ts or _trace_mod.get_cur_time_ns()

        serializable_info = None
        if abort_info is not None:
            try:
                from sglang.srt.managers.schedule_batch import BaseFinishReason

                if isinstance(abort_info, BaseFinishReason):
                    serializable_info = abort_info.to_json()
                elif isinstance(abort_info, dict):
                    serializable_info = abort_info
                else:
                    serializable_info = {"reason": str(abort_info)}
            except ImportError:
                serializable_info = {"reason": str(abort_info)}

        self._append_op({"type": "abort", "ts": ts, "abort_info": serializable_info})
        self._flush()

        if self.root_span and not self.is_copy:
            self.root_span.end(end_time=ts)
            self.root_span = None

    # -- cross-process serialization ------------------------------------

    def copy_for_thread(self) -> TraceReqContextAsync:
        if not self.tracing_enable:
            from sglang.srt.observability.trace import TraceNullContext

            return TraceNullContext()

        copied = TraceReqContextAsync.__new__(TraceReqContextAsync)
        copied.rid = self.rid
        copied.tracing_enable = True
        copied.trace_level = self.trace_level
        copied.is_copy = True
        copied._init_args = self._init_args.copy()
        copied._operations = []
        copied._flush_threshold = self._flush_threshold
        copied.root_span = None
        copied.root_span_context = self.root_span_context
        copied._root_span_carrier = self._root_span_carrier
        copied._id_rng = random.Random(time.time_ns() ^ os.getpid() ^ id(copied))
        copied._span_id_stack = []
        copied._last_span_id = self._last_span_id
        copied._caller_pid = 0
        copied._context_id = random.getrandbits(64)
        copied._init_last_span_context = None
        return copied

    def __getstate__(self) -> Dict[str, Any]:
        if not self.tracing_enable:
            return {"tracing_enable": False}

        self._flush()

        # Determine the span context to propagate for cross-process linking.
        # Prefer the outermost open slice (if any), otherwise the last
        # completed first-level slice.
        last_id = self._last_span_id
        if self._span_id_stack:
            last_id = self._span_id_stack[0]

        state: Dict[str, Any] = {
            "tracing_enable": True,
            "rid": self.rid,
            "init_args": self._init_args,
            "trace_level": self.trace_level,
            "is_async": True,
        }

        if self._root_span_carrier:
            state["root_span_carrier"] = self._root_span_carrier

        # Construct last_span_context from carrier's trace_id + last pre-generated span_id.
        if self._root_span_carrier and last_id:
            tp_parts = self._root_span_carrier.get("traceparent", "").split("-")
            if len(tp_parts) >= 2:
                try:
                    state["last_span_context"] = {
                        "trace_id": int(tp_parts[1], 16),
                        "span_id": last_id,
                    }
                except ValueError:
                    pass

        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.tracing_enable = state.get("tracing_enable", False)
        if not self.tracing_enable:
            return

        if not is_async_tracing_available():
            self.tracing_enable = False
            return

        self.rid = state["rid"]
        self.trace_level = state.get("trace_level", _trace_mod.global_trace_level)
        self._init_args = state["init_args"]
        self._operations = []
        self._flush_threshold = envs.SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD.get()
        self.is_copy = True
        self.root_span = None
        self.root_span_context = None
        self._root_span_carrier = state.get("root_span_carrier")
        self._id_rng = random.Random(time.time_ns() ^ os.getpid() ^ id(self))
        self._span_id_stack = []
        self._caller_pid = 0
        self._context_id = random.getrandbits(64)

        # Restore last_span_id for continued span ID tracking.
        lsc = state.get("last_span_context")
        self._last_span_id = lsc["span_id"] if lsc else 0

        # Pass last_span_context to the exporter so it can set up the link
        # from the destination process's first span to the source's last span.
        self._init_last_span_context = lsc

        # Send init info to exporter for this deserialized context.
        self._append_op(
            {
                "type": "init",
                "init_args": self._init_args,
            }
        )

        # Restore root_span_context for local use.
        if self._root_span_carrier:
            try:
                from opentelemetry import propagate

                self.root_span_context = propagate.extract(self._root_span_carrier)
            except Exception:
                pass

    def __del__(self):
        if not getattr(self, "tracing_enable", False):
            return
        try:
            if hasattr(self, "_operations") and self._operations is not None:
                self._operations.append(
                    {
                        "type": "abort",
                        "ts": _trace_mod.get_cur_time_ns(),
                        "abort_info": {"reason": "have unclosed span, auto closed"},
                    }
                )
                self._flush()
        except BaseException:
            pass
        if getattr(self, "root_span", None) and not getattr(self, "is_copy", False):
            try:
                self.root_span.end()
            except BaseException:
                pass
