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
"""package for sglang requests tracing"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from sglang.srt.environ import envs
from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)
opentelemetry_imported = False
opentelemetry_initialized = False
_trace_context_propagator = None
tracer: Optional[trace.Tracer] = None

global_trace_level = get_int_env_var("SGLANG_TRACE_LEVEL", 3)

# Modules allowed to emit spans (from --trace-modules); None means no filtering.
global_trace_modules: Optional[List[str]] = None

TRACE_HEADERS = ["traceparent", "tracestate"]

try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider, id_generator
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _trace_context_propagator = TraceContextTextMapPropagator()

    opentelemetry_imported = True
except ImportError:

    class id_generator:
        class IdGenerator:
            pass

    logger.debug("opentelemetry package is not installed, tracing disabled")


def extract_trace_headers(headers: Mapping[str, str]) -> Optional[Dict]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


def set_global_trace_level(level: int):
    global global_trace_level
    global_trace_level = level


@dataclass
class TraceThreadInfo:
    host_id: str
    pid: int
    thread_label: str
    tp_rank: int
    dp_rank: int
    pp_rank: int


@dataclass
class TraceEvent:
    event_name: str
    ts: int
    attrs: Dict[str, Any]


@dataclass
class TraceSliceContext:
    slice_name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    span: Optional[trace.span.Span] = None
    level: int = 1
    attrs: Optional[Dict[str, Any]] = None
    events: Optional[List[TraceEvent]] = None


@dataclass
class TraceThreadContext:
    thread_info: TraceThreadInfo
    cur_slice_stack: Optional[List[TraceSliceContext]] = None
    thread_span: Optional[trace.span.Span] = None


class TraceCustomIdGenerator(id_generator.IdGenerator):
    """Custom ID generator with support for pre-setting the next span ID.

    Why custom IDs are needed:
      The default IdGenerator may produce duplicate trace IDs across
      multiple TP scheduler processes.

    Preset mechanism (used by async tracing):
      When SGLANG_TRACE_ASYNC=1, span creation is deferred to an exporter
      process while the caller process needs to know span IDs in advance
      for cross-process span linking.  The caller pre-generates a span ID
      and sends it to the exporter.  Before calling tracer.start_span(),
      the exporter calls preset_next_span_id(id) — the next
      generate_span_id() call consumes it, then falls back to random
      generation.  This avoids modifying the standard OTel start_span()
      API while giving the caller full control over span IDs.

      Thread-safety: _preset_local is a threading.local(), so concurrent
      callers in different threads cannot interfere.  The exporter process
      is single-threaded, so no additional locking is needed.
    """

    # Thread-local storage for the next span ID to use preset.
    _preset_local = threading.local()

    def __init__(self):
        super().__init__()
        self.local_random = random.Random()
        self.local_random.seed(time.time())

    def generate_trace_id(self) -> int:
        return self.local_random.getrandbits(64)

    def generate_span_id(self) -> int:
        # If a preset span ID was injected, consume it (one-shot).
        preset = getattr(self._preset_local, "span_id", None)
        if preset is not None:
            self._preset_local.span_id = None
            return preset
        return self.local_random.getrandbits(64)

    @classmethod
    def preset_next_span_id(cls, span_id: int):
        """Inject a pre-generated span ID for the next start_span() call.

        The ID is consumed exactly once by generate_span_id() and then
        cleared.  Call this immediately before tracer.start_span().
        """
        cls._preset_local.span_id = span_id


# global variables
threads_info: Dict[int, TraceThreadInfo] = {}

# Optional callback invoked when a new thread registers its trace info.
# Used by trace_async to forward thread info to the exporter process.
_on_thread_info_set = None

get_cur_time_ns = lambda: int(time.time() * 1e9)
if hasattr(time, "time_ns"):
    get_cur_time_ns = lambda: int(time.time_ns())


def _get_host_id() -> str:
    """
    In distributed tracing systems, obtain a unique node identifier
    and inject it into all subsequently generated spans
    to prevent PID conflicts between threads on different nodes.
    """
    if os.path.exists("/etc/machine-id"):
        try:
            with open("/etc/machine-id", "r") as f:
                return f.read().strip()
        except:
            pass

    mac = uuid.getnode()
    if mac != 0:
        return uuid.UUID(int=mac).hex

    return "unknown"


# Should be called by each tracked process.
def process_tracing_init(
    otlp_endpoint, server_name, trace_modules: Optional[str] = None
):
    global opentelemetry_initialized
    global get_cur_time_ns
    global tracer
    global global_trace_modules

    if trace_modules is not None:
        global_trace_modules = [
            module.strip() for module in trace_modules.split(",") if module.strip()
        ]

    if not opentelemetry_imported:
        opentelemetry_initialized = False
        raise RuntimeError(
            "opentelemetry package is not installed!!! Please not enable tracing or install opentelemetry"
        )

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(
            resource=resource, id_generator=TraceCustomIdGenerator()
        )

        schedule_delay_millis = envs.SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS.get()
        max_export_batch_size = envs.SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE.get()

        processor = BatchSpanProcessor(
            span_exporter=get_otlp_span_exporter(otlp_endpoint),
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
        )
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        opentelemetry_initialized = False
        raise RuntimeError(
            f"initialize opentelemetry error:{e}. Please set correct otlp endpoint."
        )

    opentelemetry_initialized = True
    tracer = trace.get_tracer("sglang server")

    # Auto-start async trace exporter when SGLANG_TRACE_ASYNC=1
    if envs.SGLANG_TRACE_ASYNC.get():
        from sglang.srt.observability.trace_async import start_trace_exporter

        start_trace_exporter(otlp_endpoint, server_name, trace_modules=trace_modules)


def get_global_tracing_enabled():
    return opentelemetry_initialized


def get_otlp_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    supported_protocols = {"grpc", "http/protobuf"}

    if protocol not in supported_protocols:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' configured. "
            f"Supported protocols are: {', '.join(sorted(supported_protocols))}"
        )

    if protocol == "grpc":
        return GRPCSpanExporter(endpoint=endpoint, insecure=True)
    elif protocol == "http/protobuf":
        return HTTPSpanExporter(endpoint=endpoint)


# Should be called by each tracked thread.
def trace_set_thread_info(
    thread_label: str,
    tp_rank: Optional[int] = None,
    dp_rank: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    if not opentelemetry_initialized:
        return

    pid = threading.get_native_id()
    if pid in threads_info:
        return

    threads_info[pid] = TraceThreadInfo(
        host_id=_get_host_id(),
        pid=pid,
        thread_label=thread_label,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
    )

    if _on_thread_info_set is not None:
        _on_thread_info_set(threads_info[pid])


class TraceReqContext:
    def __init__(
        self,
        rid,
        bootstrap_room=None,
        role="unified",
        module_name="",
        external_trace_header: Optional[Dict[str, str]] = None,
        trace_level: Optional[int] = None,
    ):
        self.rid: str = str(rid)
        self.trace_level = (
            trace_level if trace_level is not None else global_trace_level
        )
        self.tracing_enable: bool = opentelemetry_initialized and self.trace_level > 0

        # Filter by --trace-modules only for explicitly named modules; contexts
        # created with the default empty module_name are always traced.
        if (
            module_name
            and global_trace_modules is not None
            and module_name not in global_trace_modules
        ):
            self.tracing_enable = False

        if not self.tracing_enable:
            return

        self.start_time_ns: Optional[int] = None
        self.thread_context: Optional[TraceThreadContext] = None
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.role: str = role
        self.module_name = module_name

        # Indicates whether this instance is a replica from the main process.
        # When True, root_span is None and only root_span_context is preserved.
        self.is_copy: bool = False
        self.root_span: Optional[trace.span.Span] = None
        self.root_span_context: Optional[context.Context] = None
        # Record the most recently completed span as the previous span for the next span to be created.
        self.last_span_context: Optional[trace.span.SpanContext] = None
        self.external_trace_header: Optional[Dict[str, str]] = external_trace_header

        self.events_cache: List[TraceEvent] = []

        self.pid: int = threading.get_native_id()

    def is_tracing_enabled(self) -> bool:
        return self.tracing_enable

    def __create_thread_context(self, ts: int):
        if self.pid not in threads_info:
            trace_set_thread_info("unknown")

        thread_info = threads_info[self.pid]
        thread_context = TraceThreadContext(
            thread_info=thread_info,
            cur_slice_stack=[],
        )

        thread_name = f"{thread_info.thread_label}"
        if thread_info.tp_rank is not None:
            thread_name += f" [TP {thread_info.tp_rank}] "
        if thread_info.pp_rank is not None:
            thread_name += f" [PP {thread_info.pp_rank}] "
        if thread_info.dp_rank is not None:
            thread_name += f" [DP {thread_info.dp_rank}] "
        thread_name += f"(host:{thread_info.host_id[:8]} | pid:{self.pid})"
        thread_context.thread_span = tracer.start_span(
            name=thread_name,
            start_time=ts,
            context=self.root_span_context,
        )

        rank_attrs = {}
        if thread_info.tp_rank is not None:
            rank_attrs["tp_rank"] = thread_info.tp_rank
        if thread_info.pp_rank is not None:
            rank_attrs["pp_rank"] = thread_info.pp_rank
        if thread_info.dp_rank is not None:
            rank_attrs["dp_rank"] = thread_info.dp_rank
        if rank_attrs:
            thread_context.thread_span.set_attributes(rank_attrs)

        thread_context.thread_span.set_attributes(
            {
                "host_id": thread_info.host_id,
                "pid": thread_info.pid,
                "thread_label": thread_info.thread_label,
            }
        )

        return thread_context

    def __getstate__(self) -> Optional[Dict[str, Any]]:
        if not self.tracing_enable:
            return {"tracing_enable": False}

        if not self.root_span_context:
            return {"tracing_enable": False}

        state = {
            "tracing_enable": self.tracing_enable,
            "rid": self.rid,
            "bootstrap_room": self.bootstrap_room,
            "start_time_ns": self.start_time_ns,
            "role": self.role,
            "trace_level": self.trace_level,
            "module_name": self.module_name,
            "is_copy": self.is_copy,
            "pid": self.pid,
            "thread_context": None,
            "root_span": None,
            "last_span_context": None,
        }

        carrier: dict[str, str] = {}
        propagate.inject(carrier, self.root_span_context)
        state["root_span_context"] = carrier

        prev_span_context = self.last_span_context
        if self.thread_context and self.thread_context.cur_slice_stack:
            cur_slice = self.thread_context.cur_slice_stack[0]
            if cur_slice.span:
                prev_span_context = cur_slice.span.get_span_context()

        if prev_span_context:
            state["last_span_context"] = {
                "span_id": prev_span_context.span_id,
                "trace_id": prev_span_context.trace_id,
            }

        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        if not opentelemetry_initialized:
            self.tracing_enable = False
        if not self.tracing_enable:
            return

        self.is_copy = True
        self.pid = threading.get_native_id()
        self.root_span_context = propagate.extract(self.root_span_context)
        if self.last_span_context:
            self.last_span_context = trace.span.SpanContext(
                trace_id=self.last_span_context["trace_id"],
                span_id=self.last_span_context["span_id"],
                is_remote=True,
            )
        self.events_cache = []

    def copy_for_thread(self) -> TraceReqContext:
        """
        Create a copy of this context for use in another thread.

        The copy shares the same root_span_context but has its own thread_context.
        This is useful for propagating trace context across threads (e.g., worker threads).

        Usage:
            # Sender (main thread)
            trace_ctx_copy = trace_ctx.copy_for_thread()
            queue.put(TransferKVChunk(..., trace_ctx=trace_ctx_copy))

            # Receiver (worker thread)
            kv_chunk = queue.get()
            kv_chunk.trace_ctx.rebuild_thread_context()
        """
        # Fast path: not tracing
        if not self.tracing_enable or not self.root_span_context:
            return TraceNullContext()

        # Extract prev_span_context from current thread state
        prev_span_context = self.last_span_context
        if self.thread_context and self.thread_context.cur_slice_stack:
            cur_slice = self.thread_context.cur_slice_stack[0]
            if cur_slice.span:
                prev_span_context = cur_slice.span.get_span_context()

        # Create new instance with shared state
        copied = TraceReqContext.__new__(TraceReqContext)
        copied.tracing_enable = self.tracing_enable
        copied.rid = self.rid
        copied.bootstrap_room = self.bootstrap_room
        copied.start_time_ns = self.start_time_ns
        copied.role = self.role
        copied.trace_level = self.trace_level
        copied.module_name = self.module_name
        copied.is_copy = True  # Mark as copy
        copied.pid = None

        # thread_context is None, will be rebuilt via rebuild_thread_context()
        copied.thread_context = None
        copied.root_span = None

        # Share root_span_context (already a context, no need to serialize)
        copied.root_span_context = self.root_span_context

        # Set prev_span_context for linking spans
        if prev_span_context:
            copied.last_span_context = trace.span.SpanContext(
                trace_id=prev_span_context.trace_id,
                span_id=prev_span_context.span_id,
                is_remote=True,
            )
        else:
            copied.last_span_context = None

        copied.events_cache = []

        return copied

    def rebuild_thread_context(
        self, ts: Optional[int] = None, pid: Optional[int] = None
    ):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()
        self.pid = pid if pid is not None else threading.get_native_id()
        self.thread_context = self.__create_thread_context(ts)

    def trace_req_start(
        self,
        ts: Optional[int] = None,
    ):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()

        # create req context and root span
        self.start_time_ns = ts

        external_trace_context = _trace_context_propagator.extract(
            self.external_trace_header or {}
        )

        # Drop the worker_id added by MultiTokenizer
        orig_rid = self.rid.split("_")[-1]
        role = "" if self.role == "unified" else self.role
        attrs = {"rid": orig_rid, "module": f"sglang::{self.module_name}"}
        if self.bootstrap_room:
            attrs["bootstrap_room"] = str(hex(self.bootstrap_room))
        root_span = tracer.start_span(
            name=f"{role} Req {orig_rid[:8]}",
            start_time=ts,
            context=external_trace_context,
            attributes=attrs,
        )

        self.root_span = root_span
        self.root_span_context = trace.set_span_in_context(root_span)

        # create thread context and thread span
        self.thread_context = self.__create_thread_context(ts)

    def trace_req_finish(
        self, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
    ):
        if not self.tracing_enable:
            return

        if not self.root_span:
            return

        ts = ts or get_cur_time_ns()

        # End all unclosed thread spans.
        self.abort()

        if attrs:
            self.root_span.set_attributes(attrs)

        self.root_span.end(end_time=ts)
        self.root_span = None

    def __check_fast_return(self, level=None):
        if not self.tracing_enable:
            return True

        if not self.thread_context:
            return True

        if level and level > self.trace_level:
            return True

        return False

    def trace_slice_start(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
    ):
        if self.__check_fast_return(level):
            return

        ts = ts or get_cur_time_ns()

        cur_slice = TraceSliceContext(
            slice_name=name,
            start_time_ns=ts,
            level=level,
            attrs={},
            events=[],
        )

        parent_span = self.thread_context.thread_span
        prev_span_context = None
        if not self.thread_context.cur_slice_stack:
            if self.last_span_context:
                prev_span_context = self.last_span_context
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span

        parent_span_context = trace.set_span_in_context(parent_span)

        span = tracer.start_span(
            name=cur_slice.slice_name,
            start_time=cur_slice.start_time_ns,
            context=parent_span_context,
        )
        cur_slice.span = span

        if prev_span_context:
            span.add_link(prev_span_context)

        self.thread_context.cur_slice_stack.append(cur_slice)

    def trace_slice_end(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        thread_finish_flag: bool = False,
    ):
        if self.__check_fast_return(level):
            return

        if not self.thread_context.cur_slice_stack:
            logger.warning(
                f"No matching with the SLICE_START event {name} is required."
            )
            return

        cur_slice = self.thread_context.cur_slice_stack[-1]
        ts = ts or get_cur_time_ns()

        # check if slice_name matching and level matching
        # unlikely path, excepting error API usage
        if cur_slice.slice_name != name or cur_slice.level != level:
            logger.warning(
                f"Slice name mismatch: {name} != {cur_slice.slice_name} or level mismatch: {level} != {cur_slice.level}"
            )
            self.thread_context.cur_slice_stack.pop()
            return

        span = cur_slice.span

        if attrs:
            span.set_attributes(attrs)

        if self.events_cache:
            new_events_cache = []
            for event in self.events_cache:
                if event.ts >= cur_slice.start_time_ns and event.ts < ts:
                    span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)
            self.events_cache = new_events_cache

        span.end(end_time=ts)

        self.thread_context.cur_slice_stack.pop()
        # only for first level slice
        if not self.thread_context.cur_slice_stack:
            self.last_span_context = span.get_span_context()

        if thread_finish_flag:
            self.abort(ts)

    def trace_slice(
        self,
        slice: TraceSliceContext,
        thread_finish_flag: bool = False,
    ):
        if self.__check_fast_return(slice.level):
            return

        parent_span = self.thread_context.thread_span
        prev_span_context = None
        if not self.thread_context.cur_slice_stack:
            if self.last_span_context:
                prev_span_context = self.last_span_context
        else:
            parent_span = self.thread_context.cur_slice_stack[-1].span

        parent_span_context = trace.set_span_in_context(parent_span)

        span = tracer.start_span(
            name=slice.slice_name,
            start_time=slice.start_time_ns,
            context=parent_span_context,
        )

        if prev_span_context:
            span.add_link(prev_span_context)

        if slice.attrs:
            span.set_attributes(slice.attrs)

        if slice.events:
            for event in slice.events:
                span.add_event(
                    name=event.event_name, timestamp=event.ts, attributes=event.attrs
                )

        if self.events_cache:
            new_events_cache = []
            for event in self.events_cache:
                if event.ts >= slice.start_time_ns and event.ts < slice.end_time_ns:
                    span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                else:
                    new_events_cache.append(event)
            self.events_cache = new_events_cache

        span.end(end_time=slice.end_time_ns)

        # only for first level slice
        if not self.thread_context.cur_slice_stack:
            self.last_span_context = span.get_span_context()

        if thread_finish_flag:
            self.abort(slice.end_time_ns)

    # Add event to the current slice on the same thread with the same rid.
    def trace_event(
        self,
        name: str,
        level: int,
        ts: Optional[int] = None,
        attrs: Dict[str, Any] = None,
    ):
        if self.__check_fast_return(level):
            return

        ts = ts or get_cur_time_ns()

        if attrs is None:
            attrs = {}
        self.events_cache.append(TraceEvent(name, ts, attrs))

    def trace_set_root_attrs(self, attrs: Dict[str, Any]):
        if not self.tracing_enable:
            return

        if self.root_span:
            self.root_span.set_attributes(attrs)

    def trace_set_thread_attrs(self, attrs: Dict[str, Any]):
        if self.__check_fast_return():
            return

        if self.thread_context.thread_span:
            self.thread_context.thread_span.set_attributes(attrs)

    def abort(self, ts=None, abort_info: Optional[Dict] = None):
        if self.__check_fast_return():
            return

        # close all slice spans (unlikely, except error API usage)
        ts = ts or get_cur_time_ns()
        while len(self.thread_context.cur_slice_stack) > 0:
            if self.thread_context.cur_slice_stack[-1].span:
                self.thread_context.cur_slice_stack[-1].span.end(end_time=ts)
            self.thread_context.cur_slice_stack.pop()

        # set abort info into thread span
        if self.thread_context.thread_span:
            if abort_info:
                from sglang.srt.managers.schedule_batch import BaseFinishReason

                if isinstance(abort_info, BaseFinishReason):
                    abort_info = abort_info.to_json()
                self.thread_context.thread_span.set_status(Status(StatusCode.ERROR))
                self.thread_context.thread_span.set_attributes(abort_info)

            if self.events_cache:
                for event in self.events_cache:
                    self.thread_context.thread_span.add_event(
                        name=event.event_name,
                        timestamp=event.ts,
                        attributes=event.attrs,
                    )
                self.events_cache = []

            self.thread_context.thread_span.end(end_time=ts)
        self.thread_context = None

    def flush(self):
        pass

    def __del__(self):
        self.abort(abort_info={"reason": "have unclosed span, auto closed"})


@dataclass
class TraceNullContext:
    tracing_enable: bool = False

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_USAGE_CACHED_TOKENS = "gen_ai.usage.cached_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"
