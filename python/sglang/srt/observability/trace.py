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

from sglang.srt.utils import get_int_env_var

logger = logging.getLogger(__name__)
opentelemetry_imported = False
opentelemetry_initialized = False
_trace_context_propagator = None
tracer: Optional[trace.Tracer] = None

global_trace_level = 3

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
    """
    The default IdGenerator may produce duplicate trace IDs across multiple TP scheduler processes,
    hence a custom IdGenerator is implemented.
    """

    def __init__(self):
        super().__init__()
        self.local_random = random.Random()
        self.local_random.seed(time.time())

    def generate_trace_id(self) -> int:
        return self.local_random.getrandbits(64)

    def generate_span_id(self) -> int:
        return self.local_random.getrandbits(64)


# global variables
threads_info: Dict[int, TraceThreadInfo] = {}

get_cur_time_ns = lambda: int(time.time() * 1e9)
if hasattr(time, "time_ns"):
    get_cur_time_ns = lambda: int(time.time_ns())


def __get_host_id() -> str:
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
def process_tracing_init(otlp_endpoint, server_name):
    global opentelemetry_initialized
    global get_cur_time_ns
    global tracer
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

        schedule_delay_millis = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", 500
        )
        max_export_batch_size = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", 64
        )

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
    thread_label: str, tp_rank: Optional[int] = None, dp_rank: Optional[int] = None
):
    if not opentelemetry_initialized:
        return

    pid = threading.get_native_id()
    if pid in threads_info:
        return

    threads_info[pid] = TraceThreadInfo(
        host_id=__get_host_id(),
        pid=pid,
        thread_label=thread_label,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
    )


class TraceReqContext:
    def __init__(
        self,
        rid,
        bootstrap_room=None,
        role="unified",
        module_name="",
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        self.rid: str = str(rid)
        self.trace_level = global_trace_level
        self.tracing_enable: bool = opentelemetry_initialized and self.trace_level > 0

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
        thread_name += f"(host:{thread_info.host_id[:8]} | pid:{self.pid})"
        thread_context.thread_span = tracer.start_span(
            name=thread_name,
            start_time=ts,
            context=self.root_span_context,
        )

        if thread_info.tp_rank is not None:
            thread_context.thread_span.set_attributes({"tp_rank": thread_info.tp_rank})

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

    def rebuild_thread_context(self, ts: Optional[int] = None):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()
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
