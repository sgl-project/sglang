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


@dataclass
class TraceThreadInfo:
    host_id: str
    pid: int
    thread_label: str
    tp_rank: int
    dp_rank: int
    tracer: trace.Tracer


@dataclass
class TraceEvent:
    event_name: str
    ts: int
    attrs: Dict[str, Any]


@dataclass
class SGLangTraceSliceContext:
    slice_name: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    span: Optional[trace.span.Span] = None
    # When True, defers slice_name assignment until trace_slice_end()
    anonymous: bool = False
    # For nested slices, if parent slice is anonymous,
    # child slice will be create lazily ultil parent slice_name is assigned.
    lazy_flag: bool = False
    level: int = 1
    attrs: Optional[Dict[str, Any]] = None
    events: Optional[List[TraceEvent]] = None
    parent_slice: Optional[SGLangTraceSliceContext] = None
    child_slices: Optional[List[SGLangTraceSliceContext]] = None
    prev_span_context: Optional[trace.span.SpanContext] = None


@dataclass
class TraceThreadContext:
    thread_info: TraceThreadInfo
    cur_slice: Optional[SGLangTraceSliceContext] = None
    thread_span: Optional[trace.span.Span] = None
    # Record the most recently completed span as the previous span for the next span to be created.
    last_span_context: Optional[trace.span.SpanContext] = None


@dataclass
class TracePropagateContext:
    root_span_context: context.Context
    prev_span_context: Optional[trace.span.SpanContext]

    def to_dict(self):
        carrier: dict[str, str] = {}
        propagate.inject(carrier, self.root_span_context)

        if self.prev_span_context:
            return {
                "root_span": carrier,
                "prev_span": {
                    "span_id": self.prev_span_context.span_id,
                    "trace_id": self.prev_span_context.trace_id,
                },
            }
        else:
            return {"root_span": carrier, "prev_span": None}

    @classmethod
    def instance_from_dict(cls, d):
        if not isinstance(d, dict):
            return None
        if "root_span" not in d or "prev_span" not in d:
            return None

        carrier = d["root_span"]
        root_span_context = propagate.extract(carrier)

        if d["prev_span"] == "None" or d["prev_span"] is None:
            prev_span_context = None
        else:
            prev_span_context = trace.span.SpanContext(
                trace_id=d["prev_span"]["trace_id"],
                span_id=d["prev_span"]["span_id"],
                is_remote=True,
            )

        return cls(root_span_context, prev_span_context)


class SGLangTraceCustomIdGenerator(id_generator.IdGenerator):
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
    if not opentelemetry_imported:
        logger.warning("opentelemetry package is not installed!!! audo disable tracing")
        opentelemetry_initialized = False
        return

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(
            resource=resource, id_generator=SGLangTraceCustomIdGenerator()
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
        logger.error(f": initialize opentelemetry error:{e}")
        logger.warning("pelease set correct otlp endpoint")
        opentelemetry_initialized = False
        return

    opentelemetry_initialized = True


def get_opentelemetry_initialized():
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
        tracer=trace.get_tracer("sglang server"),
    )


class SGLangTraceReqContext:
    def __init__(
        self,
        rid,
        bootstrap_room=None,
        role="null",
        tracing_enable=False,
        trace_level=1,
        module_name="",
    ):
        self.tracing_enable: bool = tracing_enable and opentelemetry_initialized
        self.rid: str = str(rid)
        if not self.tracing_enable:
            return

        self.start_time_ns: Optional[int] = None
        self.thread_context: Optional[TraceThreadContext] = None
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.role: str = role

        self.trace_level = trace_level
        self.module_name = module_name

        # Indicates whether this instance is a replica from the main process.
        # When True, root_span is None and only root_span_context is preserved.
        self.is_copy: bool = False
        self.root_span: Optional[trace.span.Span] = None
        self.root_span_context: Optional[context.Context] = None

        self.pid: int = threading.get_native_id()

    def is_tracing_enabled(self) -> bool:
        return self.tracing_enable

    def __create_thread_context(self, ts: int):
        if self.pid not in threads_info:
            trace_set_thread_info("unknown")

        thread_info = threads_info[self.pid]
        thread_context = TraceThreadContext(
            thread_info=thread_info,
        )

        thread_name = f"{thread_info.thread_label}"
        if thread_info.tp_rank is not None:
            thread_name += f" [TP {thread_info.tp_rank}] "
        thread_name += f"(host:{thread_info.host_id[:8]} | pid:{self.pid})"
        thread_context.thread_span = thread_context.thread_info.tracer.start_span(
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

    def trace_get_proc_propagate_context(self) -> Optional[Dict[str, Any]]:
        if not self.tracing_enable:
            return None

        if not self.root_span_context:
            return None

        prev_span_context = None
        if self.thread_context.cur_slice:
            cur_slice = self.thread_context.cur_slice
            if cur_slice.span:
                prev_span_context = cur_slice.span.get_span_context()

        if not prev_span_context:
            # may be None
            prev_span_context = self.thread_context.last_span_context

        root_span_context = self.root_span_context

        trace_context = TracePropagateContext(root_span_context, prev_span_context)
        return trace_context.to_dict()

    def trace_set_proc_propagate_context(self, trace_context: Optional[Dict[str, Any]]):
        if not self.tracing_enable:
            return

        self.start_time_ns = get_cur_time_ns()
        self.is_copy = True

        trace_context = TracePropagateContext.instance_from_dict(trace_context)
        if not trace_context:
            self.tracing_enable = False
            return
        else:
            self.root_span_context = trace_context.root_span_context

        self.thread_context = self.__create_thread_context(self.start_time_ns)
        self.thread_context.last_span_context = trace_context.prev_span_context

    def trace_req_start(
        self,
        ts: Optional[int] = None,
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()

        # create req context and root span
        self.start_time_ns = ts

        tracer = threads_info[self.pid].tracer
        external_trace_context = _trace_context_propagator.extract(
            external_trace_header
        )

        # Drop the worker_id added by MultiTokenizer
        orig_rid = self.rid.split("_")[-1]
        role = "" if self.role == "null" else self.role
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

        ts = ts or get_cur_time_ns()

        # End all unclosed thread spans.
        self.abort()

        if attrs:
            self.root_span.set_attributes(attrs)

        self.root_span.end(end_time=ts)

    def __create_slice_span(self, _slice: SGLangTraceSliceContext):
        if _slice.span:
            return

        parent_span = self.thread_context.thread_span
        if _slice.parent_slice:
            parent_span = _slice.parent_slice.span

        parent_span_context = trace.set_span_in_context(parent_span)
        span = self.thread_context.thread_info.tracer.start_span(
            name=_slice.slice_name,
            start_time=_slice.start_time_ns,
            context=parent_span_context,
        )

        if _slice.prev_span_context:
            span.add_link(_slice.prev_span_context)

        _slice.span = span

        if _slice.attrs:
            span.set_attributes(_slice.attrs)
        if _slice.events:
            for event in _slice.events:
                span.add_event(
                    name=event.event_name,
                    timestamp=event.ts,
                    attributes=event.attrs,
                )
        _slice.lazy_flag = False
        _slice.anonymous = False
        _slice.attrs = {}
        _slice.events = []

    def __end_slice_span(self, _slice: SGLangTraceSliceContext):
        # child_slices is not empty but they have not created span
        # if cur_slice.lazy_flag is True before.
        if _slice.child_slices:
            for child_slice in _slice.child_slices:
                self.__create_slice_span(child_slice)
                self.__end_slice_span(child_slice)
            _slice.child_slices = []

        _slice.span.end(end_time=_slice.end_time)
        _slice.parent_slice = None

    def trace_slice_start(
        self,
        name: str,
        ts: Optional[int] = None,
        anonymous: bool = False,
        level: int = 1,
    ):
        if not self.tracing_enable:
            return

        if not self.thread_context:
            return

        ts = ts or get_cur_time_ns()

        cur_slice = SGLangTraceSliceContext(
            slice_name=name,
            start_time_ns=ts,
            anonymous=anonymous,
            level=level,
            attrs={},
            events=[],
            parent_slice=self.thread_context.cur_slice,
            child_slices=[],
        )
        if self.thread_context.cur_slice:
            self.thread_context.cur_slice.child_slices.append(cur_slice)
        self.thread_context.cur_slice = cur_slice

        if level > self.trace_level:
            cur_slice.lazy_flag = True
            return

        # find prev span, only first level slice has previous span
        if not cur_slice.parent_slice:
            if self.thread_context.last_span_context:
                cur_slice.prev_span_context = self.thread_context.last_span_context

        # check if span creation is lazy
        if anonymous or (cur_slice.parent_slice and cur_slice.parent_slice.lazy_flag):
            cur_slice.lazy_flag = True
            return

        self.__create_slice_span(cur_slice)

    def __release_slice_reference_tree(self, _slice: SGLangTraceSliceContext):
        for child_slice in _slice.child_slices:
            self.__release_slice_reference_tree(child_slice)
        _slice.child_slices = []
        _slice.parent_slice = None

    def __trace_slice_end_flag_process(self, auto_next_anon, thread_finish_flag, ts):
        # If this is the last slice in the thread,
        # release the thread context and check whether to release the request context.
        if thread_finish_flag:
            self.abort(ts)
            return

        if auto_next_anon:
            self.trace_slice_start("", ts=ts, anonymous=True)

    def trace_slice_end(
        self,
        name: str,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        auto_next_anon: bool = False,
        thread_finish_flag: bool = False,
        level: int = 1,
    ):
        if not self.tracing_enable:
            return

        if not self.thread_context:
            return

        if not self.thread_context.cur_slice:
            logger.warning(
                f"No matching with the SLICE_START event {name} is required."
            )
            return

        cur_slice = self.thread_context.cur_slice
        ts = ts or get_cur_time_ns()

        if level > self.trace_level:
            # release obj loop references to avoid GC block
            self.thread_context.cur_slice = cur_slice.parent_slice
            if cur_slice.parent_slice:
                cur_slice.parent_slice.child_slices.remove(cur_slice)
            self.__release_slice_reference_tree(cur_slice)
            self.__trace_slice_end_flag_process(auto_next_anon, thread_finish_flag, ts)
            return

        # check if slice_name matching and level matching
        # unlikely path, excepting error API usage
        if not cur_slice.anonymous and (
            cur_slice.slice_name != name or cur_slice.level != level
        ):
            logger.warning(
                f"Slice name mismatch: {name} != {cur_slice.slice_name} or level mismatch: {level} != {cur_slice.level}"
            )
            self.thread_context.cur_slice = cur_slice.parent_slice
            if cur_slice.parent_slice:
                cur_slice.parent_slice.child_slices.remove(cur_slice)
            self.__release_slice_reference_tree(cur_slice)
            return

        cur_slice.end_time = ts
        cur_slice.slice_name = name
        cur_slice.level = level

        if cur_slice.lazy_flag:
            # check if span can be created now
            if cur_slice.parent_slice and cur_slice.parent_slice.lazy_flag:
                if attrs:
                    cur_slice.attrs.update(attrs)
                self.thread_context.cur_slice = cur_slice.parent_slice
                self.__trace_slice_end_flag_process(
                    auto_next_anon, thread_finish_flag, ts
                )
                return

            self.__create_slice_span(cur_slice)

        span = cur_slice.span

        if attrs:
            span.set_attributes(attrs)

        self.thread_context.cur_slice = cur_slice.parent_slice
        # only for first level slice
        if not cur_slice.parent_slice:
            self.thread_context.last_span_context = span.get_span_context()
        else:
            cur_slice.parent_slice.child_slices.remove(cur_slice)
        self.__end_slice_span(cur_slice)

        self.__trace_slice_end_flag_process(auto_next_anon, thread_finish_flag, ts)

    # alias
    trace_slice = trace_slice_end

    # Add event to the current slice on the same thread with the same rid.
    def trace_event(
        self, name: str, ts: Optional[int] = None, attrs: Dict[str, Any] = None
    ):
        if not self.tracing_enable:
            return

        if not self.thread_context:
            return

        if not self.thread_context.cur_slice:
            logger.warning(f"No slice is currently being traced.")
            return

        cur_slice = self.thread_context.cur_slice
        ts = ts or get_cur_time_ns()

        if cur_slice.span:
            cur_slice.span.add_event(name=name, timestamp=ts, attributes=attrs)
        else:
            cur_slice.events.append(TraceEvent(name, ts, attrs))

    # Add attrs to the current slice on the same thread with the same rid.
    def trace_slice_add_attr(self, attrs: Dict[str, Any]):
        if not self.tracing_enable:
            return

        if not self.thread_context:
            return

        if not self.thread_context.cur_slice:
            logger.warning(f"No slice is currently being traced.")
            return

        cur_slice = self.thread_context.cur_slice
        if cur_slice.span:
            cur_slice.span.set_attributes(attrs)
        else:
            cur_slice.span.attrs.update(attrs)

    def abort(self, ts=None, abort_info: Optional[Dict] = None):
        if not self.tracing_enable:
            return

        if not self.thread_context:
            return

        # close all slice spans (unlikely except error API usage)
        ts = ts or get_cur_time_ns()
        if self.thread_context.cur_slice is not None:
            if self.thread_context.cur_slice.span:
                self.thread_context.cur_slice.span.end(end_time=ts)

            # if has nested span, end them
            while self.thread_context.cur_slice.parent_slice:
                self.thread_context.cur_slice = (
                    self.thread_context.cur_slice.parent_slice
                )
                if self.thread_context.cur_slice.span:
                    self.thread_context.cur_slice.span.end(end_time=ts)

            # slice will be dropped directly if slice.lazy_flag is True
            self.__release_slice_reference_tree(self.thread_context.cur_slice)
            self.thread_context.cur_slice = None

        # set abort info into thread span
        if self.thread_context.thread_span:
            if abort_info:
                from sglang.srt.managers.schedule_batch import BaseFinishReason

                if isinstance(abort_info, BaseFinishReason):
                    abort_info = abort_info.to_json()
                self.thread_context.thread_span.set_status(Status(StatusCode.ERROR))
                self.thread_context.thread_span.set_attributes(abort_info)
            self.thread_context.thread_span.end(end_time=ts)
        self.thread_context = None

    def __del__(self):
        self.abort(abort_info={"abort_info": "have unclosed span, auto closed"})
