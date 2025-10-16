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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Req

logger = logging.getLogger(__name__)
opentelemetry_imported = False
tracing_enabled = False

try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider, id_generator
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    opentelemetry_imported = True
except ImportError:

    class id_generator:
        class IdGenerator:
            pass

    logger.info("opentelemetry package is not installed, tracing disabled")


@dataclass
class SglangTraceThreadInfo:
    host_id: str
    pid: int
    thread_label: str
    tp_rank: int
    dp_rank: int
    tracer: trace.Tracer


@dataclass
class SglangTraceSliceContext:
    slice_name: str
    span: Optional[trace.span.Span] = None
    # When True, defers slice_name assignment until trace_slice_end()
    anonymous: bool = False


@dataclass
class SglangTraceThreadContext:
    thread_info: SglangTraceThreadInfo
    cur_slice_stack: List[SglangTraceSliceContext]
    thread_span: Optional[trace.span.Span] = None
    # Record the most recently completed span as the previous span for the next span to be created.
    last_span_context: Optional[trace.span.SpanContext] = None


@dataclass
class SglangTraceReqContext:
    rid: str
    start_time_ns: int
    threads_context: Dict[int, SglangTraceThreadContext]
    bootstrap_room: Optional[int] = None

    # Indicates whether this instance is a replica from the main process.
    # When True, root_span is None and only root_span_context is preserved.
    is_copy: bool = False
    root_span: Optional[trace.span.Span] = None
    root_span_context: Optional[context.Context] = None


@dataclass
class SglangTracePropagateContext:
    root_span_context: context.Context
    prev_span_context: Optional[trace.span.SpanContext]

    def to_dict(self):
        carrier: dict[str, str] = {}
        context.attach(self.root_span_context)
        propagate.inject(carrier)

        if self.prev_span_context:
            return {
                "root_span": carrier,
                "prev_span": {
                    "span_id": self.prev_span_context.span_id,
                    "trace_id": self.prev_span_context.trace_id,
                },
            }
        else:
            return {"root_span": carrier, "prev_span": "None"}

    @classmethod
    def instance_from_dict(cls, d):
        if "root_span" not in d or "prev_span" not in d:
            return None

        carrier = d["root_span"]
        root_span_context = propagate.extract(carrier)

        if d["prev_span"] == "None":
            prev_span_context = None
        else:
            prev_span_context = trace.span.SpanContext(
                trace_id=d["prev_span"]["trace_id"],
                span_id=d["prev_span"]["span_id"],
                is_remote=True,
            )

        return cls(root_span_context, prev_span_context)


class SglangTraceCustomIdGenerator(id_generator.IdGenerator):
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
threads_info: Dict[int, SglangTraceThreadInfo] = {}
reqs_context: Dict[str, SglangTraceReqContext] = {}

__get_cur_time_ns = lambda: int(time.time() * 1e9)


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
    global tracing_enabled
    global __get_cur_time_ns
    if not opentelemetry_imported:
        tracing_enabled = False
        return

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(
            resource=resource, id_generator=SglangTraceCustomIdGenerator()
        )

        processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        )
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        logger.error(f": initialize opentelemetry error:{e}")
        logger.warning("pelease set correct otlp endpoint")
        tracing_enabled = False
        return

    if hasattr(time, "time_ns"):
        __get_cur_time_ns = lambda: int(time.time_ns())

    tracing_enabled = True


# Should be called by each tracked thread.
def trace_set_thread_info(
    thread_label: str, tp_rank: Optional[int] = None, dp_rank: Optional[int] = None
):
    if not tracing_enabled:
        return

    pid = threading.get_native_id()
    if pid in threads_info:
        return

    threads_info[pid] = SglangTraceThreadInfo(
        host_id=__get_host_id(),
        pid=pid,
        thread_label=thread_label,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        tracer=trace.get_tracer("sglang server"),
    )


def __create_thread_context(pid, req_span_context, ts: Optional[int] = None):
    if pid not in threads_info:
        trace_set_thread_info("unknown")

    thread_info = threads_info[pid]
    thread_context = SglangTraceThreadContext(
        thread_info=thread_info,
        cur_slice_stack=[],
    )

    thread_name = f"{thread_info.thread_label}"
    if thread_info.tp_rank is not None:
        thread_name += f" [TP {thread_info.tp_rank}] "
    thread_name += f"(host:{thread_info.host_id[:8]} | pid:{pid})"
    ts = ts or __get_cur_time_ns()
    thread_context.thread_span = thread_context.thread_info.tracer.start_span(
        name=thread_name,
        start_time=ts,
        context=req_span_context,
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


def trace_get_proc_propagate_context(rid) -> Optional[Dict[str, Any]]:
    if not tracing_enabled:
        return None

    rid = str(rid)
    if rid not in reqs_context or not reqs_context[rid].root_span_context:
        return None

    pid = threading.get_native_id()
    prev_span_context = None
    thread_context = reqs_context[rid].threads_context[pid]
    if thread_context.cur_slice_stack:
        cur_slice_info = thread_context.cur_slice_stack[0]
        prev_span_context = cur_slice_info.span.get_span_context()
    elif thread_context.last_span_context:
        prev_span_context = thread_context.last_span_context

    trace_context = SglangTracePropagateContext(
        reqs_context[rid].root_span_context, prev_span_context
    )
    return trace_context.to_dict()


def trace_set_proc_propagate_context(rid, trace_context: Optional[Dict[str, Any]]):
    if not tracing_enabled:
        return
    if not trace_context:
        return

    trace_context = SglangTracePropagateContext.instance_from_dict(trace_context)
    if not trace_context:
        return

    rid = str(rid)
    # Create a copy of the request context
    if rid not in reqs_context:
        reqs_context[rid] = SglangTraceReqContext(
            rid=rid,
            start_time_ns=__get_cur_time_ns(),
            threads_context={},
            root_span_context=trace_context.root_span_context,
            is_copy=True,
        )

    pid = threading.get_native_id()

    if pid in reqs_context[rid].threads_context:
        return

    # Create new thread context.
    reqs_context[rid].threads_context[pid] = __create_thread_context(
        pid,
        trace_context.root_span_context,
        reqs_context[rid].start_time_ns,
    )

    reqs_context[rid].threads_context[
        pid
    ].last_span_context = trace_context.prev_span_context


def trace_req_start(
    rid: str,
    bootstrap_room: Optional[int] = None,
    ts: Optional[int] = None,
):
    if not tracing_enabled:
        return

    rid = str(rid)

    ts = ts or __get_cur_time_ns()

    pid = threading.get_native_id()
    if pid not in threads_info:
        return

    # create req context and root span
    reqs_context[rid] = SglangTraceReqContext(
        rid=rid,
        start_time_ns=ts,
        threads_context={},
        bootstrap_room=bootstrap_room,
        is_copy=False,
    )

    # Drop the worker_id added by MultiTokenizer
    orig_rid = rid.split("_")[-1]
    tracer = threads_info[pid].tracer
    root_span = tracer.start_span(
        name=f"Req {orig_rid[:8]}",
        start_time=ts,
    )

    root_span.set_attributes(
        {
            "rid": rid,
            "bootstrap_room": bootstrap_room if bootstrap_room else "None",
        }
    )

    reqs_context[rid].root_span = root_span
    reqs_context[rid].root_span_context = trace.set_span_in_context(root_span)

    # create thread context and thread span
    reqs_context[rid].threads_context[pid] = __create_thread_context(
        pid,
        reqs_context[rid].root_span_context,
        ts,
    )


def trace_req_finish(
    rid: str, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    req_context = reqs_context[rid]
    ts = ts or __get_cur_time_ns()

    # End all unclosed thread spans.
    for thread_context in req_context.threads_context.values():
        thread_context.thread_span.end(end_time=ts)

    if attrs:
        req_context.root_span.set_attributes(attrs)

    req_context.root_span.end(end_time=ts)

    del reqs_context[rid]


def trace_slice_start(
    name: str,
    rid: str,
    ts: Optional[int] = None,
    anonymous: bool = False,
):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    ts = ts or __get_cur_time_ns()

    slice_info = SglangTraceSliceContext(
        slice_name=name,
        anonymous=anonymous,
    )

    # find prev slice
    prev_span_context = None
    if not thread_context.cur_slice_stack:
        if thread_context.last_span_context:
            prev_span_context = thread_context.last_span_context

    parent_span = thread_context.thread_span
    if thread_context.cur_slice_stack:
        parent_span = thread_context.cur_slice_stack[-1].span

    parent_span_context = trace.set_span_in_context(parent_span)
    span = thread_context.thread_info.tracer.start_span(
        name=slice_info.slice_name,
        start_time=ts,
        context=parent_span_context,
    )

    if prev_span_context:
        span.add_link(prev_span_context)

    slice_info.span = span

    thread_context.cur_slice_stack.append(slice_info)


def trace_slice_end(
    name: str,
    rid: str,
    ts: Optional[int] = None,
    attrs: Optional[Dict[str, Any]] = None,
    auto_next_anon: bool = False,
    thread_finish_flag: bool = False,
):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    if not thread_context.cur_slice_stack:
        logger.warning(f"No matching with the SLICE_START event{name} is required.")
        return

    ts = ts or __get_cur_time_ns()
    slice_info = thread_context.cur_slice_stack[-1]
    span = slice_info.span

    if slice_info.anonymous:
        span.update_name(name)
    else:
        span = slice_info.span
        if slice_info.slice_name != name:
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            logger.warning(f"Slice name mismatch: {name} != {slice_info.slice_name}")

    if attrs:
        span.set_attributes(attrs)

    span.end(end_time=ts)

    thread_context.cur_slice_stack.pop()
    if len(thread_context.cur_slice_stack) == 0:
        thread_context.last_span_context = span.get_span_context()

    # If this is the last slice in the thread,
    # release the thread context and check whether to release the request context.
    if thread_finish_flag:
        thread_context.thread_span.end(end_time=ts)
        del reqs_context[rid].threads_context[pid]
        if reqs_context[rid].is_copy and not reqs_context[rid].threads_context:
            del reqs_context[rid]
        return

    if auto_next_anon:
        trace_slice_start("", rid, ts, True)


# alias
trace_slice = trace_slice_end


# Add event to the current slice on the same thread with the same rid.
def trace_event(name: str, rid: str, ts: Optional[int] = None):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    if not thread_context.cur_slice_stack:
        logger.warning(f"No slice is currently being traced.")
        return

    ts = ts or __get_cur_time_ns()

    slice_info = thread_context.cur_slice_stack[-1]
    slice_info.span.add_event(name=name, timestamp=ts)


# Add attrs to the current slice on the same thread with the same rid.
def trace_slice_add_attr(rid: str, attrs: Dict[str, Any]):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    if not thread_context.cur_slice_stack:
        logger.warning(f"No slice is currently being traced.")
        return

    slice_info = thread_context.cur_slice_stack[-1]
    slice_info.span.set_attributes(attrs)


def trace_slice_batch(
    name: str,
    reqs: List[Req],
):
    for req in reqs:
        trace_slice(
            name,
            req.rid,
            auto_next_anon=not req.finished(),
            thread_finish_flag=req.finished(),
        )
