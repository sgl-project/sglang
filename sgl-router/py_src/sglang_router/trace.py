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
"""simple requests tracing package for mini_lb"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
opentelemetry_imported = False
tracing_enabled = False

try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    opentelemetry_imported = True
except ImportError:
    logger.info("opentelemetry package is not installed, tracing disabled")


@dataclass
class RouterTraceReqContext:
    bootstrap_room: Optional[int] = None

    bootstrap_room_span: Optional[trace.span.Span] = None
    bootstrap_room_span_context: Optional[context.Context] = None
    root_span: Optional[trace.span.Span] = None

    thread_span: Optional[trace.span.Span] = None
    cur_span: Optional[trace.span.Span] = None
    last_span_context: Optional[trace.span.SpanContext] = None


@dataclass
class RouterTracePropagateContext:
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
            return {"root_span": carrier, "prev_span": "None"}


# global variables
reqs_context: Dict[str, RouterTraceReqContext] = {}

__get_cur_time_ns = lambda: int(time.time() * 1e9)

tracer = None
host_id = None


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
    global tracer
    global host_id
    if not opentelemetry_imported:
        tracing_enabled = False
        return

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(resource=resource)

        processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True),
            schedule_delay_millis=500,
            max_export_batch_size=64,
        )
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer("sglang server")
        host_id = __get_host_id()
    except Exception as e:
        logger.error(f": initialize opentelemetry error:{e}")
        logger.warning("pelease set correct otlp endpoint")
        tracing_enabled = False
        return

    if hasattr(time, "time_ns"):
        __get_cur_time_ns = lambda: int(time.time_ns())

    tracing_enabled = True


def trace_get_remote_propagate_context(bootstrap_room_list: List[int]):
    if not tracing_enabled:
        return ""

    reqs_propagate_contexts = {}
    for bootstrap_room in bootstrap_room_list:

        if bootstrap_room not in reqs_context:
            continue

        req_context = reqs_context[bootstrap_room]

        prev_span_context = (
            req_context.last_span_context
            if req_context.cur_span is None
            else req_context.cur_span.get_span_context()
        )
        _context = RouterTracePropagateContext(
            req_context.bootstrap_room_span_context, prev_span_context
        )
        reqs_propagate_contexts[bootstrap_room] = _context.to_dict()

    json_str = json.dumps(reqs_propagate_contexts, ensure_ascii=False)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def trace_req_start(
    bootstrap_room: Optional[int] = None,
    ts: Optional[int] = None,
):
    if not tracing_enabled:
        return

    ts = ts or __get_cur_time_ns()

    pid = threading.get_native_id()

    # create req context and root span
    bootstrap_room = 0 if bootstrap_room is None else bootstrap_room
    req_context = RouterTraceReqContext(
        bootstrap_room=bootstrap_room,
    )

    reqs_context[bootstrap_room] = req_context

    # create bootstrap room span
    attrs = {"bootstrap_room": hex(bootstrap_room)}
    bootstrap_room_span = tracer.start_span(
        name=f"Bootstrap Room {hex(bootstrap_room)}",
        start_time=ts,
        attributes=attrs,
    )
    req_context.bootstrap_room_span = bootstrap_room_span
    bootstrap_room_span_context = trace.set_span_in_context(bootstrap_room_span)

    role = "router"
    attrs = {"rid": hex(bootstrap_room)}
    root_span = tracer.start_span(
        name=f"{role} Req {str(hex(bootstrap_room))[:8]}",
        start_time=ts,
        context=bootstrap_room_span_context,
        attributes=attrs,
    )
    root_span_context = trace.set_span_in_context(root_span)

    req_context.root_span = root_span
    req_context.bootstrap_room_span_context = bootstrap_room_span_context

    # create thread context and thread span
    attrs = {"host_id": host_id, "pid": pid, "thread_label": "router"}
    thread_span = tracer.start_span(
        name=f"router (host:{host_id[:8]} | pid:{pid})",
        start_time=ts,
        context=root_span_context,
        attributes=attrs,
    )
    req_context.thread_span = thread_span


def trace_req_finish(
    bootstrap_room: int,
    ts: Optional[int] = None,
    attrs: Optional[Dict[str, Any]] = None,
):
    if not tracing_enabled:
        return

    if bootstrap_room not in reqs_context:
        return

    req_context = reqs_context[bootstrap_room]
    ts = ts or __get_cur_time_ns()

    if req_context.thread_span:
        req_context.thread_span.end(end_time=ts)

    if attrs:
        req_context.root_span.set_attributes(attrs)

    req_context.root_span.end(end_time=ts)
    req_context.bootstrap_room_span.end(end_time=ts)

    del reqs_context[bootstrap_room]


def trace_slice_start(
    name: str,
    bootstrap_room: int,
    ts: Optional[int] = None,
    anonymous: bool = False,
):
    if not tracing_enabled:
        return

    if bootstrap_room not in reqs_context:
        return

    req_context = reqs_context[bootstrap_room]
    ts = ts or __get_cur_time_ns()

    prev_span_context = req_context.last_span_context
    parent_span = req_context.thread_span
    parent_span_context = trace.set_span_in_context(parent_span)
    span = tracer.start_span(
        name=name,
        start_time=ts,
        context=parent_span_context,
    )

    if prev_span_context:
        span.add_link(prev_span_context)
    req_context.cur_span = span


def trace_slice_end(
    name: str,
    bootstrap_room: int,
    ts: Optional[int] = None,
    attrs: Optional[Dict[str, Any]] = None,
    thread_finish_flag: bool = False,
    auto_next_anon: bool = False,
):
    if not tracing_enabled:
        return

    if bootstrap_room not in reqs_context:
        return

    ts = ts or __get_cur_time_ns()
    req_context = reqs_context[bootstrap_room]
    span = req_context.cur_span
    if span:
        span.update_name(name)
        if attrs:
            span.set_attributes(attrs)

        span.end(end_time=ts)
        req_context.last_span_context = span.get_span_context()
        req_context.cur_span = None

    if thread_finish_flag:
        req_context.thread_span.end(end_time=ts)
        req_context.thread_span = None
        return

    if auto_next_anon:
        trace_slice_start("", bootstrap_room, ts, True)


# alias
trace_slice = trace_slice_end
