"""Lightweight in-process OTLP collector for tracing tests.

Provides a minimal OTLP collector that receives traces via gRPC (with HTTP
fallback) and stores them in memory for test assertions, eliminating the need
for Docker-based opentelemetry-collector and file I/O.

Usage::

    collector = LightweightOtlpCollector(port=4317)
    collector.start()
    # ... run code that emits traces ...
    assert collector.has_span("my_span")
    collector.stop()
"""

import json
import logging
import threading
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a single span extracted from OTLP trace data."""

    name: str
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    start_time_ns: int = 0
    end_time_ns: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


class LightweightOtlpCollector:
    """A minimal OTLP collector that stores traces in memory for test assertions.

    This replaces the Docker-based opentelemetry-collector for testing purposes.
    It listens on a gRPC port for OTLP trace data and stores spans in memory,
    allowing tests to verify specific spans based on trace level.
    """

    def __init__(self, port: int = 4317):
        self.port = port
        self._server = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        # In-memory storage for collected spans
        self._spans: List[Span] = []
        self._raw_traces: List[Dict[str, Any]] = []

    def _try_grpc_server(self):
        """Try to start gRPC server with full OTLP protocol."""
        try:
            from grpc import server as grpc_server
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceResponse,
            )
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
                TraceServiceServicer,
                add_TraceServiceServicer_to_server,
            )

            class TraceServicer(TraceServiceServicer):
                def __init__(self, collector):
                    self.collector = collector

                def Export(self, request, context):
                    self.collector._handle_trace_request(request)
                    return ExportTraceServiceResponse()

            self._server = grpc_server(futures.ThreadPoolExecutor(max_workers=4))
            add_TraceServiceServicer_to_server(TraceServicer(self), self._server)
            self._server.add_insecure_port(f"127.0.0.1:{self.port}")
            return True
        except ImportError:
            logger.warning("Full gRPC OTLP not available, using HTTP fallback")
            return False

    def _handle_trace_request(self, request):
        """Handle incoming trace request and extract spans to memory."""
        with self._lock:
            try:
                trace_data = self._protobuf_to_dict(request)
                self._raw_traces.append(trace_data)
                # Extract spans from the trace data
                self._extract_spans(trace_data)
            except Exception as e:
                logger.error(f"Failed to process trace: {e}")

    def _protobuf_to_dict(self, proto_obj) -> Dict[str, Any]:
        """Convert protobuf message to dict."""
        result = {}
        for field, value in proto_obj.ListFields():
            if field.message_type:
                type_name = type(value).__name__
                if "Repeated" in type_name:
                    result[field.name] = [self._protobuf_to_dict(v) for v in value]
                else:
                    result[field.name] = self._protobuf_to_dict(value)
            else:
                result[field.name] = value
        return result

    def _extract_spans(self, trace_data: Dict[str, Any]):
        """Extract Span objects from OTLP trace data structure."""
        resource_spans = trace_data.get("resource_spans", [])
        for rs in resource_spans:
            scope_spans = rs.get("scope_spans", [])
            for ss in scope_spans:
                spans = ss.get("spans", [])
                for span_data in spans:
                    span = Span(
                        name=span_data.get("name", ""),
                        trace_id=span_data.get("trace_id", ""),
                        span_id=span_data.get("span_id", ""),
                        parent_span_id=span_data.get("parent_span_id", ""),
                        start_time_ns=span_data.get("start_time_unix_nano", 0),
                        end_time_ns=span_data.get("end_time_unix_nano", 0),
                        attributes=span_data.get("attributes", {}),
                        events=span_data.get("events", []),
                    )
                    self._spans.append(span)

    def _http_server_loop(self):
        """Fallback HTTP server for OTLP HTTP protocol."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class OTLPHandler(BaseHTTPRequestHandler):
            def __init__(self, request, client_address, server):
                self.collector = server.collector
                super().__init__(request, client_address, server)

            def do_POST(self):
                if self.path in ["/v1/traces", "/v1/traces/"]:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    try:
                        data = json.loads(body)
                        with self.collector._lock:
                            self.collector._raw_traces.append(data)
                            self.collector._extract_spans_http(data)
                        self.send_response(200)
                        self.end_headers()
                    except Exception as e:
                        logger.error(f"HTTP trace handling error: {e}")
                        self.send_response(500)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress HTTP server logs

        class CollectorHTTPServer(HTTPServer):
            def __init__(self, server_address, collector):
                self.collector = collector
                super().__init__(
                    server_address,
                    lambda r, a, s: OTLPHandler(r, a, s),
                )

        server = CollectorHTTPServer(("127.0.0.1", 4318), self)
        server.timeout = 0.5
        while self._running:
            server.handle_request()

    def _extract_spans_http(self, data: Dict[str, Any]):
        """Extract Span objects from OTLP HTTP JSON format."""
        resource_spans = data.get("resourceSpans", [])
        for rs in resource_spans:
            scope_spans = rs.get("scopeSpans", [])
            for ss in scope_spans:
                spans = ss.get("spans", [])
                for span_data in spans:
                    span = Span(
                        name=span_data.get("name", ""),
                        trace_id=span_data.get("traceId", ""),
                        span_id=span_data.get("spanId", ""),
                        parent_span_id=span_data.get("parentSpanId", ""),
                        start_time_ns=span_data.get("startTimeUnixNano", 0),
                        end_time_ns=span_data.get("endTimeUnixNano", 0),
                        attributes=span_data.get("attributes", {}),
                        events=span_data.get("events", []),
                    )
                    self._spans.append(span)

    def start(self):
        """Start the collector server."""
        self._running = True
        self._spans.clear()
        self._raw_traces.clear()
        if self._try_grpc_server():
            self._server.start()
            logger.info(f"OTLP gRPC collector started on port {self.port}")
        else:
            # Fallback to HTTP server in a thread
            self._thread = threading.Thread(target=self._http_server_loop, daemon=True)
            self._thread.start()
            logger.info("OTLP HTTP collector started on port 4318")

    def stop(self):
        """Stop the collector server."""
        self._running = False
        if self._server:
            self._server.stop(1)
            self._server = None
        logger.info("OTLP collector stopped")

    # ========================================================================
    # Public API for test assertions
    # ========================================================================

    def get_spans(self) -> List[Span]:
        """Get all collected spans."""
        with self._lock:
            return list(self._spans)

    def get_span_names(self) -> Set[str]:
        """Get all unique span names."""
        with self._lock:
            return {s.name for s in self._spans}

    def has_span(self, name: str) -> bool:
        """Check if a span with the given name exists."""
        return name in self.get_span_names()

    def has_any_span(self, names: List[str]) -> bool:
        """Check if any of the given span names exist."""
        span_names = self.get_span_names()
        return any(name in span_names for name in names)

    def has_all_spans(self, names: List[str]) -> bool:
        """Check if all of the given span names exist."""
        span_names = self.get_span_names()
        return all(name in span_names for name in names)

    def get_spans_by_name(self, name: str) -> List[Span]:
        """Get all spans with the given name."""
        with self._lock:
            return [s for s in self._spans if s.name == name]

    def count_spans(self) -> int:
        """Get total count of collected spans."""
        with self._lock:
            return len(self._spans)

    def clear(self):
        """Clear all collected spans."""
        with self._lock:
            self._spans.clear()
            self._raw_traces.clear()
