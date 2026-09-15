"""Unit tests for trace.py — no server, no model loading."""

import os

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
import unittest
from unittest.mock import patch

import sglang.srt.observability.trace as mod
from sglang.srt.observability.trace import (
    SpanAttributes,
    TraceEvent,
    TraceNullContext,
    TraceReqContext,
    TraceSliceContext,
    TraceThreadInfo,
    extract_trace_headers,
    get_global_trace_level,
    get_global_tracing_enabled,
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    from sglang.srt.observability.trace import get_otlp_span_exporter

    _has_otel = True
except ImportError:
    _has_otel = False

# Access the private module-level function (avoid name mangling inside classes).
_get_host_id = getattr(mod, "_get_host_id")


class TestTraceFunctions(unittest.TestCase):
    def test_extract_trace_headers(self):
        headers = {"traceparent": "abc", "tracestate": "xyz", "other": "skip"}
        result = extract_trace_headers(headers)
        self.assertEqual(result, {"traceparent": "abc", "tracestate": "xyz"})

    def test_extract_trace_headers_missing(self):
        self.assertEqual(extract_trace_headers({}), {})

    def test_set_global_trace_level(self):
        from sglang.srt.runtime_context import get_resources

        orig = get_resources().trace_level
        try:
            set_global_trace_level(5)
            self.assertEqual(get_global_trace_level(), 5)
        finally:
            get_resources().trace_level = orig

    def test_global_trace_level_env_var(self):
        # The level lives on ctx.resources and is seeded lazily from the env
        # on first read after a reset (no module reload involved).
        from sglang.srt.runtime_context import get_resources

        orig = get_resources().trace_level
        try:
            with patch.dict(os.environ, {"SGLANG_TRACE_LEVEL": "2"}):
                get_resources().trace_level = None
                self.assertEqual(get_global_trace_level(), 2)
            get_resources().trace_level = None  # SGLANG_TRACE_LEVEL unset → 3
            self.assertEqual(get_global_trace_level(), 3)
        finally:
            get_resources().trace_level = orig

    def test_get_global_tracing_enabled(self):
        self.assertEqual(get_global_tracing_enabled(), mod.opentelemetry_initialized)

    def test_get_cur_time_ns(self):
        ts = mod.get_cur_time_ns()
        self.assertIsInstance(ts, int)
        self.assertGreater(ts, 0)


class TestTraceNullContext(unittest.TestCase):
    def test_null_object_pattern(self):
        ctx = TraceNullContext()
        self.assertFalse(ctx.tracing_enable)
        # Any attribute access returns self
        self.assertIs(ctx.some_method, ctx)
        # Callable returns self
        self.assertIs(ctx("arg1", key="val"), ctx)
        # Chaining works
        self.assertIs(ctx.foo.bar.baz(1, 2, 3), ctx)


class TestSpanAttributes(unittest.TestCase):
    def test_constants_exist(self):
        self.assertEqual(SpanAttributes.GEN_AI_LATENCY_E2E, "gen_ai.latency.e2e")
        self.assertIsInstance(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, str)


# __get_host_id
class TestGetHostId(unittest.TestCase):
    def test_from_machine_id_file(self):
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                unittest.mock.mock_open(read_data="abc123\n"),
            ),
        ):
            self.assertEqual(_get_host_id(), "abc123")

    def test_from_machine_id_file_error(self):
        """Falls back to MAC address when file read fails."""
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", side_effect=IOError("read error")),
        ):
            result = _get_host_id()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_from_mac_address(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("uuid.getnode", return_value=0x112233445566),
        ):
            result = _get_host_id()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_unknown_fallback(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("uuid.getnode", return_value=0),
        ):
            self.assertEqual(_get_host_id(), "unknown")


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestGetOtlpSpanExporter(unittest.TestCase):
    def test_grpc_default(self):

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", None)
            exporter = get_otlp_span_exporter("localhost:4317")
        self.assertIsNotNone(exporter)

    def test_http_protobuf(self):

        with patch.dict(
            os.environ, {"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"}
        ):
            exporter = get_otlp_span_exporter("http://localhost:4318/v1/traces")
        self.assertIsNotNone(exporter)

    def test_invalid_protocol(self):

        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "invalid"}):
            with self.assertRaises(ValueError):
                get_otlp_span_exporter("localhost:4317")


class TestProcessTracingInit(unittest.TestCase):
    def test_raises_without_otel(self):

        orig = mod.opentelemetry_imported
        mod.opentelemetry_imported = False
        try:
            with self.assertRaises(RuntimeError):
                process_tracing_init("localhost:4317", "test")
        finally:
            mod.opentelemetry_imported = orig


class TestTraceReqContextDisabled(unittest.TestCase):
    def setUp(self):
        self.orig = mod.opentelemetry_initialized
        mod.opentelemetry_initialized = False

    def tearDown(self):
        mod.opentelemetry_initialized = self.orig

    def test_init_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        self.assertFalse(ctx.tracing_enable)
        self.assertFalse(ctx.is_tracing_enabled())

    def test_getstate_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        state = ctx.__getstate__()
        self.assertEqual(state, {"tracing_enable": False})

    def test_setstate_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.__setstate__({"tracing_enable": True, "is_copy": False})
        # opentelemetry_initialized is False → tracing forced off
        self.assertFalse(ctx.tracing_enable)

        # Should not register anything


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestTraceReqContextEnabled(unittest.TestCase):
    def setUp(self):

        self.orig_initialized = mod.opentelemetry_initialized
        self.orig_tracer = mod.tracer
        self.orig_threads = mod.threads_info.copy()
        from sglang.srt.runtime_context import get_resources

        self.orig_level = get_resources().trace_level

        # Reset OTel global TracerProvider so set_tracer_provider works each test
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
        otel_trace._TRACER_PROVIDER = None

        self.provider = TracerProvider()
        otel_trace.set_tracer_provider(self.provider)
        mod.opentelemetry_initialized = True
        mod.tracer = otel_trace.get_tracer("test")
        set_global_trace_level(3)

    def tearDown(self):
        mod.opentelemetry_initialized = self.orig_initialized
        mod.tracer = self.orig_tracer
        mod.threads_info.clear()
        mod.threads_info.update(self.orig_threads)
        from sglang.srt.runtime_context import get_resources

        get_resources().trace_level = self.orig_level

    def test_trace_set_thread_info(self):
        trace_set_thread_info("scheduler", tp_rank=0, dp_rank=0)

        pid = threading.get_native_id()
        self.assertIn(pid, mod.threads_info)
        self.assertEqual(mod.threads_info[pid].thread_label, "scheduler")

        # Second call for same thread is a no-op
        trace_set_thread_info("different_label")
        self.assertEqual(mod.threads_info[pid].thread_label, "scheduler")

    def test_module_filtering(self):
        """global_trace_modules gates only explicitly named modules."""
        orig_modules = mod.global_trace_modules
        mod.global_trace_modules = ["request"]
        try:
            # Default empty module_name is never filtered
            ctx = TraceReqContext(rid="req-1")
            self.assertTrue(ctx.tracing_enable)
            # Listed module is traced
            ctx = TraceReqContext(rid="req-1", module_name="request")
            self.assertTrue(ctx.tracing_enable)
            # Unlisted module is filtered out
            ctx = TraceReqContext(rid="req-1", module_name="mooncake")
            self.assertFalse(ctx.tracing_enable)
        finally:
            mod.global_trace_modules = orig_modules

    def test_full_lifecycle(self):
        """Start → slice_start → slice_end → finish."""
        ctx = TraceReqContext(rid="req-1", role="unified")
        self.assertTrue(ctx.tracing_enable)

        ctx.trace_req_start(ts=1000)
        self.assertEqual(ctx.start_time_ns, 1000)
        self.assertIsNotNone(ctx.root_span)
        self.assertIsNotNone(ctx.thread_context)

        ctx.trace_slice_start("prefill", level=1, ts=2000)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 1)

        ctx.trace_slice_end("prefill", level=1, ts=3000)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)
        self.assertIsNotNone(ctx.last_span_context)

        ctx.trace_req_finish(ts=4000, attrs={"tokens": 42})
        self.assertIsNone(ctx.root_span)

    def test_trace_req_start_with_bootstrap_room(self):
        ctx = TraceReqContext(rid="req-1", bootstrap_room=0xFF, role="prefill")
        ctx.trace_req_start(ts=1000)
        self.assertIsNotNone(ctx.root_span)
        ctx.trace_req_finish(ts=2000)

    def test_trace_slice_combined(self):
        """trace_slice() creates and ends a span in one call."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        s = TraceSliceContext(
            "decode",
            2000,
            end_time_ns=3000,
            level=1,
            attrs={"key": "val"},
            events=[TraceEvent("evt", 2500, {"e": 1})],
        )
        ctx.trace_slice(s)
        self.assertIsNotNone(ctx.last_span_context)
        ctx.trace_req_finish(ts=4000)

    def test_trace_slice_with_events_cache(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        # Add events to cache
        ctx.trace_event("schedule", level=1, ts=1500, attrs={"bid": "x"})
        self.assertEqual(len(ctx.events_cache), 1)

        # trace_slice_start + trace_slice_end flushes matching events
        ctx.trace_slice_start("prefill", level=1, ts=1200)
        ctx.trace_slice_end("prefill", level=1, ts=2000)
        self.assertEqual(len(ctx.events_cache), 0)

        ctx.trace_req_finish(ts=3000)

    def test_trace_slice_combined_with_events_cache(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_event("evt", level=1, ts=1500)
        s = TraceSliceContext("decode", 1200, end_time_ns=2000, level=1)
        ctx.trace_slice(s)
        self.assertEqual(len(ctx.events_cache), 0)
        ctx.trace_req_finish(ts=3000)

    def test_trace_event_no_attrs(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_event("evt", level=1, ts=1500, attrs=None)
        self.assertEqual(ctx.events_cache[0].attrs, {})
        ctx.trace_req_finish(ts=2000)

    def test_trace_slice_end_empty_stack(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        # End without start → warning, no crash
        ctx.trace_slice_end("missing", level=1, ts=2000)
        ctx.trace_req_finish(ts=3000)

    def test_trace_slice_end_name_mismatch(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("prefill", level=1, ts=1500)
        # Mismatched name → warning, slice popped
        ctx.trace_slice_end("wrong_name", level=1, ts=2000)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)
        ctx.trace_req_finish(ts=3000)

    def test_trace_slice_end_with_attrs_and_thread_finish(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("dispatch", level=2, ts=1500)
        ctx.trace_slice_end(
            "dispatch",
            level=2,
            ts=2000,
            attrs={"key": "val"},
            thread_finish_flag=True,
        )
        # thread_finish_flag triggers abort → thread_context is None
        self.assertIsNone(ctx.thread_context)

    def test_trace_slice_combined_with_thread_finish(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        s = TraceSliceContext("dispatch", 1500, end_time_ns=2000, level=2)
        ctx.trace_slice(s, thread_finish_flag=True)
        self.assertIsNone(ctx.thread_context)

    def test_nested_slices(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("outer", level=1, ts=1500)
        ctx.trace_slice_start("inner", level=2, ts=1600)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 2)
        ctx.trace_slice_end("inner", level=2, ts=1800)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 1)
        ctx.trace_slice_end("outer", level=1, ts=2000)
        ctx.trace_req_finish(ts=3000)

    def test_nested_slice_with_last_span_context(self):
        """trace_slice uses last_span_context when slice stack is empty."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        # First slice sets last_span_context
        ctx.trace_slice_start("s1", level=1, ts=1500)
        ctx.trace_slice_end("s1", level=1, ts=2000)
        self.assertIsNotNone(ctx.last_span_context)

        # Second slice uses last_span_context as link
        ctx.trace_slice_start("s2", level=1, ts=2500)
        ctx.trace_slice_end("s2", level=1, ts=3000)

        # trace_slice also uses last_span_context
        s = TraceSliceContext("s3", 3500, end_time_ns=4000, level=1)
        ctx.trace_slice(s)

        ctx.trace_req_finish(ts=5000)

    def test_abort_with_unclosed_slices(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("s1", level=1, ts=1500)
        ctx.trace_slice_start("s2", level=2, ts=1600)
        ctx.abort(ts=2000)
        self.assertIsNone(ctx.thread_context)

    def test_abort_with_events_cache(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_event("evt", level=1, ts=1500)
        ctx.abort(ts=2000)
        self.assertEqual(len(ctx.events_cache), 0)

    def test_abort_with_abort_info_dict(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.abort(ts=2000, abort_info={"reason": "cancelled"})
        self.assertIsNone(ctx.thread_context)

    def test_abort_with_base_finish_reason(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        abort_obj = FINISH_LENGTH(length=10)
        ctx.abort(ts=2000, abort_info=abort_obj)
        self.assertIsNone(ctx.thread_context)

    def test_check_fast_return_by_level(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_level = 1  # instance-level, set at init from global
        # Level 2 > trace_level 1 → fast return
        ctx.trace_slice_start("s", level=2, ts=1500)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)
        ctx.trace_level = 3
        ctx.trace_req_finish(ts=2000)

    def test_rebuild_thread_context(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        old_tc = ctx.thread_context
        ctx.rebuild_thread_context(ts=1500)
        self.assertIsNot(ctx.thread_context, old_tc)
        ctx.trace_req_finish(ts=2000)

    def test_getstate_enabled(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        state = ctx.__getstate__()
        self.assertTrue(state["tracing_enable"])
        self.assertEqual(state["rid"], "req-1")
        self.assertIn("root_span_context", state)
        ctx.trace_req_finish(ts=2000)

    def test_getstate_no_root_context(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.root_span_context = None
        state = ctx.__getstate__()
        self.assertFalse(state["tracing_enable"])
        ctx.root_span_context = True  # prevent __del__ issues
        ctx.trace_req_finish(ts=2000)

    def test_getstate_with_slice_stack(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("s1", level=1, ts=1500)
        state = ctx.__getstate__()
        self.assertIn("last_span_context", state)
        ctx.trace_req_finish(ts=2000)

    def test_setstate_enabled(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        state = ctx.__getstate__()
        ctx.trace_req_finish(ts=2000)

        ctx2 = TraceReqContext(rid="req-2")
        ctx2.__setstate__(state)
        self.assertTrue(ctx2.tracing_enable)
        self.assertTrue(ctx2.is_copy)
        self.assertIsNotNone(ctx2.root_span_context)

    def test_thread_context_with_tp_rank(self):
        """Covers tp_rank branch in __create_thread_context."""

        pid = threading.get_native_id()
        mod.threads_info[pid] = TraceThreadInfo(
            "host", pid, "sched", tp_rank=0, dp_rank=0, pp_rank=0
        )
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        self.assertIsNotNone(ctx.thread_context)
        ctx.trace_req_finish(ts=2000)

    def test_setstate_with_last_span_context(self):
        """Covers __setstate__ path where last_span_context is truthy."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("s1", level=1, ts=1500)
        ctx.trace_slice_end("s1", level=1, ts=2000)
        state = ctx.__getstate__()
        ctx.trace_req_finish(ts=3000)

        self.assertIsNotNone(state.get("last_span_context"))
        ctx2 = TraceReqContext(rid="req-2")
        ctx2.__setstate__(state)
        self.assertIsNotNone(ctx2.last_span_context)

    def test_events_cache_partial_match(self):
        """Events outside the slice time range stay in cache."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_event("early", level=1, ts=500)
        ctx.trace_event("inside", level=1, ts=1500)
        ctx.trace_event("late", level=1, ts=5000)

        ctx.trace_slice_start("s", level=1, ts=1200)
        ctx.trace_slice_end("s", level=1, ts=2000)
        # "early" (500 < 1200) and "late" (5000 >= 2000) stay in cache
        self.assertEqual(len(ctx.events_cache), 2)
        ctx.trace_req_finish(ts=6000)

    def test_trace_slice_combined_events_partial_match(self):
        """Events outside slice range stay in cache for trace_slice method."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_event("early", level=1, ts=500)
        ctx.trace_event("inside", level=1, ts=1500)

        s = TraceSliceContext("s", 1200, end_time_ns=2000, level=1)
        ctx.trace_slice(s)
        self.assertEqual(len(ctx.events_cache), 1)  # "early" stays
        ctx.trace_req_finish(ts=3000)

    def test_trace_slice_nested_parent(self):
        """trace_slice with parent from slice stack (not thread_span)."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)

        ctx.trace_slice_start("outer", level=1, ts=1500)
        s = TraceSliceContext("inner", 1600, end_time_ns=1800, level=2)
        ctx.trace_slice(s)
        ctx.trace_slice_end("outer", level=1, ts=2000)
        ctx.trace_req_finish(ts=3000)

    def test_del_triggers_abort(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        # __del__ calls abort
        ctx.__del__()
        self.assertIsNone(ctx.thread_context)


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestMergedAsyncTracing(CustomTestCase):
    """Exercise the wire protocol and real OTel replay without an exporter process."""

    def setUp(self):
        import pickle
        from types import SimpleNamespace

        import sglang.srt.observability.req_time_stats as rts
        import sglang.srt.observability.trace_async as async_mod
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        if not async_mod._zmq_available:
            self.skipTest("pyzmq not installed")
        self.async_mod = async_mod
        self.rts = rts
        self.namespace = SimpleNamespace
        self.messages = []
        self.spans = InMemorySpanExporter()
        self.provider = TracerProvider(id_generator=mod.TraceCustomIdGenerator())
        self.provider.add_span_processor(SimpleSpanProcessor(self.spans))
        self.addCleanup(self.provider.shutdown)
        self.contexts = {}
        self.callers = []
        self.exporter = async_mod._TraceExporterProcess("unused", "test", "unused")

        # Serialize at the transport boundary so later buffer mutations cannot
        # accidentally make an in-memory fake pass where ZMQ would fail.
        def send_pyobj(message, flags):
            self.assertEqual(flags, async_mod.zmq.NOBLOCK)
            self.messages.append(pickle.loads(pickle.dumps(message)))

        self.socket = SimpleNamespace(send_pyobj=send_pyobj)
        patches = [
            patch.object(async_mod, "is_async_tracing_available", return_value=True),
            patch.object(rts, "is_async_tracing_available", return_value=True),
            patch.object(async_mod, "_get_zmq_socket", return_value=self.socket),
            patch.object(mod, "opentelemetry_initialized", True),
            patch.object(mod, "global_trace_modules", None),
            patch.object(mod, "get_global_trace_level", return_value=3),
            patch.object(mod, "tracer", self.provider.get_tracer("merged-test")),
            patch.object(mod, "threads_info", {}),
            patch.dict(os.environ, {"SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD": "10000"}),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        pid = threading.get_native_id()
        mod.threads_info[pid] = TraceThreadInfo("test-host", pid, "scheduler", 0, 0, 0)
        self.addCleanup(self._close_contexts)

    def _close_contexts(self):
        for ctx, _ in self.contexts.values():
            ctx.abort(ts=10000)
        self.contexts.clear()
        for ctx in self.callers:
            if ctx.tracing_enable and ctx.root_span is not None:
                ctx.root_span.end(end_time=10000)
                ctx.root_span = None

    def _new_context(self, rid="same-request"):
        ctx = self.async_mod.TraceReqContextAsync(rid=rid)
        self.callers.append(ctx)
        return ctx

    def _replay(self, message):
        args = (
            message,
            self.contexts,
            mod.threads_info,
            mod.TraceCustomIdGenerator,
            TraceReqContext,
            TraceSliceContext,
            TraceEvent,
        )
        if message["action"] == "multi_batch":
            self.exporter._replay_multi_batch(*args)
        else:
            self.exporter._replay_batch(*args)

    def test_batch_flush_preserves_context_identity_and_pending_ops(self):
        """Same-rid contexts stay separate; duplicate request references cannot resend ops."""
        a, b = self._new_context(), self._new_context()
        with patch.object(
            self.async_mod, "is_async_tracing_available", return_value=False
        ):
            disabled = self._new_context()
        a.trace_event("first", level=1, ts=100)
        req = lambda ctx: self.namespace(time_stats=self.namespace(trace_ctx=ctx))
        reqs = [
            req(a),
            object(),
            req(TraceNullContext()),
            req(b),
            req(a),
            req(disabled),
            self.namespace(time_stats=None),
        ]
        self.rts.flush_trace_batch(reqs)
        self.assertEqual(len(self.messages), 1)
        first = self.messages[0]
        self.assertEqual(first["action"], "multi_batch")
        self.assertEqual(
            [x["context_id"] for x in first["batches"]], [a._context_id, b._context_id]
        )
        self.assertEqual(
            [op["type"] for op in first["batches"][0]["operations"]], ["init", "event"]
        )
        self.assertEqual(a._operations, [])
        self.assertEqual(b._operations, [])
        self.rts.flush_trace_batch(reqs)
        self.assertEqual(len(self.messages), 1)
        a.trace_event("next-step", level=1, ts=200)
        a.flush()
        self.assertEqual(len(self.messages), 2)
        self.assertEqual(self.messages[1]["action"], "batch")
        self.assertEqual(
            [op["name"] for op in self.messages[1]["operations"]], ["next-step"]
        )
        self.assertEqual(first["batches"][0]["operations"][-1]["name"], "first")

    def test_replay_preserves_spans_across_scheduler_steps(self):
        """Merging must preserve IDs, parents, timestamps and events across flushes."""
        callers = [self._new_context(), self._new_context()]
        expected = []
        for i, ctx in enumerate(callers):
            ctx.trace_req_start(ts=100)
            root_id = ctx.root_span.get_span_context().span_id
            ctx.trace_slice_start(f"prefill-{i}", level=1, ts=200)
            expected.append((root_id, ctx._span_id_stack[-1]))
        self.async_mod.flush_trace_contexts_merged(callers)
        self._replay(self.messages[-1])
        self.assertEqual(len(self.contexts), 2)
        for i, ctx in enumerate(callers):
            ctx.trace_event(f"scheduled-{i}", level=1, ts=250, attrs={"tokens": 16 + i})
            ctx.trace_slice_end(f"prefill-{i}", level=1, ts=300, attrs={"batch": i})
        self.async_mod.flush_trace_contexts_merged(callers)
        self._replay(self.messages[-1])
        spans = {s.name: s for s in self.spans.get_finished_spans()}
        for i, (root_id, span_id) in enumerate(expected):
            span = spans[f"prefill-{i}"]
            self.assertEqual(span.context.span_id, span_id)
            thread_span = self.contexts[callers[i]._context_id][
                0
            ].thread_context.thread_span
            self.assertEqual(
                span.parent.span_id, thread_span.get_span_context().span_id
            )
            self.assertEqual(thread_span.parent.span_id, root_id)
            self.assertEqual((span.start_time, span.end_time), (200, 300))
            self.assertEqual(span.attributes["batch"], i)
            self.assertEqual(
                [(e.name, e.timestamp, e.attributes["tokens"]) for e in span.events],
                [(f"scheduled-{i}", 250, 16 + i)],
            )
        for ctx in callers:
            ctx.trace_req_finish(ts=400)
            self._replay(self.messages[-1])
        self.assertEqual(self.contexts, {})

    def test_bad_sub_batch_does_not_block_other_requests(self):
        """A replay failure must not discard valid contexts later in the merged message."""
        for i in range(2):
            ctx = self._new_context(rid=f"request-{i}")
            ctx.trace_req_start(ts=100)
            ctx.trace_slice_start(f"prefill-{i}", level=1, ts=200)
            ctx.trace_slice_end(f"prefill-{i}", level=1, ts=300)
        self.async_mod.flush_trace_contexts_merged(self.callers)
        message = self.messages[-1]
        message["batches"].insert(1, {"rid": "malformed", "operations": []})
        with self.assertLogs(self.async_mod.logger, level="ERROR"):
            self._replay(message)
        self.assertEqual(len(self.contexts), 2)
        self.assertEqual(
            {s.name for s in self.spans.get_finished_spans()},
            {"prefill-0", "prefill-1"},
        )

    def test_failed_send_drops_detached_ops_without_resending_them(self):
        """Best-effort transport drops a failed step, but preserves the next step."""
        for failure in (None, self.async_mod.zmq.Again(), RuntimeError("closed")):
            with self.subTest(failure=type(failure).__name__):
                a, b = self._new_context(), self._new_context()

                def fail_send(message, flags):
                    raise failure

                socket = (
                    None if failure is None else self.namespace(send_pyobj=fail_send)
                )
                with patch.object(
                    self.async_mod, "_get_zmq_socket", return_value=socket
                ):
                    self.async_mod.flush_trace_contexts_merged([a, b])
                self.assertEqual((a._operations, b._operations), ([], []))
                a.trace_event("after-drop", level=1, ts=200)
                self.async_mod.flush_trace_contexts_merged([a, b])
                batches = self.messages[-1]["batches"]
                self.assertEqual(len(batches), 1)
                self.assertEqual(
                    [op["name"] for op in batches[0]["operations"]], ["after-drop"]
                )

    def test_unavailable_async_tracing_does_not_iterate_batch(self):
        """Sync-only tracing must not pay per-request batch traversal overhead."""

        class UnvisitedBatch(list):
            def __iter__(self):
                raise AssertionError("Traversed batch without async tracing")

        with patch.object(self.rts, "is_async_tracing_available", return_value=False):
            self.rts.flush_trace_batch(UnvisitedBatch([object()]))
        self.assertEqual(self.messages, [])

    def test_disabled_batch_flush_keeps_buffered_operations(self):
        ctx = self._new_context()
        req = self.namespace(time_stats=self.namespace(trace_ctx=ctx))
        before = list(ctx._operations)
        with patch.object(self.rts, "get_global_tracing_enabled", return_value=False):
            self.rts.flush_trace_batch([req])
        self.rts.flush_trace_batch(None)
        self.assertEqual(ctx._operations, before)
        self.assertEqual(self.messages, [])


if __name__ == "__main__":
    unittest.main()
