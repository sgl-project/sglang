"""Unit tests for trace.py — no server, no model loading."""

# ── Stubs for heavy transitive deps ──
import os
import sys
import types


def _ensure_module(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


_su = _ensure_module("sglang.srt.utils")
if not hasattr(_su, "get_int_env_var"):
    _su.get_int_env_var = lambda name, default=0: int(os.getenv(name, str(default)))
_ensure_module("sglang.srt.utils.common")

_ensure_module("sglang.srt.managers")
_sb = _ensure_module("sglang.srt.managers.schedule_batch")
_sb.BaseFinishReason = type("BaseFinishReason", (), {"to_json": lambda self: {}})

# ── End stubs ──

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import threading
import unittest
from unittest.mock import patch

import sglang.srt.observability.trace as mod
from sglang.srt.observability.trace import (
    SpanAttributes,
    TraceCustomIdGenerator,
    TraceEvent,
    TraceNullContext,
    TraceReqContext,
    TraceSliceContext,
    TraceThreadContext,
    TraceThreadInfo,
    extract_trace_headers,
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
_get_host_id = getattr(mod, "__get_host_id")


class TestTraceFunctions(unittest.TestCase):
    def test_extract_trace_headers(self):
        headers = {"traceparent": "abc", "tracestate": "xyz", "other": "skip"}
        result = extract_trace_headers(headers)
        self.assertEqual(result, {"traceparent": "abc", "tracestate": "xyz"})

    def test_extract_trace_headers_missing(self):
        self.assertEqual(extract_trace_headers({}), {})

    def test_set_global_trace_level(self):
        orig = mod.global_trace_level
        set_global_trace_level(5)
        self.assertEqual(mod.global_trace_level, 5)
        mod.global_trace_level = orig

    def test_get_global_tracing_enabled(self):
        self.assertEqual(get_global_tracing_enabled(), mod.opentelemetry_initialized)

    def test_get_cur_time_ns(self):
        ts = mod.get_cur_time_ns()
        self.assertIsInstance(ts, int)
        self.assertGreater(ts, 0)


class TestDataclasses(unittest.TestCase):
    def test_trace_thread_info(self):
        info = TraceThreadInfo("host", 123, "label", 0, 1)
        self.assertEqual(info.thread_label, "label")

    def test_trace_event(self):
        evt = TraceEvent("name", 100, {"k": "v"})
        self.assertEqual(evt.event_name, "name")

    def test_trace_slice_context(self):
        s = TraceSliceContext("slice", 100, end_time_ns=200, level=2, attrs={"a": 1})
        self.assertEqual(s.slice_name, "slice")

    def test_trace_thread_context(self):
        info = TraceThreadInfo("h", 1, "l", 0, 0)
        ctx = TraceThreadContext(thread_info=info, cur_slice_stack=[])
        self.assertEqual(len(ctx.cur_slice_stack), 0)


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


class TestTraceCustomIdGenerator(unittest.TestCase):
    def test_generates_nonzero_ids(self):
        gen = TraceCustomIdGenerator()
        trace_id = gen.generate_trace_id()
        span_id = gen.generate_span_id()
        self.assertIsInstance(trace_id, int)
        self.assertIsInstance(span_id, int)


# __get_host_id
class TestGetHostId(unittest.TestCase):
    def test_from_machine_id_file(self):
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open",
            unittest.mock.mock_open(read_data="abc123\n"),
        ):
            self.assertEqual(_get_host_id(), "abc123")

    def test_from_machine_id_file_error(self):
        """Falls back to MAC address when file read fails."""
        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", side_effect=IOError("read error")
        ):
            result = _get_host_id()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_from_mac_address(self):
        with patch("os.path.exists", return_value=False), patch(
            "uuid.getnode", return_value=0x112233445566
        ):
            result = _get_host_id()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_unknown_fallback(self):
        with patch("os.path.exists", return_value=False), patch(
            "uuid.getnode", return_value=0
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

    def test_all_methods_noop(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start()
        ctx.trace_req_finish()
        ctx.trace_slice_start("s", 1)
        ctx.trace_slice_end("s", 1)
        ctx.trace_slice(TraceSliceContext("s", 100))
        ctx.trace_event("e", 1)
        ctx.trace_set_root_attrs({"k": "v"})
        ctx.trace_set_thread_attrs({"k": "v"})
        ctx.abort()
        ctx.rebuild_thread_context()

    def test_getstate_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        state = ctx.__getstate__()
        self.assertEqual(state, {"tracing_enable": False})

    def test_setstate_disabled(self):
        ctx = TraceReqContext.__new__(TraceReqContext)
        ctx.__setstate__({"tracing_enable": True, "is_copy": False})
        # opentelemetry_initialized is False → tracing forced off
        self.assertFalse(ctx.tracing_enable)

    def test_trace_set_thread_info_disabled(self):
        trace_set_thread_info("test_label")
        # Should not register anything


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestTraceReqContextEnabled(unittest.TestCase):
    def setUp(self):

        self.orig_initialized = mod.opentelemetry_initialized
        self.orig_tracer = mod.tracer
        self.orig_threads = mod.threads_info.copy()
        self.orig_level = mod.global_trace_level

        self.provider = TracerProvider()
        otel_trace.set_tracer_provider(self.provider)
        mod.opentelemetry_initialized = True
        mod.tracer = otel_trace.get_tracer("test")
        mod.global_trace_level = 3

    def tearDown(self):
        mod.opentelemetry_initialized = self.orig_initialized
        mod.tracer = self.orig_tracer
        mod.threads_info.clear()
        mod.threads_info.update(self.orig_threads)
        mod.global_trace_level = self.orig_level

    def test_trace_set_thread_info(self):
        trace_set_thread_info("scheduler", tp_rank=0, dp_rank=0)

        pid = threading.get_native_id()
        self.assertIn(pid, mod.threads_info)
        self.assertEqual(mod.threads_info[pid].thread_label, "scheduler")

        # Second call for same thread is a no-op
        trace_set_thread_info("different_label")
        self.assertEqual(mod.threads_info[pid].thread_label, "scheduler")

    def test_full_lifecycle(self):
        """Start → slice_start → slice_end → finish."""
        ctx = TraceReqContext(rid="req-1", role="unified", module_name="test")
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

    def test_trace_req_finish_without_start(self):
        """finish without start is a no-op."""
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.root_span = None
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

    def test_trace_set_root_attrs(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_set_root_attrs({"model": "llama"})
        ctx.trace_req_finish(ts=2000)

    def test_trace_set_root_attrs_no_span(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.root_span = None
        ctx.trace_set_root_attrs({"model": "llama"})  # no crash

    def test_trace_set_thread_attrs(self):
        ctx = TraceReqContext(rid="req-1")
        ctx.trace_req_start(ts=1000)
        ctx.trace_set_thread_attrs({"batch_size": 32})
        ctx.trace_req_finish(ts=2000)

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
        abort_obj = _sb.BaseFinishReason()
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

        ctx2 = TraceReqContext.__new__(TraceReqContext)
        ctx2.__setstate__(state)
        self.assertTrue(ctx2.tracing_enable)
        self.assertTrue(ctx2.is_copy)
        self.assertIsNotNone(ctx2.root_span_context)

    def test_thread_context_with_tp_rank(self):
        """Covers tp_rank branch in __create_thread_context."""

        pid = threading.get_native_id()
        mod.threads_info[pid] = TraceThreadInfo(
            "host", pid, "sched", tp_rank=0, dp_rank=0
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
        ctx2 = TraceReqContext.__new__(TraceReqContext)
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


if __name__ == "__main__":
    unittest.main()
