"""Unit tests for DiffGenerator OTEL tracing integration — no server, no GPU."""

import pickle
import unittest

import sglang.srt.observability.trace as trace_mod
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    _has_otel = True
except ImportError:
    _has_otel = False


class TestTraceNullContext(unittest.TestCase):
    """TraceNullContext is the default when tracing is disabled."""

    def test_noop_calls(self):
        """All tracing method calls are no-ops and don't error."""
        ctx = TraceNullContext()
        self.assertFalse(ctx.tracing_enable)
        ctx.rebuild_thread_context()
        ctx.trace_slice_start("test", level=1)
        ctx.trace_slice_end("test", level=1)
        ctx.trace_req_start()
        ctx.trace_req_finish()

    def test_chaining(self):
        ctx = TraceNullContext()
        self.assertIs(ctx.foo.bar.baz(1, 2, 3), ctx)

    def test_picklable(self):
        ctx = TraceNullContext()
        restored = pickle.loads(pickle.dumps(ctx))
        self.assertIsInstance(restored, TraceNullContext)
        self.assertFalse(restored.tracing_enable)


class TestTraceReqContextDisabled(unittest.TestCase):
    """TraceReqContext when OTEL is not initialized."""

    def setUp(self):
        self.orig = trace_mod.opentelemetry_initialized
        trace_mod.opentelemetry_initialized = False

    def tearDown(self):
        trace_mod.opentelemetry_initialized = self.orig

    def test_tracing_disabled(self):
        ctx = TraceReqContext(rid="req-1", module_name="diffusion")
        self.assertFalse(ctx.tracing_enable)

    def test_all_methods_noop(self):
        ctx = TraceReqContext(rid="req-1", module_name="diffusion")
        ctx.trace_req_start()
        ctx.trace_slice_start("scheduler_dispatch", level=1)
        ctx.trace_slice_end("scheduler_dispatch", level=1)
        ctx.rebuild_thread_context()
        ctx.trace_req_finish()

    def test_getstate_disabled(self):
        ctx = TraceReqContext(rid="req-1")
        state = ctx.__getstate__()
        self.assertEqual(state, {"tracing_enable": False})

    def test_pickle_round_trip_disabled(self):
        ctx = TraceReqContext(
            rid="req-1",
            module_name="diffusion",
            external_trace_header={
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            },
        )
        data = pickle.dumps(ctx)
        restored = pickle.loads(data)
        self.assertFalse(restored.tracing_enable)


@unittest.skipUnless(_has_otel, "opentelemetry not installed")
class TestTraceReqContextEnabled(unittest.TestCase):
    """TraceReqContext with OTEL initialized — full span lifecycle."""

    def setUp(self):
        import threading

        from sglang.srt.observability.trace import TraceThreadInfo

        self.orig_initialized = trace_mod.opentelemetry_initialized
        self.orig_tracer = trace_mod.tracer
        self.orig_threads = trace_mod.threads_info.copy()
        self.orig_level = trace_mod.global_trace_level

        self.provider = TracerProvider()
        otel_trace.set_tracer_provider(self.provider)
        trace_mod.opentelemetry_initialized = True
        trace_mod.tracer = otel_trace.get_tracer("test")
        trace_mod.global_trace_level = 3

        pid = threading.get_native_id()
        trace_mod.threads_info[pid] = TraceThreadInfo(
            "host", pid, "DiffWorker", tp_rank=None, dp_rank=None
        )

    def tearDown(self):
        trace_mod.opentelemetry_initialized = self.orig_initialized
        trace_mod.tracer = self.orig_tracer
        trace_mod.threads_info.clear()
        trace_mod.threads_info.update(self.orig_threads)
        trace_mod.global_trace_level = self.orig_level

    def test_full_diffusion_lifecycle(self):
        """Simulates the complete trace flow: start → scheduler → gpu_forward → finish."""
        ctx = TraceReqContext(
            rid="diff-req-1",
            module_name="diffusion",
            external_trace_header={
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            },
        )
        # prepare_request calls trace_req_start
        ctx.trace_req_start(ts=1000)
        self.assertIsNotNone(ctx.root_span)
        self.assertIsNotNone(ctx.thread_context)

        # Simulate scheduler: rebuild after pickle, dispatch span
        ctx.rebuild_thread_context(ts=2000)
        ctx.trace_slice_start("scheduler_dispatch", level=1, ts=2000)

        # GPU worker: gpu_forward span nested inside
        ctx.trace_slice_start("gpu_forward", level=2, ts=3000)
        ctx.trace_slice_end("gpu_forward", level=2, ts=4000)

        ctx.trace_slice_end(
            "scheduler_dispatch", level=1, ts=5000, thread_finish_flag=True
        )

        # DiffGenerator.generate() finishes the trace
        ctx.trace_req_finish(ts=6000)
        self.assertIsNone(ctx.root_span)

    def test_external_trace_header_links_parent(self):
        """Verify external trace context is extracted and used."""
        ctx = TraceReqContext(
            rid="diff-req-2",
            module_name="diffusion",
            external_trace_header={
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            },
        )
        ctx.trace_req_start(ts=1000)
        # Root span should exist and be linked to external context
        self.assertIsNotNone(ctx.root_span)
        self.assertIsNotNone(ctx.root_span_context)
        ctx.trace_req_finish(ts=2000)

    def test_pickle_round_trip_preserves_trace(self):
        """Simulates ZMQ pickle transport between HTTP server and scheduler."""
        ctx = TraceReqContext(
            rid="diff-req-3",
            module_name="diffusion",
            external_trace_header={
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            },
        )
        ctx.trace_req_start(ts=1000)

        # Pickle (simulates ZMQ send_pyobj)
        data = pickle.dumps(ctx)
        restored = pickle.loads(data)

        self.assertTrue(restored.tracing_enable)
        self.assertTrue(restored.is_copy)
        self.assertIsNotNone(restored.root_span_context)

        # Scheduler rebuilds thread context on deserialized copy
        restored.rebuild_thread_context(ts=2000)
        restored.trace_slice_start("scheduler_dispatch", level=1, ts=2000)
        restored.trace_slice_end(
            "scheduler_dispatch", level=1, ts=3000, thread_finish_flag=True
        )

        # Clean up original
        ctx.trace_req_finish(ts=4000)

    def test_abort_on_error(self):
        """Verify abort cleans up spans when generation fails."""
        ctx = TraceReqContext(rid="diff-req-4", module_name="diffusion")
        ctx.trace_req_start(ts=1000)
        ctx.trace_slice_start("scheduler_dispatch", level=1, ts=2000)
        ctx.trace_slice_start("gpu_forward", level=2, ts=3000)

        # Simulate exception during gpu_forward
        ctx.abort(ts=3500)
        self.assertIsNone(ctx.thread_context)

        ctx.trace_req_finish(ts=4000)


class TestDiffGeneratorSignature(unittest.TestCase):
    """Verify generate() accepts external_trace_header without importing heavy deps."""

    def test_generate_has_external_trace_header_param(self):
        # Import just the module source to inspect the signature,
        # avoiding the full multimodal_gen import chain
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "diffusion_generator",
            "python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py",
        )
        source = spec.loader.get_data(spec.origin).decode()
        self.assertIn("external_trace_header", source)
        self.assertIn("dict[str, str] | None", source)


class TestServerArgsTracing(unittest.TestCase):
    """Verify tracing fields exist on ServerArgs without full instantiation."""

    def test_server_args_has_tracing_fields(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "server_args",
            "python/sglang/multimodal_gen/runtime/server_args.py",
        )
        source = spec.loader.get_data(spec.origin).decode()
        self.assertIn("enable_trace: bool = False", source)
        self.assertIn("otlp_traces_endpoint: str =", source)
        self.assertIn("--enable-trace", source)
        self.assertIn("--otlp-traces-endpoint", source)


class TestPrepareRequestTracing(unittest.TestCase):
    """Verify prepare_request accepts external_trace_header."""

    def test_prepare_request_has_trace_param(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "utils",
            "python/sglang/multimodal_gen/runtime/entrypoints/utils.py",
        )
        source = spec.loader.get_data(spec.origin).decode()
        self.assertIn("external_trace_header", source)
        self.assertIn("TraceReqContext", source)


if __name__ == "__main__":
    unittest.main()
