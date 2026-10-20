import pickle
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import multi_tokenizer_mixin as mod
from sglang.srt.observability import trace as trace_mod

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMultiTokenizerTracing(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(trace_mod, "opentelemetry_initialized", False))
        self.thread = MagicMock()
        stack.enter_context(
            patch.object(
                mod, "threading", SimpleNamespace(Thread=Mock(return_value=self.thread))
            )
        )
        for target in (
            "zmq.asyncio.Context",
            "get_zmq_socket",
            "asyncio.new_event_loop",
            "asyncio.run_coroutine_threadsafe",
            "start_disagg_service",
            "SocketMapping",
        ):
            stack.enter_context(patch(f"{mod.__name__}.{target}"))
        stack.enter_context(patch.object(mod, "zmq_reader_owner", return_value=False))
        for method in ("router_worker_obj", "handle_loop"):
            stack.enter_context(
                patch.object(mod.MultiTokenizerRouter, method, new=Mock())
            )
        self.server_args = SimpleNamespace(
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
            trace_modules="request",
        )
        self.port_args = SimpleNamespace(
            tokenizer_ipc_name="tokenizer",
            scheduler_input_ipc_name="scheduler",
            tokenizer_worker_ipc_name="workers",
        )

    @unittest.skipUnless(
        trace_mod.opentelemetry_imported, "OpenTelemetry is not installed"
    )
    def test_trace_context_survives_router_forwarding(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        provider = TracerProvider()
        self.addCleanup(provider.shutdown)
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("router-tracing-test")

        def initialize(*args, **kwargs):
            trace_mod.opentelemetry_initialized = True

        def receive_and_forward():
            # Simulate worker -> router -> scheduler with real serialization.
            forwarded = pickle.dumps(pickle.loads(payload))
            received = pickle.loads(forwarded)
            self.assertTrue(received.tracing_enable)
            received.rebuild_thread_context()
            try:
                for name in ("prefill_forward", "decode_forward"):
                    received.trace_slice_start(name, level=1)
                    received.trace_slice_end(name, level=1)
            finally:
                received.abort()

        with (
            patch.object(trace_mod, "tracer", tracer),
            patch.object(trace_mod, "global_trace_modules", None),
            patch.dict(trace_mod.threads_info, clear=True),
            patch.object(mod, "process_tracing_init", side_effect=initialize) as init,
        ):
            # The tokenizer worker has tracing initialized in its own process.
            with patch.object(trace_mod, "opentelemetry_initialized", True):
                request = trace_mod.TraceReqContext("router-tracing-test")
                request.trace_req_start()
                self.addCleanup(request.trace_req_finish)
                payload = pickle.dumps(request)

            # The parent/router process has not initialized tracing yet.
            self.assertFalse(trace_mod.get_global_tracing_enabled())
            self.thread.start.side_effect = receive_and_forward
            mod.MultiTokenizerRouter(self.server_args, self.port_args)

            init.assert_called_once_with(
                "localhost:4317", "sglang", trace_modules="request"
            )
            names = {span.name for span in exporter.get_finished_spans()}
            self.assertTrue({"prefill_forward", "decode_forward"}.issubset(names))

    def test_disabled_tracing_does_not_initialize_router(self):
        self.server_args.enable_trace = False
        with patch.object(mod, "process_tracing_init") as init:
            mod.MultiTokenizerRouter(self.server_args, self.port_args)
            init.assert_not_called()

    def test_already_initialized_tracing_is_reused(self):
        with (
            patch.object(trace_mod, "opentelemetry_initialized", True),
            patch.object(mod, "process_tracing_init") as init,
        ):
            mod.MultiTokenizerRouter(self.server_args, self.port_args)
            init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
