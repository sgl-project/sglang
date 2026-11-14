import unittest
from types import SimpleNamespace
from unittest.mock import Mock

try:
    from opentelemetry import trace
except ImportError:
    trace = None

from sglang.srt.observability.prefill_admission_trace import record_admission_wait
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _parent(sampled=True):
    return trace.set_span_in_context(
        trace.NonRecordingSpan(
            trace.SpanContext(
                trace_id=1,
                span_id=2,
                is_remote=True,
                trace_flags=trace.TraceFlags(
                    trace.TraceFlags.SAMPLED if sampled else 0
                ),
            )
        )
    )


@unittest.skipIf(trace is None, "OpenTelemetry API not installed")
class TestAdmissionTrace(unittest.TestCase):
    def test_first_occurrence_of_each_reason_has_own_snapshot(self):
        ctx = SimpleNamespace(
            tracing_enable=True,
            trace_level=3,
            root_span_context=_parent(),
            trace_slice=Mock(),
        )
        req = SimpleNamespace(time_stats=SimpleNamespace(trace_ctx=ctx))
        record_admission_wait(req, "prefill_input_budget", remaining_input_tokens=20)
        record_admission_wait(req, "prefill_input_budget", remaining_input_tokens=10)
        record_admission_wait(req, "prefill_max_requests", max_requests=2)
        self.assertEqual(ctx.trace_slice.call_count, 2)
        first = ctx.trace_slice.call_args_list[0].args[0]
        self.assertEqual(first.attrs["remaining_input_tokens"], 20)
        self.assertEqual(first.start_time_ns, first.end_time_ns)
        self.assertEqual(first.events[0].attrs, first.attrs)
        self.assertEqual(first.events[0].event_name, "prefill_admission_rejection")

    def test_w3c_unsampled_parent_does_not_build_spans_or_dedup_state(self):
        ctx = SimpleNamespace(
            tracing_enable=True,
            trace_level=3,
            root_span_context=_parent(False),
            trace_slice=Mock(),
        )
        record_admission_wait(
            SimpleNamespace(time_stats=SimpleNamespace(trace_ctx=ctx)),
            "prefill_max_requests",
            max_requests=1,
        )
        ctx.trace_slice.assert_not_called()
        self.assertFalse(hasattr(ctx, "_prefill_admission_reasons"))

    def test_rejection_is_exported_even_without_a_later_queue_slice(self):
        from unittest.mock import patch

        try:
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError:
            self.skipTest("OpenTelemetry SDK not installed")
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from sglang.srt.observability import trace as implementation

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("admission-test")
        parent = tracer.start_span("request-thread")
        ctx = implementation.TraceReqContext.__new__(implementation.TraceReqContext)
        ctx.tracing_enable = True
        ctx.trace_level = 3
        ctx.root_span_context = trace.set_span_in_context(parent)
        ctx.thread_context = SimpleNamespace(thread_span=parent, cur_slice_stack=[])
        ctx.last_span_context = None
        ctx.events_cache = []
        try:
            with patch.object(implementation, "tracer", tracer):
                record_admission_wait(
                    SimpleNamespace(time_stats=SimpleNamespace(trace_ctx=ctx)),
                    "prefill_input_budget",
                    input_tokens=12,
                    remaining_input_tokens=10,
                )
            spans = exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0].name, "prefill_admission_wait")
            self.assertEqual(spans[0].parent.span_id, parent.get_span_context().span_id)
            self.assertEqual(
                spans[0].events[0].attributes["remaining_input_tokens"], 10
            )
            self.assertEqual(ctx.events_cache, [])
        finally:
            ctx.tracing_enable = False
            parent.end()
            provider.shutdown()

    def test_unsampled_request_has_no_state_or_spans(self):
        ctx = SimpleNamespace(tracing_enable=False, trace_slice=Mock())
        record_admission_wait(
            SimpleNamespace(time_stats=SimpleNamespace(trace_ctx=ctx)),
            "prefill_max_requests",
            max_requests=1,
        )
        ctx.trace_slice.assert_not_called()
        self.assertFalse(hasattr(ctx, "_prefill_admission_reasons"))


if __name__ == "__main__":
    unittest.main()
