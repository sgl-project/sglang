"""Check request-local timing with nested and overlapping operations."""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from unittest.mock import patch

from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.observability.frontend_request_timing import (
    FrontendRequestTiming,
    FrontendTimingGroup,
    bind_frontend_timing,
    current_frontend_timing,
)
from sglang.srt.observability.tokenizer_encode_metrics import measure_tokenizer_encode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFrontendRequestTiming(CustomTestCase):
    def test_multimodal_executor_keeps_request_context_isolated(self):
        first = FrontendRequestTiming(started_at=10.0)
        second = FrontendRequestTiming(started_at=10.0)
        executor = MultimodalProcessorExecutor(lambda: {}, max_workers=1)

        def encode(duration, *, processor):
            timing = current_frontend_timing.get()
            if timing is not None:
                timing.add_span("text", 10.0, 10.0 + duration)
            return timing

        async def submit(timing, duration):
            with bind_frontend_timing(timing):
                return await executor.run(encode, duration)

        async def run():
            results = await asyncio.gather(submit(first, 0.1), submit(second, 0.2))
            self.assertEqual(results, [first, second])
            self.assertIsNone(await executor.run(encode, 0.5))

        try:
            asyncio.run(run())
        finally:
            executor.shutdown()
        self.assertAlmostEqual(first.snapshot(11.0)["text"], 0.1)
        self.assertAlmostEqual(second.snapshot(11.0)["text"], 0.2)

    def test_batch_forks_keep_shared_work_and_isolate_later_spans(self):
        parent = FrontendRequestTiming(started_at=10.0)
        parent.add_span("text", 10.0, 10.1)
        first, second = parent.fork(), parent.fork()
        FrontendTimingGroup([first, second]).add_span("text", 10.2, 10.4)
        first.add_span("multimodal", 10.5, 10.7)
        self.assertAlmostEqual(first.snapshot(11.0)["text"], 0.3)
        self.assertAlmostEqual(second.snapshot(11.0)["text"], 0.3)
        self.assertNotIn("multimodal", second.snapshot(11.0))
        self.assertAlmostEqual(parent.snapshot(11.0)["text"], 0.1)

    def test_encode_span_follows_request_into_executor(self):
        request = FrontendRequestTiming(started_at=10.0)

        class Collector:
            def observe_tokenizer_encode(self, *args, **kwargs):
                pass

        def encode():
            with measure_tokenizer_encode(Collector(), source="chat"):
                pass

        with patch(
            "sglang.srt.observability.tokenizer_encode_metrics.time.perf_counter",
            side_effect=[10.2, 10.4],
        ):
            with (
                bind_frontend_timing(request),
                ThreadPoolExecutor(max_workers=1) as pool,
            ):
                pool.submit(copy_context().run, encode).result()
        self.assertIsNone(current_frontend_timing.get())
        self.assertAlmostEqual(request.snapshot(11.0)["text"], 0.2)

    def test_request_span_union_preserves_cross_stage_overlap(self):
        request = FrontendRequestTiming(started_at=10.0)
        request.add_span("text", 10.2, 10.5)
        request.add_span("text", 10.4, 10.6)
        request.add_span("multimodal", 10.3, 10.8)
        durations = request.snapshot(11.0)
        self.assertAlmostEqual(durations["total"], 1.0)
        self.assertAlmostEqual(durations["text"], 0.4)
        self.assertAlmostEqual(durations["multimodal"], 0.5)

    def test_spans_are_clipped_to_request_lifetime(self):
        request = FrontendRequestTiming(started_at=1.0)
        request.add_span("text", 0.0, 1.5)
        request.add_span("text", 2.5, 4.0)
        self.assertEqual(request.snapshot(3.0)["text"], 1.0)

    def test_rejects_invalid_intervals(self):
        request = FrontendRequestTiming(started_at=1.0)
        with self.assertRaises(ValueError):
            request.add_span("text", 2.0, 1.0)
        with self.assertRaises(ValueError):
            request.snapshot(0.5)


if __name__ == "__main__":
    unittest.main()
