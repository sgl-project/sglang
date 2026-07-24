"""Pure-CPU unit tests for the tokenizer event-loop-lag metric.

Covers the two halves of ``sglang:event_loop_lag_seconds``:

* ``TokenizerMetricsCollector`` registers the histogram and
  ``observe_event_loop_lag`` routes the value through the collector's labels.
* ``TokenizerManager.watch_event_loop_lag`` records a lag sample roughly equal
  to how long the loop was blocked -- the real failure mode this metric exists
  to surface (a starved loop that cannot promptly stamp ``received_time``).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import time
import unittest


class _RecordingCollector:
    def __init__(self):
        self.lags = []

    def observe_event_loop_lag(self, lag):
        self.lags.append(lag)


class _FakeCounter:
    def __init__(self, name, documentation, labelnames):
        pass


class _FakeHistogram:
    by_name = {}

    def __init__(self, name, documentation, labelnames, buckets):
        self.name = name
        self.buckets = buckets
        self.observed = []
        self.last_labels = None
        _FakeHistogram.by_name[name] = self

    def labels(self, **kwargs):
        self.last_labels = kwargs
        return self

    def observe(self, value):
        self.observed.append(value)


class _StubServerArgs:
    """Minimal ServerArgs stand-in for constructing the collector.

    Only the two bucket-rule attributes are read in ``__init__``; ``None`` makes
    ``generate_buckets`` fall back to its defaults.
    """

    prompt_tokens_buckets = None
    generation_tokens_buckets = None


class TestEventLoopLagMetricWiring(unittest.TestCase):
    def test_metric_registered_and_routes_through_labels(self):
        from sglang.srt.observability.metrics_collector import (
            TokenizerMetricsCollector,
        )

        _FakeHistogram.by_name = {}

        class _DICollector(TokenizerMetricsCollector):
            _counter_cls = _FakeCounter
            _histogram_cls = _FakeHistogram

        labels = {"model_name": "m", "engine_type": "unified"}
        collector = _DICollector(server_args=_StubServerArgs(), labels=labels)

        self.assertIn("sglang:event_loop_lag_seconds", _FakeHistogram.by_name)
        hist = _FakeHistogram.by_name["sglang:event_loop_lag_seconds"]
        # Sub-millisecond floor through multi-second ceiling.
        self.assertEqual(hist.buckets[0], 0.0005)
        self.assertEqual(hist.buckets[-1], 10.0)

        collector.observe_event_loop_lag(0.7)
        self.assertEqual(hist.observed, [0.7])
        # Process-wide metric carries the collector's own labels.
        self.assertEqual(hist.last_labels, labels)


class TestWatchEventLoopLag(unittest.TestCase):
    def test_blocked_loop_records_lag(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        # Bypass the heavy __init__; the monitor only touches metrics_collector.
        tm = TokenizerManager.__new__(TokenizerManager)
        collector = _RecordingCollector()
        tm.metrics_collector = collector

        async def scenario():
            task = asyncio.create_task(tm.watch_event_loop_lag(interval=0.02))
            await asyncio.sleep(0.06)  # a few healthy ticks (lag ~0)
            time.sleep(0.4)  # synchronously block the loop
            await asyncio.sleep(0.06)  # let the overdue tick land
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())

        self.assertGreaterEqual(len(collector.lags), 2)
        self.assertTrue(
            any(lag >= 0.3 for lag in collector.lags),
            f"expected a >=0.3s lag sample from the 0.4s block, got {collector.lags}",
        )


if __name__ == "__main__":
    unittest.main()
