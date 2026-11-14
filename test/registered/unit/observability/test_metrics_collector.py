"""Unit tests for the tokenizer metrics collector's defaults: no server, no model."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from typing import ClassVar

from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.test_utils import CustomTestCase


class _RecordingMetric:
    """Keeps the constructor arguments, so a test can read the buckets."""

    instances: ClassVar[list] = []

    def __init__(self, *args, name=None, labelnames=(), buckets=None, **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.buckets = buckets
        _RecordingMetric.instances.append(self)

    def labels(self, *values, **labels):
        return self


class _RecordingTokenizerMetricsCollector(TokenizerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


def _collector(**kwargs):
    # The constructor reads the published config; a unit test has none.
    with get_context().override_server_args(
        prompt_tokens_buckets=None, generation_tokens_buckets=None
    ):
        return _RecordingTokenizerMetricsCollector(labels={"model_name": "m"}, **kwargs)


class TestDefaultBuckets(CustomTestCase):
    def setUp(self):
        _RecordingMetric.instances.clear()
        _collector()

    def tearDown(self):
        reset_context()

    def _buckets(self, name):
        return next(m.buckets for m in _RecordingMetric.instances if m.name == name)

    def test_ttft_buckets_resolve_fast_first_tokens(self):
        # A short prompt on a current GPU returns its first token in tens of
        # milliseconds; the lowest buckets have to sit below 0.1 s to tell
        # 20 ms from 90 ms.
        buckets = self._buckets("sglang:time_to_first_token_seconds")
        self.assertEqual(buckets[:6], [0.01, 0.02, 0.04, 0.06, 0.08, 0.1])
        self.assertEqual(buckets, sorted(buckets))

    def test_ttft_buckets_keep_the_long_tail(self):
        buckets = self._buckets("sglang:time_to_first_token_seconds")
        self.assertIn(1, buckets)
        self.assertIn(400, buckets)

    def test_explicit_buckets_are_used_as_given(self):
        _RecordingMetric.instances.clear()
        _collector(bucket_time_to_first_token=[0.5, 1, 2])
        self.assertEqual(
            self._buckets("sglang:time_to_first_token_seconds"), [0.5, 1, 2]
        )


if __name__ == "__main__":
    unittest.main()
