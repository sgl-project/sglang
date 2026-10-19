"""Empty prefill reports must not overwrite the last published cache hit rate."""

import types
import unittest

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
    _prefill_cache_hit_rate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_reporter():
    """Reporter with only the state the hit-rate guard touches."""
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.stats = types.SimpleNamespace(cache_hit_rate=0.0)
    return reporter


class TestPrefillCacheHitRate(CustomTestCase):
    def test_ratio_when_tokens_were_prefilled(self):
        self.assertEqual(_prefill_cache_hit_rate(96, 100), 0.96)

    def test_no_cache_hits_is_a_real_zero(self):
        # 100 tokens prefilled and none cached is a genuine 0% — still published.
        self.assertEqual(_prefill_cache_hit_rate(0, 100), 0.0)

    def test_report_without_prefill_tokens_has_no_rate(self):
        self.assertIsNone(_prefill_cache_hit_rate(0, 0))

    def test_negative_totals_have_no_rate(self):
        # effective tokens are differences and may go negative; never publish.
        self.assertIsNone(_prefill_cache_hit_rate(-5, -2))


class TestPrefillCacheHitRateReporting(CustomTestCase):
    """The guard itself: an empty report must leave the published rate alone."""

    def test_empty_report_keeps_last_valid_rate(self):
        reporter = _make_reporter()
        reporter._record_prefill_cache_hit_rate(96, 100)
        self.assertEqual(reporter.stats.cache_hit_rate, 0.96)

        # An empty report has nothing to measure and must not reset the gauge.
        reporter._record_prefill_cache_hit_rate(0, 0)
        self.assertEqual(reporter.stats.cache_hit_rate, 0.96)

    def test_genuine_zero_hit_report_is_published(self):
        reporter = _make_reporter()
        reporter._record_prefill_cache_hit_rate(96, 100)
        reporter._record_prefill_cache_hit_rate(0, 100)
        self.assertEqual(reporter.stats.cache_hit_rate, 0.0)

        # A later empty report must not error or resurrect an older value.
        reporter._record_prefill_cache_hit_rate(0, 0)
        self.assertEqual(reporter.stats.cache_hit_rate, 0.0)

    def test_fully_cached_report_publishes_one(self):
        # Every prefilled token was a hit: effective input is zero but total is not.
        reporter = _make_reporter()
        reporter._record_prefill_cache_hit_rate(100, 100)
        self.assertEqual(reporter.stats.cache_hit_rate, 1.0)

    def test_reprocessed_only_report_keeps_previous_rate(self):
        # A fully reprocessed request cancels out of the effective totals, leaving
        # both at zero — the same "nothing to measure" case as an empty report.
        reporter = _make_reporter()
        reporter._record_prefill_cache_hit_rate(80, 100)
        reporter._record_prefill_cache_hit_rate(0, 0)
        self.assertEqual(reporter.stats.cache_hit_rate, 0.8)


if __name__ == "__main__":
    unittest.main()
