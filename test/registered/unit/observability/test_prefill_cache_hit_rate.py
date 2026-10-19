"""The prefill cache-hit-rate gauge must never be published from an empty report."""

import unittest

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _prefill_cache_hit_rate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCacheHitRate(CustomTestCase):
    def test_ratio_when_tokens_were_prefilled(self):
        self.assertEqual(_prefill_cache_hit_rate(96, 100), 0.96)

    def test_no_cache_hits_is_a_real_zero(self):
        # 100 tokens prefilled and none cached is a genuine 0% — still published.
        self.assertEqual(_prefill_cache_hit_rate(0, 100), 0.0)

    def test_report_without_prefill_tokens_publishes_nothing(self):
        # total == 0 cannot be divided; returning None keeps the last known value
        # instead of exporting a fabricated 0.0 that reads as a cache miss.
        self.assertIsNone(_prefill_cache_hit_rate(0, 0))

    def test_negative_totals_are_ignored(self):
        # effective tokens are differences and may go negative; never publish.
        self.assertIsNone(_prefill_cache_hit_rate(-5, -2))


if __name__ == "__main__":
    unittest.main()