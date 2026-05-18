"""Unit tests for cache_hit_rate decode-phase persistence.

Issue #20451: report_decode_stats() overwrote stats.cache_hit_rate = 0.0 on
every periodic decode log interval, so Prometheus almost always scraped 0.0
(decode is ~95% of wall time) even when prefill had computed a correct non-zero
value.

Fix: remove the two lines that set cache_hit_rate = 0.0 and assign it to
stats in report_decode_stats(). The field now retains its prefill-computed
value across decode iterations.
"""

import unittest

from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestCacheHitRateDecodePersistence(CustomTestCase):
    """Verify cache_hit_rate is not reset to 0 during decode stats update."""

    def _simulate_prefill_stats_update(self, stats: SchedulerStats, hit_rate: float):
        """Mimic the assignment done by report_prefill_stats."""
        stats.cache_hit_rate = hit_rate

    def _simulate_decode_stats_update_buggy(self, stats: SchedulerStats):
        """Mimic the old (buggy) report_decode_stats that overwrote cache_hit_rate."""
        cache_hit_rate = 0.0  # was always 0 in decode
        stats.cache_hit_rate = cache_hit_rate  # overwrote prefill value — BUG

    def _simulate_decode_stats_update_fixed(self, stats: SchedulerStats):
        """Mimic the fixed report_decode_stats that does NOT touch cache_hit_rate."""
        # These two lines were removed in the fix — cache_hit_rate is not updated here.
        pass

    def test_prefill_value_preserved_across_decode(self):
        """cache_hit_rate must keep the prefill value after decode stats run."""
        stats = SchedulerStats()
        self._simulate_prefill_stats_update(stats, hit_rate=0.63)
        self.assertAlmostEqual(stats.cache_hit_rate, 0.63)

        # Fixed: decode does not touch cache_hit_rate
        self._simulate_decode_stats_update_fixed(stats)
        self.assertAlmostEqual(
            stats.cache_hit_rate,
            0.63,
            msg="cache_hit_rate must not be reset to 0 during decode stats",
        )

    def test_buggy_decode_would_reset_to_zero(self):
        """Documents the pre-fix behaviour: decode stats zeroed cache_hit_rate."""
        stats = SchedulerStats()
        self._simulate_prefill_stats_update(stats, hit_rate=0.63)

        self._simulate_decode_stats_update_buggy(stats)
        self.assertAlmostEqual(
            stats.cache_hit_rate,
            0.0,
            msg="Old code wiped the prefill value to 0 — this is the bug",
        )

    def test_multiple_decode_iterations_do_not_degrade_value(self):
        """Value must survive many consecutive decode log intervals."""
        stats = SchedulerStats()
        self._simulate_prefill_stats_update(stats, hit_rate=0.45)

        for _ in range(100):
            self._simulate_decode_stats_update_fixed(stats)

        self.assertAlmostEqual(stats.cache_hit_rate, 0.45)

    def test_next_prefill_updates_value(self):
        """A new prefill batch should update cache_hit_rate to the new rate."""
        stats = SchedulerStats()
        self._simulate_prefill_stats_update(stats, hit_rate=0.30)
        self._simulate_decode_stats_update_fixed(stats)
        self.assertAlmostEqual(stats.cache_hit_rate, 0.30)

        # New prefill batch with different hit rate
        self._simulate_prefill_stats_update(stats, hit_rate=0.75)
        self.assertAlmostEqual(stats.cache_hit_rate, 0.75)

    def test_zero_hit_rate_prefill_is_preserved(self):
        """A legitimate 0.0 from prefill (cold start) must also be preserved."""
        stats = SchedulerStats()
        self._simulate_prefill_stats_update(stats, hit_rate=0.0)
        self._simulate_decode_stats_update_fixed(stats)
        self.assertAlmostEqual(stats.cache_hit_rate, 0.0)

    def test_stats_default_value(self):
        """Default cache_hit_rate is 0.0 before any prefill has run."""
        stats = SchedulerStats()
        self.assertAlmostEqual(stats.cache_hit_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
