from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.observability.metrics_collector import SchedulerStats

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _stats_for(pool_stats: PoolStats) -> SchedulerStats:
    stats = SchedulerStats()
    pool_stats.update_scheduler_stats(stats)
    return stats


def _plain(full_usage: float) -> PoolStats:
    return PoolStats(
        full_num_used=int(full_usage * 1000),
        full_token_usage=full_usage,
        full_available_size=1000 - int(full_usage * 1000),
        full_evictable_size=0,
    )


def _hybrid_swa(full_usage: float, swa_usage: float) -> PoolStats:
    return PoolStats(
        full_num_used=int(full_usage * 1000),
        full_token_usage=full_usage,
        full_available_size=1000 - int(full_usage * 1000),
        full_evictable_size=0,
        is_hybrid_swa=True,
        swa_num_used=int(swa_usage * 1000),
        swa_token_usage=swa_usage,
        swa_available_size=1000 - int(swa_usage * 1000),
        swa_evictable_size=0,
    )


def _with_mamba(pool_stats: PoolStats, mamba_usage: float) -> PoolStats:
    pool_stats.is_hybrid_ssm = True
    pool_stats.mamba_num_used = int(mamba_usage * 100)
    pool_stats.mamba_usage = mamba_usage
    pool_stats.mamba_available_size = 100 - int(mamba_usage * 100)
    pool_stats.mamba_evictable_size = 0
    return pool_stats


class TestKvCacheUsagePercSemantics(CustomTestCase):
    """kv_cache_usage_perc is max(full, swa) with the Mamba pool excluded;
    aliasing it to token_usage would report Mamba pressure as KV usage."""

    def test_plain_equals_full_usage(self):
        stats = _stats_for(_plain(0.72))
        self.assertEqual(stats.kv_cache_usage_perc, 0.72)
        self.assertEqual(stats.token_usage, 0.72)

    def test_hybrid_swa_full_dominant(self):
        stats = _stats_for(_hybrid_swa(0.5, 0.3))
        self.assertEqual(stats.kv_cache_usage_perc, 0.5)

    def test_hybrid_swa_swa_dominant(self):
        stats = _stats_for(_hybrid_swa(0.3, 0.5))
        self.assertEqual(stats.kv_cache_usage_perc, 0.5)

    def test_hybrid_ssm_excludes_mamba(self):
        # Turns red if the gauge is ever re-aliased to stats.token_usage.
        stats = _stats_for(_with_mamba(_plain(0.5), 0.9))
        self.assertEqual(stats.kv_cache_usage_perc, 0.5)
        self.assertEqual(stats.token_usage, 0.9)
        self.assertNotEqual(stats.kv_cache_usage_perc, stats.token_usage)

    def test_hybrid_swa_ssm_excludes_mamba(self):
        stats = _stats_for(_with_mamba(_hybrid_swa(0.5, 0.3), 0.9))
        self.assertEqual(stats.kv_cache_usage_perc, 0.5)
        self.assertEqual(stats.token_usage, 0.9)

    def test_unrounded(self):
        stats = _stats_for(_plain(0.123456))
        self.assertEqual(stats.kv_cache_usage_perc, 0.123456)


if __name__ == "__main__":
    import unittest

    unittest.main()
