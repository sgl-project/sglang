import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _PrefixCachingMetrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestPrefixCachingMetrics(CustomTestCase):
    """Sliding window over the most recent admitted requests (vLLM CachingMetrics)."""

    def test_hit_rate_is_token_weighted(self):
        window = _PrefixCachingMetrics()
        window.observe(requests=2, query_tokens=100, hit_tokens=50)
        window.observe(requests=1, query_tokens=300, hit_tokens=0)
        self.assertEqual(window.requests, 3)
        self.assertAlmostEqual(window.hit_rate, 50 / 400)

    def test_evicts_oldest_samples(self):
        window = _PrefixCachingMetrics(max_recent_requests=3)
        window.observe(requests=2, query_tokens=200, hit_tokens=200)
        window.observe(requests=2, query_tokens=200, hit_tokens=0)
        # 4 requests > 3: the first sample is dropped whole, never split.
        self.assertEqual(window.requests, 2)
        self.assertEqual(window.hit_rate, 0.0)

    def test_keeps_newest_sample_even_when_over_window(self):
        window = _PrefixCachingMetrics(max_recent_requests=1)
        window.observe(requests=5, query_tokens=100, hit_tokens=10)
        window.observe(requests=5, query_tokens=100, hit_tokens=90)
        self.assertEqual(window.requests, 5)
        self.assertAlmostEqual(window.hit_rate, 0.9)

    def test_empty_samples_are_skipped(self):
        window = _PrefixCachingMetrics()
        window.observe(requests=0, query_tokens=0, hit_tokens=0)
        self.assertTrue(window.empty)
        self.assertEqual(window.hit_rate, 0.0)

    def test_reset(self):
        window = _PrefixCachingMetrics()
        window.observe(requests=1, query_tokens=10, hit_tokens=5)
        window.reset()
        self.assertTrue(window.empty)


class TestPrefixCacheSampling(CustomTestCase):
    """Which admissions feed the window."""

    def _adder(self, enable: bool):
        from sglang.srt.managers.schedule_policy import PrefillAdder

        # Bypass __init__: it needs a live scheduler and tree cache.
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.enable_prefix_cache_hit_rate_window = enable
        adder.log_cache_num_reqs = 0
        adder.log_cache_query_tokens = 0
        adder.log_cache_hit_tokens = 0
        adder.log_device_hit_tokens = 0
        adder.log_host_hit_tokens = 0
        adder.log_storage_hit_tokens = 0
        adder.tree_cache = SimpleNamespace(
            finish_storage_prefetch_admission=lambda *a, **k: None,
            discard_storage_prefetch_accounting=lambda *a, **k: None,
        )
        return adder

    def _req(self, retracted: bool, prompt_len: int = 100):
        return SimpleNamespace(
            retracted_stain=retracted,
            full_untruncated_fill_ids=[0] * prompt_len,
            materialized_host_hit_len=lambda: 0,
            storage_hit_length=0,
            storage_hit_start=0,
            host_hit_is_storage=False,
            fulfilled_storage_hit_len=lambda prefix_len: 0,
            needs_host_load_back=lambda: False,
            host_loaded_length=0,
            host_hit_length=0,
            cache_request_handle=None,
        )

    def test_nothing_is_recorded_when_the_flag_is_off(self):
        adder = self._adder(enable=False)
        adder._record_prefix_cache_sample(req=self._req(retracted=False), hit_tokens=40)
        self.assertEqual(adder.log_cache_num_reqs, 0)
        self.assertEqual(adder.log_cache_query_tokens, 0)
        self.assertEqual(adder.log_cache_hit_tokens, 0)

    def test_retracted_requests_are_excluded(self):
        # vLLM routes preempted requests to separate counters instead of the
        # window, so counting them here would report re-done work as a hit.
        adder = self._adder(enable=True)
        adder._account_prefill_cache_admission(
            req=self._req(retracted=False), prefix_len=40
        )
        self.assertEqual(adder.log_cache_num_reqs, 1)
        self.assertEqual(adder.log_cache_query_tokens, 100)
        self.assertEqual(adder.log_cache_hit_tokens, 40)

        adder._account_prefill_cache_admission(
            req=self._req(retracted=True), prefix_len=40
        )
        self.assertEqual(adder.log_cache_num_reqs, 1)


if __name__ == "__main__":
    unittest.main()
