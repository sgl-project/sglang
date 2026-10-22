import argparse
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, prompt_len: int, prefix_len: int):
        self.full_untruncated_fill_ids = [0] * prompt_len
        self.prefix_len = prefix_len


class TestPrefixCachingMetrics(CustomTestCase):
    """Sliding window of the most recent requests (vLLM CachingMetrics semantics)."""

    def _window(self, max_recent_requests: int = 1000):
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            _PrefixCachingMetrics,
        )

        return _PrefixCachingMetrics(max_recent_requests=max_recent_requests)

    def test_hit_rate_is_token_weighted(self):
        window = self._window()
        window.observe(requests=2, query_tokens=100, hit_tokens=50)
        window.observe(requests=1, query_tokens=300, hit_tokens=0)
        self.assertEqual(window.requests, 3)
        self.assertAlmostEqual(window.hit_rate, 50 / 400)

    def test_evicts_oldest_samples(self):
        window = self._window(max_recent_requests=3)
        window.observe(requests=2, query_tokens=200, hit_tokens=200)
        window.observe(requests=2, query_tokens=200, hit_tokens=0)
        # 4 requests > 3: the first sample is dropped entirely, keeping a whole
        # sample rather than a partial one.
        self.assertEqual(window.requests, 2)
        self.assertEqual(window.hit_rate, 0.0)

    def test_keeps_newest_sample_even_when_over_window(self):
        window = self._window(max_recent_requests=1)
        window.observe(requests=5, query_tokens=100, hit_tokens=10)
        window.observe(requests=5, query_tokens=100, hit_tokens=90)
        self.assertEqual(window.requests, 5)
        self.assertAlmostEqual(window.hit_rate, 0.9)

    def test_empty_samples_are_skipped(self):
        window = self._window()
        window.observe(requests=0, query_tokens=0, hit_tokens=0)
        self.assertEqual(window.requests, 0)
        self.assertEqual(window.hit_rate, 0.0)

    def test_reset(self):
        window = self._window()
        window.observe(requests=1, query_tokens=10, hit_tokens=5)
        window.reset()
        self.assertEqual(window.requests, 0)
        self.assertEqual(window.hit_rate, 0.0)


class TestCacheHitRateFlag(CustomTestCase):
    """--enable-cache-hit-rate is off by default and gates the sampling."""

    def test_flag_defaults_to_off(self):
        from sglang.srt.arg_groups.fields.observability import Observability

        self.assertFalse(Observability().enable_cache_hit_rate)

    def test_cli_flag_exists_and_defaults_to_false(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        actions = [
            a for a in parser._actions if "--enable-cache-hit-rate" in a.option_strings
        ]
        self.assertEqual(len(actions), 1)
        self.assertFalse(actions[0].default)

    def test_window_gauge_unset_until_flag_is_on(self):
        from sglang.srt.observability.metrics_collector import SchedulerStats

        self.assertIsNone(SchedulerStats().cache_hit_rate_window)


class TestPrefillAdderSamplingGate(CustomTestCase):
    """The adder only samples when --enable-cache-hit-rate is on."""

    def _adder(self, enable: bool):
        from sglang.srt.managers.schedule_policy import PrefillAdder

        # Bypass __init__: it needs a live scheduler/tree cache.
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.enable_cache_hit_rate = enable
        adder.log_cache_num_reqs = 0
        adder.log_cache_query_tokens = 0
        adder.log_cache_hit_tokens = 0
        return adder

    def test_no_sampling_when_flag_off(self):
        adder = self._adder(enable=False)
        adder._record_prefix_cache_sample(_FakeReq(prompt_len=100, prefix_len=40))
        self.assertEqual(adder.log_cache_num_reqs, 0)
        self.assertEqual(adder.log_cache_query_tokens, 0)
        self.assertEqual(adder.log_cache_hit_tokens, 0)

    def test_samples_request_level_tokens_when_flag_on(self):
        adder = self._adder(enable=True)
        adder._record_prefix_cache_sample(_FakeReq(prompt_len=100, prefix_len=40))
        self.assertEqual(adder.log_cache_num_reqs, 1)
        self.assertEqual(adder.log_cache_query_tokens, 100)
        self.assertEqual(adder.log_cache_hit_tokens, 40)


if __name__ == "__main__":
    unittest.main()
