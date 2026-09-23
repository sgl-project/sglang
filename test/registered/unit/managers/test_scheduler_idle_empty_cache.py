"""Idle cache reclamation must run on every rank, not only the socket owner.

SGLANG_EMPTY_CACHE_INTERVAL reclaims cached allocator blocks while the server
is idle; gating that on the rank that owns the tokenizer/rpc sockets leaves the
other ranks' cached blocks pinned until the process OOMs. CPU-only: builds a
bare Scheduler with mocked collaborators, like test_scheduler_on_idle_load.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_EMPTY_CACHE = "sglang.srt.managers.scheduler.current_platform.empty_cache"
_MONOTONIC = "sglang.srt.managers.scheduler.time.monotonic"


class TestOnIdleEmptyCache(CustomTestCase):
    def _idle_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = MagicMock()
        s.enable_hicache_storage = False
        s.enable_unified_memory = False
        s.enable_hisparse = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.idle_sleeper = None  # every rank except the one owning the ipc sockets
        s._last_idle_empty_cache_time = 0.0
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=True)
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s.kv_events_publisher = MagicMock()
        s.new_token_ratio_tracker = MagicMock()
        s.pool_stats_observer = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        s.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
        s.invariant_checker = MagicMock()
        s.invariant_checker._check_all_pools.return_value = (False, [])
        return s

    def test_idle_reclaims_on_a_rank_without_an_idle_sleeper(self):
        s = self._idle_scheduler()
        with (
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(600),
            patch(_EMPTY_CACHE) as empty_cache,
            patch(_MONOTONIC, return_value=1000.0),
        ):
            s.on_idle()
        self.assertEqual(empty_cache.call_count, 1)

    def test_disabled_interval_never_reclaims(self):
        s = self._idle_scheduler()
        with (
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(-1),
            patch(_EMPTY_CACHE) as empty_cache,
            patch(_MONOTONIC, return_value=1e9),
        ):
            s.on_idle()
        self.assertEqual(empty_cache.call_count, 0)

    def test_reclaims_at_most_once_per_interval(self):
        s = self._idle_scheduler()
        with (
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(600),
            patch(_EMPTY_CACHE) as empty_cache,
            patch(_MONOTONIC) as monotonic,
        ):
            monotonic.return_value = 1000.0
            s.on_idle()
            self.assertEqual(empty_cache.call_count, 1)

            monotonic.return_value = 1599.0  # still inside the interval
            s.on_idle()
            self.assertEqual(empty_cache.call_count, 1)

            monotonic.return_value = 1601.0
            s.on_idle()
            self.assertEqual(empty_cache.call_count, 2)
        self.assertEqual(s._last_idle_empty_cache_time, 1601.0)


if __name__ == "__main__":
    unittest.main()
