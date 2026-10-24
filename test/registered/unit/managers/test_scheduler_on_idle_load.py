"""on_idle's stalled-path load publish is wall-clock bounded.

A non-storage no-batch stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. CPU-only: builds a bare Scheduler with mocked collaborators, like
test_scheduler_flush_cache.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.enable_hicache_storage = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_lmcache = False
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s.dp_attn_adapter = MagicMock()
        s._last_stall_publish_ts = float("-inf")
        return s

    def test_stall_publish_is_wall_clock_bounded_by_the_floor(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            # A spinning stall inside the floor publishes once.
            mono.return_value = 100.0
            for _ in range(100):
                s.on_idle()
            self.assertEqual(s.publish_load_snapshot.call_count, 1)
            self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)
            # After the floor elapses the next stall publishes again.
            mono.return_value = 100.10  # > LOAD_STALL_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)

    def test_is_fully_idle_with_lmcache_and_disabled_radix_cache(self):
        """A disabled radix cache builds a plain UnifiedRadixCache even when
        enable_lmcache is set; is_fully_idle must not crash on missing LMCache
        hooks."""
        from types import SimpleNamespace

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(
            enable_lmcache=True, disable_radix_cache=True
        )
        override.install()
        self.addCleanup(override.restore)

        s = Scheduler.__new__(Scheduler)
        s.running_batch = SimpleNamespace(is_empty=lambda: True)
        s.chunked_req = None
        s.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        s.last_batch = None
        s.enable_overlap = False
        s.result_queue = []
        s.waiting_queue = []
        s.grammar_manager = SimpleNamespace(grammar_queue=[])
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_hisparse = False
        s.enable_hierarchical_cache = False
        s.enable_lmcache = True
        s.disable_radix_cache = True
        s.tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)

        self.assertTrue(s.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
