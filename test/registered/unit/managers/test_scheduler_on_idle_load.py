"""Regression coverage for scheduler idle-state ownership.

A non-storage no-batch stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. Work held outside the admitted request queues (async SHM
materialization) must keep the scheduler non-idle. CPU-only: builds a bare
Scheduler with mocked collaborators, like test_scheduler_flush_cache.
"""

import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.models.unlimited_ocr_v2_support.async_shm import AsyncShmMaterializer
from sglang.srt.observability.scheduler_stage_metrics import (
    SchedulerStageMetricsRecorder,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class TestAsyncShmIdleAccounting(CustomTestCase):
    def _materializer(self):
        self.enterContext(envs.SGLANG_UOCR_ASYNC_SHM_MAX_MB.override(1))
        materializer = AsyncShmMaterializer(
            max_recv_per_poll=1, discard=Mock(), max_items=2, workers=1
        )
        future = Future()
        self.addCleanup(materializer.shutdown)
        self.addCleanup(lambda: future.done() or future.set_result(None))
        self.assertTrue(future.set_running_or_notify_cancel())
        self.enterContext(
            patch.object(materializer.executor, "submit", return_value=future)
        )
        value = object()
        self.assertTrue(materializer.reserve_receive())
        materializer.submit(
            [value], finalize=Mock(), has_shm=lambda values: bool(values)
        )
        return materializer, future, value

    def _idle_scheduler(self, materializer):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler_stage_metrics = SchedulerStageMetricsRecorder(enabled=False)
        scheduler.maybe_send_health_check_signal = MagicMock()
        scheduler.is_fully_idle = Scheduler.is_fully_idle.__get__(scheduler)
        scheduler.on_idle = Scheduler.on_idle.__wrapped__.__get__(scheduler)
        scheduler.metrics_reporter = MagicMock()
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_lmcache = False
        scheduler.dp_attn_adapter = MagicMock()
        scheduler._engine_paused = False
        scheduler.async_shm_materializer = materializer
        scheduler._last_stall_publish_ts = 0.0
        scheduler.publish_load_snapshot = MagicMock(return_value=None)
        scheduler.load_publisher = MagicMock()
        scheduler.load_inquirer = MagicMock()
        scheduler.enable_unified_memory = False
        scheduler.invariant_checker = MagicMock()
        scheduler.invariant_checker._check_all_pools.return_value = (False, [])
        scheduler.pool_stats_observer = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
        scheduler.kv_events_publisher = MagicMock()
        scheduler.new_token_ratio_tracker = MagicMock()
        scheduler.idle_sleeper = MagicMock()
        return scheduler

    def test_async_shm_entries_block_idle_until_drained(self):
        """Undelivered materialized requests must not enter idle sleep."""
        materializer, future, value = self._materializer()
        progress_wait = self.enterContext(
            patch.object(
                materializer,
                "wait_for_progress",
                wraps=materializer.wait_for_progress,
            )
        )
        scheduler = self._idle_scheduler(materializer)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=0.0):
            self.assertFalse(scheduler.is_fully_idle())
            self.assertTrue(scheduler.is_fully_idle(for_health_check=True))
            scheduler.on_idle()
            scheduler.idle_sleeper.maybe_sleep.assert_not_called()
            self.assertEqual(progress_wait.call_count, 1)

            future.set_result(None)
            self.assertFalse(scheduler.is_fully_idle())
            scheduler.on_idle()
            scheduler.idle_sleeper.maybe_sleep.assert_not_called()
            self.assertEqual(progress_wait.call_count, 2)

        self.assertEqual(materializer.pop_ready(), [value])
        self.assertTrue(scheduler.is_fully_idle())
        scheduler.on_idle()
        scheduler.idle_sleeper.maybe_sleep.assert_called_once_with()
        self.assertEqual(progress_wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
