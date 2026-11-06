"""Regression coverage for scheduler idle-state ownership and idle publishing.

Both a no-batch-but-not-idle stall and a fully idle scheduler spin on_idle
without sleeping, so the gate must cap the O(queue) get_loads for both the
DP-balancing writer and the load socket. The busy->idle transition still
publishes immediately. Work held outside the admitted request queues (async
SHM materialization) must keep the scheduler non-idle. CPU-only: builds a bare
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

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.models.unlimited_ocr_v2_support.async_shm import AsyncShmMaterializer
from sglang.srt.models.unlimited_ocr_v2_support.request_receiver import (
    UocrRequestReceiver,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SchedulerStageMetricsRecorder,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.deferred_responses = DeferredResponses(
            disaggregation_mode=DisaggregationMode.NULL,
            output_streamer=None,
            send_abort=None,
        )
        s.scheduler_stage_metrics = None
        s.enable_hicache_storage = False
        s.enable_unified_cache_external_linker = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_lmcache = False
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s.dp_attn_adapter = MagicMock()
        s.request_receiver = object.__new__(SchedulerRequestReceiver)
        s._last_spin_publish_ts = float("-inf")
        s._was_fully_idle = False
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
            mono.return_value = 100.10  # > LOAD_SPIN_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


class OnIdleIdlePublishTestCase(CustomTestCase):
    def _fully_idle_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.deferred_responses = DeferredResponses(
            disaggregation_mode=DisaggregationMode.NULL,
            output_streamer=None,
            send_abort=None,
        )
        s.scheduler_stage_metrics = SchedulerStageMetricsRecorder(enabled=False)
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=True)
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s.dp_attn_adapter = MagicMock()
        s.request_receiver = object.__new__(SchedulerRequestReceiver)
        s._last_spin_publish_ts = float("-inf")
        s._was_fully_idle = False
        s.enable_fpm = False
        s.enable_hicache_storage = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.batch_result_processor = MagicMock()
        s.running_reqs = MagicMock()
        s.prefill_launched_reqs = MagicMock()
        s._record_step_counters = MagicMock()
        s._maybe_clear_mm_inputs = MagicMock()
        s.tree_cache = MagicMock()
        s.enable_unified_memory = False
        s.enable_hisparse = False
        s.invariant_checker = MagicMock()
        s.invariant_checker._check_all_pools.return_value = (False, [])
        s.pool_stats_observer = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        s.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
        s.kv_events_publisher = MagicMock()
        s.new_token_ratio_tracker = MagicMock()
        s.maybe_sleep_on_idle = MagicMock()
        return s


class TestOnIdleIdlePublish(OnIdleIdlePublishTestCase):
    def test_spinning_idle_publishes_once_within_the_floor(self):
        s = self._fully_idle_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(1000):
                s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)
        self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)

    def test_idle_transition_publishes_immediately(self):
        s = self._fully_idle_scheduler()
        s._last_spin_publish_ts = 100.0
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)

    def test_idle_publishes_again_after_the_floor_elapses(self):
        s = self._fully_idle_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            mono.return_value = 100.0
            s.on_idle()
            mono.return_value = 100.10  # > LOAD_SPIN_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)

    def test_every_return_to_idle_publishes(self):
        s = self._fully_idle_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(3):
                s.is_fully_idle.return_value = False
                s.on_idle()
                s.is_fully_idle.return_value = True
                s.on_idle()
        # One stall publish (first busy pass, floor empty) + one per transition.
        self.assertEqual(s.publish_load_snapshot.call_count, 4)

    def test_real_batch_makes_the_next_idle_pass_publish(self):
        s = self._fully_idle_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            s.on_idle()
            self.assertEqual(s.publish_load_snapshot.call_count, 1)
            s.mark_engine_busy()
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


def _batch_mock(*, idle: bool) -> MagicMock:
    forward_mode = MagicMock()
    forward_mode.is_idle.return_value = idle
    forward_mode.is_extend.return_value = False
    forward_mode.is_decode.return_value = False
    forward_mode.is_prebuilt.return_value = False
    batch = MagicMock()
    batch.forward_mode = forward_mode
    batch.reqs = []
    return batch


class TestBatchResultClearsTheTransitionFlag(OnIdleIdlePublishTestCase):
    def test_real_batch_clears_the_flag(self):
        s = self._fully_idle_scheduler()
        s._was_fully_idle = True
        s.process_batch_result(_batch_mock(idle=False), MagicMock())
        self.assertFalse(s._was_fully_idle)

    def test_idle_filler_batch_clears_the_flag_too(self):
        """A dp-attention rank runs a filler IDLE batch to join the MLP-sync
        collective while its own queues can hold requests, so it is a real step
        and must let the next idle pass publish immediately."""
        s = self._fully_idle_scheduler()
        s._was_fully_idle = True
        s.process_batch_result(_batch_mock(idle=True), MagicMock())
        self.assertFalse(s._was_fully_idle)


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
        scheduler.deferred_responses = DeferredResponses(
            disaggregation_mode=DisaggregationMode.NULL,
            output_streamer=None,
            send_abort=None,
        )
        scheduler.scheduler_stage_metrics = SchedulerStageMetricsRecorder(enabled=False)
        scheduler.maybe_send_health_check_signal = MagicMock()
        scheduler.is_fully_idle = Scheduler.is_fully_idle.__get__(scheduler)
        scheduler.on_idle = Scheduler.on_idle.__wrapped__.__get__(scheduler)
        scheduler.metrics_reporter = MagicMock()
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hisparse = False
        scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        scheduler.running_reqs = []
        scheduler.dllm_config = None
        scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None
        scheduler.hisparse_coordinator = None
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_lmcache = False
        scheduler.dp_attn_adapter = MagicMock()
        scheduler._engine_paused = False
        receiver = object.__new__(UocrRequestReceiver)
        object.__setattr__(receiver, "async_shm_materializer", materializer)
        scheduler.request_receiver = receiver
        scheduler._last_spin_publish_ts = 0.0
        scheduler._was_fully_idle = False
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
