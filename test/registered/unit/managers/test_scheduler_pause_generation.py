import unittest
from collections import deque
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=9, suite="base-c-test-cpu")


class TestSchedulerPauseGeneration(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler._pending_pause = None
        scheduler.enable_overlap = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        # cur_batch / last_batch are event-loop locals; out-of-loop state is
        # the last_iter snapshot, cleared when a pending pause is applied.
        scheduler.last_iter = MagicMock()
        scheduler.chunked_req = None
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.running_batch.batch_is_full = False
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.protected_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock()
        scheduler.result_queue = deque()
        # Support _kv_snap diagnostic logging in patched schedulers
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 1000
        scheduler.max_total_num_tokens = 1000
        scheduler._get_token_info = MagicMock(
            return_value=PoolStats(
                full_num_used=0,
                full_token_usage=0,
                full_available_size=1000,
                full_evictable_size=0,
            )
        )
        # pause_generation zeros gen_throughput and flushes KV events.
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
        scheduler.kv_events_publisher = MagicMock()
        return scheduler

    def test_inplace_only_sets_flag(self):
        """in_place pause should only set _engine_paused, with no pending pause."""
        scheduler = self._new_scheduler()
        scheduler.chunked_req = MagicMock()
        original_chunked_req = scheduler.chunked_req

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        # in_place must not enqueue any batch manipulation for the loop to apply.
        self.assertIsNone(scheduler._pending_pause)
        # State must be preserved — no mutation.
        self.assertIs(scheduler.chunked_req, original_chunked_req)

    def test_inplace_does_not_drain_overlap_queue(self):
        """in_place should not enqueue work that would drain the result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))
        # Applying any pending pause is a no-op for in_place.
        last_batch = scheduler._maybe_apply_pending_pause(MagicMock())

        self.assertTrue(scheduler._engine_paused)
        self.assertIsNotNone(last_batch)
        self.assertEqual(len(scheduler.result_queue), 1)

    def test_inplace_does_not_merge_batch(self):
        """in_place should not filter or merge last_batch into running_batch."""
        scheduler = self._new_scheduler()
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))
        scheduler._maybe_apply_pending_pause(last_batch)

        last_batch.filter_batch.assert_not_called()
        scheduler.running_batch.merge_batch.assert_not_called()

    def test_abort_clears_state(self):
        """abort mode should consume last_batch and clear the last_iter snapshot."""
        scheduler = self._new_scheduler()
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = False

        scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))
        self.assertTrue(scheduler._engine_paused)
        self.assertIsNotNone(scheduler._pending_pause)

        applied = scheduler._maybe_apply_pending_pause(last_batch)

        self.assertIsNone(applied)
        self.assertIsNone(scheduler.last_iter)
        self.assertIsNone(scheduler._pending_pause)

    def test_retract_clears_running_batch(self):
        """retract mode should retract all requests from running_batch."""
        scheduler = self._new_scheduler()
        scheduler.running_batch.reqs = [MagicMock(), MagicMock()]
        scheduler.running_batch.__len__ = lambda self: len(self.reqs)
        scheduler.running_batch.is_empty.return_value = False
        scheduler.waiting_queue = []
        scheduler._add_request_to_queue = MagicMock()

        retracted = [MagicMock(), MagicMock()]
        scheduler.running_batch.retract_all.return_value = retracted
        scheduler.running_batch.filter_batch = MagicMock()
        scheduler.server_args = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
        scheduler._maybe_apply_pending_pause(None)

        self.assertTrue(scheduler._engine_paused)
        scheduler.running_batch.retract_all.assert_called_once()
        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        self.assertIsNone(scheduler.chunked_req)

    def test_abort_drains_overlap_queue(self):
        """abort with overlap enabled should drain the result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = False
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])
        scheduler.process_batch_result = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))
        scheduler._maybe_apply_pending_pause(last_batch)

        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 0)


if __name__ == "__main__":
    unittest.main()
