import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.schedule_batch import ReqPhase
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=9, suite="base-c-test-cpu")


class TestSchedulerPauseGeneration(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.running_batch.batch_is_full = False
        scheduler.active_reqs = {}
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.protected_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock()
        scheduler.hisparse_coordinator = None
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
        """in_place pause should only set _engine_paused and return."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.cur_batch = MagicMock()

        original_last_batch = scheduler.last_batch
        original_cur_batch = scheduler.cur_batch

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        # All state must be preserved — no mutation
        self.assertIs(scheduler.last_batch, original_last_batch)
        self.assertIs(scheduler.cur_batch, original_cur_batch)

    def test_inplace_does_not_drain_overlap_queue(self):
        """in_place should not process the overlap result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        scheduler.last_batch = MagicMock()
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(len(scheduler.result_queue), 1)

    def test_inplace_does_not_merge_batch(self):
        """in_place should not filter or merge last_batch into running_batch."""
        scheduler = self._new_scheduler()
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True
        scheduler.last_batch = last_batch

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        last_batch.filter_batch.assert_not_called()
        scheduler.running_batch.merge_batch.assert_not_called()

    def test_abort_clears_state(self):
        """abort mode should clear last_batch and cur_batch."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.forward_mode.is_extend.return_value = False
        scheduler.cur_batch = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))

        self.assertTrue(scheduler._engine_paused)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.cur_batch)

    def test_retract_clears_running_batch(self):
        """retract mode retracts running_batch decode reqs AND active mid-chunk
        (EXTEND_NON_LAST) reqs that never entered running_batch."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = None
        # Two decode reqs already merged into running_batch.
        req_a = MagicMock(rid="req-a", phase=ReqPhase.DECODE)
        req_b = MagicMock(rid="req-b", phase=ReqPhase.DECODE)
        # A partially-extended (mid-chunk) req lives ONLY in active_reqs: the
        # last_batch merge excludes it via skip_extend_intermediate, so when it
        # is the single in-flight chunked-prefill req the retract path must still
        # release it through partially_extended_reqs(). Dropping that term would
        # leak req_c's KV / req-pool row across the pause.
        req_c = MagicMock(rid="req-c", phase=ReqPhase.EXTEND_NON_LAST)
        scheduler.running_batch.reqs = [req_a, req_b]
        scheduler.running_batch.__len__ = lambda self: len(self.reqs)
        scheduler.running_batch.is_empty.return_value = False
        scheduler.active_reqs = {
            req_a.rid: req_a,
            req_b.rid: req_b,
            req_c.rid: req_c,
        }
        scheduler.waiting_queue = []
        scheduler._add_request_to_queue = MagicMock()

        scheduler.running_batch.filter_batch = MagicMock()
        scheduler.server_args = MagicMock()

        with patch("sglang.srt.managers.scheduler.release_req") as mock_release_req:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler._engine_paused)
        # All three reqs (2 from running_batch + 1 partially-extended) get
        # released, deactivated, and re-queued.
        self.assertEqual(mock_release_req.call_count, 3)
        self.assertEqual(scheduler._add_request_to_queue.call_count, 3)
        self.assertEqual(scheduler.active_reqs, {})

    def test_abort_drains_overlap_queue(self):
        """abort with overlap enabled should drain the result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        mock_batch = MagicMock()
        mock_batch.forward_mode.is_extend.return_value = False
        scheduler.last_batch = mock_batch
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])
        scheduler.process_batch_result = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))

        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 0)


if __name__ == "__main__":
    unittest.main()
