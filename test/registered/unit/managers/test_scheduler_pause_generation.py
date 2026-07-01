import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)
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
        scheduler.chunked_req = None
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.running_batch.batch_is_full = False
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.protected_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock()
        scheduler.result_queue = deque()
        scheduler.disaggregation_mode = DisaggregationMode.NULL
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
        scheduler.chunked_req = MagicMock()

        original_last_batch = scheduler.last_batch
        original_cur_batch = scheduler.cur_batch
        original_chunked_req = scheduler.chunked_req

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        # All state must be preserved — no mutation
        self.assertIs(scheduler.last_batch, original_last_batch)
        self.assertIs(scheduler.cur_batch, original_cur_batch)
        self.assertIs(scheduler.chunked_req, original_chunked_req)

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
        """retract mode should retract all requests from running_batch."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = None
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

        self.assertTrue(scheduler._engine_paused)
        scheduler.running_batch.retract_all.assert_called_once()
        self.assertEqual(scheduler._add_request_to_queue.call_count, 2)
        self.assertIsNone(scheduler.chunked_req)

    def test_pd_decode_retract_requeues_for_rebootstrap(self):
        """PD decode retract should rebootstrap instead of resuming stale CPU KV."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.last_batch = None
        scheduler.running_batch.reqs = [MagicMock()]
        scheduler.running_batch.is_empty.return_value = False
        scheduler._add_request_to_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue = MagicMock()

        req = SimpleNamespace(
            output_ids=[10, 11, 12],
            time_stats=MagicMock(),
            kv_cache_cpu=object(),
        )
        scheduler.running_batch.retract_all.return_value = [req]
        scheduler.running_batch.filter_batch = MagicMock()
        scheduler.server_args = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler._add_request_to_queue.assert_not_called()
        scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.assert_called_once_with(
            req
        )
        self.assertEqual(req.output_ids, [10, 11])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 12)
        self.assertTrue(req.pd_rebootstrap_in_progress)
        self.assertFalse(hasattr(req, "kv_cache_cpu"))

    def test_pd_decode_continue_releases_held_rebootstrap(self):
        """continue_generation must enqueue staged rebootstrap reqs on resume."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler._engine_paused = True

        scheduler.continue_generation(
            ContinueGenerationReqInput(torch_empty_cache=False)
        )

        scheduler.disagg_decode_prealloc_queue.enqueue_held_rebootstrap.assert_called_once_with()
        self.assertFalse(scheduler._engine_paused)

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
