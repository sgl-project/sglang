import unittest
from collections import deque
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="base-c-test-cpu")


def _make_req(is_retracted: bool) -> MagicMock:
    req = MagicMock()
    req.is_retracted = is_retracted
    return req


class TestSchedulerFlushCacheAfterRetract(unittest.TestCase):
    """Regression coverage for flush_cache() no-oping right after
    pause_generation(mode="retract") repopulates waiting_queue with requests
    that hold no KV/pool state (see RETRACT_MODE_FLUSH_CACHE_FIX.md in miles)."""

    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = True
        scheduler.enable_overlap = False
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.chunked_req = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.result_queue = deque()
        scheduler.waiting_queue = []
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.grammar_queue = []
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_hisparse = False
        scheduler.tree_cache = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.draft_worker = None
        scheduler.metrics_reporter = MagicMock()
        return scheduler

    def test_flush_cache_succeeds_when_only_retracted_reqs_are_waiting(self):
        scheduler = self._new_scheduler()
        scheduler.waiting_queue = [_make_req(is_retracted=True)]

        self.assertTrue(scheduler.flush_cache())
        scheduler.tree_cache.reset.assert_called_once()
        scheduler.req_to_token_pool.clear.assert_called_once()
        scheduler.token_to_kv_pool_allocator.clear.assert_called_once()

    def test_flush_cache_still_blocks_on_a_genuinely_new_request(self):
        """Closes the vacuous-fix gap: a request that arrived independently of
        retract (is_retracted=False) must still defer the flush."""
        scheduler = self._new_scheduler()
        scheduler.waiting_queue = [_make_req(is_retracted=False)]

        self.assertFalse(scheduler.flush_cache())
        scheduler.tree_cache.reset.assert_not_called()

    def test_flush_cache_blocks_on_mixed_queue(self):
        scheduler = self._new_scheduler()
        scheduler.waiting_queue = [
            _make_req(is_retracted=True),
            _make_req(is_retracted=False),
        ]

        self.assertFalse(scheduler.flush_cache())
        scheduler.tree_cache.reset.assert_not_called()

    def test_is_fully_idle_default_ignores_nothing(self):
        """Regression guard: every other is_fully_idle() caller passes no args,
        so a retracted-only waiting_queue must still report not-idle by default."""
        scheduler = self._new_scheduler()
        scheduler.waiting_queue = [_make_req(is_retracted=True)]

        self.assertFalse(scheduler.is_fully_idle())
        self.assertTrue(scheduler.is_fully_idle(ignore_retracted=True))

    def test_flush_cache_does_not_ignore_retracted_reqs_while_unpaused(self):
        """ignore_retracted is gated on _engine_paused inside flush_cache — a
        flush issued without pausing first must not be silently permissive."""
        scheduler = self._new_scheduler()
        scheduler._engine_paused = False
        scheduler.waiting_queue = [_make_req(is_retracted=True)]

        self.assertFalse(scheduler.flush_cache())
        scheduler.tree_cache.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
