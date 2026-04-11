"""Test that batch_is_full is reset after filtering prefill-only batches.

Regression test for https://github.com/sgl-project/sglang/issues/22518
Embedding/reward models deadlock after the first full batch because
filter_batch() on a prefill-only running_batch did not reset batch_is_full.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class FakeReq:
    """Minimal request stub for testing."""

    def __init__(self, finished=False):
        self._finished = finished

    def finished(self):
        return self._finished


class FakeRunningBatch:
    """Minimal ScheduleBatch stub that tracks batch_is_full and reqs."""

    def __init__(self, reqs, is_prefill_only=False, batch_is_full=False):
        self.reqs = list(reqs)
        self.is_prefill_only = is_prefill_only
        self.batch_is_full = batch_is_full

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def filter_batch(self, chunked_req_to_exclude=None):
        """Remove finished requests (simplified version of ScheduleBatch.filter_batch)."""
        self.reqs = [r for r in self.reqs if not r.finished()]

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class TestEmbeddingBatchIsFullReset(unittest.TestCase):
    """Verify batch_is_full is correctly reset for prefill-only models."""

    def _new_scheduler(self) -> Scheduler:
        """Create a minimal scheduler for testing get_next_batch_to_run."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.chunked_req = None
        scheduler.enable_hisparse = False
        scheduler.dllm_config = None
        scheduler.waiting_queue = []
        scheduler.result_queue = []
        scheduler.require_mlp_sync = False
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.protected_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 1000
        scheduler.max_total_num_tokens = 1000
        scheduler._get_token_info = MagicMock(return_value=(0, 0, 1000, 0))
        scheduler.server_args = MagicMock()
        scheduler.server_args.waiting_queue_timeout = None
        scheduler.server_args.running_queue_timeout = None
        return scheduler

    def test_batch_is_full_reset_after_prefill_only_filter(self):
        """After filtering finished reqs from a prefill-only batch,
        batch_is_full should be reset so new batches can be scheduled.

        This is the exact scenario from issue #22518:
        1. Embedding requests fill max_running_requests -> batch_is_full=True
        2. All requests complete (prefill-only, no decode step)
        3. filter_batch() removes completed requests
        4. batch_is_full must be reset to False
        """
        # All requests are finished (embedding requests complete after prefill)
        finished_reqs = [FakeReq(finished=True), FakeReq(finished=True)]
        running_batch = FakeRunningBatch(
            reqs=finished_reqs,
            is_prefill_only=True,
            batch_is_full=True,  # Set by previous scheduling round
        )

        scheduler = self._new_scheduler()
        scheduler.running_batch = running_batch

        # Patch methods we don't want to actually run
        with patch.object(scheduler, "_abort_on_waiting_timeout"), \
             patch.object(scheduler, "_abort_on_running_timeout"), \
             patch.object(scheduler, "get_new_batch_prefill", return_value=None):
            scheduler.get_next_batch_to_run()

        self.assertFalse(
            running_batch.batch_is_full,
            "batch_is_full should be False after all prefill-only requests "
            "are filtered out. Otherwise, the scheduler deadlocks and never "
            "schedules new batches (issue #22518).",
        )

    def test_batch_is_full_stays_true_when_no_requests_filtered(self):
        """If no requests were filtered, batch_is_full should remain unchanged."""
        # Requests still running (not finished)
        active_reqs = [FakeReq(finished=False), FakeReq(finished=False)]
        running_batch = FakeRunningBatch(
            reqs=active_reqs,
            is_prefill_only=True,
            batch_is_full=True,
        )

        scheduler = self._new_scheduler()
        scheduler.running_batch = running_batch

        with patch.object(scheduler, "_abort_on_waiting_timeout"), \
             patch.object(scheduler, "_abort_on_running_timeout"), \
             patch.object(scheduler, "get_new_batch_prefill", return_value=None):
            scheduler.get_next_batch_to_run()

        self.assertTrue(
            running_batch.batch_is_full,
            "batch_is_full should remain True when no requests were filtered.",
        )

    def test_batch_is_full_reset_partial_filter(self):
        """When some (but not all) requests finish, batch_is_full should reset."""
        reqs = [FakeReq(finished=True), FakeReq(finished=False)]
        running_batch = FakeRunningBatch(
            reqs=reqs,
            is_prefill_only=True,
            batch_is_full=True,
        )

        scheduler = self._new_scheduler()
        scheduler.running_batch = running_batch

        with patch.object(scheduler, "_abort_on_waiting_timeout"), \
             patch.object(scheduler, "_abort_on_running_timeout"), \
             patch.object(scheduler, "get_new_batch_prefill", return_value=None):
            scheduler.get_next_batch_to_run()

        self.assertFalse(
            running_batch.batch_is_full,
            "batch_is_full should be reset when batch size decreases after filter.",
        )

    def test_non_prefill_only_batch_not_affected(self):
        """For non-prefill-only batches, the prefill-only filter path is skipped."""
        finished_reqs = [FakeReq(finished=True), FakeReq(finished=True)]
        running_batch = FakeRunningBatch(
            reqs=finished_reqs,
            is_prefill_only=False,  # Normal decode model
            batch_is_full=True,
        )

        scheduler = self._new_scheduler()
        scheduler.running_batch = running_batch

        with patch.object(scheduler, "_abort_on_waiting_timeout"), \
             patch.object(scheduler, "_abort_on_running_timeout"), \
             patch.object(scheduler, "get_new_batch_prefill", return_value=None), \
             patch.object(scheduler, "update_running_batch", return_value=running_batch):
            scheduler.get_next_batch_to_run()

        # For non-prefill-only, the prefill-only code path is not taken,
        # so batch_is_full is not reset here (it's handled elsewhere).
        self.assertTrue(
            running_batch.batch_is_full,
            "Non-prefill-only batch should not be affected by the prefill-only filter path.",
        )


if __name__ == "__main__":
    unittest.main()
