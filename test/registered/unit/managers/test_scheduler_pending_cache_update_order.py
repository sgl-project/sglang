import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StopScheduling(Exception):
    pass


class TestSchedulerPendingCacheUpdateOrder(unittest.TestCase):
    def _scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.gracefully_exit = False
        scheduler.request_receiver = MagicMock()
        scheduler.request_receiver.recv_requests.return_value = []
        scheduler.process_input_requests = MagicMock()
        scheduler._engine_paused = False
        scheduler._commit_queued_cache_updates = MagicMock()
        scheduler.running_batch = MagicMock()
        scheduler.last_batch = MagicMock()
        return scheduler

    def _assert_commit_precedes_planning(
        self,
        *,
        loop_name: str,
        planner_name: str,
        prepare=None,
    ) -> None:
        scheduler = self._scheduler()
        if prepare is not None:
            prepare(scheduler)

        def stop_after_order_check(*args, **kwargs):
            scheduler._commit_queued_cache_updates.assert_called_once_with()
            raise _StopScheduling

        setattr(scheduler, planner_name, MagicMock(side_effect=stop_after_order_check))

        with self.assertRaises(_StopScheduling):
            getattr(Scheduler, loop_name)(scheduler)

    def test_normal_overlap_commits_before_planning(self):
        self._assert_commit_precedes_planning(
            loop_name="event_loop_overlap",
            planner_name="get_next_batch_to_run",
        )

    def test_disagg_prefill_overlap_commits_before_planning(self):
        def prepare(scheduler):
            scheduler.waiting_queue = []
            scheduler.disagg_prefill_bootstrap_queue = MagicMock()
            scheduler.disagg_prefill_bootstrap_queue.pop_bootstrapped.return_value = []

        self._assert_commit_precedes_planning(
            loop_name="event_loop_overlap_disagg_prefill",
            planner_name="get_next_disagg_prefill_batch_to_run",
            prepare=prepare,
        )

    def test_disagg_decode_overlap_commits_before_planning(self):
        def prepare(scheduler):
            scheduler.process_decode_queue = MagicMock()

        self._assert_commit_precedes_planning(
            loop_name="event_loop_overlap_disagg_decode",
            planner_name="get_next_disagg_decode_batch_to_run",
            prepare=prepare,
        )


if __name__ == "__main__":
    unittest.main()
