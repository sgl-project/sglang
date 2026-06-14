import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class TestSchedulerAbortRequest(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        # Mock just enough state for abort_request to reach the running-batch and
        # dLLM blocks without touching the waiting-queue / grammar / disagg paths.
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = []
        scheduler.grammar_manager = MagicMock()
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.cur_batch = scheduler.running_batch
        scheduler.dllm_config = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.waiting_queue = []
        return scheduler

    def _make_req(self, rid: str) -> MagicMock:
        req = MagicMock()
        req.rid = rid
        req.finished.return_value = False
        req.to_finish = None
        return req

    def test_abort_targets_dllm_waiting_queue_req_not_in_current_batch(self):
        """A dLLM request in waiting_queue but absent from the current batch is still aborted."""
        scheduler = self._new_scheduler()
        scheduler.dllm_config = MagicMock()  # dLLM enabled
        req_a = self._make_req("req-a")
        req_b = self._make_req("req-b")
        # Both are in-flight dLLM reqs living only in dllm_manager.waiting_queue;
        # neither is in running_batch / cur_batch.
        scheduler.dllm_manager.waiting_queue = [req_a, req_b]

        scheduler.abort_request(AbortReq(rid="req-a"))

        self.assertIsInstance(req_a.to_finish, FINISH_ABORT)
        self.assertIsNone(req_b.to_finish)

    def test_abort_all_targets_every_dllm_waiting_queue_req(self):
        """abort_all marks every in-flight dLLM request in the waiting_queue."""
        scheduler = self._new_scheduler()
        scheduler.dllm_config = MagicMock()
        req_a = self._make_req("req-a")
        req_b = self._make_req("req-b")
        scheduler.dllm_manager.waiting_queue = [req_a, req_b]

        scheduler.abort_request(AbortReq(rid="", abort_all=True))

        self.assertIsInstance(req_a.to_finish, FINISH_ABORT)
        self.assertIsInstance(req_b.to_finish, FINISH_ABORT)

    def test_dllm_branch_skipped_when_disabled(self):
        """With dLLM disabled (dllm_config is None) the waiting_queue is left untouched."""
        scheduler = self._new_scheduler()  # dllm_config is None
        req_a = self._make_req("req-a")
        scheduler.dllm_manager.waiting_queue = [req_a]

        scheduler.abort_request(AbortReq(rid="req-a"))

        self.assertIsNone(req_a.to_finish)


if __name__ == "__main__":
    unittest.main()
