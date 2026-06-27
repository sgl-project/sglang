import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="base-b-test-cpu")


def _dllm_req(rid: str, finished: bool = False) -> MagicMock:
    """Build a minimal in-flight DLLM req mock for abort_request."""
    req = MagicMock(rid=rid)
    req.finished.return_value = finished
    req.to_finish = None
    return req


class TestSchedulerAbortRequest(unittest.TestCase):
    def _new_scheduler(self, dllm_enabled: bool = True) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting_queue = []
        scheduler.enable_hicache_storage = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.grammar_manager = MagicMock()
        scheduler.active_reqs = {}
        scheduler.dllm_config = MagicMock() if dllm_enabled else None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.waiting_queue = []
        return scheduler

    def test_abort_aborts_matching_inflight_dllm_request(self):
        """abort_request must reach DLLM reqs owned by the manager (not in active_reqs)."""
        scheduler = self._new_scheduler()
        target = _dllm_req("dllm-1")
        other = _dllm_req("dllm-2")
        scheduler.dllm_manager.waiting_queue = [target, other]

        scheduler.abort_request(AbortReq(rid="dllm-1"))

        self.assertIsInstance(target.to_finish, FINISH_ABORT)
        self.assertIsNone(other.to_finish)

    def test_abort_all_aborts_unfinished_dllm_requests(self):
        """abort_all should abort every unfinished DLLM req and skip finished ones."""
        scheduler = self._new_scheduler()
        running = _dllm_req("dllm-1")
        done = _dllm_req("dllm-2", finished=True)
        scheduler.dllm_manager.waiting_queue = [running, done]

        scheduler.abort_request(AbortReq(abort_all=True))

        self.assertIsInstance(running.to_finish, FINISH_ABORT)
        self.assertIsNone(done.to_finish)

    def test_abort_skips_dllm_queue_when_disabled(self):
        """With DLLM unconfigured the manager queue is not folded into reqs."""
        scheduler = self._new_scheduler(dllm_enabled=False)
        untouched = _dllm_req("dllm-1")
        scheduler.dllm_manager.waiting_queue = [untouched]

        scheduler.abort_request(AbortReq(abort_all=True))

        self.assertIsNone(untouched.to_finish)


if __name__ == "__main__":
    unittest.main()
