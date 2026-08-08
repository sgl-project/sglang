import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerIdleGate(unittest.TestCase):
    def _new_scheduler(self, *, overlap: bool, overlap_mlx: bool, queued: int):
        s = Scheduler.__new__(Scheduler)
        s.running_batch = MagicMock(is_empty=MagicMock(return_value=True))
        s.chunked_req = None
        s.dllm_manager = MagicMock(any_staging_reqs=MagicMock(return_value=False))
        s.last_batch = None
        s.enable_overlap = overlap
        s.enable_overlap_mlx = overlap_mlx
        s.result_queue = [object()] * queued
        s._pp_microbatches_drained = MagicMock(return_value=True)
        s.waiting_queue = []
        return s

    def test_mlx_overlap_with_queued_result_is_not_idle(self):
        s = self._new_scheduler(overlap=False, overlap_mlx=True, queued=1)
        self.assertFalse(s.is_fully_idle(for_health_check=True))

    def test_torch_overlap_unchanged(self):
        s = self._new_scheduler(overlap=True, overlap_mlx=False, queued=1)
        self.assertFalse(s.is_fully_idle(for_health_check=True))

    def test_mlx_overlap_with_empty_queue_is_idle(self):
        s = self._new_scheduler(overlap=False, overlap_mlx=True, queued=0)
        self.assertTrue(s.is_fully_idle(for_health_check=True))


if __name__ == "__main__":
    unittest.main()
