import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerCaptureGradMode(unittest.TestCase):
    def test_init_all_cuda_graphs_disables_grad(self):
        grad_enabled = []
        worker = SimpleNamespace(
            init_cuda_graphs=lambda: grad_enabled.append(torch.is_grad_enabled())
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = worker
        scheduler.draft_worker = worker

        with torch.enable_grad():
            scheduler.init_all_cuda_graphs()

        self.assertEqual(grad_enabled, [False, False])


if __name__ == "__main__":
    unittest.main()
