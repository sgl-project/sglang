"""Cpu-vs-gpu parity of the confidence-metrics accumulators.

Guards device-dependent numeric drift in PerPositionConfidenceMetrics
(in-place accumulation order, dtype promotion); the pure-CPU behavior is
covered by test_dspark_confidence_metrics.py.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")


class TestConfidenceMetricsDeviceParity(CustomTestCase):
    def test_on_device_accumulation_matches_cpu(self):
        from sglang.srt.speculative.dspark_components.dspark_observability import (
            PerPositionConfidenceMetrics,
        )

        torch.manual_seed(0)
        bs, gamma = 24, 4
        survival = torch.rand(bs, gamma)
        prefix_mask = (torch.rand(bs, gamma) < 0.5).to(torch.float64)

        cpu = PerPositionConfidenceMetrics(gamma=gamma, device=torch.device("cpu"))
        cpu.update(survival=survival, prefix_mask=prefix_mask)

        gpu = PerPositionConfidenceMetrics(gamma=gamma, device=DEVICE)
        gpu.update(survival=survival.cuda(), prefix_mask=prefix_mask.cuda())

        for cpu_row, gpu_row in zip(cpu.compute(), gpu.compute()):
            for key in ("ece", "auc", "brier", "pred_mean", "target_mean"):
                self.assertAlmostEqual(cpu_row[key], gpu_row[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
