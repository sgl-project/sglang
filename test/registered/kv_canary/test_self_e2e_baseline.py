from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _BaselineBase(CanaryE2EBase):
    """No perturb, kv-canary=log, sweep off. Server should run clean with no canary
    violations and every request must come back 200."""

    kv_canary_mode = "log"
    extra_env = {}

    def test_no_violation(self) -> None:
        """Verify the baseline canary run completes without violations."""
        self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)
        if self.model_mode == "swa":
            self.assert_swa_divergence_observed()


class TestBaselineMha(_BaselineBase, unittest.TestCase):
    model_mode = "mha"


class TestBaselineSwa(_BaselineBase, unittest.TestCase):
    model_mode = "swa"


if __name__ == "__main__":
    unittest.main()
