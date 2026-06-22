from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.consts import SWA_POOL_SERVER_ARGS
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=106, stage="extra-a", runner_config="1-gpu-small")


class _BaselineBase(CanaryE2EBase):
    """No perturb, kv-canary=log, sweep off. Server should run clean with no canary
    violations and every request must come back 200."""

    kv_canary_mode = CanaryMode.LOG
    extra_env = {}

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _BaselineBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set model_mode")
        super().setUpClass()

    def test_no_violation(self) -> None:
        """Verify the baseline canary run completes without violations."""
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)
        self.maybe_assert_swa_divergence_observed()


class TestBaselineMha(_BaselineBase):
    model_mode = "mha"


class TestBaselineSwa(_BaselineBase):
    model_mode = "swa"
    extra_server_args = SWA_POOL_SERVER_ARGS


if __name__ == "__main__":
    unittest.main()
