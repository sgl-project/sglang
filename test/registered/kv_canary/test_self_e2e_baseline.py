from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _BaselineBase(CanaryE2EBase):
    """No perturb, kv-canary=log, sweep off. Server should run clean with no canary
    violations and every request must come back 200."""

    __test__ = False

    kv_canary_mode = CanaryMode.LOG
    extra_env = {}

    def test_no_violation(self) -> None:
        """Verify the baseline canary run completes without violations."""
        self.send_parallel_requests(
            n=self.workload_n_requests,
            max_concurrent=self.workload_max_concurrent,
        )
        self.assert_no_violation(wait_seconds=2.0)
        self.maybe_assert_swa_divergence_observed()


class TestBaselineMha(_BaselineBase):
    __test__ = True

    model_mode = "mha"


class TestBaselineSwa(_BaselineBase):
    __test__ = True

    model_mode = "swa"
    # Tight SWA pool (≈19K slots) forces window reorganization across sequential batches
    # so swa_full_idx_divergence > 0. Concurrency 4 keeps SWA in-flight footprint below
    # the pool cap; 16 sequential requests cycle the pool enough to surface divergence.
    extra_server_args = ("--swa-full-tokens-ratio", "0.3")
    workload_n_requests = 16
    workload_max_concurrent = 4


if __name__ == "__main__":
    unittest.main()
