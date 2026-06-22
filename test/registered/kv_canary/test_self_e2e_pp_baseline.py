from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.pp_fixture import CanaryPPFixture

register_cuda_ci(est_time=193, stage="extra-a", runner_config="2-gpu-large")


class TestPPBaselineSwa(CanaryPPFixture):

    kv_canary_mode = CanaryMode.LOG

    def test_no_violation(self) -> None:
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)
        self.maybe_assert_swa_divergence_observed()


if __name__ == "__main__":
    unittest.main()
