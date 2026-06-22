from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.pd_fixture import CanaryPDFixture

register_cuda_ci(est_time=116, stage="extra-a", runner_config="2-gpu-large")


class TestPDBaselineMha(CanaryPDFixture):
    model_mode = "mha"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation(side="prefill")
        self.assert_no_violation(side="decode")


class TestPDBaselineSwa(CanaryPDFixture):
    model_mode = "swa"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation(side="prefill")
        self.assert_no_violation(side="decode")


if __name__ == "__main__":
    unittest.main()
