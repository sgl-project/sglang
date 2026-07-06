from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.pp_fixture import CanaryPPFixture

register_cuda_ci(est_time=322, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=298, stage="extra-a", runner_config="2-gpu-large-amd")


class TestPPPerturbSwaSwa(CanaryPPFixture):

    kv_canary_mode = CanaryMode.LOG
    target_group: ClassVar[TargetGroupKind] = TargetGroupKind.SWA
    extra_server_args = ("--kv-canary-real-data", "partial")

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0.1",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        super().setUpClass()

    def test_real_kv_used_perturbation_reports_real_kv_hash_violation(self) -> None:
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_per_forward_violation_reported(
            fail_reason="verify_real_kv_hash",
            target_group=self.target_group,
        )
        self.maybe_assert_swa_divergence_observed()


if __name__ == "__main__":
    unittest.main()
