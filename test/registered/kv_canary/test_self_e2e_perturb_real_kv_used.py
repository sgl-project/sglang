from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


_SWA_TIGHT_POOL_ARGS = ("--max-total-tokens", "8192")


class _PerturbRealKvUsedBase(CanaryE2EBase):
    __test__ = False

    kv_canary_mode = CanaryMode.LOG
    extra_server_args = ("--kv-canary-real-data", "partial")

    target_group: ClassVar[TargetGroupKind]

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0.1",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
            "SGLANG_KV_CANARY_ENABLE_MHA_V": "1",
        }
        super().setUpClass()

    def test_real_kv_used_perturbation_reports_real_kv_hash_violation(self) -> None:
        """Verify active real KV perturbation reports a real KV hash violation."""
        self.send_parallel_requests()
        self.assert_per_forward_violation_reported(
            fail_reason="real_kv_hash",
            target_group=self.target_group,
        )
        self.maybe_assert_swa_divergence_observed()


class TestPerturbRealKvUsedMhaFull(_PerturbRealKvUsedBase):
    __test__ = True

    model_mode = "mha"
    target_group = TargetGroupKind.FULL


class TestPerturbRealKvUsedSwaFull(_PerturbRealKvUsedBase):
    __test__ = True

    model_mode = "swa"
    target_group = TargetGroupKind.FULL
    extra_server_args = (*_PerturbRealKvUsedBase.extra_server_args, *_SWA_TIGHT_POOL_ARGS)


class TestPerturbRealKvUsedSwaSwa(_PerturbRealKvUsedBase):
    __test__ = True

    model_mode = "swa"
    target_group = TargetGroupKind.SWA
    extra_server_args = (*_PerturbRealKvUsedBase.extra_server_args, *_SWA_TIGHT_POOL_ARGS)


if __name__ == "__main__":
    unittest.main()
