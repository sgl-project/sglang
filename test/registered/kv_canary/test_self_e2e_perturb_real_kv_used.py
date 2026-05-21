from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _PerturbRealKvUsedBase(CanaryE2EBase):
    """Perturb point (b): flip the first byte of an active req's currently-used KV slot.

    With sweep OFF, the only way to surface this corruption is the per-forward
    HEAD/TAIL real_kv_hash check on the targeted group (FULL or SWA). Subclasses set
    ``model_mode`` and ``target_group``; the FULL/SWA suffix in the violation launch_tag must
    match ``target_group``.
    """

    kv_canary_mode = "log"
    extra_server_args = ("--kv-canary-real-data", "partial")

    target_group: ClassVar[TargetGroupKind]

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0.1",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        super().setUpClass()

    def test_real_kv_used_perturbation_reports_real_kv_hash_violation(self) -> None:
        """Verify active real KV perturbation reports a real KV hash violation."""
        self.send_parallel_requests(n=4)
        self.assert_per_forward_violation_reported(
            fail_reason="real_kv_hash",
            target_group=self.target_group,
        )


class TestPerturbRealKvUsedMhaFull(_PerturbRealKvUsedBase, unittest.TestCase):
    model_mode = "mha"
    target_group = TargetGroupKind.FULL


class TestPerturbRealKvUsedSwaFull(_PerturbRealKvUsedBase, unittest.TestCase):
    model_mode = "swa"
    target_group = TargetGroupKind.FULL


class TestPerturbRealKvUsedSwaSwa(_PerturbRealKvUsedBase, unittest.TestCase):
    model_mode = "swa"
    target_group = TargetGroupKind.SWA


if __name__ == "__main__":
    unittest.main()
