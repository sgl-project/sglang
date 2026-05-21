from __future__ import annotations

import unittest
from typing import ClassVar, Literal

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class _PerturbRealKvUsedBase(CanaryE2EBase):
    """Perturb point (b): flip the first byte of an active req's currently-used KV slot.

    With sweep OFF, the only way to surface this corruption is the per-forward
    HEAD/TAIL real_kv_hash check on the targeted group (FULL or SWA). Subclasses set
    ``mode`` and ``target_group``; the FULL/SWA suffix in the violation launch_tag must
    match ``target_group``.
    """

    kv_canary_mode = "log"

    target_group: ClassVar[Literal["full", "swa"]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.perturb_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0.1",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": cls.target_group,
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        super().setUpClass()

    def test_real_kv_hash_violation_observed(self) -> None:
        self.send_parallel_requests(n=4, max_new_tokens=200)
        suffix = "FULL" if self.target_group == "full" else "SWA"
        try:
            self.assert_violation_logged(
                launch_tag_pattern=f"HEAD_*_{suffix}",
                fail_reason="real_kv_hash",
            )
        except AssertionError:
            self.assert_violation_logged(
                launch_tag_pattern=f"TAIL_*_{suffix}",
                fail_reason="real_kv_hash",
            )


class TestPerturbRealKvUsedMhaFull(_PerturbRealKvUsedBase, unittest.TestCase):
    mode = "mha"
    target_group = "full"


class TestPerturbRealKvUsedSwaFull(_PerturbRealKvUsedBase, unittest.TestCase):
    mode = "swa"
    target_group = "full"


class TestPerturbRealKvUsedSwaSwa(_PerturbRealKvUsedBase, unittest.TestCase):
    mode = "swa"
    target_group = "swa"


if __name__ == "__main__":
    unittest.main()
