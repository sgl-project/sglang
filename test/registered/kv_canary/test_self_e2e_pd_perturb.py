from __future__ import annotations

import unittest
from typing import ClassVar, Literal

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.pd_fixture import CanaryPDFixture

register_cuda_ci(est_time=180, stage="extra-a", runner_config="2-gpu-large")


class TestPDBaselineMha(CanaryPDFixture, unittest.TestCase):
    model_mode = "mha"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation_on("prefill")
        self.assert_no_violation_on("decode")


class TestPDBaselineSwa(CanaryPDFixture, unittest.TestCase):
    model_mode = "swa"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation_on("prefill")
        self.assert_no_violation_on("decode")


class _PDPerturbBase(CanaryPDFixture):
    """Perturb point (d): P-side post-forward flip on a slot in out_cache_loc just
    before send_kv_chunk transfers it to D. D's first decode forward HEAD/TAIL
    kernel verifies against the canary written before the flip, so the
    real_kv_hash mismatch shows up on the D-side log only.
    """

    target_group: ClassVar[Literal["full", "swa"]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_prefill_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB": "1.0",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": cls.target_group,
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        super().setUpClass()

    def test_p_side_perturb_surfaces_real_kv_hash_violation_on_d_first_forward(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_d_per_forward_violation_reported(
            fail_reason="real_kv_hash",
            target_group=self.target_group,
        )


class TestPDPerturbMhaFull(_PDPerturbBase, unittest.TestCase):
    model_mode = "mha"
    target_group = "full"


class TestPDPerturbSwaFull(_PDPerturbBase, unittest.TestCase):
    model_mode = "swa"
    target_group = "full"


class TestPDPerturbSwaSwa(_PDPerturbBase, unittest.TestCase):
    model_mode = "swa"
    target_group = "swa"


if __name__ == "__main__":
    unittest.main()
