from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.pd_fixture import CanaryPDFixture

register_cuda_ci(est_time=180, stage="extra-a", runner_config="2-gpu-large")


class _PDPerturbBase(CanaryPDFixture):
    """Perturb point (d): P-side post-forward flip; both P and D surface the mismatch.

    Both servers run kv-canary in log mode. P-side detects the flip when its
    post-forward bonus-token forward re-verifies the prompt prefix (canary verify plan
    covers the full prefix_len, not only out_cache_loc). D-side detects the same flip
    when its decode forwards re-verify the transferred prefix slots. ``max_new_tokens``
    is bumped well above one so D actually runs decode forwards that exercise the
    verify path.
    """

    __test__ = (
        False  # pytest must not collect the abstract base (target_group is unset)
    )

    target_group: ClassVar[TargetGroupKind]

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_prefill_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB": "1.0",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        super().setUpClass()

    def test_p_side_perturb_surfaces_real_kv_hash_violation_on_both_sides(
        self,
    ) -> None:
        # max_new_tokens=100 forces D-side to actually run decode forwards (not just
        # return the prefill bonus token), giving canary verify on the transferred
        # prefix many chances to fire.
        self.send_parallel_short_requests(n=4, max_new_tokens=100)
        for side in ("prefill", "decode"):
            self.assert_per_forward_violation_reported(
                fail_reason="real_kv_hash",
                target_group=self.target_group,
                side=side,
                flush_wait_seconds=4.0,
            )


class TestPDPerturbMhaFull(_PDPerturbBase):
    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    model_mode = "mha"
    target_group = TargetGroupKind.FULL


class TestPDPerturbSwaFull(_PDPerturbBase):
    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    model_mode = "swa"
    target_group = TargetGroupKind.FULL


class TestPDPerturbSwaSwa(_PDPerturbBase):
    __test__ = True  # re-enable collection (base sets __test__ = False, inherited)
    model_mode = "swa"
    target_group = TargetGroupKind.SWA


if __name__ == "__main__":
    unittest.main()
