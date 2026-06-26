from __future__ import annotations

import unittest
from typing import ClassVar

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.pd_fixture import CanaryPDFixture

register_cuda_ci(est_time=180, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=231, stage="extra-a", runner_config="2-gpu-large-amd")


class _PDPerturbBase(CanaryPDFixture):
    target_group: ClassVar[TargetGroupKind]

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _PDPerturbBase:
            raise unittest.SkipTest(
                "abstract base; concrete subclasses set model_mode + target_group"
            )
        cls.extra_prefill_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB": "1.0",
            "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": str(cls.target_group),
            "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        }
        cls.extra_decode_env = {
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB": "0",
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0",
            "SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB": "0",
            "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "0",
        }
        super().setUpClass()

    def test_p_side_perturb_surfaces_real_kv_hash_violation_on_decode_side(
        self,
    ) -> None:
        # send_parallel_short_requests defaults to max_new_tokens=100 so D-side runs
        # decode forwards that exercise canary verify on the transferred prefix.
        #
        # distinct_prompts is required for the violation to surface reliably:
        # with a shared prompt, P-side radix caching rewrites each request's
        # req_to_token row to the first-inserted (canonical) copy's slots in
        # cache_unfinished_req BEFORE send_kv_chunk snapshots the indices, so a
        # flip on a deduped duplicate slot is freed untransferred and never
        # re-verified on either side.
        self.send_parallel_short_requests(n=4, distinct_prompts=True)
        # D-side: first decode forward re-verifies the transferred prefix slots,
        # so the flip MUST surface as real_kv_hash violation.
        self.assert_per_forward_violation_reported(
            fail_reason="verify_real_kv_hash",
            target_group=self.target_group,
            side="decode",
            flush_wait_seconds=4.0,
        )
        # P-side: flip happens post-TAIL of the prefill forward, and PD prefill
        # does not run another forward on P that would verify the perturbed slot,
        # so P MUST stay silent (no false-positive violations) for this perturb
        # point. If a future canary feature adds post-prefill verify on P, this
        # assert will start failing and should be upgraded to assert the
        # violation on P-side too.
        self.assert_no_violation(side="prefill", wait_seconds=0.5)


class TestPDPerturbMhaFull(_PDPerturbBase):
    model_mode = "mha"
    target_group = TargetGroupKind.FULL


class TestPDPerturbSwaFull(_PDPerturbBase):
    model_mode = "swa"
    target_group = TargetGroupKind.FULL


class TestPDPerturbSwaSwa(_PDPerturbBase):
    model_mode = "swa"
    target_group = TargetGroupKind.SWA


if __name__ == "__main__":
    unittest.main()
