from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=48, stage="extra-a", runner_config="1-gpu-small")


class TestPerturbRaiseMha(CanaryE2EBase):
    model_mode = "mha"
    kv_canary_mode = CanaryMode.RAISE
    extra_server_args = ("--kv-canary-real-data", "partial", "--skip-server-warmup")
    extra_env = {
        "SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB": "0.1",
        "SGLANG_KV_CANARY_PERTURB_TARGET_GROUP": "full",
        "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
    }

    def test_real_kv_used_perturbation_raises_in_raise_mode(self) -> None:
        """Verify raise mode surfaces real KV perturbation as a logged violation."""
        try:
            self.send_parallel_requests(
                n=4,
                assert_all_success=False,
                timeout=30.0,
            )
        except Exception:
            pass
        self.assert_per_forward_violation_reported(
            fail_reason="verify_real_kv_hash",
            target_group=TargetGroupKind.FULL,
            flush_wait_seconds=3.0,
        )


if __name__ == "__main__":
    unittest.main()
