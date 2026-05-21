from __future__ import annotations

import unittest

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


class TestPerturbRaiseMha(CanaryE2EBase, unittest.TestCase):
    """kv-canary=raise: the first violation must abort the server.

    Uses perturb point (b) real_kv_used on FULL as the simplest deterministic trigger.
    We don't assert process exit directly (race-prone — the tee thread may still be
    draining); instead we assert the violation line landed in the captured log before
    the abort propagated. Client-side, the request may either time out or come back
    with a dropped connection, so the /generate call is wrapped in a best-effort try.
    """

    model_mode = "mha"
    kv_canary_mode = "raise"
    extra_server_args = ("--kv-canary-real-data", "partial")
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
            fail_reason="real_kv_hash",
            target_group=TargetGroupKind.FULL,
            flush_wait_seconds=3.0,
        )


if __name__ == "__main__":
    unittest.main()
