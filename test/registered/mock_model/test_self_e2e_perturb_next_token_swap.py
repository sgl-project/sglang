from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.perturb_e2e_base import MockModelPerturbE2EBase

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


class TestPerturbNextTokenSwap(MockModelPerturbE2EBase):
    """Mock-model self-test: swap two reqs' sampled next tokens at the sampler exit.

    KV path is untouched, so kv_canary KV-side fail_reasons stay silent. The
    token-oracle input check downstream MUST report fail_reason=write_token.
    Validates the input-check link is genuinely active.
    """

    extra_env = {
        "SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB": "0.1",
        "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
    }
    extra_server_args = ("--skip-server-warmup",)

    def test_swap_triggers_input_check_violation_but_kv_paths_silent(self) -> None:
        """Verify next_token swap fires write_token violation while KV reasons stay silent."""
        try:
            self.send_parallel_requests(n=4, timeout=30.0)
        except Exception:
            pass
        self.assert_log_contains("mock_perturb next_token_swap: swapped")
        self.assert_violation_reported(fail_reason="write_token")
        self.assert_violation_absent(fail_reason="real_kv_hash")
        self.assert_violation_absent(fail_reason="position")
        self.assert_violation_absent(fail_reason="chain_hash")


if __name__ == "__main__":
    unittest.main()
