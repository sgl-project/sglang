from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.perturb_e2e_base import MockModelPerturbE2EBase

register_cuda_ci(est_time=43, stage="extra-a", runner_config="1-gpu-small")


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
        self.send_parallel_requests(n=4, timeout=30.0)
        self.assert_log_contains("kv_canary perturb next_token_swap: swapped")
        self.assert_any_launch_tag_violation_reported(fail_reason="write_token")
        self.assert_any_launch_tag_violation_absent(fail_reason="verify_real_kv_hash")
        self.assert_any_launch_tag_violation_absent(fail_reason="verify_position")
        self.assert_any_launch_tag_violation_absent(fail_reason="verify_chain_hash")


if __name__ == "__main__":
    unittest.main()
