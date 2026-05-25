from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase
from sglang.test.kv_canary.swa_test_pool_config import SWA_POOL_SERVER_ARGS

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")


class _PerturbReqToTokenBase(CanaryE2EBase):
    __test__ = False

    kv_canary_mode = CanaryMode.LOG
    extra_env = {
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "0.1",
        "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
    }

    def test_req_to_token_perturbation_reports_chain_hash_violation(self) -> None:
        """Verify req_to_token perturbation reports a chain hash violation."""
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_per_forward_violation_reported(fail_reason="chain_hash")
        self.maybe_assert_swa_divergence_observed()


class TestPerturbReqToTokenMha(_PerturbReqToTokenBase):
    __test__ = True

    model_mode = "mha"


class TestPerturbReqToTokenSwa(_PerturbReqToTokenBase):
    __test__ = True

    model_mode = "swa"
    extra_server_args = SWA_POOL_SERVER_ARGS


if __name__ == "__main__":
    unittest.main()
