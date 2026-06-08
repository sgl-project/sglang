from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.consts import SWA_POOL_SERVER_ARGS
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=160, stage="extra-a", runner_config="1-gpu-small")


class _PerturbReqToTokenBase(CanaryE2EBase):
    kv_canary_mode = CanaryMode.LOG
    extra_env = {
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "0.1",
        "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        # req_to_token perturbation deliberately corrupts the slot mapping
        # by design, which the scheduler's on-idle invariant checker reports
        # as a pool memory leak (perturbed slot is freed, original slot
        # still looks busy). That's expected for this test; disable strict
        # mode so the leak warning doesn't crash the scheduler.
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
    }

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _PerturbReqToTokenBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set model_mode")
        super().setUpClass()

    def test_req_to_token_perturbation_reports_chain_hash_violation(self) -> None:
        """Verify req_to_token perturbation reports a chain hash violation."""
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_per_forward_violation_reported(fail_reason="verify_chain_hash")
        self.maybe_assert_swa_divergence_observed()


class TestPerturbReqToTokenMha(_PerturbReqToTokenBase):
    model_mode = "mha"


class TestPerturbReqToTokenSwa(_PerturbReqToTokenBase):
    model_mode = "swa"
    extra_server_args = SWA_POOL_SERVER_ARGS


if __name__ == "__main__":
    unittest.main()
