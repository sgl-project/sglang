"""FP8 KV-cache acceptance for the direct symmetric-memory DCP backend."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci

from test_dcp_symm_a2a import SymmA2ATestBase

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


class TestDCPSymmA2AFP8KV(SymmA2ATestBase):
    required_gpus = 2

    def test_fp8_kv_matches_ag_rs_or_fails_with_actionable_fallback(self):
        baseline = self._run_case(
            tp_size=2,
            backend="ag_rs",
            disable_cuda_graph=False,
            kv_cache_dtype="fp8_e4m3",
        )
        self.assertIsNotNone(baseline.outputs)
        self._assert_finite_logprobs(baseline.outputs)

        actual = self._run_case(
            tp_size=2,
            backend="symm_a2a",
            disable_cuda_graph=False,
            kv_cache_dtype="fp8_e4m3",
            allow_actionable_fp8_failure=True,
        )
        if actual.outputs is not None:
            self._assert_backend_parity(baseline.outputs, actual.outputs)
        else:
            self.assertIsNotNone(actual.unsupported_reason)


if __name__ == "__main__":
    unittest.main()
