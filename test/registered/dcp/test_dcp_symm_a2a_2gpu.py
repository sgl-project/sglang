"""TP2/DCP2 parity and FP8 KV-cache acceptance for the direct symmetric-memory DCP backend.

Both cases run on the same 2-GPU runner, so a single CI registration covers
them and they share one server-launch harness from ``SymmA2ATestBase``.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci

from test_dcp_symm_a2a import SymmA2ATestBase

register_cuda_ci(est_time=540, stage="base-b", runner_config="2-gpu-large")


class TestDCPSymmA2ATP2(SymmA2ATestBase):
    required_gpus = 2

    def test_bf16_matches_ag_rs_with_graph_and_eager(self):
        for disable_cuda_graph in (False, True):
            with self.subTest(disable_cuda_graph=disable_cuda_graph):
                baseline = self._run_case(
                    tp_size=2,
                    backend="ag_rs",
                    disable_cuda_graph=disable_cuda_graph,
                )
                actual = self._run_case(
                    tp_size=2,
                    backend="symm_a2a",
                    disable_cuda_graph=disable_cuda_graph,
                )
                self.assertIsNotNone(baseline.outputs)
                self.assertIsNotNone(actual.outputs)
                self._assert_backend_parity(baseline.outputs, actual.outputs)


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
