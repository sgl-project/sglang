"""TP2/DCP2 parity for the direct symmetric-memory DCP backend."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci

from test_dcp_symm_a2a import SymmA2ATestBase

register_cuda_ci(est_time=360, stage="base-b", runner_config="2-gpu-large")


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


if __name__ == "__main__":
    unittest.main()
