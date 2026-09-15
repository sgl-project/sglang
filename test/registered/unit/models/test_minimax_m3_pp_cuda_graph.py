import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.minimax_m3 import (
    _normalize_scattered_pp_proxy_tensors_for_cuda_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxM3PPCudaGraph(unittest.TestCase):
    def test_slices_graph_dummy_proxy_to_local_tp_tokens(self):
        proxy = PPProxyTensors(
            {
                "hidden_states": torch.arange(24).reshape(8, 3),
                "residual": torch.arange(24).reshape(8, 3),
            }
        )

        normalized = _normalize_scattered_pp_proxy_tensors_for_cuda_graph(
            proxy,
            positions=torch.zeros(8, dtype=torch.int64),
            attn_tp_size=4,
        )

        self.assertEqual(normalized["hidden_states"].shape, (2, 3))
        self.assertEqual(normalized["residual"].shape, (2, 3))
        torch.testing.assert_close(normalized["residual"], proxy["residual"][:2])

    def test_keeps_runtime_proxy_that_is_already_scattered(self):
        proxy = PPProxyTensors(
            {
                "hidden_states": torch.zeros((2, 3)),
                "residual": torch.zeros((2, 3)),
            }
        )

        normalized = _normalize_scattered_pp_proxy_tensors_for_cuda_graph(
            proxy,
            positions=torch.zeros(8, dtype=torch.int64),
            attn_tp_size=4,
        )

        self.assertIs(normalized, proxy)


if __name__ == "__main__":
    unittest.main()
