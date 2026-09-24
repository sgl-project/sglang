import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBiasedGroupedTopkMusa(CustomTestCase):
    def test_musa_routes_through_mate_moe_fused_gate(self):
        """On MUSA, biased grouped top-k raised UnboundLocalError instead of
        calling mate's moe_fused_gate, because a later branch of the same
        function imported another kernel under that name."""
        expected = (torch.zeros(4, 8), torch.zeros(4, 8, dtype=torch.int32))
        mate_moe_fused_gate = mock.Mock(return_value=expected)
        with (
            mock.patch.multiple(
                topk_module,
                _is_cuda=False,
                _is_hip=False,
                _use_aiter=False,
                _is_musa=True,
            ),
            mock.patch.object(
                topk_module, "moe_fused_gate", mate_moe_fused_gate, create=True
            ),
        ):
            topk_weights, topk_ids = topk_module.biased_grouped_topk_gpu(
                hidden_states=torch.randn(4, 16),
                gating_output=torch.randn(4, 64),
                correction_bias=torch.zeros(64),
                topk=8,
                renormalize=True,
                num_expert_group=8,
                topk_group=4,
            )

        mate_moe_fused_gate.assert_called_once()
        self.assertIs(topk_weights, expected[0])
        self.assertIs(topk_ids, expected[1])


if __name__ == "__main__":
    unittest.main()
