import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestFusedMoeNative(CustomTestCase):
    def test_noncontiguous_topk_ids(self):
        torch.manual_seed(0)
        layer = SimpleNamespace(
            num_experts=3,
            w13_weight=torch.randn(3, 4, 2),
            w2_weight=torch.randn(3, 2, 2),
        )
        hidden_states = torch.randn(3, 2)
        topk_weights = torch.tensor(
            [[0.75, 0.25], [0.4, 0.6], [0.9, 0.1]], dtype=torch.float32
        )
        padded_topk_ids = torch.tensor(
            [[0, -1, 2, -1], [1, -1, 0, -1], [2, -1, 1, -1]],
            dtype=torch.int64,
        )
        topk_ids = padded_topk_ids[:, ::2]
        self.assertFalse(topk_ids.is_contiguous())

        config = SimpleNamespace(
            activation="silu",
            apply_router_weight_on_input=False,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        )
        output = moe_forward_native(
            layer,
            hidden_states,
            (topk_weights, topk_ids, None),
            config,
        )
        reference = moe_forward_native(
            layer,
            hidden_states,
            (topk_weights, topk_ids.contiguous(), None),
            config,
        )

        torch.testing.assert_close(output, reference)


if __name__ == "__main__":
    unittest.main()
