import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestTPMoERoutingWeights(unittest.TestCase):
    def test_dispatch_preserves_small_fp32_probabilities(self):
        dispatcher = AscendTPDispatcher(MoeRunnerConfig(num_experts=4, top_k=2))
        weights = torch.tensor(
            [[0.000101, 0.10011], [0.20021, 0.30031]], dtype=torch.float32
        )
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        topk = StandardTopKOutput(weights, ids, torch.zeros(2, 4))
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                hidden = torch.zeros((2, 64), dtype=dtype)
                routing_output = (
                    hidden.repeat_interleave(2, dim=0),
                    torch.arange(4, dtype=torch.int32),
                    torch.ones(4, dtype=torch.int64),
                    None,
                )
                with patch.object(
                    dispatcher.init, "_init_routing", return_value=routing_output
                ):
                    output = dispatcher.dispatch(hidden, topk)
                self.assertEqual(output.topk_weights.dtype, torch.float32)
                torch.testing.assert_close(output.topk_weights, weights, atol=0, rtol=0)
                self.assertFalse(torch.equal(weights, weights.to(dtype).float()))
                self.assertEqual(output.topk_ids.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
