"""Unit tests for the FlashInfer MoE token dispatcher."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher import flashinfer as flashinfer_module
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashinferDispatcher(CustomTestCase):
    @patch.object(flashinfer_module, "get_dp_global_num_tokens", return_value=None)
    def test_dispatch_casts_topk_ids_to_int32(self, _get_dp_global_num_tokens):
        dispatcher = FlashinferDispatcher.__new__(FlashinferDispatcher)
        dispatcher.quant_config = {}
        dispatcher.ep_size = 1
        dispatcher.max_num_tokens = 8
        dispatcher.invalid_token_expert_id = 4
        dispatcher.payload_in_workspace = False
        dispatcher.moe_a2a = MagicMock()
        dispatcher.moe_a2a.dispatch.side_effect = (
            lambda topk_ids, payloads, *_args, **_kwargs: (
                payloads[0],
                payloads[1],
                payloads[2],
            )
        )

        hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
        input_topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            topk_ids=input_topk_ids,
            router_logits=torch.zeros((2, 4), dtype=torch.float32),
        )

        output = dispatcher.dispatch(hidden_states, topk_output)

        call = dispatcher.moe_a2a.dispatch.call_args
        self.assertEqual(call.args[0].dtype, torch.int32)
        self.assertEqual(call.args[1][1].dtype, torch.int32)
        self.assertEqual(output.topk_output.topk_ids.dtype, torch.int32)
        self.assertEqual(input_topk_ids.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
