import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingMoeAlltoAll:
    def dispatch(self, token_selected_experts, payloads, *_args, **kwargs):
        self.token_selected_experts = token_selected_experts
        self.payloads = payloads
        self.expert_id_payload_index = kwargs["expert_id_payload_index"]
        return payloads


def _make_dispatcher(*, dispatch_type, max_num_tokens, quant_config):
    dispatcher = object.__new__(FlashinferDispatcher)
    dispatcher.dispatch_type = dispatch_type
    dispatcher.hidden_size = 128
    dispatcher.max_num_tokens = max_num_tokens
    dispatcher.ep_size = 1
    dispatcher.invalid_token_expert_id = 8
    dispatcher.payload_in_workspace = False
    dispatcher.quant_config = quant_config
    dispatcher.moe_a2a = _RecordingMoeAlltoAll()
    return dispatcher


def _dispatch_without_dp(*, dispatcher, hidden_states, topk_output):
    with (
        patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
            return_value=None,
        ),
        patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.is_dp_attention_enabled",
            return_value=False,
        ),
    ):
        return dispatcher.dispatch(hidden_states=hidden_states, topk_output=topk_output)


class TestFlashinferDispatcher(CustomTestCase):
    def test_empty_mxfp8_dispatch_uses_same_payload_dtype_as_nonempty_rank(self):
        dispatcher = _make_dispatcher(
            dispatch_type=FlashinferA2ADispatchType.MXFP8,
            max_num_tokens=0,
            quant_config={"use_mxfp8": True},
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.empty((0, 1), dtype=torch.float32),
            topk_ids=torch.empty((0, 1), dtype=torch.int32),
            router_logits=None,
        )

        output = _dispatch_without_dp(
            dispatcher=dispatcher,
            hidden_states=torch.empty((0, 128), dtype=torch.bfloat16),
            topk_output=topk_output,
        )

        self.assertEqual(
            [payload.dtype for payload in dispatcher.moe_a2a.payloads],
            [torch.float8_e4m3fn, torch.uint8, torch.int32, torch.float32],
        )
        self.assertEqual(output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.hidden_states_scale.dtype, torch.uint8)

    def test_int64_expert_ids_reach_moe_a2a_as_int32(self):
        """Expert IDs reach MoeAlltoAll as int32 even when routing emits int64,
        as static EP dispatch does; MoeAlltoAll rejects any other dtype."""
        dispatcher = _make_dispatcher(
            dispatch_type=FlashinferA2ADispatchType.BF16,
            max_num_tokens=4,
            quant_config={},
        )
        topk_ids = torch.tensor([[0, 5], [7, 2], [3, 3]], dtype=torch.int64)
        topk_output = StandardTopKOutput(
            topk_weights=torch.full((3, 2), 0.5, dtype=torch.float32),
            topk_ids=topk_ids,
            router_logits=None,
        )

        _dispatch_without_dp(
            dispatcher=dispatcher,
            hidden_states=torch.zeros((3, 128), dtype=torch.bfloat16),
            topk_output=topk_output,
        )

        moe_a2a = dispatcher.moe_a2a
        self.assertEqual(moe_a2a.token_selected_experts.dtype, torch.int32)
        self.assertEqual(
            moe_a2a.payloads[moe_a2a.expert_id_payload_index].dtype, torch.int32
        )
        self.assertTrue(
            torch.equal(moe_a2a.token_selected_experts, topk_ids.to(torch.int32))
        )

    def test_bypassed_topk_is_routed_before_moe_a2a(self):
        """Runners that fuse routing hand over bypassed top-k; MoeAlltoAll still
        needs explicit expert IDs and weights, so routing must run first."""
        dispatcher = _make_dispatcher(
            dispatch_type=FlashinferA2ADispatchType.BF16,
            max_num_tokens=4,
            quant_config={},
        )
        hidden_states = torch.zeros((3, 128), dtype=torch.bfloat16)
        router_logits = torch.tensor(
            [[0.1, 3.0, 0.2, 2.0], [5.0, 0.0, 4.0, 0.1], [0.0, 0.1, 0.2, 9.0]]
        )
        topk_output = BypassedTopKOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=TopKConfig(top_k=2, torch_native=True),
        )

        _dispatch_without_dp(
            dispatcher=dispatcher,
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        moe_a2a = dispatcher.moe_a2a
        expert_ids = moe_a2a.token_selected_experts
        self.assertEqual(expert_ids.dtype, torch.int32)
        self.assertEqual(
            expert_ids.sort(dim=-1).values.tolist(), [[1, 3], [0, 2], [2, 3]]
        )
        topk_weights = moe_a2a.payloads[moe_a2a.expert_id_payload_index + 1]
        self.assertEqual(topk_weights.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
