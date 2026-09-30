import sys
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_empty_mxfp8_dispatch_uses_same_payload_dtype_as_nonempty_rank():
    class FakeMoeAlltoAll:
        def dispatch(self, _topk_ids, payloads, *_args, **_kwargs):
            self.payload_dtypes = [payload.dtype for payload in payloads]
            return payloads

    dispatcher = object.__new__(FlashinferDispatcher)
    dispatcher.dispatch_type = FlashinferA2ADispatchType.MXFP8
    dispatcher.hidden_size = 128
    dispatcher.max_num_tokens = 0
    dispatcher.ep_size = 1
    dispatcher.invalid_token_expert_id = 8
    dispatcher.payload_in_workspace = False
    dispatcher.quant_config = {"use_mxfp8": True}
    dispatcher.moe_a2a = FakeMoeAlltoAll()

    hidden_states = torch.empty((0, 128), dtype=torch.bfloat16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.empty((0, 1), dtype=torch.float32),
        topk_ids=torch.empty((0, 1), dtype=torch.int32),
        router_logits=None,
    )

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
        output = dispatcher.dispatch(hidden_states, topk_output)

    assert dispatcher.moe_a2a.payload_dtypes == [
        torch.float8_e4m3fn,
        torch.uint8,
        torch.int32,
        torch.float32,
    ]
    assert output.hidden_states.dtype == torch.float8_e4m3fn
    assert output.hidden_states_scale.dtype == torch.uint8


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
