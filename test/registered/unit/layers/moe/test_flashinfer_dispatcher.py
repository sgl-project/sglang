import sys
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeMoeAlltoAll:
    def dispatch(self, _topk_ids, payloads, *_args, **_kwargs):
        self.payload_dtypes = [payload.dtype for payload in payloads]
        return payloads


def run_dispatch(dispatch_type, hidden_states, quant_config=None):
    dispatcher = object.__new__(FlashinferDispatcher)
    dispatcher.dispatch_type = dispatch_type
    dispatcher.quant_config = quant_config
    dispatcher.hidden_size = 128
    dispatcher.ep_size = 1
    dispatcher.invalid_token_expert_id = 8
    dispatcher.payload_in_workspace = False
    dispatcher.moe_a2a = FakeMoeAlltoAll()

    batch_size = hidden_states.shape[0]
    topk_output = StandardTopKOutput(
        topk_weights=torch.empty((batch_size, 1), dtype=torch.float32),
        topk_ids=torch.empty((batch_size, 1), dtype=torch.int32),
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
    return dispatcher, output


def test_empty_mxfp8_dispatch_uses_same_payload_dtype_as_nonempty_rank():
    dispatcher, output = run_dispatch(
        FlashinferA2ADispatchType.MXFP8,
        torch.empty((0, 128), dtype=torch.bfloat16),
    )

    assert dispatcher.moe_a2a.payload_dtypes == [
        torch.float8_e4m3fn,
        torch.uint8,
        torch.int32,
        torch.float32,
    ]
    assert output.hidden_states.dtype == torch.float8_e4m3fn
    assert output.hidden_states_scale.dtype == torch.uint8


def test_nvfp4_dispatch_without_static_scale_falls_back_to_bf16():
    dispatcher, output = run_dispatch(
        FlashinferA2ADispatchType.NVFP4,
        torch.empty((1, 128), dtype=torch.bfloat16),
        {"input_global_scale": None},
    )

    assert dispatcher.moe_a2a.payload_dtypes == [
        torch.bfloat16,
        torch.int32,
        torch.float32,
    ]
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.hidden_states_scale is None


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
