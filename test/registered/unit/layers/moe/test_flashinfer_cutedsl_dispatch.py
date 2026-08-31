import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl as cutedsl_runner
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_flashinfer_prefill_returns_standard_combine_input():
    dispatch_output = StandardDispatchOutput(
        hidden_states=torch.empty(2, 16, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.empty(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            router_logits=None,
        ),
    )
    expected_output = torch.empty(2, 16, dtype=torch.bfloat16)
    wrapper = Mock()
    wrapper.run.return_value = expected_output
    quant_info = SimpleNamespace(
        wrapper=wrapper,
        use_per_token_activation=False,
        a1_scale=torch.tensor(1.0),
        a2_scale=torch.tensor(1.0),
        w13_weight=object(),
        w13_weight_sf=object(),
        w1_alpha=object(),
        w2_weight=object(),
        w2_weight_sf=object(),
        w2_alpha=object(),
    )
    runner_config = SimpleNamespace(activation="silu")

    with patch(
        "sglang.srt.layers.quantization.fp4_utils.fp4_quantize",
        return_value=(
            torch.empty(2, 8, dtype=torch.uint8),
            torch.empty(2, 1, dtype=torch.float8_e4m3fn),
        ),
    ):
        result = cutedsl_runner.fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
            dispatch_output, quant_info, runner_config
        )

    assert isinstance(result, StandardCombineInput)
    assert result.hidden_states is expected_output
    wrapper.run.assert_called_once()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
