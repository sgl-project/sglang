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
        quant_mode="w4a4",
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


@pytest.mark.parametrize(
    "dispatch_kind", ["none", "flashinfer_prefill", "flashinfer_decode"]
)
@pytest.mark.parametrize("per_token_activation", [False, True])
def test_nvfp4_online_w4a16_keeps_bf16_activations(dispatch_kind, per_token_activation):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.layers.quantization.nvfp4_online import (
        ModelOptNvFp4OnlineFusedMoEMethod,
        NvFp4OnlineConfig,
    )
    from sglang.srt.runtime_context import get_flags, override_platform

    use_flashinfer = dispatch_kind != "none"
    is_decode = dispatch_kind == "flashinfer_decode"
    dispatch_type = FlashinferDispatchOutput if is_decode else StandardDispatchOutput
    fused_func = (
        cutedsl_runner.fused_experts_flashinfer_to_flashinfer_cutedsl_fp4
        if use_flashinfer
        else cutedsl_runner.fused_experts_none_to_flashinfer_cutedsl_fp4
    )
    hidden_states = torch.empty(2, 16, dtype=torch.bfloat16)
    dispatch_output = dispatch_type(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            router_logits=None,
        ),
    )
    expected_output = torch.empty_like(hidden_states)
    wrapper = Mock(quant_mode="w4a16")
    wrapper.run.return_value = expected_output
    scale = torch.ones(1)
    layer = SimpleNamespace(
        _cutedsl_wrapper=wrapper,
        _cutedsl_scales=(scale, scale, scale),
        _cutedsl_input_scale=scale,
        w13_weight=object(),
        w2_weight=object(),
        w13_blockscale_swizzled=object(),
        w2_blockscale_swizzled=object(),
    )
    runner_config = SimpleNamespace(activation="silu")
    activation_quantizer = Mock(
        side_effect=AssertionError("W4A16 must not quantize activations")
    )
    with (
        override_platform(is_blackwell=True),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTEDSL,
            a2a_backend=(
                MoeA2ABackend.FLASHINFER if use_flashinfer else MoeA2ABackend.NONE
            ),
        ),
        envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True),
        envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.override(
            per_token_activation
        ),
        patch(
            "sglang.srt.layers.quantization.fp4_utils.fp4_quantize",
            activation_quantizer,
        ),
        patch.dict(
            sys.modules,
            {
                "flashinfer": SimpleNamespace(
                    SfLayout=SimpleNamespace(layout_linear=None),
                    nvfp4_quantize=activation_quantizer,
                )
            },
        ),
    ):
        config = NvFp4OnlineConfig()
        assert config.use_per_token_activation
        method = ModelOptNvFp4OnlineFusedMoEMethod(config, "model.layers.0.mlp.experts")
        method.moe_runner_config = runner_config
        method.runner = SimpleNamespace(
            run=Mock(
                side_effect=lambda output, info: fused_func(output, info, runner_config)
            )
        )
        result = method.apply(layer, dispatch_output)

    quant_info = method.runner.run.call_args.args[1]
    assert quant_info.quant_mode == "w4a16"
    assert not quant_info.use_per_token_activation
    activation_quantizer.assert_not_called()
    wrapper.run.assert_called_once()
    call = wrapper.run.call_args.kwargs
    assert call["x"] is hidden_states
    assert call["x_sf"] is None
    assert call["per_token_scale"] is None
    assert call["fc2_input_scale"] is None
    assert isinstance(
        result, FlashinferCombineInput if is_decode else StandardCombineInput
    )
    assert result.hidden_states is expected_output


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
