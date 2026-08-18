from contextlib import nullcontext

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as runner_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _dispatch_output(topk_weights, hidden_states_scale=None):
    hidden_states = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    topk_ids = torch.tensor([[1], [0]], dtype=torch.int32)
    topk_output = StandardTopKOutput(topk_weights, topk_ids, None)
    return StandardDispatchOutput(hidden_states, hidden_states_scale, topk_output)


def test_prescale_router_weight_on_input_top1():
    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.bfloat16)
    dispatch_output = _dispatch_output(topk_weights)
    config = MoeRunnerConfig(apply_router_weight_on_input=True)

    result = runner_module._prescale_router_weight_on_input(dispatch_output, config)

    expected = dispatch_output.hidden_states * topk_weights
    torch.testing.assert_close(result.hidden_states, expected)
    assert result.topk_output.topk_weights.dtype == torch.float32
    assert torch.equal(
        result.topk_output.topk_weights,
        torch.ones_like(topk_weights, dtype=torch.float32),
    )
    assert torch.equal(
        result.topk_output.topk_ids, dispatch_output.topk_output.topk_ids
    )


def test_prescale_router_weight_on_input_rejects_unsupported_inputs():
    config = MoeRunnerConfig(apply_router_weight_on_input=True)

    with pytest.raises(AssertionError, match="requires topk=1"):
        runner_module._prescale_router_weight_on_input(
            _dispatch_output(torch.ones(2, 2, dtype=torch.float32)), config
        )

    with pytest.raises(NotImplementedError, match="fp4 all-gather path"):
        runner_module._prescale_router_weight_on_input(
            _dispatch_output(
                torch.ones(2, 1, dtype=torch.float32),
                hidden_states_scale=torch.ones(2, 1, dtype=torch.uint8),
            ),
            config,
        )


def test_run_passes_float32_token_final_scales(monkeypatch):
    captured = {}

    def fake_cutlass_fused_moe(**kwargs):
        captured.update(kwargs)
        return (kwargs["output"],)

    monkeypatch.setattr(
        runner_module,
        "_flashinfer_cutlass_fused_moe",
        lambda: (fake_cutlass_fused_moe, None),
    )
    monkeypatch.setattr(runner_module, "_activation_type", lambda config: None)
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: None)
    monkeypatch.setattr(runner_module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        runner_module, "use_symmetric_memory", lambda *args, **kwargs: nullcontext()
    )

    topk_weights = torch.tensor([[0.25], [0.75]], dtype=torch.bfloat16)
    dispatch_output = _dispatch_output(topk_weights)
    quant_info = runner_module.FlashInferCutlassMoeQuantInfo(
        quant_type="bf16",
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
    )
    config = MoeRunnerConfig(apply_router_weight_on_input=True)

    runner_module._run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=config,
    )

    assert captured["token_final_scales"].dtype == torch.float32
    assert torch.equal(
        captured["token_final_scales"],
        torch.ones_like(topk_weights, dtype=torch.float32),
    )
    torch.testing.assert_close(
        captured["input"], dispatch_output.hidden_states * topk_weights
    )
