from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_gelu_moe_skips_amx_weight_packing():
    method = UnquantizedFusedMoEMethod()
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(activation="gelu"),
        w13_weight=torch.nn.Parameter(torch.empty(1)),
        w2_weight=torch.nn.Parameter(torch.empty(1)),
    )

    with (
        patch("sglang.srt.layers.quantization.unquant._is_cpu", True),
        patch("sglang.srt.layers.quantization.unquant._is_cpu_amx_available", True),
        patch(
            "sglang.srt.layers.quantization.unquant._amx_process_weight_after_loading"
        ) as pack,
    ):
        method.process_weights_after_loading(layer)

    pack.assert_not_called()


def test_gelu_moe_uses_native_cpu_forward():
    method = UnquantizedFusedMoEMethod()
    method.moe_runner_config = SimpleNamespace(activation="gelu")
    hidden_states = torch.randn(2, 4)
    dispatch_output = SimpleNamespace(hidden_states=hidden_states, topk_output=object())

    with (
        patch(
            "sglang.srt.layers.quantization.unquant.use_intel_amx_backend",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.moe.fused_moe_native.moe_forward_native",
            return_value=hidden_states,
        ) as forward,
    ):
        output = method.forward_cpu(SimpleNamespace(), dispatch_output)

    forward.assert_called_once()
    assert output.hidden_states is hidden_states
