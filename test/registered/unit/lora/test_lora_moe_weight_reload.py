from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.model_loader.loader import postprocess_weight, restore_weight
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeQuantMethod:
    def __init__(self):
        self.wrapper = None

    def _assert_stale_references_released(self):
        assert not self.wrapper._parameters
        assert self.wrapper._quant_info is None

    def restore_weights_before_loading(self, layer):
        self._assert_stale_references_released()
        parameter = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        parameter.weight_loader = lambda *args: None
        layer.weight = parameter

    def process_weights_after_loading(self, layer):
        self._assert_stale_references_released()
        layer.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def _make_lora_moe_wrapper():
    base_layer = torch.nn.Module()
    base_layer.weight = torch.nn.Parameter(torch.full((2,), -1.0), requires_grad=False)
    quant_method = _FakeQuantMethod()
    base_layer.quant_method = quant_method

    wrapper = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(wrapper)
    wrapper.base_layer = base_layer
    wrapper.register_parameter("weight", base_layer.weight)
    wrapper._quant_info = {"weight": base_layer.weight}
    quant_method.wrapper = wrapper
    wrapper._refresh_quant_info = MethodType(
        lambda self: setattr(self, "quant_info_refreshed", True), wrapper
    )
    return wrapper


def test_lora_moe_parameters_follow_quantized_weight_reload():
    wrapper = _make_lora_moe_wrapper()

    restore_weight(wrapper, torch.device("cpu"))

    assert wrapper.weight is wrapper.base_layer.weight
    assert hasattr(wrapper.weight, "weight_loader")

    postprocess_weight(wrapper, torch.device("cpu"))

    assert wrapper.weight is wrapper.base_layer.weight
    assert torch.equal(wrapper.weight, torch.ones(2))
    assert wrapper.quant_info_refreshed


@pytest.mark.parametrize(
    ("gemm1_alpha", "gate_up_interleaved", "expected_rows"),
    [
        (1.702, True, [6, 22, 7, 23]),
        (4.0, False, [6, 7, 22, 23]),
        (None, True, [6, 7, 22, 23]),
    ],
)
def test_moe_lora_gate_up_layout_follows_runner_config(
    gemm1_alpha, gate_up_interleaved, expected_rows
):
    base_layer = torch.nn.Module()
    base_layer.quant_method = SimpleNamespace(
        get_marlin_quant_info=lambda _: SimpleNamespace()
    )
    base_layer.moe_runner_config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=8,
        hidden_size=32,
        intermediate_size_per_partition=2,
        top_k=4,
        gemm1_alpha=gemm1_alpha,
        gate_up_interleaved=gate_up_interleaved,
    )
    base_layer.dispatcher = SimpleNamespace()
    base_layer.num_local_experts = 8
    base_layer.should_fuse_routed_scaling_factor_in_topk = False
    base_layer.moe_tp_size = 8
    base_layer.moe_tp_rank = 3
    base_layer.intermediate_size_per_partition = 2
    lora_backend = SimpleNamespace(is_moe_lora=False)

    with mock.patch(
        "sglang.srt.layers.moe.utils.get_moe_runner_backend",
        return_value=MoeRunnerBackend.MARLIN,
    ):
        wrapper = FusedMoEWithLoRA(base_layer, lora_backend)

    gate_up_b = torch.arange(32).reshape(32, 1)
    actual = wrapper.slice_moe_lora_b_weights(
        gate_up_b, tp_rank=3, target_module="gate_up_proj_moe"
    )

    assert actual[:, 0].tolist() == expected_rows
