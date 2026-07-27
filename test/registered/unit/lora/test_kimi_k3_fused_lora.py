from contextlib import nullcontext
from types import MethodType

import torch
from torch import nn

import sglang.srt.models.kimi_k3 as kimi_k3
from sglang.srt.lora.layers import (
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _lora_wrapper(wrapper_cls, weight: torch.Tensor, delta: float):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    nn.Module.__init__(wrapper)
    wrapper.weight = nn.Parameter(weight, requires_grad=False)
    wrapper.set_lora = True
    wrapper.lora_inputs = []

    def apply_lora(self, base_output, input_):
        self.lora_inputs.append(input_.clone())
        base_output.add_(delta)
        return base_output

    wrapper.apply_lora = MethodType(apply_lora, wrapper)
    return wrapper


class _SharedExperts(nn.Module):
    def __init__(self, gate_up_proj, down_proj=None):
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.forward_gate_up = None

    def forward_from_gate_up(self, gate_up):
        self.forward_gate_up = gate_up.clone()
        return gate_up[:, :2]


def _empty_moe():
    moe = KimiK3MoE.__new__(KimiK3MoE)
    nn.Module.__init__(moe)
    return moe


class _IdentityQuantMethod:
    def apply(self, layer, input_, bias=None):
        return input_.clone()


def test_row_parallel_lora_preserves_k3_input_transform():
    base_layer = nn.Module()
    base_layer.input_is_parallel = True
    base_layer.tp_size = 1
    base_layer.tp_rank = 0
    base_layer.bias = None
    base_layer.skip_bias_add = False
    base_layer.reduce_results = False
    base_layer.quant_method = _IdentityQuantMethod()
    base_layer.lora_input_transform = lambda tensor: tensor * 3

    wrapper = RowParallelLinearWithLoRA.__new__(RowParallelLinearWithLoRA)
    nn.Module.__init__(wrapper)
    wrapper.base_layer = base_layer
    wrapper.set_lora = False

    output, output_bias = wrapper(torch.tensor([[1.0, 2.0]]))

    assert torch.equal(output, torch.tensor([[3.0, 6.0]]))
    assert output_bias is None
