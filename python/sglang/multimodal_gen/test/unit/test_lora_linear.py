import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import RowParallelLinearWithLoRA


class _DummyQuantMethod:
    def __init__(self) -> None:
        self.last_bias = None

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
        self.last_bias = bias
        return F.linear(x, layer.weight, bias)


class _DummyRowParallelLinear(nn.Module):
    def __init__(self, *, skip_bias_add: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False
        )
        self.bias = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        self.input_is_parallel = True
        self.tp_size = 1
        self.tp_rank = 0
        self.input_size_per_partition = 16
        self.reduce_results = True
        self.skip_bias_add = skip_bias_add
        self.tp_group = None
        self.quant_method = _DummyQuantMethod()

    def forward(self, input_: torch.Tensor):
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_, bias=bias_)
        output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def test_row_parallel_lora_wrapper_matches_base_bias_semantics() -> None:
    base = _DummyRowParallelLinear(skip_bias_add=False)
    wrapper = RowParallelLinearWithLoRA(base)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)

    expected_out, expected_bias = base(x)
    actual_out, actual_bias = wrapper(x)

    assert base.quant_method.last_bias is base.bias
    assert expected_bias is None
    assert actual_bias is None
    assert torch.equal(actual_out, expected_out)


def test_row_parallel_lora_wrapper_preserves_skip_bias_add() -> None:
    base = _DummyRowParallelLinear(skip_bias_add=True)
    wrapper = RowParallelLinearWithLoRA(base)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)

    expected_out, expected_bias = base(x)
    actual_out, actual_bias = wrapper(x)

    assert base.quant_method.last_bias is None
    assert torch.equal(actual_out, expected_out)
    assert actual_bias is expected_bias is base.bias
