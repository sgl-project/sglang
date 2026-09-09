from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    MergedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)

_RANK_PATCH = "sglang.multimodal_gen.runtime.layers.lora.linear.get_tp_rank"


class _FakeLinearMethod:
    def apply(
        self,
        layer: nn.Module,
        input_: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(input_, layer.weight, bias)


class _FakeParallelLinear(nn.Module):
    def __init__(
        self,
        weight_shape: tuple[int, int],
        *,
        output_sizes: list[int] | None = None,
        output_partition_sizes: list[int] | None = None,
        input_size_per_partition: int | None = None,
        input_is_parallel: bool = True,
        reduce_results: bool = False,
        tp_size: int = 2,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(weight_shape))
        self.output_sizes = output_sizes
        self.output_partition_sizes = output_partition_sizes
        self.input_size_per_partition = input_size_per_partition
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.tp_size = tp_size
        self.bias = None
        self.skip_bias_add = False
        self.gather_output = False
        self.quant_method = _FakeLinearMethod()


def test_stacked_merged_column_lora_merges_tp_local_delta():
    base = _FakeParallelLinear(
        (12, 6),
        output_sizes=[8, 8, 8],
        output_partition_sizes=[4, 4, 4],
    )
    layer = MergedColumnParallelLinearWithLoRA(base, snapshot_base=False)
    lora_a = torch.arange(36, dtype=torch.float32).reshape(3, 2, 6)
    lora_b = torch.arange(48, dtype=torch.float32).reshape(3, 8, 2)
    data = torch.zeros(12, 6)
    lora_entry = (
        nn.Parameter(lora_a),
        nn.Parameter(lora_b),
        None,
        1.0,
        2,
        2,
        None,
    )

    with patch(_RANK_PATCH, return_value=1):
        layer._merge_lora_into_data(data, [lora_entry])

    expected = (lora_b[:, 4:8] @ lora_a).reshape(12, 6)
    torch.testing.assert_close(data, expected)


def test_stacked_merged_column_lora_applies_tp_local_dynamic_delta():
    base = _FakeParallelLinear(
        (12, 6),
        output_sizes=[8, 8, 8],
        output_partition_sizes=[4, 4, 4],
    )
    layer = MergedColumnParallelLinearWithLoRA(base, snapshot_base=False)
    layer.lora_A = nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(3, 2, 6))
    layer.lora_B = nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(3, 8, 2))
    layer.lora_rank = 2
    layer.lora_alpha = 2
    layer.disable_lora = False

    input_ = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    with patch(_RANK_PATCH, return_value=1):
        output, output_bias = layer(input_)

    expected_b = layer.lora_B[:, 4:8]
    expected = torch.einsum(
        "...nr,nor->...no",
        torch.einsum("...i,nri->...nr", input_, layer.lora_A),
        expected_b,
    ).flatten(start_dim=-2)
    torch.testing.assert_close(output, expected)
    assert output_bias is None


def test_row_parallel_lora_applies_tp_local_dynamic_delta():
    base = _FakeParallelLinear((5, 3), input_size_per_partition=3)
    layer = RowParallelLinearWithLoRA(base, snapshot_base=False)
    layer.lora_A = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(2, 6))
    layer.lora_B = nn.Parameter(torch.arange(10, dtype=torch.float32).reshape(5, 2))
    layer.lora_rank = 2
    layer.lora_alpha = 2
    layer.disable_lora = False

    input_ = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    with patch(_RANK_PATCH, return_value=1):
        output, output_bias = layer(input_)

    expected = input_ @ layer.lora_A[:, 3:6].T @ layer.lora_B.T
    torch.testing.assert_close(output, expected)
    assert output_bias is None
