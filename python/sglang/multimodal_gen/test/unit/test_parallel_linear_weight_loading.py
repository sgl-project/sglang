from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)


def _per_tensor_scale(values: list[float]) -> PerTensorScaleParameter:
    return PerTensorScaleParameter(
        data=torch.tensor(values, dtype=torch.float32),
        weight_loader=lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize("loaded_weight", [torch.tensor(0.25), torch.tensor([0.25])])
def test_merged_column_parallel_scalar_scale_load_fills_fused_slots(loaded_weight):
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    layer.tp_size = 2
    param = _per_tensor_scale([-1.0, -2.0])

    layer.weight_loader_v2(param, loaded_weight)

    assert torch.equal(param.data, torch.tensor([0.25, 0.25]))


@pytest.mark.parametrize("loaded_weight", [torch.tensor(0.5), torch.tensor([0.5])])
def test_qkv_parallel_scalar_scale_load_fills_fused_slots(loaded_weight):
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    layer.tp_size = 2
    param = _per_tensor_scale([-1.0, -2.0, -3.0])

    layer.weight_loader_v2(param, loaded_weight)

    assert torch.equal(param.data, torch.tensor([0.5, 0.5, 0.5]))


def test_merged_column_parallel_full_scale_vector_loads_all_fused_slots():
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    layer.tp_size = 1
    param = _per_tensor_scale([-1.0, -2.0])

    layer.weight_loader_v2(param, torch.tensor([0.25, 0.75]))

    assert torch.equal(param.data, torch.tensor([0.25, 0.75]))


def test_qkv_parallel_full_scale_vector_loads_all_fused_slots():
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    layer.tp_size = 1
    param = _per_tensor_scale([-1.0, -2.0, -3.0])

    layer.weight_loader_v2(param, torch.tensor([0.25, 0.5, 0.75]))

    assert torch.equal(param.data, torch.tensor([0.25, 0.5, 0.75]))


def test_merged_replicated_linear_loads_independent_weight_shards():
    layer = MergedReplicatedLinear(3, [2, 1, 1], bias=False)
    first = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    second = torch.tensor([[7.0, 8.0, 9.0]])
    third = torch.tensor([[10.0, 11.0, 12.0]])

    layer.weight_loader(layer.weight, first, 0)
    layer.weight_loader(layer.weight, second, 1)
    layer.weight_loader(layer.weight, third, "v")

    assert torch.equal(layer.weight, torch.cat((first, second, third)))


def test_merged_replicated_linear_loads_independent_scalar_shards():
    layer = MergedReplicatedLinear(3, [2, 1, 1], bias=False)
    scales = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    scales.needs_scalar_to_array = True

    layer.weight_loader(scales, torch.tensor(0.25), "q")
    layer.weight_loader(scales, torch.tensor(0.5), "k")
    layer.weight_loader(scales, torch.tensor(0.75), "v")

    assert torch.equal(scales, torch.tensor([0.25, 0.5, 0.75]))


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_parallel_independent_input_segments(tp_size, dtype):
    sizes = [8, 16]
    weight = torch.arange(96, dtype=dtype).reshape(4, 24)
    inputs = torch.arange(48, dtype=dtype).reshape(2, 24)
    bias = torch.arange(4, dtype=dtype)
    outputs = []
    for rank in range(tp_size):
        layer = RowParallelLinear(
            24,
            4,
            input_sizes=sizes,
            params_dtype=dtype,
            tp_group=SimpleNamespace(world_size=tp_size, rank_in_group=rank),
            reduce_results=False,
            skip_bias_add=True,
        )
        layer.weight.weight_loader(layer.weight, weight)
        layer.bias.weight_loader(layer.bias, bias)
        expected_weight = torch.cat(
            [part.chunk(tp_size, dim=1)[rank] for part in weight.split(sizes, dim=1)],
            dim=1,
        )
        torch.testing.assert_close(layer.weight, expected_weight, atol=0, rtol=0)
        local_input = torch.cat(
            [part.chunk(tp_size, dim=1)[rank] for part in inputs.split(sizes, dim=1)],
            dim=1,
        )
        output, output_bias = layer(local_input)
        torch.testing.assert_close(output_bias, bias, atol=0, rtol=0)
        outputs.append(output.float())
        # already-sharded checkpoints must not be sliced a second time
        layer.weight.is_sharded_weight = True
        layer.weight.weight_loader(layer.weight, expected_weight)
        torch.testing.assert_close(layer.weight, expected_weight, atol=0, rtol=0)
    torch.testing.assert_close(
        sum(outputs) + bias.float(),
        torch.nn.functional.linear(inputs.float(), weight.float(), bias.float()),
        rtol=0.005 if dtype == torch.bfloat16 else 0,
        atol=0,
    )


@pytest.mark.parametrize("factor", [1, 2, 16])
@pytest.mark.parametrize("input_dim", [0, 1])
def test_row_parallel_quantized_input_segments(factor, input_dim):
    layer = RowParallelLinear(
        128,
        4,
        input_sizes=[32, 96],
        bias=False,
        tp_group=SimpleNamespace(world_size=2, rank_in_group=1),
    )
    checkpoint = torch.arange(512 // factor).reshape(4, 128 // factor)
    expected = torch.cat(
        (checkpoint[:, 16 // factor : 32 // factor], checkpoint[:, 80 // factor :]),
        dim=1,
    )
    if input_dim == 0:
        checkpoint, expected = checkpoint.T, expected.T
    param = ModelWeightParameter(
        data=torch.empty_like(expected),
        input_dim=input_dim,
        output_dim=1 - input_dim,
        weight_loader=layer.weight_loader_v2,
    )
    param.weight_loader(param, checkpoint)
    torch.testing.assert_close(param, expected, atol=0, rtol=0)


@pytest.mark.parametrize("sizes", [[3, 5], [4, 8]])
def test_row_parallel_rejects_invalid_input_segments(sizes):
    with pytest.raises((AssertionError, ValueError)):
        RowParallelLinear(
            8,
            4,
            input_sizes=sizes,
            tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        )
