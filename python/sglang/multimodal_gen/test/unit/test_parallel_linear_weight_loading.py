import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
)
from sglang.multimodal_gen.runtime.models.parameter import PerTensorScaleParameter


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
