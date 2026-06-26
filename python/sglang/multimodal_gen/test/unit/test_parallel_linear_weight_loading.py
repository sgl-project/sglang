import pytest
import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
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
