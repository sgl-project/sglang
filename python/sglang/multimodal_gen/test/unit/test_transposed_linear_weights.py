# SPDX-License-Identifier: Apache-2.0
"""Linear weights stored as [in, out] compute what the [out, in] layout does."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import sglang.multimodal_gen.runtime.layers.linear as linear
from sglang.kernels.ops.diffusion.sites import fused_linear_gelu_site
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    linear_weight_in_out,
    linear_weight_out_in,
    transpose_linear_weights,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.loader.component_loaders import transformer_loader
from sglang.multimodal_gen.runtime.models.dits.flux import FluxTransformer2DModel
from sglang.multimodal_gen.runtime.models.dits.flux_2 import Flux2Transformer2DModel
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    QwenImage21Transformer2DModel,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _load_weights_into_module,
)


class _FakeTpGroup:
    pass


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr(linear, "get_tp_group", lambda: _FakeTpGroup())
    monkeypatch.setattr(linear, "get_group_rank", lambda group: 0)
    monkeypatch.setattr(linear, "get_group_size", lambda group: 1)


def _model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    model.replicated = ReplicatedLinear(12, 20, bias=True)
    model.column = ColumnParallelLinear(12, 12, bias=False)
    model.row = RowParallelLinear(20, 12, bias=True)
    model.merged = MergedColumnParallelLinear(12, [8, 8], bias=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return model


def _outputs(model: nn.Module) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    outputs = {}
    with torch.no_grad():
        for name, layer in model.named_children():
            x = torch.randn(3, 5, layer.input_size, generator=generator)
            outputs[name] = layer(x)[0]
    return outputs


def test_transposed_weights_compute_the_same():
    model = _model()
    expected = _outputs(model)
    weights = {name: layer.weight.clone() for name, layer in model.named_children()}

    assert transpose_linear_weights(model) == 4

    for name, layer in model.named_children():
        assert layer.weight_transposed
        assert layer.weight.is_contiguous()
        assert layer.weight.shape == weights[name].t().shape
        assert torch.equal(linear_weight_out_in(layer), weights[name])
        assert torch.equal(linear_weight_in_out(layer), weights[name].t())
    for name, output in _outputs(model).items():
        torch.testing.assert_close(output, expected[name], msg=name)


def test_transposing_twice_changes_nothing():
    model = _model()
    transpose_linear_weights(model)
    weights = [layer.weight.data for layer in model.children()]

    assert transpose_linear_weights(model) == 0
    for layer, weight in zip(model.children(), weights):
        assert layer.weight.data_ptr() == weight.data_ptr()


def test_a_weight_that_shares_storage_keeps_its_layout():
    model = _model()
    packed = torch.cat([model.column.weight.data, model.column.weight.data.clone()])
    model.column.weight.data = packed[:12]

    transpose_linear_weights(model)

    assert not model.column.weight_transposed
    assert model.column.weight.data_ptr() == packed.data_ptr()


def test_a_weight_update_writes_the_checkpoint_layout():
    """A square [out, in] update must not land transposed; its shape check cannot catch that."""
    model = _model()
    transpose_linear_weights(model)
    torch.manual_seed(2)
    new_weight = torch.randn(12, 12)

    _load_weights_into_module(model, iter([("column.weight", new_weight)]))

    assert model.column.weight_transposed
    assert torch.equal(linear_weight_out_in(model.column), new_weight)
    x = torch.randn(4, 12)
    torch.testing.assert_close(model.column(x)[0], F.linear(x, new_weight))


def test_wrapping_a_lora_restores_the_standard_layout():
    model = _model()
    weight = model.column.weight.clone()
    transpose_linear_weights(model)

    wrapper = BaseLayerWithLoRA(model.column)

    assert not model.column.weight_transposed
    assert torch.equal(wrapper.weight, weight)
    assert torch.equal(wrapper.cpu_weight, weight)


def test_the_fused_gelu_epilogue_skips_a_transposed_weight():
    if not fused_linear_gelu_site._HAS_ADDMM_ACTIVATION:
        pytest.skip("torch._addmm_activation unavailable")
    layer = ReplicatedLinear(12, 20, bias=True, params_dtype=torch.bfloat16)
    assert fused_linear_gelu_site._static_reject_reason(layer) is None

    transpose_linear_weights(layer)

    assert (
        fused_linear_gelu_site._static_reject_reason(layer)
        == "weight stored as [in, out]"
    )


@pytest.mark.parametrize(
    ("model_cls", "rocm", "arch", "expected"),
    [
        (ZImageTransformer2DModel, True, "gfx942:sramecc+:xnack-", True),
        (Flux2Transformer2DModel, True, "gfx942:sramecc+:xnack-", True),
        (ZImageTransformer2DModel, True, "gfx950:sramecc+:xnack-", False),
        (ZImageTransformer2DModel, False, "", False),
        (FluxTransformer2DModel, True, "gfx942:sramecc+:xnack-", False),
        (QwenImageTransformer2DModel, True, "gfx942:sramecc+:xnack-", False),
        (QwenImage21Transformer2DModel, True, "gfx942:sramecc+:xnack-", False),
    ],
)
def test_only_opted_in_models_on_their_archs_transpose(
    monkeypatch, model_cls, rocm, arch, expected
):
    platform = SimpleNamespace(is_rocm=lambda: rocm, get_gcn_arch_name=lambda: arch)
    monkeypatch.setattr(transformer_loader, "current_platform", platform)
    model = model_cls.__new__(model_cls)

    assert transformer_loader._gemm_prefers_transposed_linear_weights(model) is expected
