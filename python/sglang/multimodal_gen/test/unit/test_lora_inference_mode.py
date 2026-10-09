from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    _compute_lora_delta,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
    WeightOnlyFP8Linear,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context


@pytest.mark.parametrize("kind", ["linear", "replicated", "fp8"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_request_lora_scale_preserves_delta_offset_and_base(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if kind == "replicated":
        base = ReplicatedLinear(4, 3, bias=False)
    elif kind == "fp8":
        base = WeightOnlyFP8Linear(4, 3, bias=False, enable_fused_w8a8=False)
    else:
        base = nn.Linear(4, 3, bias=False)
    base = base.to(device)
    with torch.no_grad():
        base.weight.copy_(torch.ones(3, 4, device=device))
        if kind == "fp8":
            base.weight_scale.fill_(1.0)
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    layer.set_lora_weights(
        torch.ones(2, 4, device=device),
        torch.ones(3, 2, device=device),
        strength=0.5,
        merge_weights=False,
        output_offset=torch.full((3,), 2.0, device=device),
    )
    x = torch.ones(2, 4, device=device)
    base_weight = base.weight.detach().clone()
    batch = SimpleNamespace(runtime_lora_scale=0.0)
    for scale in (0.0, 1.0, 0.5, 0.0):
        batch.runtime_lora_scale = scale
        with set_forward_context(0, None, forward_batch=batch):
            actual = layer(x)
        if kind == "replicated":
            actual, output_bias = actual
            assert output_bias is None
        torch.testing.assert_close(
            actual.float(), torch.full((2, 3), 4.0 + 5.0 * scale, device=device)
        )
        torch.testing.assert_close(base.weight.float(), base_weight.float())
    if kind == "fp8":
        assert not layer.can_merge_base_weight
        with pytest.raises(ValueError, match="only supports dynamic"):
            layer.set_lora_weights(torch.ones(2, 4), torch.ones(3, 2))


def test_stacked_lora_delta_preserves_projection_order():
    x = torch.tensor([[2.0, 3.0]])
    lora_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    lora_b = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    actual = _compute_lora_delta(x, lora_a, lora_b)

    torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0, 9.0, 12.0]]))


def test_lora_merge_unmerge_handles_inference_base_weight():
    with torch.inference_mode():
        base_layer = nn.Linear(4, 3, bias=False)

    layer = LinearWithLoRA(base_layer, lora_rank=2, lora_alpha=2)
    base_weight = layer.cpu_weight.clone()

    assert layer.base_layer.weight.is_inference()
    assert not base_weight.is_inference()

    lora_a = torch.ones(2, 4)
    lora_b = torch.full((3, 2), 0.5)
    expected_merged = base_weight + lora_b @ lora_a

    with torch.inference_mode(False):
        layer.set_lora_weights(
            lora_a,
            lora_b,
            clear_existing=True,
            merge_weights=True,
        )

    assert layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, expected_merged)

    with torch.inference_mode(False):
        layer.unmerge_lora_weights()

    assert not layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, base_weight)
