import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import LinearWithLoRA


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
