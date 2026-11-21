"""A device-resident LoRA layer takes its unmerge snapshot (pinned) only when first merged."""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    wrap_with_lora_layer,
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@torch.no_grad()
def test_dynamic_lora_takes_no_snapshot_and_merge_restores_base(device):
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16).to(device)
    original = base.weight.detach().clone()
    if device == "cpu":
        layer = LinearWithLoRA(base, lora_rank=2, lora_alpha=2, defer_snapshot=True)
    else:
        layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    assert layer._cpu_weight is None

    A = torch.randn(2, 8, dtype=torch.bfloat16, device=device)
    B = torch.randn(16, 2, dtype=torch.bfloat16, device=device)
    layer.set_lora_weights(A, B, clear_existing=True, merge_weights=False)
    assert layer._cpu_weight is None

    layer.set_lora_weights(A, B, clear_existing=True, merge_weights=True)
    assert layer.merged
    assert layer._cpu_weight is not None
    assert layer.cpu_weight.device.type == "cpu"
    if device == "cuda":
        # Pinned, so the unmerge restore is a fast host-to-device copy.
        assert layer.cpu_weight.is_pinned()
    torch.testing.assert_close(layer.cpu_weight, original.cpu(), rtol=0, atol=0)
    expected = original.float() + B.float() @ A.float()
    torch.testing.assert_close(base.weight.float(), expected, rtol=2e-2, atol=2e-2)

    layer.unmerge_lora_weights()
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)


def test_cpu_layer_still_snapshots_at_wrap():
    base = torch.nn.Linear(4, 4, bias=False)
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    assert layer._cpu_weight is not None
