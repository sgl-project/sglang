from types import SimpleNamespace

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    _compute_lora_delta,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.upsampling import (
    LTX2LoRASwitchStage,
)


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


def test_dynamic_lora_reuses_inference_weights_without_autograd_tracking():
    base_layer = nn.Linear(4, 3, bias=False)
    layer = LinearWithLoRA(base_layer, lora_rank=2, lora_alpha=2)

    with torch.inference_mode():
        layer.set_lora_weights(
            torch.ones(2, 4),
            torch.ones(3, 2),
            clear_existing=True,
            merge_weights=False,
        )

    assert layer.lora_A.is_inference()
    assert layer.lora_B.is_inference()
    assert not layer.lora_A.requires_grad
    assert not layer.lora_B.requires_grad
    with torch.no_grad():
        sharded_view = layer.lora_B[:2, :]
    assert sharded_view.shape == (2, 2)


def test_ltx2_lora_switch_creates_versioned_adapter_tensors():
    class _Pipeline:
        adapter = None

        @staticmethod
        def should_skip_ltx2_lora_switch_stage():
            return False

        def switch_lora_phase(self, phase, *, batch):
            self.adapter = torch.ones(1)

    pipeline = _Pipeline()
    stage = LTX2LoRASwitchStage(pipeline, "stage2")
    batch = SimpleNamespace(extra={})

    with torch.inference_mode():
        stage.forward(batch, SimpleNamespace())

    assert pipeline.adapter is not None
    assert not pipeline.adapter.is_inference()
    assert batch.extra["ltx2_phase"] == "stage2"
