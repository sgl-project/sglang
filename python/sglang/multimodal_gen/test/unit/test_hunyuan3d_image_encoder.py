# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

import sglang.multimodal_gen.runtime.models.encoders.clip as clip_module
import sglang.multimodal_gen.runtime.models.encoders.dinov2 as dinov2_module
from sglang.multimodal_gen.configs.models.encoders.clip import (
    CLIPTextArchConfig,
    CLIPTextConfig,
    CLIPVisionArchConfig,
)
from sglang.multimodal_gen.configs.models.encoders.clip import (
    CLIPVisionConfig as NativeCLIPVisionConfig,
)
from sglang.multimodal_gen.runtime.models.encoders.dinov2 import Dinov2Model
from sglang.multimodal_gen.runtime.models.encoders.hunyuan3d import (
    CLIPImageEncoder,
    DinoImageEncoder,
    SingleImageEncoder,
)


class _TestAttention(nn.Module):
    def __init__(self, *args, softmax_scale=None, causal=None, **kwargs):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(self, query, key, value):
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
        )
        return output.transpose(1, 2)


def _dino_config() -> Dinov2Config:
    return Dinov2Config(
        hidden_size=16,
        image_size=8,
        patch_size=4,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2,
        use_swiglu_ffn=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
    )


def _initialize_finite(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.05, 0.05)


def test_native_dinov2_preserves_checkpoint_layout_and_layerwise_path(monkeypatch):
    monkeypatch.setattr(dinov2_module, "LocalAttention", _TestAttention)
    model = Dinov2Model(_dino_config()).eval()
    _initialize_finite(model)

    state_keys = set(model.state_dict())
    layer_prefix = "encoder.layer.0"
    assert f"{layer_prefix}.attention.attention.query.weight" in state_keys
    assert f"{layer_prefix}.attention.attention.key.weight" in state_keys
    assert f"{layer_prefix}.attention.attention.value.weight" in state_keys
    assert f"{layer_prefix}.mlp.weights_in.weight" in state_keys
    assert f"{layer_prefix}.layer_scale1.lambda1" in state_keys
    assert model.layer_names == ["encoder.layer"]

    output = model(torch.zeros(1, 3, 8, 8))
    assert output.last_hidden_state.shape == (1, 5, 16)
    assert output.pooler_output.shape == (1, 16)


def test_hunyuan_dino_loader_requires_complete_checkpoint(monkeypatch):
    monkeypatch.setattr(dinov2_module, "LocalAttention", _TestAttention)
    config = _dino_config().to_dict()
    source = DinoImageEncoder(config=config, image_size=8)
    target = DinoImageEncoder(config=config, image_size=8)
    _initialize_finite(source)
    weights = [
        (f"model.{name}", tensor.clone())
        for name, tensor in source.model.state_dict().items()
    ]

    loaded = target.load_weights(weights)

    assert loaded == {f"model.{name}" for name in source.model.state_dict()}
    for name, tensor in source.model.state_dict().items():
        torch.testing.assert_close(target.model.state_dict()[name], tensor)

    incomplete = weights[:-1]
    try:
        target.load_weights(incomplete)
    except RuntimeError as exc:
        assert "checkpoint is missing" in str(exc)
    else:
        raise AssertionError("Incomplete DINOv2 checkpoint should be rejected")


def test_single_image_encoder_dispatches_nested_weights(monkeypatch):
    monkeypatch.setattr(dinov2_module, "LocalAttention", _TestAttention)
    config = {
        "type": "DinoImageEncoder",
        "kwargs": {"config": _dino_config().to_dict(), "image_size": 8},
    }
    source = SingleImageEncoder(config)
    target = SingleImageEncoder(config)
    weights = [(name, tensor.clone()) for name, tensor in source.state_dict().items()]

    loaded = target.load_weights(weights)

    assert loaded == set(source.state_dict())


def test_hunyuan_clip_reuses_native_clip_without_post_norm(monkeypatch):
    class FakeNativeCLIPVisionModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    monkeypatch.setattr(
        CLIPImageEncoder,
        "MODEL_CLASS",
        FakeNativeCLIPVisionModel,
    )
    config = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=8,
        patch_size=4,
    )

    model = CLIPImageEncoder._build_model(config)

    assert model.config.hidden_size == 16
    assert model.config.patch_size == 4
    assert model.config.require_post_norm is False


def test_native_clip_attention_is_causal_only_for_text(monkeypatch):
    class FakeParallelLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(clip_module, "QKVParallelLinear", FakeParallelLinear)
    monkeypatch.setattr(clip_module, "RowParallelLinear", FakeParallelLinear)
    monkeypatch.setattr(clip_module, "LocalAttention", _TestAttention)
    monkeypatch.setattr(clip_module, "get_tp_world_size", lambda: 1)

    vision_config = NativeCLIPVisionConfig(
        arch_config=CLIPVisionArchConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
        )
    )
    text_config = CLIPTextConfig(
        arch_config=CLIPTextArchConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
        )
    )

    vision_attention = clip_module.CLIPAttention(vision_config)
    text_attention = clip_module.CLIPAttention(text_config)

    assert vision_attention.causal is False
    assert vision_attention.attn.causal is False
    assert text_attention.causal is True
    assert text_attention.attn.causal is True
