# SPDX-License-Identifier: Apache-2.0
"""Quantization-detection contract for the MiniMax H3 Qwen3-VL text encoder.

H3 points `--text-encoder-path` at a stock Qwen3-VL release, so the encoder has
to adopt whatever scheme that checkpoint ships -- BF16 or a pre-quantized FP8
drop-in such as Qwen3-VL-32B-Instruct-FP8.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Qwen3VLConfig,
)
from sglang.multimodal_gen.runtime.models.encoders import (
    minimax_h3_qwen3vl as encoder_module,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)

# Mirrors the quantization_config block of Qwen3-VL-32B-Instruct-FP8.
FP8_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "ignored_layers": [
        "lm_head",
        "model.visual.merger.linear_fc1",
        "visual.blocks.0.attn.qkv",
    ],
    "weight_block_size": [128, 128],
}


def _config(quantization_config=None) -> MiniMaxH3Qwen3VLConfig:
    config = MiniMaxH3Qwen3VLConfig()
    arch = config.arch_config
    arch.text_config = SimpleNamespace(
        hidden_size=5120,
        intermediate_size=25600,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=64,
        output_hidden_states=True,
        use_cache=True,
    )
    if quantization_config is not None:
        arch.quantization_config = quantization_config
    return config


def test_bf16_checkpoint_stays_unquantized():
    config = _config()
    config.post_diffusers_config_update()

    assert config.quant_config is None
    assert config.arch_config.num_hidden_layers == MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER


def test_fp8_checkpoint_builds_block_quant_config():
    config = _config(FP8_QUANT_CONFIG)
    config.post_diffusers_config_update()

    quant_config = config.quant_config
    assert quant_config is not None
    assert quant_config.get_name() == "fp8"
    assert quant_config.is_checkpoint_fp8_serialized
    assert quant_config.weight_block_size == [128, 128]
    assert quant_config.activation_scheme == "dynamic"
    # The layer trim still applies: H3 reads hidden_states[50] either way.
    assert (
        config.arch_config.text_config.num_hidden_layers
        == MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    )


def test_fp8_ignored_layers_cover_the_vision_tower():
    """Vision modules must stay unquantized -- they ship no scales."""
    config = _config(FP8_QUANT_CONFIG)
    config.post_diffusers_config_update()

    # Fp8Config strips the leading "model." so prefixes match module paths.
    ignored = config.quant_config.ignored_layers
    assert "visual.merger.linear_fc1" in ignored
    assert "visual.blocks.0.attn.qkv" in ignored


def test_unsupported_quant_method_is_rejected():
    config = _config({**FP8_QUANT_CONFIG, "quant_method": "awq"})
    with pytest.raises(ValueError, match="only supports 'fp8'"):
        config.post_diffusers_config_update()


def test_quant_config_object_form_is_accepted():
    """transformers may hand back a config object rather than a dict."""

    class _QuantConfigObject:
        def to_dict(self):
            return dict(FP8_QUANT_CONFIG)

    config = _config(_QuantConfigObject())
    config.post_diffusers_config_update()

    assert config.quant_config is not None
    assert config.quant_config.weight_block_size == [128, 128]


class _DeviceRecordingQuantMethod:
    """Stands in for Fp8LinearMethod, recording where it was asked to run."""

    def __init__(self) -> None:
        self.seen_devices: list[torch.device] = []

    def process_weights_after_loading(self, layer) -> None:
        self.seen_devices.append(layer.weight.device)


class _FakeQuantizedEncoder(nn.Module):
    """Borrows the methods under test; a real encoder needs the full checkpoint."""

    _module_param_device = staticmethod(
        MiniMaxH3Qwen3VLEncoder._module_param_device,
    )
    _process_weights_after_loading = (
        MiniMaxH3Qwen3VLEncoder._process_weights_after_loading
    )

    def __init__(self, quant_config) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.quantized = nn.Linear(8, 8, bias=False)
        self.quantized.quant_method = _DeviceRecordingQuantMethod()
        self.plain = nn.Linear(8, 8, bias=False)


def _patch_platform(monkeypatch, *, is_cpu: bool, device: torch.device) -> None:
    monkeypatch.setattr(
        encoder_module.current_platform, "is_cpu", lambda: is_cpu, raising=False
    )
    monkeypatch.setattr(
        encoder_module, "get_local_torch_device", lambda: device, raising=True
    )


def test_unquantized_encoder_skips_the_post_load_pass(monkeypatch):
    _patch_platform(monkeypatch, is_cpu=False, device=torch.device("cpu"))
    model = _FakeQuantizedEncoder(quant_config=None)

    model._process_weights_after_loading()

    assert model.quantized.quant_method.seen_devices == []


def test_cpu_platform_leaves_weights_in_place(monkeypatch):
    """On a CPU-only deployment the host tensors are already the right target."""
    _patch_platform(monkeypatch, is_cpu=True, device=torch.device("cpu"))
    model = _FakeQuantizedEncoder(quant_config=object())

    model._process_weights_after_loading()

    assert model.quantized.quant_method.seen_devices == [torch.device("cpu")]
    assert model.quantized.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_offloaded_weights_are_postprocessed_on_the_accelerator(monkeypatch):
    """Marlin and the ue8m0 requant are CUDA kernels; host weights must be staged."""
    device = torch.device("cuda", 0)
    _patch_platform(monkeypatch, is_cpu=False, device=device)
    model = _FakeQuantizedEncoder(quant_config=object())
    assert model.quantized.weight.device.type == "cpu"

    model._process_weights_after_loading()

    assert model.quantized.quant_method.seen_devices == [device]
    # Handed straight back, so layerwise offload still owns residency.
    assert model.quantized.weight.device.type == "cpu"
    assert model.plain.weight.device.type == "cpu"
