# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 drops the text encoder layers after the last hidden state it reads.

FLUX.2-dev reads hidden states 10, 20 and 30 of its 40-layer Mistral3 encoder,
and FLUX.2-klein reads 9, 18 and 27 of its 36-layer Qwen3 encoder. CPU only --
no weights, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinBasePipelineConfig,
    Flux2KleinPipelineConfig,
    Flux2PipelineConfig,
    FluxPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux_finetuned import (
    Flux2FinetunedPipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    _drop_unread_encoder_layers,
    _maybe_drop_unread_encoder_layers,
)
from sglang.multimodal_gen.runtime.models.encoders.mistral_3 import (
    Mistral3ForConditionalGeneration,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3ForCausalLM

FLUX2_CONFIGS = [
    Flux2PipelineConfig,
    Flux2KleinPipelineConfig,
    Flux2KleinBasePipelineConfig,
    Flux2FinetunedPipelineConfig,
]


@pytest.mark.parametrize("config_cls", FLUX2_CONFIGS)
def test_postprocess_reads_exactly_the_declared_layers(config_cls):
    config = config_cls()
    layers = config.text_encoder_output_layers
    torch.manual_seed(0)
    hidden_states = [torch.randn(2, 5, 3) for _ in range(41)]

    out = config.postprocess_text_funcs[0](
        SimpleNamespace(hidden_states=hidden_states), None
    )

    expected = torch.cat([hidden_states[k] for k in layers], dim=-1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_flux1_does_not_declare_output_layers():
    assert getattr(FluxPipelineConfig(), "text_encoder_output_layers", None) is None


class _AddLayer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, position_ids, hidden_states, residual, attention_lengths):
        return hidden_states * 2 + self.i + 1, residual


class _ScaleNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states * 10, None


def _fake_qwen3(num_layers):
    model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(output_hidden_states=True)
    model.layers = nn.ModuleList(_AddLayer(i) for i in range(num_layers))
    model.norm = _ScaleNorm()
    model.get_input_embeddings = lambda ids: torch.ones(*ids.shape, 4)
    return model


def test_qwen3_keeps_the_read_hidden_states():
    model = _fake_qwen3(36)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    read = Flux2KleinPipelineConfig().text_encoder_output_layers

    full = model(input_ids=input_ids).hidden_states
    assert _drop_unread_encoder_layers(model, read) == 36 - (max(read) + 1)
    trimmed = model(input_ids=input_ids).hidden_states

    assert len(model.layers) == max(read) + 1
    for k in read:
        torch.testing.assert_close(trimmed[k], full[k], rtol=0, atol=0)


def test_keeping_only_max_layers_would_read_the_normed_state():
    """Layer max(k) must stay: the last hidden state is the normed output."""
    model = _fake_qwen3(8)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    full = model(input_ids=input_ids).hidden_states

    del model.layers[6:]
    trimmed = model(input_ids=input_ids).hidden_states

    assert not torch.equal(trimmed[6], full[6])


def test_mistral3_layers_are_found_through_layer_names():
    model = Mistral3ForConditionalGeneration.__new__(Mistral3ForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.layers = nn.ModuleList(nn.Identity() for _ in range(40))
    read = Flux2PipelineConfig().text_encoder_output_layers

    assert _drop_unread_encoder_layers(model, read) == 40 - (max(read) + 1)
    assert len(model.language_model.layers) == max(read) + 1


@pytest.mark.parametrize(
    "output_layers, component_name, expected_layers",
    [
        ((2, 4), "text_encoder", 5),
        ((2, 4), "text_encoder_2", 8),
        (None, "text_encoder", 8),
        ((2, -1), "text_encoder", 8),
        ((20,), "text_encoder", 8),
    ],
)
def test_drop_is_gated_on_the_pipeline_config(
    output_layers, component_name, expected_layers
):
    model = _fake_qwen3(8)
    pipeline_config = SimpleNamespace()
    if output_layers is not None:
        pipeline_config.text_encoder_output_layers = output_layers
    server_args = SimpleNamespace(pipeline_config=pipeline_config)

    _maybe_drop_unread_encoder_layers(model, server_args, component_name)

    assert len(model.layers) == expected_layers
