"""The Gemma-4-Unified encoder (LTX-2.5) matches transformers and is layerwise-offloadable."""

from unittest.mock import patch

import pytest
import torch

transformers = pytest.importorskip("transformers")
gemma4_unified = pytest.importorskip("transformers.models.gemma4_unified")

from sglang.multimodal_gen.configs.models.encoders import Gemma4UnifiedConfig
from sglang.multimodal_gen.runtime.managers.memory_managers import layerwise_offload
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    _host_resident_tables,
    detach_host_resident_tables,
    restore_host_resident_tables,
)
from sglang.multimodal_gen.runtime.models.encoders.gemma_4_unified import (
    NATIVE_LAYERWISE_ADAPTERS,
    Gemma4UnifiedForConditionalGeneration,
)


def _tiny_model(**text_overrides) -> torch.nn.Module:
    text_config = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        vocab_size=512,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
    )
    text_config.update(text_overrides)
    config = gemma4_unified.Gemma4UnifiedConfig(
        text_config=text_config,
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        ),
        audio_config=None,
    )
    torch.manual_seed(0)
    model = transformers.AutoModel.from_config(config).eval()
    # checkpoint values, not the all-ones init
    with torch.no_grad():
        for layer in model.language_model.layers:
            layer.layer_scalar.uniform_(0.05, 1.0)
            for norm in (layer.input_layernorm, layer.post_feedforward_layernorm):
                norm.weight.uniform_(0.5, 1.5)
    return model


def _encode(model: torch.nn.Module, input_ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    return torch.stack(out.hidden_states)


def _native_model(reference: torch.nn.Module) -> torch.nn.Module:
    config = Gemma4UnifiedConfig()
    # what TextEncoderLoader.build_model_config copies from the HF config
    for key, value in reference.config.__dict__.items():
        setattr(config.arch_config, key, value)
    model = Gemma4UnifiedForConditionalGeneration(config).eval()
    loaded = model.load_weights(
        (f"model.{name}", tensor)
        for name, tensor in reference.state_dict().items()
        if model.should_materialize_checkpoint_weight(f"model.{name}")
    )
    assert loaded == {name for name, _ in model.named_parameters()}
    return model


@pytest.mark.usefixtures("single_process_model_parallel")
def test_native_encoder_matches_transformers_bit_for_bit():
    # k_eq_v global layers with one KV head, wider global heads under
    # proportional RoPE, and a prompt longer than the sliding window
    reference = _tiny_model(
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        global_head_dim=64,
        attention_k_eq_v=True,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
    )
    model = _native_model(reference)
    assert isinstance(model, LayerwiseOffloadableModuleMixin)
    k_eq_v = [layer.self_attn.k_eq_v for layer in model.language_model.layers]
    assert k_eq_v == [False, False, True]

    input_ids = torch.randint(1, 500, (2, 40))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :7] = 0  # left padding, as LTX tokenizes
    expected = _encode(reference, input_ids, attention_mask)
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    assert len(out.hidden_states) == reference.config.text_config.num_hidden_layers + 1
    torch.testing.assert_close(torch.stack(out.hidden_states), expected, rtol=0, atol=0)
    torch.testing.assert_close(out.last_hidden_state, expected[-1], rtol=0, atol=0)

    # the vocab table can stay on the host under layerwise offload
    threshold = f"{layerwise_offload.__name__}.HOST_RESIDENT_TABLE_MIN_BYTES"
    with patch(threshold, 1024):
        assert _host_resident_tables(model) == [model.language_model.embed_tokens]
        restore_host_resident_tables(detach_host_resident_tables(model), "cpu")
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    torch.testing.assert_close(torch.stack(out.hidden_states), expected, rtol=0, atol=0)


def test_adapter_keeps_outputs_and_exposes_the_decoder_stack():
    model = _tiny_model()
    input_ids = torch.randint(1, 500, (1, 12))
    reference = _encode(model, input_ids)

    adapter = NATIVE_LAYERWISE_ADAPTERS[type(model).__name__]
    model = adapter(model)

    assert isinstance(model, LayerwiseOffloadableModuleMixin)
    assert type(model).__name__ == "Gemma4UnifiedModel"
    assert isinstance(
        dict(model.named_modules())[model.layer_names[0]], torch.nn.ModuleList
    )
    torch.testing.assert_close(_encode(model, input_ids), reference, rtol=0, atol=0)
    # idempotent
    assert type(adapter(model)) is type(model)


def test_host_resident_embedding_matches_resident_embedding():
    model = NATIVE_LAYERWISE_ADAPTERS["Gemma4UnifiedModel"](_tiny_model())
    input_ids = torch.randint(1, 500, (1, 12))
    reference = _encode(model, input_ids)

    threshold = f"{layerwise_offload.__name__}.HOST_RESIDENT_TABLE_MIN_BYTES"
    with patch(threshold, 1024):
        assert _host_resident_tables(model) == [model.language_model.embed_tokens]
        restore_host_resident_tables(detach_host_resident_tables(model), "cpu")

    torch.testing.assert_close(_encode(model, input_ids), reference, rtol=0, atol=0)
