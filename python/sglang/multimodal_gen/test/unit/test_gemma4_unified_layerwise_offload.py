"""The native Gemma-4-Unified encoder (LTX-2.5) is layerwise-offloadable and exact."""

from unittest.mock import patch

import pytest
import torch

transformers = pytest.importorskip("transformers")
gemma4_unified = pytest.importorskip("transformers.models.gemma4_unified")

from sglang.multimodal_gen.runtime.managers.memory_managers import layerwise_offload
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    _host_resident_tables,
    detach_host_resident_tables,
    restore_host_resident_tables,
)
from sglang.multimodal_gen.runtime.models.encoders.gemma_4_unified import (
    NATIVE_LAYERWISE_ADAPTERS,
)


def _tiny_model() -> torch.nn.Module:
    config = gemma4_unified.Gemma4UnifiedConfig(
        text_config=dict(
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
        ),
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
    return transformers.AutoModel.from_config(config).eval()


def _encode(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
        )
    return torch.stack(out.hidden_states)


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
