# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import CLIPTextConfig, CLIPTextModel

from sglang.multimodal_gen.configs.models.encoders import (
    CLIPTextConfig as NativeCLIPTextConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.models.encoders.clip import (
    CLIPTextModel as NativeCLIPTextModel,
)
from sglang.multimodal_gen.runtime.models.encoders.clip import (
    CLIPTextModelWithProjection,
)
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_pipeline import (
    Kandinsky6TI2VAPipeline,
)
from sglang.srt.runtime_context import get_parallel


def test_clip_uses_shared_native_loader():
    assert "text_encoder_2" not in Kandinsky6TI2VAPipeline.component_loaders
    assert "text_encoder_2" in Kandinsky6TI2VAPipelineConfig().native_only_components
    for architecture in ("CLIPTextModel", "Qwen2_5_VLForConditionalGeneration"):
        loader = ComponentLoader.for_component_type(
            "text_encoder", "transformers", architecture
        )
        assert type(loader) is TextEncoderLoader


@pytest.mark.parametrize("eos_token_id", [2, 31])
@pytest.mark.parametrize("masked", [False, True])
def test_clip_bare_and_prefixed_checkpoints_match(tmp_path, eos_token_id, masked):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = (
        CLIPTextModel(
            CLIPTextConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=8,
                bos_token_id=1,
                eos_token_id=eos_token_id,
            )
        )
        .to(device)
        .eval()
    )
    config = NativeCLIPTextConfig()
    config.update_model_arch(reference.config.to_dict())
    config.post_diffusers_config_update()
    weights = {name: value.cpu() for name, value in reference.state_dict().items()}
    save_file(weights, str(tmp_path / "prefixed.safetensors"))
    save_file(
        {name.removeprefix("text_model."): value for name, value in weights.items()},
        str(tmp_path / "bare.safetensors"),
    )
    ids = torch.tensor([[1, 5, 31, 31, 0], [1, 6, 7, 31, 0]], device=device)
    mask = (
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], device=device)
        if masked
        else None
    )
    with get_parallel().override(
        tp_size=1, tp_rank=0, tp_group=None, attn_tp_size=1, attn_tp_rank=0
    ):
        models = []
        for checkpoint in ("prefixed", "bare"):
            model = NativeCLIPTextModel(config).to(device).eval()
            loaded = model.load_weights(
                load_file(str(tmp_path / f"{checkpoint}.safetensors")).items()
            )
            assert loaded == set(dict(model.named_parameters()))
            models.append(model)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(
                models[1].state_dict()[name], value, rtol=0, atol=0
            )
        with torch.no_grad():
            expected = reference(ids, attention_mask=mask, output_hidden_states=True)
            outputs = [
                model(ids, attention_mask=mask, output_hidden_states=True)
                for model in models
            ]
    for prefixed, bare, reference_output in (
        (
            outputs[0].last_hidden_state,
            outputs[1].last_hidden_state,
            expected.last_hidden_state,
        ),
        (outputs[0].pooler_output, outputs[1].pooler_output, expected.pooler_output),
        (
            tuple(outputs[0].hidden_states),
            tuple(outputs[1].hidden_states),
            expected.hidden_states,
        ),
    ):
        torch.testing.assert_close(prefixed, bare, rtol=0, atol=0)
        torch.testing.assert_close(prefixed, reference_output, rtol=2e-5, atol=2e-5)


def test_bare_text_checkpoint_preserves_projection_weight():
    config = NativeCLIPTextConfig()
    config.update_model_arch(
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "projection_dim": 8,
        }
    )
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        model = CLIPTextModelWithProjection(config)
    weight = torch.randn_like(model.text_projection.weight)
    assert model.load_weights([("text_projection.weight", weight)]) == {
        "text_projection.weight"
    }
    torch.testing.assert_close(model.text_projection.weight, weight, rtol=0, atol=0)
