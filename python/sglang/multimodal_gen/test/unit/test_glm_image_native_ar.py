import torch
from transformers.models.glm_image.configuration_glm_image import GlmImageConfig
from transformers.models.glm_image.modeling_glm_image import (
    GlmImageForConditionalGeneration as HFGlmImageForConditionalGeneration,
)

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.glm_image import (
    GlmImageForConditionalGeneration,
    GlmImageTextRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _config() -> GlmImageConfig:
    config = GlmImageConfig(
        image_start_token_id=100,
        image_end_token_id=101,
        image_token_id=102,
        text_config={
            "attention_bias": True,
            "attention_dropout": 0.0,
            "eos_token_id": 31,
            "pad_token_id": 0,
            "hidden_act": "silu",
            "hidden_size": 32,
            "intermediate_size": 64,
            "max_position_embeddings": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {
                "rope_theta": 10_000,
                "rope_type": "default",
                "mrope_section": [1, 1],
                "partial_rotary_factor": 0.5,
            },
            "use_cache": True,
            "vision_vocab_size": 32,
            "vocab_size": 128,
        },
        vision_config={
            "attention_bias": True,
            "attention_dropout": 0.0,
            "depth": 2,
            "hidden_act": "gelu",
            "hidden_size": 16,
            "image_size": 4,
            "in_channels": 3,
            "intermediate_size": 32,
            "layer_norm_eps": 1e-6,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
        },
        vq_config={
            "embed_dim": 8,
            "in_channels": 3,
            "latent_channels": 16,
            "num_embeddings": 32,
        },
    )
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    return config


def _models():
    torch.manual_seed(23)
    config = _config()
    reference = HFGlmImageForConditionalGeneration(config).eval()
    native = GlmImageForConditionalGeneration(config).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    return reference, native


def test_native_glm_image_matches_hf_text_prefill_cache_and_generation():
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        reference, native = _models()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            reference_output = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            native_output = native(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        torch.testing.assert_close(native_output.logits, reference_output.logits)

        next_token = torch.tensor([[5]])
        decode_mask = torch.ones(1, 5, dtype=torch.long)
        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            reference_decode = reference(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=reference_output.past_key_values,
                use_cache=True,
            )
            native_decode = native(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=native_output.past_key_values,
                use_cache=True,
            )
        torch.testing.assert_close(native_decode.logits, reference_decode.logits)

        with torch.no_grad():
            reference_ids = reference.generate(
                input_ids, max_new_tokens=2, do_sample=False
            )
        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            native_ids = native.generate(input_ids, max_new_tokens=2, do_sample=False)
        torch.testing.assert_close(native_ids, reference_ids)


def test_native_glm_image_keeps_rotary_frequencies_in_fp32():
    config = _config().text_config
    rotary = GlmImageTextRotaryEmbedding(config)
    hidden_states = torch.zeros(1, 2, config.hidden_size)
    position_ids = torch.tensor([[[0, 10_000]], [[0, 10_000]], [[0, 10_000]]])

    expected = rotary(hidden_states, position_ids)
    actual = rotary.to(dtype=torch.bfloat16)(
        hidden_states.to(torch.bfloat16), position_ids
    )

    torch.testing.assert_close(actual[0], expected[0].to(torch.bfloat16))
    torch.testing.assert_close(actual[1], expected[1].to(torch.bfloat16))


def test_native_glm_image_clears_multimodal_positions_for_new_text_request():
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        _, native = _models()
        native.model.rope_deltas = torch.ones(1, 1, dtype=torch.long)
        native.model._cached_decode_position_ids = torch.ones(1, 3, 1, dtype=torch.long)
        native.model._prefill_len = 1

        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            output = native(input_ids=torch.tensor([[1, 2]]), use_cache=True)

        assert output.rope_deltas is None
        assert native.model._cached_decode_position_ids is None
        assert native.model._prefill_len is None


def test_native_glm_image_matches_hf_vision_vq_and_i2i_prefill():
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        reference, native = _models()
        pixel_values = torch.randn(4, 12)
        source_grid = torch.tensor([[1, 2, 2]])

        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            reference_features = reference.get_image_features(pixel_values, source_grid)
            native_features = native.get_image_features(pixel_values, source_grid)
        torch.testing.assert_close(
            native_features.last_hidden_state,
            reference_features.last_hidden_state,
        )
        native_tokens = native.get_image_tokens(
            torch.cat(native_features.pooler_output), source_grid
        )
        reference_tokens = reference.get_image_tokens(
            torch.cat(reference_features.pooler_output), source_grid
        )
        torch.testing.assert_close(native_tokens, reference_tokens)

        input_ids = torch.tensor([[100, 102, 102, 102, 102, 101, 7, 100]])
        attention_mask = torch.ones_like(input_ids)
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]])
        with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None
        ):
            reference_output = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=True,
            )
            native_output = native(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=True,
            )
        torch.testing.assert_close(native_output.logits, reference_output.logits)


def test_native_glm_image_exposes_all_transformer_layers_for_offload():
    assert GlmImageForConditionalGeneration.layer_names == [
        "model.language_model.layers",
        "model.visual.blocks",
    ]
