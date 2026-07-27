# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.bagel import (
    BagelDiTArchConfig,
    BagelDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import TimestepEmbedder
from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
    BagelTransformer,
)


@pytest.fixture
def tiny_transformer() -> BagelTransformer:
    arch = BagelDiTArchConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
        latent_patch_size=2,
        max_latent_size=2,
        latent_channel=1,
        latent_downsample=2,
        timestep_frequency_embedding_size=4,
        latent_position_embedding_rows=4,
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )
    torch.manual_seed(0)
    model = BagelTransformer(BagelDiTConfig(arch_config=arch)).eval()
    with torch.no_grad():
        model.latent_pos_embed.pos_embed.zero_()
    return model


def _build_context(model: BagelTransformer, token_ids: list[int]):
    return model.build_context(
        torch.tensor(token_ids),
        None,
        height=4,
        width=4,
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )


def test_timestep_embedder_reuses_upstream_class_and_maps_keys(
    tiny_transformer: BagelTransformer,
) -> None:
    assert isinstance(tiny_transformer.time_embedder, TimestepEmbedder)

    weights = {
        "time_embedder.mlp.0.weight": torch.full((8, 4), 1.0),
        "time_embedder.mlp.0.bias": torch.full((8,), 2.0),
        "time_embedder.mlp.2.weight": torch.full((8, 8), 3.0),
        "time_embedder.mlp.2.bias": torch.full((8,), 4.0),
    }
    loaded = tiny_transformer.load_weights(weights.items(), strict=False)

    assert loaded == {
        "time_embedder.mlp.fc_in.weight",
        "time_embedder.mlp.fc_in.bias",
        "time_embedder.mlp.fc_out.weight",
        "time_embedder.mlp.fc_out.bias",
    }
    torch.testing.assert_close(
        tiny_transformer.time_embedder.mlp.fc_in.weight,
        weights["time_embedder.mlp.0.weight"],
    )
    torch.testing.assert_close(
        tiny_transformer.time_embedder.mlp.fc_out.bias,
        weights["time_embedder.mlp.2.bias"],
    )


def test_position_table_is_loaded_and_shape_checked(
    tiny_transformer: BagelTransformer,
) -> None:
    table = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    loaded = tiny_transformer.load_weights(
        [("latent_pos_embed.pos_embed", table)], strict=False
    )
    assert loaded == {"latent_pos_embed.pos_embed"}
    torch.testing.assert_close(tiny_transformer.latent_pos_embed.pos_embed, table)

    with pytest.raises(ValueError, match="shape mismatch"):
        tiny_transformer.load_weights(
            [("latent_pos_embed.pos_embed", torch.zeros(3, 8))], strict=False
        )


def test_context_and_forward_are_request_isolated(
    tiny_transformer: BagelTransformer,
) -> None:
    context_a = _build_context(tiny_transformer, [1, 2])
    context_b = _build_context(tiny_transformer, [2, 1, 2])

    assert context_a.conditional_kv is not context_b.conditional_kv
    assert context_a.unconditional_kv is not context_b.unconditional_kv
    assert context_a.conditional_kv.sequence_length == 2
    assert context_b.conditional_kv.sequence_length == 3
    assert context_a.unconditional_kv.sequence_length == 0
    assert not hasattr(tiny_transformer, "main_kv")
    assert not hasattr(tiny_transformer, "_img_shape")

    torch.manual_seed(1)
    latents = torch.randn(4, 4)
    timestep = torch.tensor([0.5])
    output_a_first = tiny_transformer(
        latents,
        timestep,
        bagel_context=context_a,
        guidance_scale=2.0,
    )
    tiny_transformer(
        latents,
        timestep,
        bagel_context=context_b,
        guidance_scale=2.0,
    )
    output_a_second = tiny_transformer(
        latents,
        timestep,
        bagel_context=context_a,
        guidance_scale=2.0,
    )

    torch.testing.assert_close(output_a_first, output_a_second)
    assert output_a_first.shape == latents.shape
    assert torch.isfinite(output_a_first).all()


def test_build_context_requires_explicit_valid_image_token_ids(
    tiny_transformer: BagelTransformer,
) -> None:
    with pytest.raises(ValueError, match="inside the vocabulary"):
        tiny_transformer.build_context(
            torch.tensor([1]),
            None,
            height=4,
            width=4,
            start_of_image_token_id=32,
            end_of_image_token_id=4,
        )

    with pytest.raises(ValueError, match="do not match the checkpoint"):
        tiny_transformer.build_context(
            torch.tensor([1]),
            None,
            height=4,
            width=4,
            start_of_image_token_id=5,
            end_of_image_token_id=6,
        )


def test_meta_initialization_materializes_all_state_for_streaming_load(
    tiny_transformer: BagelTransformer,
) -> None:
    with torch.device("meta"):
        model = BagelTransformer(tiny_transformer.config)

    weights = [
        (name, torch.zeros(tuple(parameter.shape), dtype=parameter.dtype))
        for name, parameter in model.named_parameters()
    ]
    loaded = model.load_weights(iter(weights))
    model.to("cpu")

    assert loaded == {name for name, _ in weights}
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(buffer.is_meta for buffer in model.buffers())


def test_attention_backend_selection_is_explicit(
    tiny_transformer: BagelTransformer,
) -> None:
    sdpa_model = BagelTransformer(
        tiny_transformer.config, attention_backend="torch_sdpa"
    )
    assert sdpa_model.layers[0].attn.attention_backend.name == "TORCH_SDPA"
    assert sdpa_model.layers[0].attn.backend.name == "TORCH_SDPA"

    with pytest.raises(ValueError, match="Unsupported BAGEL attention backend"):
        BagelTransformer(tiny_transformer.config, attention_backend="sage_attn")


def test_bf16_rms_norm_matches_official_cast_order(
    tiny_transformer: BagelTransformer,
) -> None:
    norm = tiny_transformer.layers[0].und_in_norm.to(torch.bfloat16)
    hidden_states = torch.randn(3, 8, dtype=torch.bfloat16)

    normalized = hidden_states.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (normalized * torch.rsqrt(variance + norm.eps)).to(
        hidden_states.dtype
    )

    torch.testing.assert_close(norm(hidden_states), expected, rtol=0, atol=0)


def test_internal_cfg_matches_bagel_global_renorm() -> None:
    conditional = torch.tensor([[2.0, 0.0]])
    unconditional = torch.tensor([[1.0, 0.0]])

    output = BagelTransformer._apply_cfg(
        conditional,
        unconditional,
        4.0,
        renorm_min=0.0,
        renorm_type="global",
    )

    # Raw CFG is [5, 0]; BAGEL's global renorm caps it to the conditional norm.
    torch.testing.assert_close(output, conditional)


def test_editing_context_has_three_request_owned_prefixes_with_shared_image_storage(
    tiny_transformer: BagelTransformer,
) -> None:
    torch.manual_seed(2)
    context = tiny_transformer.build_editing_context(
        vae_patches=torch.randn(4, 4),
        vae_position_ids=torch.tensor([0, 1, 2, 3]),
        vision_embeddings=torch.randn(3, 8),
        text_input_ids=torch.tensor([1, 2]),
        height=4,
        width=4,
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )

    assert context.is_editing
    assert context.conditional_kv.sequence_length == 13
    assert context.unconditional_kv.sequence_length == 11
    assert context.image_unconditional_kv is not None
    assert context.image_unconditional_kv.sequence_length == 2
    assert context.conditional_rope_offset == 4
    assert context.unconditional_rope_offset == 2
    assert context.image_unconditional_rope_offset == 2
    assert context.conditional_kv is not context.unconditional_kv
    main_key = context.conditional_kv.key_cache[0]
    image_key = context.unconditional_kv.key_cache[0]
    assert main_key is not None and image_key is not None
    assert (
        main_key.untyped_storage().data_ptr() == image_key.untyped_storage().data_ptr()
    )

    latents = torch.randn(4, 4)
    prediction = tiny_transformer(
        latents,
        torch.tensor([0.5]),
        bagel_context=context,
        guidance_scale=4.0,
        image_guidance_scale=2.0,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_type="text_channel",
    )
    assert prediction.shape == latents.shape
    assert torch.isfinite(prediction).all()


def test_three_way_cfg_matches_official_text_channel_order() -> None:
    main = torch.tensor([[2.0, 0.0]])
    image_only = torch.tensor([[1.0, 0.0]])
    text_only = torch.tensor([[0.5, 0.0]])

    output = BagelTransformer._apply_cfg_three_way(
        main,
        image_only,
        text_only,
        4.0,
        2.0,
        renorm_min=0.0,
        renorm_type="text_channel",
    )

    torch.testing.assert_close(output, torch.tensor([[3.5, 0.0]]))


def test_generation_qk_stays_fp32_until_attention_cast(
    tiny_transformer: BagelTransformer,
) -> None:
    model = tiny_transformer.to(torch.bfloat16)
    attention = model.layers[0].attn
    hidden_states = torch.randn(4, 8, dtype=torch.bfloat16)
    text_indexes = torch.tensor([0, 3])
    latent_indexes = torch.tensor([1, 2])

    query, key, value = attention._project_generation(
        hidden_states, text_indexes, latent_indexes
    )

    assert query.dtype == torch.float32
    assert key.dtype == torch.float32
    assert value.dtype == torch.bfloat16
    expected_text_query = (
        attention.und_q_proj(hidden_states[text_indexes])
        .view(-1, attention.num_heads, attention.head_dim)
        .float()
    )
    variance = expected_text_query.pow(2).mean(-1, keepdim=True)
    expected_text_query = attention.und_q_norm.weight * (
        expected_text_query * torch.rsqrt(variance + attention.und_q_norm.eps)
    ).to(expected_text_query.dtype)
    torch.testing.assert_close(query[text_indexes], expected_text_query)

    context = _build_context(model, [1, 2])
    prediction = model(
        torch.randn(4, 4, dtype=torch.bfloat16),
        torch.tensor([0.5]),
        bagel_context=context,
        guidance_scale=1.0,
    )
    assert prediction.dtype == torch.float32
