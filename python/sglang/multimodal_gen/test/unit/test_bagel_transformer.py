# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.bagel import (
    BagelDiTArchConfig,
    BagelDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import TimestepEmbedder
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
    BagelTransformer,
    _sdpa_attention,
)


def _tp2_config(*, load_lm_head: bool = False) -> BagelDiTConfig:
    arch = BagelDiTArchConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=2,
        vocab_size=128,
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
    return BagelDiTConfig(arch_config=arch, load_lm_head=load_lm_head)


@contextmanager
def _fake_tp2(rank: int) -> Iterator[SimpleNamespace]:
    fake_tp_group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        # Cache-shape tests only need the collective to preserve tensor shape.
        all_reduce=lambda tensor: tensor,
    )
    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.model_parallel_is_initialized",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.get_tp_world_size",
            return_value=2,
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=fake_tp_group,
        ),
        patch(
            "sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding.get_tp_group",
            return_value=fake_tp_group,
        ),
    ):
        yield fake_tp_group


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


def _build_thinking_transformer(
    tiny_transformer: BagelTransformer,
) -> BagelTransformer:
    config = BagelDiTConfig(
        arch_config=tiny_transformer.config.arch_config,
        load_lm_head=True,
    )
    torch.manual_seed(0)
    return BagelTransformer(config).eval()


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


def test_tp2_uses_local_attention_heads_and_linear_shard_shapes() -> None:
    with _fake_tp2(rank=1), torch.device("meta"):
        model = BagelTransformer(_tp2_config(load_lm_head=True))

    attention = model.layers[0].attn
    mlp = model.layers[0].mlp
    assert model.tp_size == 2
    assert isinstance(model.embed_tokens, VocabParallelEmbedding)
    assert tuple(model.embed_tokens.weight.shape) == (64, 8)
    assert model.lm_head is not None
    assert tuple(model.lm_head.weight.shape) == (64, 8)
    assert model.lm_head.gather_output is True
    assert attention.num_heads == 2
    assert attention.num_kv_heads == 1
    assert tuple(attention.und_q_proj.weight.shape) == (4, 8)
    assert tuple(attention.und_k_proj.weight.shape) == (2, 8)
    assert tuple(attention.und_v_proj.weight.shape) == (2, 8)
    assert tuple(attention.und_o_proj.weight.shape) == (8, 4)
    assert tuple(mlp.und_gate.weight.shape) == (6, 8)
    assert tuple(mlp.und_up.weight.shape) == (6, 8)
    assert tuple(mlp.und_down.weight.shape) == (8, 6)


def test_tp2_rank1_loads_full_checkpoint_column_and_row_shards() -> None:
    embedding_weight = torch.arange(1024, dtype=torch.float32).reshape(128, 8)
    query_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    query_bias = torch.arange(8, dtype=torch.float32)
    output_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 100
    lm_head_weight = torch.arange(1024, dtype=torch.float32).reshape(128, 8) + 1000
    with _fake_tp2(rank=1):
        model = BagelTransformer(_tp2_config(load_lm_head=True))
        loaded = model.load_weights(
            [
                ("language_model.model.embed_tokens.weight", embedding_weight),
                (
                    "language_model.model.layers.0.self_attn.q_proj.weight",
                    query_weight,
                ),
                (
                    "language_model.model.layers.0.self_attn.q_proj.bias",
                    query_bias,
                ),
                (
                    "language_model.model.layers.0.self_attn.o_proj.weight",
                    output_weight,
                ),
                ("language_model.lm_head.weight", lm_head_weight),
            ],
            strict=False,
        )

    attention = model.layers[0].attn
    assert model.lm_head is not None
    assert loaded == {
        "embed_tokens.weight",
        "layers.0.attn.und_q_proj.weight",
        "layers.0.attn.und_q_proj.bias",
        "layers.0.attn.und_o_proj.weight",
        "lm_head.weight",
    }
    torch.testing.assert_close(model.embed_tokens.weight, embedding_weight[64:])
    torch.testing.assert_close(attention.und_q_proj.weight, query_weight[4:])
    torch.testing.assert_close(attention.und_q_proj.bias, query_bias[4:])
    torch.testing.assert_close(attention.und_o_proj.weight, output_weight[:, 4:])
    torch.testing.assert_close(model.lm_head.weight, lm_head_weight[64:])


def test_tp2_meta_load_materializes_local_shards_and_preserves_loader_attrs() -> None:
    embedding_weight = torch.arange(1024, dtype=torch.float32).reshape(128, 8)
    query_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    output_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 100
    lm_head_weight = torch.arange(1024, dtype=torch.float32).reshape(128, 8) + 1000
    with _fake_tp2(rank=1):
        with torch.device("meta"):
            model = BagelTransformer(_tp2_config(load_lm_head=True))
        model.load_weights(
            [
                ("language_model.model.embed_tokens.weight", embedding_weight),
                (
                    "language_model.model.layers.0.self_attn.q_proj.weight",
                    query_weight,
                ),
                (
                    "language_model.model.layers.0.self_attn.o_proj.weight",
                    output_weight,
                ),
                ("language_model.lm_head.weight", lm_head_weight),
            ],
            strict=False,
        )

    embedding_parameter = model.embed_tokens.weight
    query_parameter = model.layers[0].attn.und_q_proj.weight
    output_parameter = model.layers[0].attn.und_o_proj.weight
    assert model.lm_head is not None
    lm_head_parameter = model.lm_head.weight
    assert not embedding_parameter.is_meta
    assert not query_parameter.is_meta
    assert not output_parameter.is_meta
    assert not lm_head_parameter.is_meta
    assert callable(embedding_parameter.weight_loader)
    assert callable(query_parameter.weight_loader)
    assert callable(output_parameter.weight_loader)
    assert callable(lm_head_parameter.weight_loader)
    assert embedding_parameter.output_dim == 0
    assert query_parameter.output_dim == 0
    assert output_parameter.input_dim == 1
    assert lm_head_parameter.output_dim == 0
    torch.testing.assert_close(embedding_parameter, embedding_weight[64:])
    torch.testing.assert_close(query_parameter, query_weight[4:])
    torch.testing.assert_close(output_parameter, output_weight[:, 4:])
    torch.testing.assert_close(lm_head_parameter, lm_head_weight[64:])


def test_tp2_prefix_cache_stores_only_local_kv_heads() -> None:
    with _fake_tp2(rank=0):
        model = BagelTransformer(_tp2_config(), attention_backend="torch_sdpa").eval()
        prefix = model.prefill_context(torch.tensor([1, 2, 3]))

    key = prefix.kv_cache.key_cache[0]
    value = prefix.kv_cache.value_cache[0]
    assert key is not None
    assert value is not None
    assert key.shape == (3, 1, 2)
    assert value.shape == (3, 1, 2)


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
    assert context.secondary_unconditional_kv is not None
    assert context.secondary_unconditional_kv.sequence_length == 2
    assert context.conditional_rope_offset == 4
    assert context.unconditional_rope_offset == 2
    assert context.secondary_unconditional_rope_offset == 2
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


def test_prefix_decode_uses_bottom_right_causal_mask() -> None:
    query = torch.zeros(1, 1, 1)
    key = torch.zeros(3, 1, 1)
    value = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])

    output = _sdpa_attention(query, key, value, causal=True)

    torch.testing.assert_close(output, torch.tensor([[[2.0]]]))


def test_optional_lm_head_is_strictly_mapped_and_loaded(
    tiny_transformer: BagelTransformer,
) -> None:
    assert tiny_transformer.lm_head is None
    thinking = _build_thinking_transformer(tiny_transformer)
    assert thinking.lm_head is not None
    head_weight = torch.randn_like(thinking.lm_head.weight)

    loaded = thinking.load_weights(
        [("language_model.lm_head.weight", head_weight)], strict=False
    )

    assert loaded == {"lm_head.weight"}
    torch.testing.assert_close(thinking.lm_head.weight, head_weight)

    with torch.device("meta"):
        meta_model = BagelTransformer(thinking.config)
    weights_without_head = [
        (name, torch.zeros(parameter.shape, dtype=parameter.dtype))
        for name, parameter in meta_model.named_parameters()
        if name != "lm_head.weight"
    ]
    with pytest.raises(ValueError, match="lm_head.weight"):
        meta_model.load_weights(iter(weights_without_head))


def test_text_generation_matches_official_eos_and_max_length_semantics(
    tiny_transformer: BagelTransformer,
) -> None:
    model = _build_thinking_transformer(tiny_transformer)
    assert model.lm_head is not None
    model.lm_head.weight.data.zero_()
    prefix = model.prefill_context(torch.tensor([1, 2]))

    stopped = model.generate_text(
        prefix,
        bos_token_id=5,
        eos_token_id=0,
        max_length=4,
    )
    capped = model.generate_text(
        prefix,
        bos_token_id=5,
        eos_token_id=31,
        max_length=3,
    )

    assert stopped.tolist() == [5]
    assert capped.tolist() == [5, 0, 0]
    assert prefix.kv_cache.sequence_length == 2
    assert prefix.rope_offset == 2


def test_sampled_text_generation_is_request_local_and_deterministic(
    tiny_transformer: BagelTransformer,
) -> None:
    model = _build_thinking_transformer(tiny_transformer)
    prefix = model.prefill_context(torch.tensor([1, 2]))
    torch.manual_seed(1234)
    global_state = torch.random.get_rng_state().clone()
    diffusion_generator = torch.Generator("cpu").manual_seed(77)
    diffusion_state = diffusion_generator.get_state().clone()

    first = model.generate_text(
        prefix,
        bos_token_id=5,
        eos_token_id=31,
        max_length=4,
        do_sample=True,
        temperature=0.7,
        seed=9,
    )
    second = model.generate_text(
        prefix,
        bos_token_id=5,
        eos_token_id=31,
        max_length=4,
        do_sample=True,
        temperature=0.7,
        seed=9,
    )

    assert first.tolist() == second.tolist()
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.equal(diffusion_generator.get_state(), diffusion_state)


def test_thinking_context_has_official_three_request_owned_prefixes(
    tiny_transformer: BagelTransformer,
) -> None:
    model = _build_thinking_transformer(tiny_transformer)
    system_prefix, user_prefix = model.prepare_thinking_prefixes(
        torch.tensor([5, 1, 6]), torch.tensor([5, 2, 6])
    )

    context = model.build_thinking_context(
        system_prefix,
        user_prefix,
        torch.tensor([5, 7, 8, 6]),
        height=4,
        width=4,
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )

    assert context.is_thinking
    assert not context.is_editing
    assert context.has_three_way_cfg
    assert context.conditional_kv.sequence_length == 10
    assert context.unconditional_kv.sequence_length == 3
    assert context.secondary_unconditional_kv is not None
    assert context.secondary_unconditional_kv.sequence_length == 6
    assert system_prefix.kv_cache.sequence_length == 3
    assert user_prefix.kv_cache.sequence_length == 6
