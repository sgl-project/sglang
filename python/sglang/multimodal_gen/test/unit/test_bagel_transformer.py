# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
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
from sglang.multimodal_gen.runtime.models.dits.bagel_taylorseer import (
    BagelTaylorSeerContext,
    TaylorSeerConfig,
)
from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
    BagelTransformer,
    _apply_bagel_qk_norm,
    _BagelRMSNorm,
    _interleave_prefix_and_query,
    _run_varlen_attention,
    _sdpa_attention,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


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


def _eager_qk_norm_for_test(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: _BagelRMSNorm,
    k_norm: _BagelRMSNorm,
    head_dim: int,
    allow_inplace: bool,
    cast_x_before_out_mul: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``apply_qk_norm`` fallback while exposing dispatch arguments."""
    del head_dim, allow_inplace, cast_x_before_out_mul
    return q_norm(q), k_norm(k)


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


@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_dynamic_batch_matches_sequential_with_variable_prefixes(
    tiny_transformer: BagelTransformer,
    guidance_scale: float,
) -> None:
    contexts = [
        _build_context(tiny_transformer, [1, 2]),
        _build_context(tiny_transformer, [2, 1, 2]),
    ]
    packed_context = tiny_transformer.pack_contexts(contexts)
    torch.manual_seed(3)
    latents = torch.randn(2, 4, 4)
    timestep = torch.tensor([0.5, 0.5])

    batched = tiny_transformer(
        latents,
        timestep,
        bagel_context=packed_context,
        guidance_scale=guidance_scale,
        cfg_interval=(0.0, 1.0),
    )
    sequential = torch.stack(
        [
            tiny_transformer(
                latents[index],
                timestep[index : index + 1],
                bagel_context=context,
                guidance_scale=guidance_scale,
                cfg_interval=(0.0, 1.0),
            )
            for index, context in enumerate(contexts)
        ]
    )

    assert packed_context.batch_size == 2
    assert packed_context.conditional_kv_lens.tolist() == [2, 3]
    assert packed_context.unconditional_kv_lens.tolist() == [0, 0]
    assert packed_context.conditional_rope_offset.tolist() == [2, 3]
    assert packed_context.conditional_kv.sequence_length == 5
    assert batched.shape == latents.shape
    torch.testing.assert_close(batched, sequential, rtol=1e-5, atol=1e-6)


def test_batch_one_preserves_two_and_three_dimensional_forward_contract(
    tiny_transformer: BagelTransformer,
) -> None:
    context = _build_context(tiny_transformer, [1, 2])
    latents = torch.randn(4, 4)

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._interleave_prefix_and_query",
        side_effect=AssertionError("B=1 must use the legacy singleton path"),
    ):
        unbatched = tiny_transformer(
            latents,
            torch.tensor([0.5]),
            bagel_context=context,
            guidance_scale=2.0,
        )
        batched = tiny_transformer(
            latents.unsqueeze(0),
            torch.tensor([0.5]),
            bagel_context=context,
            guidance_scale=2.0,
        )

    assert unbatched.shape == latents.shape
    assert batched.shape == (1, *latents.shape)
    torch.testing.assert_close(batched[0], unbatched, rtol=0, atol=0)


def test_disabled_taylorseer_preserves_pre_acceleration_singleton_math(
    tiny_transformer: BagelTransformer,
) -> None:
    context = _build_context(tiny_transformer, [1, 2])
    latents = torch.randn(4, 4)
    timestep = torch.tensor([0.5])
    normalized_timestep = tiny_transformer._normalize_timestep(
        timestep,
        batch_size=1,
        token_count=latents.shape[0],
        device=latents.device,
    )
    conditional = tiny_transformer._generation_step_single(
        latents,
        normalized_timestep,
        context.conditional_kv,
        context.conditional_kv_lens,
        context.conditional_rope_offset,
        context,
        None,
    )
    unconditional = tiny_transformer._generation_step_single(
        latents,
        normalized_timestep,
        context.unconditional_kv,
        context.unconditional_kv_lens,
        context.unconditional_rope_offset,
        context,
        None,
    )
    expected = tiny_transformer._apply_cfg(
        conditional,
        unconditional,
        2.0,
        renorm_min=0.0,
        renorm_type="global",
    )

    actual = tiny_transformer(
        latents,
        timestep,
        bagel_context=context,
        guidance_scale=2.0,
        cfg_interval=(0.0, 1.0),
        taylorseer_context=None,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_request_owned_taylorseer_skips_layers_between_refreshes(
    tiny_transformer: BagelTransformer,
) -> None:
    context = _build_context(tiny_transformer, [1, 2])
    taylorseer = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=4,
        has_secondary=False,
        config=TaylorSeerConfig(
            max_order=2,
            fresh_threshold=3,
            first_enhance=1,
        ),
    )
    layer_calls: list[int] = []
    hook = tiny_transformer.layers[0].register_forward_hook(
        lambda *_args: layer_calls.append(1)
    )
    call_counts = []
    try:
        for sigma in (1.0, 0.8, 0.6, 0.4):
            output = tiny_transformer(
                torch.randn(4, 4),
                torch.tensor([sigma]),
                bagel_context=context,
                guidance_scale=2.0,
                cfg_interval=(0.0, 1.0),
                taylorseer_context=taylorseer,
            )
            assert output.shape == (4, 4)
            assert torch.isfinite(output).all()
            call_counts.append(len(layer_calls))
    finally:
        hook.remove()

    # Conditional and unconditional branches each execute on refresh steps.
    assert call_counts == [2, 2, 2, 4]
    assert taylorseer.conditional is not taylorseer.unconditional
    assert taylorseer.get_stats()["conditional"] == {
        "total_steps": 4,
        "full_steps": 2,
        "taylor_steps": 2,
    }


def test_taylorseer_supports_packed_dynamic_batch(
    tiny_transformer: BagelTransformer,
) -> None:
    contexts = [
        _build_context(tiny_transformer, [1, 2]),
        _build_context(tiny_transformer, [2, 1, 2]),
    ]
    packed_context = tiny_transformer.pack_contexts(contexts)
    taylorseer = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=2,
        has_secondary=False,
        config=TaylorSeerConfig(
            max_order=1,
            fresh_threshold=3,
            first_enhance=1,
        ),
    )
    latents = torch.randn(2, 4, 4)

    refreshed = tiny_transformer(
        latents,
        torch.tensor([1.0, 1.0]),
        bagel_context=packed_context,
        guidance_scale=2.0,
        cfg_interval=(0.0, 1.0),
        taylorseer_context=taylorseer,
    )
    forecast = tiny_transformer(
        latents,
        torch.tensor([0.5, 0.5]),
        bagel_context=packed_context,
        guidance_scale=2.0,
        cfg_interval=(0.0, 1.0),
        taylorseer_context=taylorseer,
    )

    assert refreshed.shape == latents.shape
    assert forecast.shape == latents.shape
    assert taylorseer.conditional.completed_steps == 2
    assert taylorseer.unconditional.completed_steps == 2

    sequential_states = [
        BagelTaylorSeerContext.create(
            num_layers=1,
            num_steps=2,
            has_secondary=False,
            config=TaylorSeerConfig(
                max_order=1,
                fresh_threshold=3,
                first_enhance=1,
            ),
        )
        for _ in contexts
    ]
    sequential_outputs = []
    for sigma in (1.0, 0.5):
        sequential_outputs.append(
            torch.stack(
                [
                    tiny_transformer(
                        latents[index],
                        torch.tensor([sigma]),
                        bagel_context=context,
                        guidance_scale=2.0,
                        cfg_interval=(0.0, 1.0),
                        taylorseer_context=sequential_states[index],
                    )
                    for index, context in enumerate(contexts)
                ]
            )
        )

    torch.testing.assert_close(refreshed, sequential_outputs[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(forecast, sequential_outputs[1], rtol=1e-5, atol=1e-6)


def test_taylorseer_advances_only_cfg_branches_that_execute(
    tiny_transformer: BagelTransformer,
) -> None:
    context = _build_context(tiny_transformer, [1, 2])
    taylorseer = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=2,
        has_secondary=False,
    )

    for sigma in (1.0, 0.5):
        tiny_transformer(
            torch.randn(4, 4),
            torch.tensor([sigma]),
            bagel_context=context,
            guidance_scale=1.0,
            taylorseer_context=taylorseer,
        )

    assert taylorseer.conditional.completed_steps == 2
    assert taylorseer.unconditional.completed_steps == 0


def test_taylorseer_poison_invalidates_all_cfg_branches_after_failure(
    tiny_transformer: BagelTransformer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context(tiny_transformer, [1, 2])
    taylorseer = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=2,
        has_secondary=False,
    )
    layer = tiny_transformer.layers[0]
    original_forward = layer.forward
    call_count = 0

    def fail_unconditional(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected unconditional failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(layer, "forward", fail_unconditional)
    with pytest.raises(RuntimeError, match="injected unconditional failure"):
        tiny_transformer(
            torch.randn(4, 4),
            torch.tensor([1.0]),
            bagel_context=context,
            guidance_scale=2.0,
            cfg_interval=(0.0, 1.0),
            taylorseer_context=taylorseer,
        )

    assert taylorseer.is_failed
    assert taylorseer.conditional.completed_steps == 1
    assert taylorseer.unconditional.completed_steps == 0
    with pytest.raises(RuntimeError, match="invalid after a failed"):
        tiny_transformer(
            torch.randn(4, 4),
            torch.tensor([0.5]),
            bagel_context=context,
            guidance_scale=2.0,
            cfg_interval=(0.0, 1.0),
            taylorseer_context=taylorseer,
        )


def test_taylorseer_poison_clears_partial_two_layer_update(
    tiny_transformer: BagelTransformer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arch = replace(tiny_transformer.config.arch_config, num_hidden_layers=2)
    torch.manual_seed(0)
    model = BagelTransformer(BagelDiTConfig(arch_config=arch)).eval()
    context = _build_context(model, [1, 2])
    taylorseer = BagelTaylorSeerContext.create(
        num_layers=2,
        num_steps=1,
        has_secondary=False,
    )

    def fail_second_layer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected second-layer failure")

    monkeypatch.setattr(model.layers[1], "forward", fail_second_layer)
    with pytest.raises(RuntimeError, match="injected second-layer failure"):
        model(
            torch.randn(4, 4),
            torch.tensor([1.0]),
            bagel_context=context,
            guidance_scale=1.0,
            taylorseer_context=taylorseer,
        )

    assert taylorseer.is_failed
    assert taylorseer.conditional.completed_steps == 0
    assert taylorseer.conditional._layers == []
    assert taylorseer.unconditional._layers == []


def test_dynamic_batch_keeps_requests_attention_isolated(
    tiny_transformer: BagelTransformer,
) -> None:
    context_a = _build_context(tiny_transformer, [1, 2])
    context_b = _build_context(tiny_transformer, [2, 1, 2])
    latents = torch.randn(2, 4, 4)
    first = tiny_transformer(
        latents,
        torch.tensor([0.5, 0.5]),
        bagel_context=tiny_transformer.pack_contexts([context_a, context_b]),
        guidance_scale=1.0,
    )

    changed_context_b = _build_context(tiny_transformer, [7, 8, 9, 10])
    changed_latents = latents.clone()
    changed_latents[1].mul_(10)
    second = tiny_transformer(
        changed_latents,
        torch.tensor([0.5, 0.5]),
        bagel_context=tiny_transformer.pack_contexts([context_a, changed_context_b]),
        guidance_scale=1.0,
    )

    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


def test_dynamic_batch_rejects_mismatched_timesteps(
    tiny_transformer: BagelTransformer,
) -> None:
    contexts = [
        _build_context(tiny_transformer, [1]),
        _build_context(tiny_transformer, [2]),
    ]
    with pytest.raises(ValueError, match="same timestep"):
        tiny_transformer(
            torch.randn(2, 4, 4),
            torch.tensor([0.5, 0.4]),
            bagel_context=tiny_transformer.pack_contexts(contexts),
            guidance_scale=1.0,
        )


def test_normalizes_scalar_request_and_token_timesteps() -> None:
    scalar = BagelTransformer._normalize_timestep(
        torch.tensor(0.5), 2, 3, torch.device("cpu")
    )
    per_request = BagelTransformer._normalize_timestep(
        torch.tensor([0.5, 0.25]), 2, 3, torch.device("cpu")
    )
    per_token = BagelTransformer._normalize_timestep(
        torch.arange(6), 2, 3, torch.device("cpu")
    )

    assert scalar.tolist() == [0.5] * 6
    assert per_request.tolist() == [0.5, 0.5, 0.5, 0.25, 0.25, 0.25]
    assert per_token.tolist() == list(range(6))


def test_interleaves_prefix_and_query_per_request() -> None:
    prefix = torch.tensor([[10.0], [20.0], [21.0]])
    query = torch.tensor([[100.0], [101.0], [200.0]])

    packed = _interleave_prefix_and_query(
        prefix,
        query,
        torch.tensor([1, 2]),
        torch.tensor([2, 1]),
    )

    assert packed[:, 0].tolist() == [10.0, 100.0, 101.0, 20.0, 21.0, 200.0]


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

    with pytest.raises(ValueError, match="Unsupported BAGEL attention backend"):
        BagelTransformer(tiny_transformer.config, attention_backend="flashinfer")


def test_varlen_attention_prefers_flash_attention() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn_like(key)
    lengths = torch.tensor([3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            return_value=expected,
        ) as flash_attention,
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention"
        ) as flashinfer_attention,
    ):
        actual = _run_varlen_attention(
            query,
            key,
            value,
            lengths,
            lengths,
            causal=True,
            attention_backend=AttentionBackendEnum.FA,
        )

    assert actual is expected
    flash_attention.assert_called_once()
    flashinfer_attention.assert_not_called()


def test_varlen_attention_uses_per_request_flashinfer_fallback() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(5, 1, 4)
    value = torch.randn_like(key)
    query_lens = torch.tensor([2, 1], dtype=torch.int32)
    key_lens = torch.tensor([3, 2], dtype=torch.int32)

    def fake_single_request(
        query_chunk: torch.Tensor,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        assert key_chunk.shape == value_chunk.shape
        assert causal is False
        return query_chunk + key_chunk.shape[0]

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            side_effect=RuntimeError("FA4 unavailable"),
        ) as flash_attention,
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_single_flashinfer_attention",
            side_effect=fake_single_request,
        ) as single_request,
    ):
        first = _run_varlen_attention(
            query,
            key,
            value,
            query_lens,
            key_lens,
            causal=False,
            attention_backend=AttentionBackendEnum.FA,
        )
        second = _run_varlen_attention(
            query,
            key,
            value,
            query_lens,
            key_lens,
            causal=False,
            attention_backend=AttentionBackendEnum.FA,
        )

    expected = torch.cat((query[:2] + 3, query[2:] + 2), dim=0)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert flash_attention.call_count == 1
    assert single_request.call_count == 4
    assert [call.args[0].shape[0] for call in single_request.call_args_list] == [
        2,
        1,
        2,
        1,
    ]


def test_varlen_attention_falls_back_to_per_request_sdpa() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(5, 1, 4)
    value = torch.randn_like(key)
    query_lens = torch.tensor([2, 1], dtype=torch.int32)
    key_lens = torch.tensor([3, 2], dtype=torch.int32)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            side_effect=RuntimeError("FA4 unavailable"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention",
            side_effect=RuntimeError("FlashInfer unavailable"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._sdpa_attention",
            side_effect=lambda query_chunk, _key, _value, _causal: query_chunk,
        ) as sdpa_attention,
    ):
        actual = _run_varlen_attention(
            query,
            key,
            value,
            query_lens,
            key_lens,
            causal=False,
            attention_backend=AttentionBackendEnum.FA,
        )

    torch.testing.assert_close(actual, query)
    assert sdpa_attention.call_count == 2


def test_varlen_attention_propagates_fa_failure_after_success() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn_like(key)
    lengths = torch.tensor([3], dtype=torch.int32)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            side_effect=[query, RuntimeError("FA launch failed")],
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention"
        ) as flashinfer_attention,
    ):
        torch.testing.assert_close(
            _run_varlen_attention(
                query,
                key,
                value,
                lengths,
                lengths,
                causal=False,
                attention_backend=AttentionBackendEnum.FA,
            ),
            query,
        )
        with pytest.raises(RuntimeError, match="FA launch failed"):
            _run_varlen_attention(
                query,
                key,
                value,
                lengths,
                lengths,
                causal=False,
                attention_backend=AttentionBackendEnum.FA,
            )

    flashinfer_attention.assert_not_called()


def test_varlen_attention_propagates_flashinfer_failure_after_success() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn_like(key)
    lengths = torch.tensor([3], dtype=torch.int32)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            side_effect=RuntimeError("FA4 unavailable"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention",
            side_effect=[query, RuntimeError("FlashInfer launch failed")],
        ),
    ):
        torch.testing.assert_close(
            _run_varlen_attention(
                query,
                key,
                value,
                lengths,
                lengths,
                causal=False,
                attention_backend=AttentionBackendEnum.FA,
            ),
            query,
        )
        with pytest.raises(RuntimeError, match="FlashInfer launch failed"):
            _run_varlen_attention(
                query,
                key,
                value,
                lengths,
                lengths,
                causal=False,
                attention_backend=AttentionBackendEnum.FA,
            )


def test_varlen_attention_skips_flashinfer_before_blackwell() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn_like(key)
    lengths = torch.tensor([3], dtype=torch.int32)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=3,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            side_effect=RuntimeError("FA3 unavailable"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention"
        ) as flashinfer_attention,
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._sdpa_attention",
            return_value=query,
        ),
    ):
        actual = _run_varlen_attention(
            query,
            key,
            value,
            lengths,
            lengths,
            causal=False,
            attention_backend=AttentionBackendEnum.FA,
        )

    torch.testing.assert_close(actual, query)
    flashinfer_attention.assert_not_called()


def test_varlen_attention_explicit_sdpa_bypasses_cuda_backends() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn_like(key)
    lengths = torch.tensor([3], dtype=torch.int32)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention"
        ) as flash_attention,
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention"
        ) as flashinfer_attention,
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._sdpa_attention",
            return_value=query,
        ) as sdpa_attention,
    ):
        actual = _run_varlen_attention(
            query,
            key,
            value,
            lengths,
            lengths,
            causal=False,
            attention_backend=AttentionBackendEnum.TORCH_SDPA,
        )

    torch.testing.assert_close(actual, query)
    flash_attention.assert_not_called()
    flashinfer_attention.assert_not_called()
    sdpa_attention.assert_called_once_with(query, key, value, False)


@pytest.mark.parametrize("failing_backend", ["flash_attention", "flashinfer"])
def test_varlen_attention_never_hides_cuda_oom(failing_backend: str) -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    value = torch.randn_like(key)
    lengths = torch.tensor([3], dtype=torch.int32)
    flash_attention_error: Exception | None = None
    flashinfer_error: Exception | None = None
    if failing_backend == "flash_attention":
        flash_attention_error = torch.cuda.OutOfMemoryError("FA OOM")
    else:
        flash_attention_error = RuntimeError("FA4 unavailable")
        flashinfer_error = torch.cuda.OutOfMemoryError("FlashInfer OOM")

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASH_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._BAGEL_FLASHINFER_ATTENTION_STATE",
            {},
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_cuda_attention",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._bagel_flash_attention_version",
            return_value=4,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_bagel_flash_attention",
            side_effect=flash_attention_error,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._run_flashinfer_varlen_attention",
            side_effect=flashinfer_error,
        ),
    ):
        with pytest.raises(torch.cuda.OutOfMemoryError):
            _run_varlen_attention(
                query,
                key,
                value,
                lengths,
                lengths,
                causal=False,
                attention_backend=AttentionBackendEnum.FA,
            )


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


def test_bagel_rms_norm_dispatches_jit_with_flattened_input() -> None:
    norm = _BagelRMSNorm(512, eps=1e-6).to(torch.bfloat16)
    hidden_states = torch.randn(2, 3, 512, dtype=torch.bfloat16)
    kernel_output = torch.randn(6, 512, dtype=torch.bfloat16)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_rmsnorm_jit",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.rmsnorm_hf",
            return_value=kernel_output,
        ) as rmsnorm_kernel,
    ):
        actual = norm(hidden_states)

    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, kernel_output.reshape_as(hidden_states))
    flat_input, weight, eps = rmsnorm_kernel.call_args.args
    assert flat_input.shape == (6, 512)
    assert flat_input.is_contiguous()
    assert weight is norm.weight
    assert eps == norm.eps


def test_bagel_rms_norm_falls_back_when_jit_capability_is_unavailable() -> None:
    norm = _BagelRMSNorm(512, eps=1e-6).to(torch.bfloat16)
    hidden_states = torch.randn(3, 512, dtype=torch.bfloat16)
    normalized = hidden_states.float()
    expected = norm.weight * (
        normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + norm.eps)
    ).to(hidden_states.dtype)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_rmsnorm_jit",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.rmsnorm_hf",
        ) as rmsnorm_kernel,
    ):
        actual = norm(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rmsnorm_kernel.assert_not_called()


def test_bagel_rms_norm_propagates_kernel_launch_failures() -> None:
    norm = _BagelRMSNorm(512, eps=1e-6).to(torch.float16)
    hidden_states = torch.randn(2, 512, dtype=torch.float16)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_rmsnorm_jit",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.rmsnorm_hf",
            side_effect=RuntimeError("kernel launch failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="kernel launch failed"):
            norm(hidden_states)


def test_bagel_rms_norm_never_hides_cuda_oom() -> None:
    norm = _BagelRMSNorm(512, eps=1e-6).to(torch.float16)
    hidden_states = torch.randn(2, 512, dtype=torch.float16)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_bagel_rmsnorm_jit",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.rmsnorm_hf",
            side_effect=torch.cuda.OutOfMemoryError("out of memory"),
        ),
    ):
        with pytest.raises(torch.cuda.OutOfMemoryError, match="out of memory"):
            norm(hidden_states)


def test_bagel_rms_norm_keeps_cpu_and_fp32_on_eager_path() -> None:
    bf16_norm = _BagelRMSNorm(512, eps=1e-6).to(torch.bfloat16)
    fp32_norm = _BagelRMSNorm(512, eps=1e-6)

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.rmsnorm_hf"
    ) as rmsnorm_kernel:
        bf16_norm(torch.randn(2, 512, dtype=torch.bfloat16))
        fp32_norm(torch.randn(2, 512))

    rmsnorm_kernel.assert_not_called()


def test_bagel_qk_norm_requests_hf_cast_order_on_fallback() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    query_norm = _BagelRMSNorm(4, eps=1e-6)
    key_norm = _BagelRMSNorm(4, eps=1e-6)

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.apply_qk_norm",
        side_effect=_eager_qk_norm_for_test,
    ) as qk_norm:
        actual_query, actual_key = _apply_bagel_qk_norm(
            query, key, query_norm, key_norm, 4
        )

    torch.testing.assert_close(actual_query, query_norm(query))
    torch.testing.assert_close(actual_key, key_norm(key))
    assert qk_norm.call_args.kwargs["cast_x_before_out_mul"] is True
    assert qk_norm.call_args.kwargs["allow_inplace"] is True


def test_bagel_qk_norm_dispatches_fp32_pair_kernel() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    query_norm = _BagelRMSNorm(4, eps=1e-6)
    key_norm = _BagelRMSNorm(4, eps=1e-6)
    expected_query = query + 1
    expected_key = key + 2

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._USE_FUSED_FP32_QK_NORM",
            None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_fused_fp32_qk_norm",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._fused_fp32_qk_norm",
            return_value=(expected_query, expected_key),
        ) as fused_qk_norm,
    ):
        actual_query, actual_key = _apply_bagel_qk_norm(
            query, key, query_norm, key_norm, 4
        )

    assert actual_query is expected_query
    assert actual_key is expected_key
    fused_qk_norm.assert_called_once_with(query, key, query_norm, key_norm)


def test_bagel_qk_norm_falls_back_when_fp32_pair_kernel_is_unavailable() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 1, 4)
    query_norm = _BagelRMSNorm(4, eps=1e-6)
    key_norm = _BagelRMSNorm(4, eps=1e-6)

    with (
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._USE_FUSED_FP32_QK_NORM",
            None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._can_use_fused_fp32_qk_norm",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._fused_fp32_qk_norm",
            side_effect=ImportError("Triton backend unavailable"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.bagel_transformer.apply_qk_norm",
            side_effect=_eager_qk_norm_for_test,
        ) as fallback_qk_norm,
    ):
        actual_query, actual_key = _apply_bagel_qk_norm(
            query, key, query_norm, key_norm, 4
        )

    torch.testing.assert_close(actual_query, query_norm(query))
    torch.testing.assert_close(actual_key, key_norm(key))
    fallback_qk_norm.assert_called_once()


def test_bagel_qk_norm_rejects_mismatched_tokens() -> None:
    norm = _BagelRMSNorm(4, eps=1e-6)
    with pytest.raises(ValueError, match="matching token counts"):
        _apply_bagel_qk_norm(torch.randn(3, 2, 4), torch.randn(2, 1, 4), norm, norm, 4)


def test_understanding_projection_uses_pair_qk_norm(
    tiny_transformer: BagelTransformer,
) -> None:
    attention = tiny_transformer.layers[0].attn
    hidden_states = torch.randn(3, 8)
    expected_value = attention.und_v_proj(hidden_states).view(
        3, attention.num_kv_heads, attention.head_dim
    )

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._apply_bagel_qk_norm",
        wraps=_apply_bagel_qk_norm,
    ) as qk_norm:
        query, key, value = attention._project_understanding(hidden_states)

    qk_norm.assert_called_once()
    assert query.shape == (3, attention.num_heads, attention.head_dim)
    assert key.shape == (3, attention.num_kv_heads, attention.head_dim)
    torch.testing.assert_close(value, expected_value)


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


def test_batched_global_cfg_renorm_is_sample_local() -> None:
    conditional = torch.tensor(
        [
            [[2.0, -0.75, 0.25], [0.5, 1.25, -1.5]],
            [[-0.5, 4.0, 0.75], [1.5, -2.25, 0.125]],
        ],
        dtype=torch.bfloat16,
    )
    unconditional = torch.tensor(
        [
            [[1.0, 0.5, -0.25], [-0.75, 0.25, 1.0]],
            [[0.25, 1.0, -1.25], [0.5, -0.5, 0.375]],
        ],
        dtype=torch.bfloat16,
    )
    vector_norm = torch.linalg.vector_norm

    def fp32_vector_norm(
        value: torch.Tensor,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        return vector_norm(value.float(), dim=dim, keepdim=keepdim)

    # CUDA autocast executes vector_norm in FP32. Emulate that promotion on
    # CPU so this test protects the batch/sequential velocity dtype contract.
    with patch.object(torch.linalg, "vector_norm", side_effect=fp32_vector_norm):
        batched = BagelTransformer._apply_cfg(
            conditional,
            unconditional,
            4.0,
            renorm_min=0.0,
            renorm_type="global",
        )
        sequential = torch.stack(
            [
                BagelTransformer._apply_cfg(
                    conditional[index],
                    unconditional[index],
                    4.0,
                    renorm_min=0.0,
                    renorm_type="global",
                )
                for index in range(2)
            ]
        )
        channel_output = BagelTransformer._apply_cfg(
            conditional[0],
            unconditional[0],
            4.0,
            renorm_min=0.0,
            renorm_type="channel",
        )

    assert batched.dtype == torch.bfloat16
    assert sequential.dtype == torch.bfloat16
    assert channel_output.dtype == torch.float32
    torch.testing.assert_close(batched, sequential, rtol=0, atol=0)


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

    taylorseer = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=9,
        has_secondary=True,
    )
    tiny_transformer(
        latents,
        torch.tensor([0.5]),
        bagel_context=context,
        guidance_scale=4.0,
        image_guidance_scale=1.0,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_type="text_channel",
        taylorseer_context=taylorseer,
    )
    assert taylorseer.conditional.completed_steps == 1
    assert taylorseer.unconditional.completed_steps == 1
    assert taylorseer.secondary_unconditional is not None
    assert taylorseer.secondary_unconditional.completed_steps == 0

    for sigma in (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        tiny_transformer(
            latents,
            torch.tensor([sigma]),
            bagel_context=context,
            guidance_scale=4.0,
            image_guidance_scale=2.0,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_type="text_channel",
            taylorseer_context=taylorseer,
        )
    assert taylorseer.conditional.completed_steps == 9
    assert taylorseer.unconditional.completed_steps == 9
    assert taylorseer.secondary_unconditional.completed_steps == 8
    assert taylorseer.secondary_unconditional.get_stats() == {
        "total_steps": 8,
        "full_steps": 6,
        "taylor_steps": 2,
    }


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

    with patch(
        "sglang.multimodal_gen.runtime.models.dits.bagel_transformer._apply_bagel_qk_norm",
        wraps=_apply_bagel_qk_norm,
    ) as qk_norm:
        query, key, value = attention._project_generation(
            hidden_states, text_indexes, latent_indexes
        )

    assert query.dtype == torch.float32
    assert key.dtype == torch.float32
    assert value.dtype == torch.bfloat16
    assert qk_norm.call_count == 2
    assert all(call.args[0].dtype == torch.float32 for call in qk_norm.call_args_list)
    assert all(call.args[1].dtype == torch.float32 for call in qk_norm.call_args_list)
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
        guidance_scale=2.0,
    )
    assert prediction.dtype == torch.bfloat16


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
