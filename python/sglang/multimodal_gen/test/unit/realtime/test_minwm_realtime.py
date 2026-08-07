# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_ACTION_LABELS_CONDITION,
    MINWM_ACTION_WEIGHTS_CONDITION,
    MINWM_CHUNK_SEED_CONDITION,
    MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION,
    MINWM_CONDITION_SWITCH_CONDITION,
    MINWM_PROMPT_UPDATED_CONDITION,
    MINWM_TOTAL_CHUNKS_CONDITION,
    MINWM_TOTAL_LATENT_FRAMES_CONDITION,
    MinWMCausalDMDConfig,
    minwm_t5_postprocess_text,
)
from sglang.multimodal_gen.configs.sample.minwm import MinWMSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.minwm_realtime_adapter import (
    MinWMRealtimeAdapter,
    MinWMRealtimeState,
)
from sglang.multimodal_gen.runtime.models.dits.minwm import (
    MinWMCausalSelfAttention,
    MinWMCausalTransformer3DModel,
    MinWMPatchEmbed,
    MinWMRMSNorm,
    _frame_gate,
    _frame_modulation,
    _minwm_adaln_modulation,
    _minwm_adaln_op,
    _minwm_apply_qk_op,
    _minwm_frame_indices,
    _minwm_layer_norm,
    _minwm_packed_attention_backend,
    _minwm_project_output_in_reference_row_bucket,
    _minwm_qk_norm_op,
    _minwm_qk_norm_rope_op,
    _minwm_should_restore_reference_output_projection,
    _minwm_uniform_cu_seqlens,
    _minwm_uniform_frame_indices,
    apply_minwm_rotary_embedding,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    PrimitiveRoPETokenResidualActionEncoder,
    PrimitiveTokenResidualActionEncoder,
    action_labels_to_primitive_bits,
    key_state_to_action_label,
    validate_action_labels,
    validate_action_weights,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_kv_cache import (
    MinWMCausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.pipelines.minwm_causal_dmd_pipeline import (
    MinWMCausalDMDPipeline,
    MinWMCausalUniPCPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising import (
    MinWMCausalDMDDenoisingStage,
    MinWMCausalUniPCDenoisingStage,
    MinWMCausalVaeDecodingStage,
    MinWMChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.tools.convert_minwm_checkpoint import (
    DEFAULT_SOURCE_URI,
    TRANSFORMER_CONFIG,
    build_transformer_config,
)


@pytest.mark.parametrize(
    ("requested_mode", "first_frame", "expected_mode"),
    [
        (None, None, "t2v"),
        (None, b"image", "i2v"),
        ("t2v", None, "t2v"),
        ("i2v", b"image", "i2v"),
    ],
)
def test_minwm_normalizes_realtime_generation_mode(
    requested_mode, first_frame, expected_mode
):
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="test",
        generation_mode=requested_mode,
        first_frame=first_frame,
    )

    MinWMRealtimeAdapter._normalize_generation_mode(request)

    assert request.generation_mode == expected_mode


@pytest.mark.parametrize(
    ("requested_mode", "first_frame", "message"),
    [
        ("i2v", None, "I2V requires first_frame"),
        ("t2v", b"image", "T2V does not accept first_frame"),
    ],
)
def test_minwm_rejects_generation_mode_input_mismatch(
    requested_mode, first_frame, message
):
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="test",
        generation_mode=requested_mode,
        first_frame=first_frame,
    )

    with pytest.raises(ValueError, match=message):
        MinWMRealtimeAdapter._normalize_generation_mode(request)


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ([], 0),
        (["w"], 9),
        (["s"], 18),
        (["a"], 27),
        (["d"], 36),
        (["i"], 3),
        (["j"], 2),
        (["k"], 4),
        (["l"], 1),
        (["w", "l"], 10),
        (["w", "a"], 45),
        (["i", "j"], 7),
        (["up", "right"], 5),
    ],
)
def test_minwm_action_key_ontology(keys, expected):
    assert key_state_to_action_label(keys) == expected


def test_minwm_action_label_bits_match_wasd_ijkl_order():
    labels = torch.tensor([[0, 9, 1, 10, 45, 7]])
    bits = action_labels_to_primitive_bits(labels).to(torch.int64)
    assert bits.tolist() == [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ]
    ]


@pytest.mark.parametrize(
    "encoder_cls",
    [PrimitiveTokenResidualActionEncoder, PrimitiveRoPETokenResidualActionEncoder],
)
def test_minwm_action_label_table_is_a_nonpersistent_model_buffer(encoder_cls):
    encoder = encoder_cls(dim=24, embed_dim=8, hidden_dim=16, kernel_size=3)
    assert "_label_to_bits" in dict(encoder.named_buffers())
    assert "_label_to_bits" not in encoder.state_dict()
    assert encoder._label_to_bits.device == encoder.proj.weight.device
    labels = torch.tensor([[0, 9, 10, 1]])
    torch.testing.assert_close(
        action_labels_to_primitive_bits(labels, label_to_bits=encoder._label_to_bits),
        action_labels_to_primitive_bits(labels),
    )

    with torch.device("meta"):
        meta_encoder = encoder_cls(dim=24, embed_dim=8, hidden_dim=16, kernel_size=3)
    assert meta_encoder.proj.weight.device.type == "meta"
    assert meta_encoder._label_to_bits.device.type == "cpu"


@pytest.mark.parametrize(
    ("key", "expected_label"),
    [
        ("w", 9),
        ("s", 18),
        ("a", 27),
        ("d", 36),
        ("i", 3),
        ("j", 2),
        ("k", 4),
        ("l", 1),
    ],
)
def test_minwm_realtime_state_preserves_single_key_direction(key, expected_label):
    state = MinWMRealtimeState()
    state.receive_camera_state([key], event_id=17)
    assert state.sample_action_labels(4) == [expected_label] * 4
    assert state.latest_sampled_event_id == 17


def test_minwm_realtime_action_switch_reaches_next_chunk():
    state = MinWMRealtimeState()
    session = SimpleNamespace(
        adapter_state=state,
        request=SimpleNamespace(prompt="street", max_chunks=None),
    )
    adapter = MinWMRealtimeAdapter()
    server_args = SimpleNamespace()

    idle = adapter.sample_chunk_inputs(
        session, server_args, SimpleNamespace(index=0), chunk_size=4
    )
    assert idle.condition_inputs[MINWM_ACTION_LABELS_CONDITION] == [0, 0, 0, 0]

    state.receive_camera_state(["l"], event_id=21)
    turning = adapter.sample_chunk_inputs(
        session, server_args, SimpleNamespace(index=1), chunk_size=4
    )
    assert turning.condition_inputs[MINWM_ACTION_LABELS_CONDITION] == [1, 1, 1, 1]
    assert adapter.get_realtime_event_id(session) == 21


def test_minwm_prompt_switch_reports_prompt_event_after_older_camera_event():
    state = MinWMRealtimeState()
    session = SimpleNamespace(
        adapter_state=state,
        request=SimpleNamespace(prompt="day", max_chunks=None),
    )
    adapter = MinWMRealtimeAdapter()
    server_args = SimpleNamespace()

    state.receive_camera_state(["w"], event_id=17)
    adapter.sample_chunk_inputs(
        session, server_args, SimpleNamespace(index=0), chunk_size=4
    )
    state.receive_prompt("snowy night", event_id=23)
    switched = adapter.sample_chunk_inputs(
        session, server_args, SimpleNamespace(index=1), chunk_size=4
    )

    assert switched.prompt == "snowy night"
    assert switched.condition_inputs[MINWM_PROMPT_UPDATED_CONDITION] is True
    assert adapter.get_realtime_event_id(session) == 23


def test_minwm_action_validation_is_exact_integer_labels():
    assert validate_action_labels([0, 80], expected_frames=2) == [0, 80]
    for invalid in ([True], [81], [-1], [1.0], torch.tensor([1])):
        with pytest.raises(ValueError):
            validate_action_labels(invalid)


def test_minwm_action_weight_validation_preserves_fractional_amplitude():
    row = [0.8, 0, 0, 0, 0, 0, 0, 0]
    assert validate_action_weights([row], expected_frames=1) == [
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    for invalid in ([[1.1] + [0] * 7], [[float("nan")] + [0] * 7], [[1] * 7]):
        with pytest.raises(ValueError):
            validate_action_weights(invalid)


def test_minwm_binary_weight_windows_are_label_path_degenerate_case():
    torch.manual_seed(17)
    encoder = PrimitiveTokenResidualActionEncoder(
        dim=24, embed_dim=8, hidden_dim=16, kernel_size=3
    )
    labels = torch.tensor([[0, 9, 10, 1]])
    windows = action_labels_to_primitive_bits(labels).unsqueeze(2).expand(-1, -1, 4, -1)
    # The two forms are mathematically equivalent. Their different GEMM batch
    # shapes can select slightly different CPU kernels, so this is a numerical
    # semantic check; end-to-end parity compares the same weighted path.
    torch.testing.assert_close(
        encoder.frame_states(windows),
        encoder.frame_states(labels),
        rtol=3e-5,
        atol=3e-7,
    )


def test_minwm_primitive_rope_action_binary_windows_match_labels():
    torch.manual_seed(29)
    encoder = PrimitiveRoPETokenResidualActionEncoder(
        dim=24, embed_dim=8, hidden_dim=16, kernel_size=3
    )
    labels = torch.tensor([[0, 9, 10, 1]])
    windows = action_labels_to_primitive_bits(labels).unsqueeze(2).expand(-1, -1, 4, -1)
    torch.testing.assert_close(
        encoder.frame_states(windows),
        encoder.frame_states(labels),
        rtol=0,
        atol=0,
    )


def _make_minwm_test_cache(
    *,
    cache_size=6,
    sink_tokens=1,
    rope_position_mode="absolute",
    rope_max_frame_gap=1,
    prompt_first_frame_pin_enabled=False,
    scene_cut_rope_offset=0,
    scene_cut_sink_enabled=False,
):
    return MinWMCausalSelfAttentionKVCache(
        k=torch.zeros(1, cache_size, 1, 2),
        v=torch.zeros(1, cache_size, 1, 2),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        cache_size=cache_size,
        sink_tokens=sink_tokens,
        attention_window_size=cache_size,
        rope_position_mode=rope_position_mode,
        rope_max_frame_gap=rope_max_frame_gap,
        prompt_first_frame_pin_enabled=prompt_first_frame_pin_enabled,
        scene_cut_rope_offset=scene_cut_rope_offset,
        scene_cut_sink_enabled=scene_cut_sink_enabled,
    )


def _append_minwm_test_frames(cache, frames, *, token_start):
    frames = torch.tensor(frames, dtype=torch.long)
    position_ids = torch.stack(
        [frames, torch.zeros_like(frames), torch.zeros_like(frames)], dim=1
    )
    values = (
        torch.arange(token_start, token_start + len(frames), dtype=torch.float32)
        .view(1, -1, 1, 1)
        .expand(-1, -1, -1, 2)
    )
    return cache.update_and_get_attention_kv(
        key=values,
        value=-values,
        current_chunk_start=token_start,
        position_ids=position_ids,
    )


def test_minwm_raw_k_cache_overwrites_active_chunk_without_appending():
    cache = _make_minwm_test_cache(cache_size=4, sink_tokens=0)
    first = _append_minwm_test_frames(cache, [0, 1], token_start=0)
    replacement = torch.full((1, 2, 1, 2), 7.0)
    cache.update_and_get_attention_kv(
        key=replacement,
        value=-replacement,
        current_chunk_start=0,
        position_ids=torch.tensor([[0, 0, 0], [1, 0, 0]]),
    )
    assert cache._read_indices() == (2, 2)
    assert cache.token_ids.tolist() == [0, 1]
    torch.testing.assert_close(cache.k[:, :2], replacement, rtol=0, atol=0)
    assert first.key_position_ids[:, 0].tolist() == [0, 1]


def test_minwm_cache_plan_is_shared_across_layers_and_reused_for_recompute():
    first_cache = _make_minwm_test_cache(cache_size=6, sink_tokens=1)
    second_cache = _make_minwm_test_cache(cache_size=6, sink_tokens=1)
    position_ids = torch.tensor([[0, 0, 0], [1, 0, 0]])
    plan = first_cache.prepare_attention_plan(
        current_chunk_start=0,
        position_ids=position_ids,
    )
    first_cache.set_prepared_attention_plan(plan)
    second_cache.set_prepared_attention_plan(plan)
    first_values = torch.ones(1, 2, 1, 2)
    second_values = torch.full((1, 2, 1, 2), 2.0)
    first_cache.update_and_get_attention_kv(
        key=first_values,
        value=-first_values,
        current_chunk_start=0,
    )
    second_cache.update_and_get_attention_kv(
        key=second_values,
        value=-second_values,
        current_chunk_start=0,
    )

    assert first_cache.position_ids is second_cache.position_ids
    assert first_cache.token_ids is second_cache.token_ids
    torch.testing.assert_close(first_cache.k[:, :2], first_values)
    torch.testing.assert_close(second_cache.k[:, :2], second_values)

    recompute = first_cache.prepare_attention_plan(
        current_chunk_start=0,
        position_ids=position_ids,
    )
    assert recompute.is_recompute
    assert (
        first_cache.prepare_attention_plan(
            current_chunk_start=0,
            position_ids=position_ids,
        )
        is recompute
    )


def test_minwm_cache_append_does_not_repack_visible_history(monkeypatch):
    cache = _make_minwm_test_cache(cache_size=6, sink_tokens=1)

    def fail_if_selected(*_args, **_kwargs):
        raise AssertionError("non-evicting append must not select and repack history")

    monkeypatch.setattr(
        MinWMCausalSelfAttentionKVCache,
        "_select_kv_with_plan",
        staticmethod(fail_if_selected),
    )
    _append_minwm_test_frames(cache, [0, 1], token_start=0)
    _append_minwm_test_frames(cache, [2, 3], token_start=2)

    assert cache.last_attention_plan.preserves_all_history
    assert cache.token_ids.tolist() == [0, 1, 2, 3]
    assert cache.k[0, :4, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_minwm_fixed_shape_metadata_is_cached():
    _minwm_uniform_cu_seqlens.cache_clear()
    cu_seqlens = _minwm_uniform_cu_seqlens(2, 7, torch.device("cpu"))
    assert cu_seqlens.tolist() == [0, 7, 14]
    assert _minwm_uniform_cu_seqlens(2, 7, torch.device("cpu")) is cu_seqlens

    _minwm_uniform_frame_indices.cache_clear()
    frame_indices = _minwm_uniform_frame_indices(6, 3, torch.device("cpu"))
    assert frame_indices.tolist() == [0, 0, 1, 1, 2, 2]
    assert _minwm_uniform_frame_indices(6, 3, torch.device("cpu")) is frame_indices


def test_minwm_block_relative_rope_clamps_visible_frame_gaps():
    cache = _make_minwm_test_cache(
        cache_size=6,
        sink_tokens=1,
        rope_position_mode="block_relative",
        rope_max_frame_gap=3,
    )
    view = None
    for token_start, frame in enumerate([0, 1, 2, 10, 11, 12, 13, 14]):
        view = _append_minwm_test_frames(cache, [frame], token_start=token_start)
    assert cache.position_ids[:, 0].tolist() == [0, 10, 11, 12, 13, 14]
    assert view.key_position_ids[:, 0].tolist() == [0, 3, 4, 5, 6, 7]
    assert view.query_position_ids[:, 0].tolist() == [7]


def test_minwm_prompt_first_frame_promotes_only_when_leaving_tail():
    cache = _make_minwm_test_cache(
        cache_size=6,
        sink_tokens=1,
        rope_position_mode="block_relative",
        prompt_first_frame_pin_enabled=True,
    )
    _append_minwm_test_frames(cache, [0, 1], token_start=0)
    cache.mark_prompt_switch()
    _append_minwm_test_frames(cache, [2], token_start=2)
    for frame in (3, 4, 5, 6):
        _append_minwm_test_frames(cache, [frame], token_start=frame)
    assert cache.token_ids.tolist() == [0, 2, 3, 4, 5, 6]
    assert cache.pinned_token_start is None
    _append_minwm_test_frames(cache, [7], token_start=7)
    assert cache.token_ids.tolist() == [0, 2, 4, 5, 6, 7]
    assert (cache.pinned_token_start, cache.pinned_token_end) == (2, 3)


def test_minwm_scene_cut_updates_absolute_rope_and_pins_sink_prefix():
    cache = _make_minwm_test_cache(
        cache_size=6,
        sink_tokens=2,
        scene_cut_rope_offset=11,
        scene_cut_sink_enabled=True,
    )
    _append_minwm_test_frames(cache, [0, 1], token_start=0)
    cache.mark_scene_cut()
    view = _append_minwm_test_frames(cache, [2, 3, 4], token_start=2)
    assert view.query_position_ids[:, 0].tolist() == [13, 14, 15]
    assert (cache.pinned_token_start, cache.pinned_token_end) == (2, 4)


def test_minwm_action_history_chunk_matches_full_sequence():
    torch.manual_seed(7)
    encoder = PrimitiveTokenResidualActionEncoder(
        dim=24, embed_dim=8, hidden_dim=16, kernel_size=3
    )
    labels = torch.tensor([[0, 9, 10, 1, 45, 7, 36, 4, 80]])
    full = encoder.frame_states(labels)
    chunk_window = encoder.frame_states(labels[:, -8:])
    # Convolution kernels can accumulate the full sequence and the sliced
    # history window in a different order even on CPU. The causal receptive
    # fields are mathematically identical; observed PyTorch drift is < 6e-8.
    torch.testing.assert_close(chunk_window[:, -4:], full[:, -4:], rtol=1e-6, atol=1e-7)
    token_residual = encoder.token_residual(
        labels[:, -8:],
        num_current_frames=4,
        tokens_per_frame=6,
        dtype=torch.float32,
    )
    assert token_residual.is_contiguous()
    assert token_residual.stride(-1) == 1


def test_minwm_bounded_session_presamples_reference_and_full_horizon(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising as stage_module

    monkeypatch.setattr(
        stage_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(num_frames_per_block=4, out_channels=2)
        )
    )
    stage = MinWMChunkLatentPreparationStage(transformer)
    session = RealtimeSession()
    generator = torch.Generator().manual_seed(123)
    condition = torch.zeros(1, 2, 1, 1, 3)

    expected_generator = torch.Generator().manual_seed(123)
    expected = torch.randn(
        (1, 9, 2, 1, 3), generator=expected_generator, dtype=torch.float32
    )[:, 1:]

    def make_batch(block_idx):
        return SimpleNamespace(
            latents=None,
            image_latent=condition,
            realtime_chunk_size=4,
            generator=generator,
            session=session,
            block_idx=block_idx,
            condition_inputs={MINWM_TOTAL_CHUNKS_CONDITION: 2},
            raw_latent_shape=None,
        )

    first = stage.forward(make_batch(0), SimpleNamespace())
    second = stage.forward(make_batch(1), SimpleNamespace())
    torch.testing.assert_close(
        first.latents,
        expected[:, :4].permute(0, 2, 1, 3, 4),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        second.latents,
        expected[:, 4:].permute(0, 2, 1, 3, 4),
        rtol=0,
        atol=0,
    )
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_minwm_t2v_uses_first_regular_and_remainder_chunk_sizes():
    adapter = MinWMRealtimeAdapter()
    session = SimpleNamespace(request=SimpleNamespace(first_frame=None, num_frames=725))
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    num_frame_first_block=1,
                    num_frames_per_block=4,
                )
            ),
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            ),
        )
    )

    assert adapter.get_chunk_size(session, server_args, SimpleNamespace(index=0)) == 1
    assert adapter.get_chunk_size(session, server_args, SimpleNamespace(index=1)) == 4
    assert adapter.get_chunk_size(session, server_args, SimpleNamespace(index=45)) == 4
    assert adapter.get_chunk_size(session, server_args, SimpleNamespace(index=46)) == 1

    session.request.first_frame = "/tmp/reference.png"
    assert adapter.get_chunk_size(session, server_args, SimpleNamespace(index=0)) == 4


def test_minwm_t2v_presamples_exact_horizon_without_reference_slot(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising as stage_module

    monkeypatch.setattr(
        stage_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                num_frame_first_block=1,
                num_frames_per_block=4,
                out_channels=2,
            )
        )
    )
    stage = MinWMChunkLatentPreparationStage(transformer)
    session = RealtimeSession()
    generator = torch.Generator().manual_seed(123)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_spatial=16)
            )
        )
    )

    expected_generator = torch.Generator().manual_seed(123)
    expected = torch.randn(
        (1, 6, 2, 1, 2), generator=expected_generator, dtype=torch.float32
    )

    def make_batch(block_idx, chunk_size):
        return SimpleNamespace(
            latents=None,
            image_latent=None,
            realtime_chunk_size=chunk_size,
            generator=generator,
            session=session,
            block_idx=block_idx,
            condition_inputs={
                MINWM_TOTAL_CHUNKS_CONDITION: 3,
                MINWM_TOTAL_LATENT_FRAMES_CONDITION: 6,
            },
            raw_latent_shape=None,
            height=16,
            width=32,
            batch_size=1,
            prompt_embeds=[torch.zeros(1, 1, 1)],
        )

    first = stage.forward(make_batch(0, 1), server_args)
    middle = stage.forward(make_batch(1, 4), server_args)
    final = stage.forward(make_batch(2, 1), server_args)
    actual = torch.cat([first.latents, middle.latents, final.latents], dim=2)
    torch.testing.assert_close(
        actual,
        expected.permute(0, 2, 1, 3, 4),
        rtol=0,
        atol=0,
    )
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_minwm_director_chunk_seed_replays_prefix_rng_before_tail(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising as stage_module

    monkeypatch.setattr(
        stage_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(num_frames_per_block=4, out_channels=2)
        )
    )
    stage = MinWMChunkLatentPreparationStage(transformer)
    prefix_frames = 5
    chunk_frames = 4
    seed = 729003
    batch = SimpleNamespace(
        latents=None,
        image_latent=None,
        realtime_chunk_size=chunk_frames,
        generator=torch.Generator().manual_seed(1),
        session=RealtimeSession(),
        block_idx=2,
        condition_inputs={
            MINWM_CHUNK_SEED_CONDITION: seed,
            MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION: prefix_frames,
        },
        raw_latent_shape=None,
        height=16,
        width=32,
        batch_size=1,
        prompt_embeds=[torch.zeros(1, 1, 1)],
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_spatial=16)
            )
        )
    )
    expected_generator = torch.Generator().manual_seed(seed)
    expected = torch.randn(
        (1, prefix_frames + chunk_frames, 2, 1, 2),
        generator=expected_generator,
    )[:, prefix_frames:]
    output = stage.forward(batch, server_args)
    torch.testing.assert_close(
        output.latents,
        expected.permute(0, 2, 1, 3, 4),
        rtol=0,
        atol=0,
    )


def test_minwm_default_kv_horizon_retains_complete_bounded_session():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                sink_size=0,
                local_attn_size=-1,
                sliding_window_num_frames=128,
            )
        )
    )
    stage.sink_size = 0
    stage.sliding_window_num_frames = 128
    stage.num_frames_per_block = 4
    batch = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
        condition_inputs={MINWM_TOTAL_CHUNKS_CONDITION: 8},
        image_latent=torch.empty(1),
    )
    pipeline_config = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
    )
    stage._apply_causal_cache_overrides(
        batch, SimpleNamespace(pipeline_config=pipeline_config)
    )
    assert stage.sliding_window_num_frames == 33
    assert stage._minwm_unbounded_cache is True

    batch.realtime_causal_kv_cache_num_frames = 45
    stage._apply_causal_cache_overrides(
        batch, SimpleNamespace(pipeline_config=pipeline_config)
    )
    assert stage.sliding_window_num_frames == 45
    assert stage._minwm_unbounded_cache is False


def test_minwm_causal_cache_overrides_do_not_leak_between_requests():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                sink_size=0,
                local_attn_size=-1,
                sliding_window_num_frames=128,
            )
        )
    )
    stage.sink_size = 0
    stage.sliding_window_num_frames = 128
    stage.num_frames_per_block = 4
    pipeline_config = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
    )
    server_args = SimpleNamespace(pipeline_config=pipeline_config)

    bounded_request = SimpleNamespace(
        realtime_causal_sink_size=9,
        realtime_causal_kv_cache_num_frames=18,
        condition_inputs={MINWM_TOTAL_CHUNKS_CONDITION: 8},
        image_latent=torch.empty(1),
    )
    stage._apply_causal_cache_overrides(bounded_request, server_args)
    assert stage.sink_size == 9
    assert stage.sliding_window_num_frames == 18
    assert stage._minwm_unbounded_cache is False

    default_request = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
        condition_inputs={MINWM_TOTAL_CHUNKS_CONDITION: 8},
        image_latent=torch.empty(1),
    )
    stage._apply_causal_cache_overrides(default_request, server_args)
    assert stage.sink_size == 0
    assert stage.sliding_window_num_frames == 33
    assert stage._minwm_unbounded_cache is True


def test_minwm_t2v_default_kv_horizon_uses_exact_latent_count():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                sink_size=0,
                local_attn_size=-1,
                sliding_window_num_frames=128,
                num_frame_first_block=1,
            )
        )
    )
    stage.sink_size = 0
    stage.sliding_window_num_frames = 128
    stage.num_frames_per_block = 4
    batch = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
        condition_inputs={MINWM_TOTAL_LATENT_FRAMES_CONDITION: 182},
        image_latent=None,
    )
    pipeline_config = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
    )

    stage._apply_causal_cache_overrides(
        batch, SimpleNamespace(pipeline_config=pipeline_config)
    )

    assert stage.sliding_window_num_frames == 182
    assert stage._minwm_unbounded_cache is True


def test_minwm_model_bounded_window_is_not_expanded_to_request_horizon():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                sink_size=8,
                local_attn_size=32,
                sliding_window_num_frames=32,
            )
        )
    )
    stage.sink_size = 8
    stage.sliding_window_num_frames = 32
    stage.num_frames_per_block = 4
    batch = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
        condition_inputs={MINWM_TOTAL_LATENT_FRAMES_CONDITION: 273},
        image_latent=None,
    )
    pipeline_config = SimpleNamespace(
        realtime_causal_sink_size=None,
        realtime_causal_kv_cache_num_frames=None,
    )
    stage._apply_causal_cache_overrides(
        batch, SimpleNamespace(pipeline_config=pipeline_config)
    )
    assert stage.sliding_window_num_frames == 32
    assert stage.sink_size == 8
    assert stage._minwm_unbounded_cache is False


def test_minwm_t2v_decoder_does_not_prepend_a_reference(monkeypatch):
    seen = []

    def fake_forward(_self, batch, _server_args):
        seen.append(batch.latents.clone())
        return batch.latents

    monkeypatch.setattr(CausalVaeDecodingStage, "forward", fake_forward)
    stage = MinWMCausalVaeDecodingStage.__new__(MinWMCausalVaeDecodingStage)
    generated = torch.ones(1, 2, 1, 1, 1)
    t2v_batch = SimpleNamespace(
        block_idx=0,
        image_latent=None,
        latents=generated,
    )
    i2v_batch = SimpleNamespace(
        block_idx=0,
        image_latent=torch.zeros_like(generated),
        latents=generated,
    )

    stage.forward(t2v_batch, SimpleNamespace())
    stage.forward(i2v_batch, SimpleNamespace())

    assert seen[0].shape[2] == 1
    assert seen[1].shape[2] == 2
    assert t2v_batch.latents is generated
    assert i2v_batch.latents is generated


def test_minwm_t2v_decoder_reseeds_one_latent_first_block(monkeypatch):
    seen = []

    def fake_forward(_self, batch, _server_args):
        seen.append((batch.block_idx, batch.latents.clone()))
        pixel_frames = 1 + 4 * (batch.latents.shape[2] - 1)
        return OutputBatch(
            output=torch.arange(pixel_frames).reshape(1, 1, pixel_frames, 1, 1)
        )

    monkeypatch.setattr(CausalVaeDecodingStage, "forward", fake_forward)
    stage = MinWMCausalVaeDecodingStage.__new__(MinWMCausalVaeDecodingStage)
    session = RealtimeSession()
    first_latent = torch.full((1, 2, 1, 1, 1), 1.0)
    regular_latents = torch.full((1, 2, 4, 1, 1), 2.0)
    first_batch = SimpleNamespace(
        block_idx=0,
        image_latent=None,
        latents=first_latent,
        session=session,
    )
    regular_batch = SimpleNamespace(
        block_idx=1,
        image_latent=None,
        latents=regular_latents,
        session=session,
    )

    first_output = stage.forward(first_batch, SimpleNamespace())
    regular_output = stage.forward(regular_batch, SimpleNamespace())

    assert seen[0][0] == 0
    assert seen[0][1].shape[2] == 1
    assert seen[1][0] == 0
    assert seen[1][1].shape[2] == 5
    torch.testing.assert_close(seen[1][1][:, :, :1], first_latent)
    assert first_output.output.shape[2] == 1
    assert regular_output.output.shape[2] == 16
    assert regular_output.output.flatten()[0].item() == 1
    assert first_batch.block_idx == 0
    assert regular_batch.block_idx == 1
    assert first_batch.latents is first_latent
    assert regular_batch.latents is regular_latents


def test_minwm_remote_vae_preserves_output_index_and_t2v_trim(monkeypatch):
    def fake_forward(_self, batch, _server_args):
        return OutputBatch(
            remote_vae_request={
                "block_idx": batch.block_idx,
                "output_block_idx": batch.block_idx,
                "trim_leading_frames": 0,
            }
        )

    monkeypatch.setattr(CausalVaeDecodingStage, "forward", fake_forward)
    stage = MinWMCausalVaeDecodingStage.__new__(MinWMCausalVaeDecodingStage)
    session = RealtimeSession()
    first_latent = torch.full((1, 2, 1, 1, 1), 1.0)
    regular_latents = torch.full((1, 2, 4, 1, 1), 2.0)
    first_batch = SimpleNamespace(
        block_idx=0,
        image_latent=None,
        latents=first_latent,
        session=session,
        realtime_event_id=10,
    )
    regular_batch = SimpleNamespace(
        block_idx=1,
        image_latent=None,
        latents=regular_latents,
        session=session,
        realtime_event_id=11,
    )

    stage.forward(first_batch, SimpleNamespace())
    result = stage.forward(regular_batch, SimpleNamespace())

    assert result.remote_vae_request["block_idx"] == 0
    assert result.remote_vae_request["output_block_idx"] == 1
    assert result.remote_vae_request["trim_leading_frames"] == 1
    assert result.realtime_output_chunk_index_start == 1
    assert result.realtime_output_event_id == 11
    assert regular_batch.block_idx == 1
    assert regular_batch.latents is regular_latents


def test_minwm_unbounded_kv_policy_reaches_cache_allocation():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.num_transformer_blocks = 2
    stage.num_token_per_frame = 3
    stage.sliding_window_num_frames = 5
    stage.local_attn_size = -1
    stage.sink_size = 0
    stage.transformer = SimpleNamespace(
        num_attention_heads=2,
        attention_head_dim=4,
    )

    stage._initialize_kv_cache(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        allow_growth=True,
    )

    assert len(stage.causal_kv_cache) == 2
    assert all(cache.allow_growth for cache in stage.causal_kv_cache)
    assert stage.causal_kv_cache[0].k.shape == (1, 15, 2, 4)


def _minwm_cuda_graph_cache_stage(*, allow_growth, rope_position_mode):
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage._minwm_cuda_graph_enabled = True
    stage._minwm_unbounded_cache = allow_growth
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                rope_position_mode=rope_position_mode,
                rope_max_frame_gap=1,
                prompt_first_frame_pin_enabled=False,
                scene_cut_rope_offset=0,
                scene_cut_sink_enabled=False,
            )
        )
    )
    return stage


def test_minwm_cuda_graph_accepts_bounded_block_relative_cache():
    stage = _minwm_cuda_graph_cache_stage(
        allow_growth=False,
        rope_position_mode="block_relative",
    )
    kwargs = stage._causal_kv_cache_kwargs(
        SimpleNamespace(sequence_shard_enabled=True, expected_cache_tokens=32)
    )

    assert kwargs["allow_growth"] is False
    assert kwargs["rope_position_mode"] == "block_relative"


@pytest.mark.parametrize(
    ("allow_growth", "rope_position_mode", "message"),
    [
        (True, "block_relative", "bounded realtime KV window"),
        (False, "absolute", "block_relative RoPE"),
    ],
)
def test_minwm_cuda_graph_rejects_dynamic_cache_contracts(
    allow_growth, rope_position_mode, message
):
    stage = _minwm_cuda_graph_cache_stage(
        allow_growth=allow_growth,
        rope_position_mode=rope_position_mode,
    )

    with pytest.raises(ValueError, match=message):
        stage._causal_kv_cache_kwargs(
            SimpleNamespace(sequence_shard_enabled=False, expected_cache_tokens=32)
        )


def test_minwm_unipc_scheduler_matches_native_shift_contract():
    pipeline = MinWMCausalUniPCPipeline.__new__(MinWMCausalUniPCPipeline)
    pipeline.modules = {}
    pipeline.initialize_pipeline(
        SimpleNamespace(pipeline_config=SimpleNamespace(flow_shift=5.0))
    )
    scheduler = pipeline.modules["scheduler"]

    assert scheduler.__class__.__name__ == "MinWMFlowUniPCParityScheduler"
    assert scheduler.config.shift == 1.0
    scheduler.set_timesteps(4, device="cpu", shift=5.0)
    assert scheduler.sigmas.device.type == "cpu"
    # This is the exact NumPy FP64 schedule produced by minWM 4220c8a from
    # its FP32 sigma bounds. The previous 936/832 expectation was stale.
    assert scheduler.timesteps.tolist() == [999, 937, 833, 624]


def test_minwm_unipc_stage_steps_in_native_bfchw_layout():
    stage = MinWMCausalUniPCDenoisingStage.__new__(MinWMCausalUniPCDenoisingStage)
    stage._build_causal_attn_metadata = lambda *args, **kwargs: None
    transformer_inputs = []

    def fake_transformer(_batch, **kwargs):
        transformer_inputs.append(kwargs["latent_model_input"].clone())
        return torch.ones_like(kwargs["latent_model_input"])

    stage._forward_causal_transformer = fake_transformer

    class Scheduler:
        def __init__(self):
            self.sample_shapes = []

        def step(self, model_output, _timestep, sample, return_dict):
            assert return_dict is False
            assert model_output.shape == (1, 2, 3, 1, 1)
            self.sample_shapes.append(sample.shape)
            return (sample + 1,)

    scheduler = Scheduler()
    chunk_latents = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2, 1, 1)
    output, metadata = stage._denoise_causal_dmd_chunk(
        SimpleNamespace(),
        SimpleNamespace(),
        chunk_latents=chunk_latents,
        scheduler=scheduler,
        timesteps=torch.tensor([999, 624]),
        prompt_embeds=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start_tokens=0,
        start_frame=0,
        image_kwargs={},
        pos_cond_kwargs={},
        target_dtype=torch.float32,
        autocast_enabled=False,
        device=torch.device("cpu"),
        attn_raw_latent_shape=(2, 1, 1),
        prepare_model_input=lambda value: value,
    )

    assert scheduler.sample_shapes == [(1, 2, 3, 1, 1)] * 2
    assert len(transformer_inputs) == 2
    torch.testing.assert_close(output, chunk_latents + 2, rtol=0, atol=0)
    assert metadata is None


def test_minwm_t2v_starts_fractional_actions_without_reference_history():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(arch_config=SimpleNamespace(action_history_frames=4))
    )
    window = [[0.8, 0, 0, 0, 0, 0, 0, 0]] * 4
    batch = SimpleNamespace(
        session=RealtimeSession(),
        block_idx=0,
        latents=torch.zeros(1, 2, 4, 1, 1),
        condition_inputs={MINWM_ACTION_WEIGHTS_CONDITION: [window] * 4},
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            )
        )
    )
    action = stage._prepare_causal_dmd_pos_cond_kwargs(
        batch, server_args, torch.bfloat16
    )["action"]
    assert action.shape == (1, 4, 4, 8)
    torch.testing.assert_close(
        action[:, :, :, 0], torch.full((1, 4, 4), 0.8), rtol=0, atol=0
    )


def test_minwm_i2v_starts_fractional_actions_after_reference_history():
    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        config=SimpleNamespace(arch_config=SimpleNamespace(action_history_frames=4))
    )
    window = [[0.8, 0, 0, 0, 0, 0, 0, 0]] * 4
    batch = SimpleNamespace(
        session=RealtimeSession(),
        block_idx=0,
        latents=torch.zeros(1, 2, 4, 1, 1),
        image_latent=torch.zeros(1, 2, 1, 1, 1),
        condition_inputs={MINWM_ACTION_WEIGHTS_CONDITION: [window] * 4},
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            )
        )
    )
    action = stage._prepare_causal_dmd_pos_cond_kwargs(
        batch, server_args, torch.bfloat16
    )["action"]
    assert action.shape == (1, 5, 4, 8)
    assert torch.count_nonzero(action[:, :1]).item() == 0
    torch.testing.assert_close(
        action[:, 1:, :, 0], torch.full((1, 4, 4), 0.8), rtol=0, atol=0
    )


def test_minwm_realtime_adapter_groups_pixel_weights_by_vae_factor():
    state = MinWMRealtimeState()
    row = [0.8, 0, 0, 0, 0, 0, 0, 0]
    state.receive_action_weights([row] * 16)
    session = SimpleNamespace(
        adapter_state=state,
        request=SimpleNamespace(
            prompt="test",
            max_chunks=8,
            first_frame="/tmp/reference.png",
            num_frames=None,
        ),
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            )
        )
    )
    inputs = MinWMRealtimeAdapter().sample_chunk_inputs(
        session,
        server_args,
        SimpleNamespace(index=0),
        chunk_size=4,
    )
    windows = inputs.condition_inputs[MINWM_ACTION_WEIGHTS_CONDITION]
    assert len(windows) == 4
    assert all(len(window) == 4 for window in windows)
    assert windows == [[row] * 4] * 4
    assert inputs.condition_inputs[MINWM_TOTAL_CHUNKS_CONDITION] == 8


def test_minwm_t2v_first_latent_is_noop_without_consuming_pixel_actions():
    state = MinWMRealtimeState()
    first_action = [0.8, 0, 0, 0, 0, 0, 0, 0]
    state.receive_action_weights([first_action] * 16)
    session = SimpleNamespace(
        adapter_state=state,
        request=SimpleNamespace(
            prompt="test",
            max_chunks=2,
            first_frame=None,
            num_frames=17,
        ),
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    num_frame_first_block=1,
                    num_frames_per_block=4,
                )
            ),
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            ),
        )
    )
    adapter = MinWMRealtimeAdapter()
    first = adapter.sample_chunk_inputs(
        session, server_args, SimpleNamespace(index=0), chunk_size=1
    )
    assert first.condition_inputs[MINWM_ACTION_WEIGHTS_CONDITION] == [[[0.0] * 8] * 4]
    second = adapter.sample_chunk_inputs(
        session, server_args, SimpleNamespace(index=1), chunk_size=4
    )
    assert (
        second.condition_inputs[MINWM_ACTION_WEIGHTS_CONDITION]
        == [[first_action] * 4] * 4
    )


def test_minwm_scheduled_prompt_and_seed_target_exact_chunk():
    state = MinWMRealtimeState()
    state.receive_prompt_schedule({3: ("night", "prompt")})
    state.receive_chunk_seeds([729001, 729002, 729003, 729004])
    session = SimpleNamespace(
        adapter_state=state,
        request=SimpleNamespace(
            prompt="day",
            max_chunks=4,
            first_frame=None,
            num_frames=53,
        ),
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    num_frame_first_block=1,
                    num_frames_per_block=4,
                )
            ),
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(scale_factor_temporal=4)
            ),
        )
    )
    adapter = MinWMRealtimeAdapter()
    for chunk_index, expected_seed in enumerate(range(729001, 729005)):
        chunk_size = 1 if chunk_index == 0 else 4
        inputs = adapter.sample_chunk_inputs(
            session,
            server_args,
            SimpleNamespace(index=chunk_index),
            chunk_size=chunk_size,
        )
        assert inputs.condition_inputs[MINWM_CHUNK_SEED_CONDITION] == expected_seed
        expected_prefix = 0 if chunk_index == 0 else 1 + (chunk_index - 1) * 4
        assert (
            inputs.condition_inputs[MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION]
            == expected_prefix
        )
        if chunk_index == 3:
            assert inputs.prompt == "night"
            assert inputs.condition_inputs[MINWM_PROMPT_UPDATED_CONDITION] is True
            assert inputs.condition_inputs[MINWM_CONDITION_SWITCH_CONDITION] == "prompt"
        else:
            assert inputs.prompt == "day"


def test_minwm_prompt_schedule_validation_accepts_scene_cut_kind():
    assert MinWMRealtimeAdapter._validate_prompt_schedule(
        [
            {"target_chunk": 2, "prompt": "snow", "kind": "scene_cut"},
            {"target_chunk": 5, "prompt": "night"},
        ]
    ) == {2: ("snow", "scene_cut"), 5: ("night", "prompt")}
    with pytest.raises(ValueError, match="duplicate"):
        MinWMRealtimeAdapter._validate_prompt_schedule(
            [
                {"target_chunk": 2, "prompt": "snow"},
                {"target_chunk": 2, "prompt": "night"},
            ]
        )


def test_minwm_text_context_keeps_zero_padded_512_contract():
    hidden = torch.randn(1, 1024, 8)
    attention_mask = torch.zeros(1, 1024, dtype=torch.long)
    attention_mask[:, :17] = 1
    output = minwm_t5_postprocess_text(
        SimpleNamespace(last_hidden_state=hidden, attention_mask=attention_mask),
        None,
    )
    assert output.prompt_seq_lens == [512]
    assert output.prompt_embeds_mask.sum().item() == 512
    torch.testing.assert_close(
        output.prompt_embeds[:, :17], hidden[:, :17], rtol=0, atol=0
    )
    assert torch.count_nonzero(output.prompt_embeds[:, 17:]).item() == 0


def test_minwm_native_hf_text_output_uses_tokenizer_attention_mask():
    hidden = torch.randn(1, 1024, 8)
    attention_mask = torch.zeros(1, 1024, dtype=torch.long)
    attention_mask[:, :23] = 1
    output = minwm_t5_postprocess_text(
        SimpleNamespace(last_hidden_state=hidden),
        {"attention_mask": attention_mask},
    )
    assert output.prompt_seq_lens == [512]
    assert output.prompt_embeds_mask.sum().item() == 512
    assert torch.count_nonzero(output.prompt_embeds[:, 23:]).item() == 0


def test_minwm_requires_baseline_native_text_and_vae_components():
    config = MinWMCausalDMDConfig()
    assert config.native_component_names == ("text_encoder", "vae")
    assert config.enable_autocast is False


def test_minwm_explicit_parallel_vae_lane_uses_sglang_vae(monkeypatch):
    monkeypatch.setenv("MINWM_VAE_LANE", "parallel")
    monkeypatch.setenv("MINWM_NATIVE_COMPONENTS", "text_encoder,vae")

    config = MinWMCausalDMDConfig()

    assert config.native_component_names == ("text_encoder",)
    assert config.vae_config.use_parallel_encode is False


def test_minwm_explicit_parity_vae_lane_uses_native_vae(monkeypatch):
    monkeypatch.setenv("MINWM_VAE_LANE", "parity")
    monkeypatch.setenv("MINWM_NATIVE_COMPONENTS", "")

    config = MinWMCausalDMDConfig()

    assert config.native_component_names == ("text_encoder", "vae")


def test_minwm_rejects_unknown_vae_lane(monkeypatch):
    monkeypatch.setenv("MINWM_VAE_LANE", "unknown")

    with pytest.raises(ValueError, match="MINWM_VAE_LANE"):
        MinWMCausalDMDConfig()


def test_minwm_converter_defaults_to_requested_0721_checkpoint():
    assert (
        "wan22-5B-stage3-dmd-8-0721-6a531f0e067/global_step_003200"
        in DEFAULT_SOURCE_URI
    )


def test_minwm_converter_records_explicit_cache_policy():
    config = build_transformer_config(
        local_attn_size=32,
        sink_size=8,
        sliding_window_num_frames=32,
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
        prompt_first_frame_pin_enabled=True,
    )
    assert config["local_attn_size"] == 32
    assert config["sink_size"] == 8
    assert config["sliding_window_num_frames"] == 32
    assert config["rope_position_mode"] == "block_relative"
    assert config["rope_max_frame_gap"] == 12
    assert config["prompt_first_frame_pin_enabled"] is True
    assert TRANSFORMER_CONFIG["local_attn_size"] == -1
    assert TRANSFORMER_CONFIG["num_frame_first_block"] == 1
    with pytest.raises(ValueError, match="smaller"):
        build_transformer_config(
            local_attn_size=18,
            sink_size=18,
            sliding_window_num_frames=18,
        )
    with pytest.raises(ValueError, match="equal"):
        build_transformer_config(
            local_attn_size=32,
            sink_size=8,
            sliding_window_num_frames=128,
        )


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_minwm_accepts_supported_ulysses_sequence_parallelism(degree):
    MinWMCausalDMDPipeline._validate_sequence_parallelism_args(
        SimpleNamespace(sp_degree=1, ulysses_degree=1, ring_degree=1)
    )
    MinWMCausalDMDPipeline._validate_sequence_parallelism_args(
        SimpleNamespace(
            sp_degree=degree,
            ulysses_degree=degree,
            ring_degree=1,
            tp_size=1,
        )
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            SimpleNamespace(sp_degree=2, ulysses_degree=1, ring_degree=2),
            "ring-degree 1",
        ),
        (
            SimpleNamespace(sp_degree=4, ulysses_degree=2, ring_degree=1),
            "sp-degree == --ulysses-degree",
        ),
        (
            SimpleNamespace(sp_degree=5, ulysses_degree=5, ring_degree=1),
            "must be divisible",
        ),
        (
            SimpleNamespace(sp_degree=2, ulysses_degree=2, ring_degree=1, tp_size=2),
            "tensor parallelism",
        ),
        (
            SimpleNamespace(
                sp_degree=2,
                ulysses_degree=2,
                ring_degree=1,
                use_fsdp_inference=True,
            ),
            "FSDP",
        ),
        (
            SimpleNamespace(
                sp_degree=2,
                ulysses_degree=2,
                ring_degree=1,
                enable_torch_compile=True,
            ),
            "torch.compile",
        ),
    ],
)
def test_minwm_rejects_unsupported_parallelism_combinations(args, message):
    with pytest.raises(ValueError, match=message):
        MinWMCausalDMDPipeline._validate_sequence_parallelism_args(args)


def test_minwm_sp_request_cannot_disable_sequence_sharding(monkeypatch):
    monkeypatch.setattr(
        MinWMSamplingParams.__mro__[1],
        "_adjust",
        lambda _self, _server_args: None,
    )
    params = MinWMSamplingParams(enable_sequence_shard=False)
    with pytest.raises(ValueError, match="requires enable_sequence_shard=True"):
        params._adjust(SimpleNamespace(sp_degree=2))


def test_minwm_sp_request_enables_sequence_sharding(monkeypatch):
    monkeypatch.setattr(
        MinWMSamplingParams.__mro__[1],
        "_adjust",
        lambda _self, _server_args: None,
    )
    params = MinWMSamplingParams(enable_sequence_shard=None)
    params._adjust(SimpleNamespace(sp_degree=4))
    assert params.enable_sequence_shard is True
    assert params.adjust_frames is False


def test_minwm_sequence_shard_frame_indices_support_mid_frame_boundaries(monkeypatch):
    import sglang.multimodal_gen.runtime.models.dits.minwm as minwm_module

    forward_batch = SimpleNamespace(
        enable_sequence_shard=True,
        sequence_shard_frame_indices=torch.tensor([1, 1, 2]),
    )
    monkeypatch.setattr(
        minwm_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch),
    )
    monkeypatch.setattr(minwm_module, "get_ulysses_parallel_world_size", lambda: 2)

    hidden_states = torch.zeros(1, 3, 8)
    assert _minwm_frame_indices(hidden_states, 4).tolist() == [1, 1, 2]
    assert _minwm_frame_indices(hidden_states, 1).tolist() == [0, 0, 0]


def test_minwm_sequence_shard_rope_uses_flattened_token_positions():
    class CaptureRotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.positions = None

        def forward_uncached(self, positions):
            self.positions = positions
            return positions.float(), -positions.float()

    model = MinWMCausalTransformer3DModel.__new__(MinWMCausalTransformer3DModel)
    torch.nn.Module.__init__(model)
    model._sequence_shard_rotary_emb = CaptureRotaryEmbedding()

    cos, sin = model._compute_sequence_shard_rope(
        local_seq_len=4,
        token_start=4,
        frame_stride=6,
        width=3,
        start_frame=7,
        device=torch.device("cpu"),
    )

    expected_positions = torch.tensor(
        [
            [7, 1, 1],
            [7, 1, 2],
            [8, 0, 0],
            [8, 0, 1],
        ]
    )
    torch.testing.assert_close(
        model._sequence_shard_rotary_emb.positions,
        expected_positions,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(cos, expected_positions.float(), rtol=0, atol=0)
    torch.testing.assert_close(sin, -expected_positions.float(), rtol=0, atol=0)


@pytest.mark.parametrize("rank", range(3))
def test_minwm_output_projection_restores_reference_row_bucket(rank):
    class CaptureProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None

        def forward(self, value):
            self.input = value
            return value * 2

    seq_splits = (3, 2, 2)
    projection = CaptureProjection()
    hidden_states = torch.arange(
        seq_splits[rank] * 2,
        dtype=torch.float32,
    ).reshape(1, seq_splits[rank], 2)

    output = _minwm_project_output_in_reference_row_bucket(
        projection,
        hidden_states,
        seq_splits,
        rank,
    )

    row_start = sum(seq_splits[:rank])
    expected_input = torch.zeros(1, sum(seq_splits), 2)
    expected_input[:, row_start : row_start + seq_splits[rank]] = hidden_states
    torch.testing.assert_close(projection.input, expected_input, rtol=0, atol=0)
    torch.testing.assert_close(output, hidden_states * 2, rtol=0, atol=0)


def test_minwm_output_projection_rejects_mismatched_shard():
    with pytest.raises(ValueError, match="does not match split"):
        _minwm_project_output_in_reference_row_bucket(
            torch.nn.Identity(),
            torch.zeros(1, 2, 4),
            (3, 2),
            0,
        )


def test_minwm_output_projection_matches_global_linear_for_nonuniform_sp8():
    torch.manual_seed(7)
    seq_splits = (3, 3, 2, 2, 2, 2, 2, 2)
    projection = torch.nn.Linear(4, 3, bias=True)
    global_hidden_states = torch.randn(2, sum(seq_splits), 4)
    expected = projection(global_hidden_states)

    local_outputs = []
    row_start = 0
    for rank, local_seq_len in enumerate(seq_splits):
        local_hidden_states = global_hidden_states.narrow(1, row_start, local_seq_len)
        local_outputs.append(
            _minwm_project_output_in_reference_row_bucket(
                projection,
                local_hidden_states,
                seq_splits,
                rank,
            )
        )
        row_start += local_seq_len

    actual = torch.cat(local_outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("is_cuda", "dtype", "splits", "capability", "hip", "expected"),
    [
        (True, torch.bfloat16, (1,) * 8, (9, 0), None, True),
        (True, torch.bfloat16, (3, 3, 2, 2, 2, 2, 2, 2), (9, 0), None, True),
        (True, torch.bfloat16, None, (9, 0), None, False),
        (True, torch.bfloat16, (4, 4), (9, 0), None, False),
        (True, torch.bfloat16, (2,) * 4, (9, 0), None, False),
        (True, torch.bfloat16, (1,) * 8, (10, 0), None, False),
        (True, torch.bfloat16, (1,) * 8, (9, 4), "6.3", False),
        (True, torch.float16, (1,) * 8, (9, 0), None, False),
        (False, torch.bfloat16, (1,) * 8, (9, 0), None, False),
    ],
)
def test_minwm_output_projection_reference_bucket_policy(
    monkeypatch,
    is_cuda,
    dtype,
    splits,
    capability,
    hip,
    expected,
):
    hidden_states = SimpleNamespace(
        is_cuda=is_cuda,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: capability,
    )

    assert (
        _minwm_should_restore_reference_output_projection(hidden_states, splits)
        is expected
    )


def test_minwm_causal_cache_uses_local_ulysses_heads(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm.minwm_causal_denoising as stage_module

    stage = MinWMCausalDMDDenoisingStage.__new__(MinWMCausalDMDDenoisingStage)
    stage.transformer = SimpleNamespace(
        num_attention_heads=24,
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                rope_position_mode="block_relative",
                rope_max_frame_gap=12,
                prompt_first_frame_pin_enabled=True,
                scene_cut_rope_offset=0,
                scene_cut_sink_enabled=False,
            )
        ),
    )
    stage._minwm_unbounded_cache = True
    monkeypatch.setattr(stage_module, "get_ulysses_parallel_world_size", lambda: 4)
    monkeypatch.setattr(stage_module, "get_ring_parallel_world_size", lambda: 1)

    assert stage._causal_sequence_shard_enabled(
        SimpleNamespace(enable_sequence_shard=True)
    )
    assert stage._num_causal_cache_attention_heads(sequence_shard_enabled=True) == 6
    assert stage._use_causal_cache_int_indices(sequence_shard_enabled=True)
    assert stage._causal_kv_cache_kwargs(
        SimpleNamespace(sequence_shard_enabled=True, expected_cache_tokens=123)
    ) == {
        "sequence_shard_enabled": True,
        "kv_cache_size": 123,
        "allow_growth": True,
        "rope_position_mode": "block_relative",
        "rope_max_frame_gap": 12,
        "prompt_first_frame_pin_enabled": True,
        "scene_cut_rope_offset": 0,
        "scene_cut_sink_enabled": False,
    }


@pytest.mark.parametrize("batch_size", [1, 2])
def test_minwm_ulysses_qkv_cpu_fallback_uses_reusable_peer_first_buffers(
    monkeypatch, batch_size
):
    import sglang.multimodal_gen.runtime.layers.usp as usp_module

    monkeypatch.setattr(usp_module, "get_ulysses_parallel_world_size", lambda: 2)
    query = torch.arange(8 * batch_size, dtype=torch.float32).reshape(
        batch_size, 2, 4, 1
    )
    assert not query.is_cuda
    key = query + 10
    value = query + 20
    send_buffer = torch.empty(3 * query.numel())
    receive_buffer = torch.empty_like(send_buffer)

    def fake_exchange(packed, output_buffer=None):
        assert packed.data_ptr() == send_buffer.data_ptr()
        assert packed.shape == (2, batch_size, 2, 2, 3)
        assert output_buffer is receive_buffer
        received = output_buffer.view_as(packed)
        # Simulate rank 0 receiving its head shard from two identical peers.
        received[0].copy_(packed[0])
        received[1].copy_(packed[0])
        return received

    monkeypatch.setattr(usp_module, "_usp_all_to_all_single", fake_exchange)
    output = usp_module._usp_input_all_to_all_qkv(
        query,
        key,
        value,
        input_buffer=send_buffer,
        output_buffer=receive_buffer,
    )

    packed_qkv = torch.cat((query, key, value), dim=-1)
    expected = torch.cat((packed_qkv[:, :, :2], packed_qkv[:, :, :2]), dim=1)
    assert torch.equal(output, expected)
    if batch_size == 1:
        assert output.data_ptr() == receive_buffer.data_ptr()


def test_minwm_ulysses_qkv_peer_first_layout_round_trips_exactly(monkeypatch):
    import sglang.multimodal_gen.runtime.layers.usp as usp_module

    world_size = 2
    batch_size = 2
    local_seq = 3
    global_heads = 4
    head_dim = 2
    local_heads = global_heads // world_size
    monkeypatch.setattr(
        usp_module, "get_ulysses_parallel_world_size", lambda: world_size
    )

    qkv_by_rank = []
    input_wire_by_rank = []
    for rank in range(world_size):
        query = (
            torch.arange(
                batch_size * local_seq * global_heads * head_dim,
                dtype=torch.float32,
            ).reshape(batch_size, local_seq, global_heads, head_dim)
            + rank * 1000
        )
        qkv = torch.cat((query, query + 100, query + 200), dim=-1)
        qkv_by_rank.append(qkv)
        input_wire_by_rank.append(
            qkv.unflatten(2, (world_size, local_heads))
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )

    active_rank = 0

    def fake_input_exchange(packed, output_buffer=None):
        assert output_buffer is None
        assert torch.equal(packed, input_wire_by_rank[active_rank])
        return torch.stack(
            [input_wire_by_rank[source][active_rank] for source in range(world_size)]
        )

    monkeypatch.setattr(usp_module, "_usp_all_to_all_single", fake_input_exchange)
    gathered_by_rank = []
    for active_rank in range(world_size):
        qkv = qkv_by_rank[active_rank]
        query, key, value = qkv.chunk(3, dim=-1)
        gathered = usp_module._usp_input_all_to_all_qkv(query, key, value)
        expected = torch.cat(
            [
                source[
                    :,
                    :,
                    active_rank * local_heads : (active_rank + 1) * local_heads,
                ]
                for source in qkv_by_rank
            ],
            dim=1,
        )
        assert gathered.shape == (
            batch_size,
            local_seq * world_size,
            local_heads,
            3 * head_dim,
        )
        assert torch.equal(gathered, expected)
        gathered_by_rank.append(gathered)

    output_wire_by_rank = [
        gathered.permute(1, 0, 2, 3).contiguous() for gathered in gathered_by_rank
    ]

    def fake_output_exchange(packed, output_buffer=None):
        assert output_buffer is None
        assert torch.equal(packed, output_wire_by_rank[active_rank])
        chunks = [
            output_wire_by_rank[source].flatten().chunk(world_size)[active_rank]
            for source in range(world_size)
        ]
        return torch.cat(chunks).reshape_as(packed)

    monkeypatch.setattr(usp_module, "_usp_all_to_all_single", fake_output_exchange)
    for active_rank in range(world_size):
        round_trip = usp_module._usp_output_all_to_all(
            gathered_by_rank[active_rank], head_dim=2
        )
        assert round_trip.shape == qkv_by_rank[active_rank].shape
        assert torch.equal(round_trip, qkv_by_rank[active_rank])


@pytest.mark.parametrize("seq_splits", [(2, 2), (3, 2)])
def test_minwm_causal_attention_packs_one_ulysses_collective(monkeypatch, seq_splits):
    import sglang.multimodal_gen.runtime.models.dits.minwm as minwm_module

    local_seq = seq_splits[0]
    global_seq = sum(seq_splits)
    local_heads = 2
    head_dim = 2
    forward_batch = SimpleNamespace(
        enable_sequence_shard=True,
        sequence_shard_splits=seq_splits,
    )
    monkeypatch.setattr(
        minwm_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch),
    )
    monkeypatch.setattr(minwm_module, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(minwm_module, "_MINWM_ATTENTION_IMPL", "packed")

    calls = {"input": 0, "output": 0}
    communication_buffers = {}

    def fake_input(query, key, value, *args, **kwargs):
        calls["input"] += 1
        assert query.shape == key.shape == value.shape
        assert query.shape == (1, local_seq, 4, head_dim)
        communication_buffers["qkv_send"] = kwargs.get("input_buffer")
        communication_buffers["qkv_recv"] = kwargs.get("output_buffer")
        return torch.cat(
            [
                torch.ones(1, global_seq, local_heads, head_dim),
                torch.full((1, global_seq, local_heads, head_dim), 2.0),
                torch.full((1, global_seq, local_heads, head_dim), 3.0),
            ],
            dim=-1,
        )

    def fake_output(output, *args, **kwargs):
        calls["output"] += 1
        assert output.shape == (1, global_seq, local_heads, head_dim)
        communication_buffers["attention_recv"] = kwargs.get("output_buffer")
        return torch.zeros(1, local_seq, 4, head_dim)

    if seq_splits[0] == seq_splits[1]:
        monkeypatch.setattr(minwm_module, "_usp_input_all_to_all_qkv", fake_input)
        monkeypatch.setattr(minwm_module, "_usp_output_all_to_all", fake_output)
    else:
        monkeypatch.setattr(
            minwm_module, "_usp_input_all_to_all_varlen_qkv", fake_input
        )
        monkeypatch.setattr(minwm_module, "_usp_output_all_to_all_varlen", fake_output)

    attention = MinWMCausalSelfAttention.__new__(MinWMCausalSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.head_start = 0
    attention.ulysses_workspace = minwm_module._MinWMUlyssesWorkspace()

    class IdentityRotary:
        @staticmethod
        def forward_uncached(position_ids):
            shape = (position_ids.shape[0], head_dim // 2)
            return torch.ones(shape), torch.zeros(shape)

    attention._minwm_rotary_emb = IdentityRotary()
    cache = MinWMCausalSelfAttentionKVCache(
        k=torch.zeros(1, global_seq, local_heads, head_dim),
        v=torch.zeros(1, global_seq, local_heads, head_dim),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        cache_size=global_seq,
        attention_window_size=global_seq,
    )
    cache.set_current_position_ids(
        torch.stack(
            [
                torch.arange(global_seq),
                torch.zeros(global_seq, dtype=torch.long),
                torch.zeros(global_seq, dtype=torch.long),
            ],
            dim=1,
        )
    )

    def fake_attention(query, key, value):
        assert torch.count_nonzero(query != 1).item() == 0
        assert torch.count_nonzero(key != 2).item() == 0
        assert torch.count_nonzero(value != 3).item() == 0
        return query

    monkeypatch.setattr(minwm_module, "_minwm_packed_varlen_attention", fake_attention)
    output = attention.forward(
        torch.ones(1, local_seq, 4, head_dim),
        torch.full((1, local_seq, 4, head_dim), 2.0),
        torch.full((1, local_seq, 4, head_dim), 3.0),
        (torch.empty(0), torch.empty(0)),
        block_mask=None,
        kv_cache=cache,
        current_start=17,
        qk_already_roped=False,
    )

    assert output.shape == (1, local_seq, 4, head_dim)
    assert calls == {"input": 1, "output": 1}
    if seq_splits[0] == seq_splits[1]:
        qkv_numel = 3 * local_seq * 4 * head_dim
        assert communication_buffers["qkv_send"].numel() == qkv_numel
        assert communication_buffers["qkv_recv"].numel() == qkv_numel
        assert communication_buffers["attention_recv"].numel() == (
            1 * global_seq * local_heads * head_dim
        )
        assert set(attention.ulysses_workspace._buffers) == {
            "qkv_send",
            "qkv_recv",
            "attention_recv",
        }
    else:
        assert communication_buffers == {
            "qkv_send": None,
            "qkv_recv": None,
            "attention_recv": None,
        }
        assert attention.ulysses_workspace._buffers == {}
    assert cache.k[:, :global_seq].shape == (
        1,
        global_seq,
        local_heads,
        head_dim,
    )
    assert cache.token_ids.tolist() == list(range(17, 17 + global_seq))


@pytest.mark.parametrize(
    ("capability", "available", "expected"),
    [
        (10, {"flash_attn.cute", "flash_attn"}, "fa4"),
        (9, {"flash_attn_interface", "flash_attn"}, "fa3"),
        (9, {"flash_attn"}, "fa2"),
    ],
)
def test_minwm_attention_backend_matches_source_device_fallback(
    monkeypatch, capability, available, expected
):
    import sglang.multimodal_gen.runtime.models.dits.minwm as minwm_module

    monkeypatch.setattr(
        minwm_module.torch.cuda,
        "get_device_capability",
        lambda _device: (capability, 0),
    )
    monkeypatch.setattr(
        minwm_module.importlib.util,
        "find_spec",
        lambda name: object() if name in available else None,
    )
    assert _minwm_packed_attention_backend(torch.device("cuda")) == expected


def test_minwm_allows_benchmark_component_ablation(monkeypatch):
    monkeypatch.setenv("MINWM_NATIVE_COMPONENTS", "")
    config = MinWMCausalDMDConfig()
    assert config.native_component_names == ()


def test_minwm_rms_norm_rounds_before_weight_multiply():
    layer = MinWMRMSNorm(8, eps=1e-6).to(torch.bfloat16)
    hidden = torch.tensor(
        [[[0.1, -0.2, 0.3, -0.4, 1.5, -2.0, 3.25, -4.5]]],
        dtype=torch.bfloat16,
    )
    layer.weight.data.copy_(
        torch.tensor([0.7, 0.9, 1.1, 1.3, 0.8, 1.2, 1.4, 0.6], dtype=torch.bfloat16)
    )
    normalized = hidden.float() * torch.rsqrt(
        hidden.float().pow(2).mean(dim=-1, keepdim=True) + layer.eps
    )
    expected = normalized.to(hidden.dtype) * layer.weight
    multiply_before_rounding = (normalized * layer.weight).to(hidden.dtype)
    torch.testing.assert_close(layer(hidden), expected, rtol=0, atol=0)
    assert not torch.equal(expected, multiply_before_rounding)


def test_minwm_patch_embed_uses_native_conv3d_path():
    torch.manual_seed(5)
    layer = MinWMPatchEmbed(
        patch_size=(1, 2, 2), in_chans=3, embed_dim=7, flatten=False
    )
    hidden = torch.randn(1, 3, 2, 6, 8)
    expected = layer.proj(hidden)
    torch.testing.assert_close(layer(hidden), expected, rtol=0, atol=0)


def test_minwm_adaln_uses_bf16_modulation_sum_and_fp32_layer_norm():
    hidden = torch.tensor(
        [
            [
                [0.25, -0.5, 1.0, -2.0],
                [0.5, -1.0, 2.0, -4.0],
                [0.75, -1.5, 3.0, -6.0],
                [1.0, -2.0, 4.0, -8.0],
            ]
        ],
        dtype=torch.bfloat16,
    )
    model_value = torch.tensor([[0.101, -0.202, 0.303, -0.404]], dtype=torch.float32)
    timestep_value = torch.tensor(
        [[[0.011, -0.022, 0.033, -0.044], [0.055, -0.066, 0.077, -0.088]]],
        dtype=torch.bfloat16,
    )
    actual_modulation = _frame_modulation(
        hidden, model_value, timestep_value, num_frames=2
    )
    expected_frame_values = model_value.to(torch.bfloat16) + timestep_value
    expected_modulation = (
        expected_frame_values.unsqueeze(2).expand(-1, -1, 2, -1).flatten(1, 2)
    )
    torch.testing.assert_close(actual_modulation, expected_modulation, rtol=0, atol=0)

    actual_gate = _frame_gate(hidden, model_value, timestep_value, num_frames=2)
    expected_gate = (
        (model_value.to(torch.bfloat16).float() + timestep_value.float())
        .unsqueeze(2)
        .expand(-1, -1, 2, -1)
        .flatten(1, 2)
    )
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0, atol=0)

    expected_norm = torch.nn.functional.layer_norm(
        hidden.float(), (hidden.shape[-1],), eps=1e-6
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        _minwm_layer_norm(hidden, eps=1e-6), expected_norm, rtol=0, atol=0
    )

    generator = torch.Generator().manual_seed(23)
    affine_hidden = torch.randn(2, 11, 32, generator=generator).to(torch.bfloat16)
    affine_weight = torch.randn(32, generator=generator)
    affine_bias = torch.randn(32, generator=generator)
    expected_affine_norm = torch.nn.functional.layer_norm(
        affine_hidden.float(),
        (affine_hidden.shape[-1],),
        affine_weight.to(affine_hidden.dtype).float(),
        affine_bias.to(affine_hidden.dtype).float(),
        1e-6,
    ).to(affine_hidden.dtype)
    actual_affine_norm = _minwm_layer_norm(
        affine_hidden,
        eps=1e-6,
        weight=affine_weight,
        bias=affine_bias,
    )
    torch.testing.assert_close(actual_affine_norm, expected_affine_norm, rtol=0, atol=0)
    fp32_parameter_norm = torch.nn.functional.layer_norm(
        affine_hidden.float(),
        (affine_hidden.shape[-1],),
        affine_weight,
        affine_bias,
        1e-6,
    ).to(affine_hidden.dtype)
    assert not torch.equal(actual_affine_norm, fp32_parameter_norm)

    expected_adaln = (
        torch.nn.functional.layer_norm(hidden.float(), (hidden.shape[-1],), eps=1e-6)
        * (1 + actual_modulation.float())
        + actual_modulation.float()
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        _minwm_adaln_modulation(
            hidden,
            actual_modulation,
            actual_modulation,
            eps=1e-6,
        ),
        expected_adaln,
        rtol=0,
        atol=0,
    )


def test_minwm_fused_segments_match_main_eager_formulas():
    torch.manual_seed(11)
    hidden = torch.randn(1, 6, 8, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    model = torch.randn(1, 8, dtype=torch.bfloat16)
    timestep = torch.randn_like(hidden)
    expected_residual = (
        hidden.float() + residual.float() * (model.float() + timestep.float())
    ).to(hidden.dtype)
    actual_residual = _minwm_adaln_op(
        hidden,
        y=residual,
        m_gate=model,
        e_gate=timestep,
    )
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)

    query = torch.randn(1, 6, 8, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    query_weight = torch.randn(8, dtype=torch.bfloat16)
    key_weight = torch.randn(8, dtype=torch.bfloat16)
    angles = torch.randn(6, 2, dtype=torch.float32)
    rope = torch.stack((angles.cos(), angles.sin()), dim=-1)
    actual_query, actual_key = _minwm_qk_norm_rope_op(
        query, key, query_weight, key_weight, 1e-6, rope, 2
    )
    raw_query, raw_key = _minwm_qk_norm_op(
        query, key, query_weight, key_weight, 1e-6, 2
    )
    separated_query = apply_minwm_rotary_embedding(
        raw_query, rope[..., 0], rope[..., 1]
    )
    separated_key = apply_minwm_rotary_embedding(raw_key, rope[..., 0], rope[..., 1])
    torch.testing.assert_close(separated_query, actual_query, rtol=0, atol=0)
    torch.testing.assert_close(separated_key, actual_key, rtol=0, atol=0)

    def expected(value, weight):
        value_float = value.float()
        value = (
            value_float
            * torch.rsqrt(value_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ).to(value.dtype) * weight
        value = value.reshape(1, 6, 2, 4)
        real, imaginary = value[..., 0::2].float(), value[..., 1::2].float()
        cos = rope[..., 0].view(1, 6, 1, 2)
        sin = rope[..., 1].view(1, 6, 1, 2)
        return (
            torch.stack(
                (real * cos - imaginary * sin, real * sin + imaginary * cos),
                dim=-1,
            )
            .flatten(-2)
            .to(value.dtype)
        )

    torch.testing.assert_close(
        actual_query, expected(query, query_weight), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_key, expected(key, key_weight), rtol=0, atol=0)


def test_minwm_cache_qk_norm_stays_eager(monkeypatch):
    import sglang.multimodal_gen.runtime.models.dits.minwm as minwm_module

    compile_calls = []

    def fake_get(_cls, operation, use_compile):
        compile_calls.append(use_compile)
        return operation

    monkeypatch.setattr(
        minwm_module._MinWMSegmentCompile,
        "get",
        classmethod(fake_get),
    )

    def operation(value):
        return value + 1

    value = torch.tensor(2)

    assert (
        _minwm_apply_qk_op(
            operation,
            [value],
            use_cache=True,
            use_compile=True,
        ).item()
        == 3
    )
    assert compile_calls == []
    assert (
        _minwm_apply_qk_op(
            operation,
            [value],
            use_cache=False,
            use_compile=True,
        ).item()
        == 3
    )
    assert compile_calls == [True]


def test_minwm_cuda_graph_disables_segment_compile(monkeypatch):
    import sglang.multimodal_gen.runtime.models.dits.minwm as minwm_module

    def operation(value):
        return value

    monkeypatch.setattr(minwm_module, "_MINWM_SEGMENT_COMPILE", True)
    monkeypatch.setattr(minwm_module, "_MINWM_CUDA_GRAPH_ACTIVE", False)
    monkeypatch.setattr(minwm_module._MinWMSegmentCompile, "_compiled", {})

    minwm_module.set_minwm_cuda_graph_active(True)

    assert minwm_module._MinWMSegmentCompile.get(operation, True) is operation
    assert minwm_module._MinWMSegmentCompile._compiled == {}


def test_minwm_rotary_embedding_matches_main_explicit_formula():
    torch.manual_seed(9)
    hidden = torch.randn(1, 5, 3, 8, dtype=torch.bfloat16)
    angles = torch.randn(5, 4, dtype=torch.float64)
    cos = angles.cos().float()
    sin = angles.sin().float()
    real = hidden[..., 0::2].float()
    imaginary = hidden[..., 1::2].float()
    expected = torch.stack(
        (
            real * cos.view(1, 5, 1, 4) - imaginary * sin.view(1, 5, 1, 4),
            real * sin.view(1, 5, 1, 4) + imaginary * cos.view(1, 5, 1, 4),
        ),
        dim=-1,
    ).flatten(-2)
    expected = expected.to(hidden.dtype)
    torch.testing.assert_close(
        apply_minwm_rotary_embedding(hidden, cos, sin), expected, rtol=0, atol=0
    )


def test_minwm_vae_pixel_and_latent_arithmetic_matches_main():
    config = MinWMCausalDMDConfig()
    assert config.preprocess_vae_encode_before_dtype_cast is True
    pixels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 16, 16)
    generic_normalized = pixels.float().div(255.0).mul(2.0).sub(1.0)
    actual_pixels = config.preprocess_vae_encode(generic_normalized, None)
    expected_pixels = pixels.to(torch.bfloat16).div(127.5).sub(1.0)
    torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)

    vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=[0.125], latents_std=[0.75])
    )
    posterior = torch.tensor([[[[[-1.5, 0.5]]]]], dtype=torch.bfloat16)
    normalized = config.normalize_vae_encode(posterior, vae)
    expected_normalized = (
        posterior - torch.tensor(0.125, dtype=torch.bfloat16)
    ) / torch.tensor(0.75, dtype=torch.bfloat16)
    expected_normalized = (
        expected_normalized.float().to(torch.float16).to(torch.bfloat16)
    )
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)
    decoded = config.preprocess_decoding(normalized, vae=vae)
    expected_decoded = normalized * torch.tensor(
        0.75, dtype=torch.bfloat16
    ) + torch.tensor(0.125, dtype=torch.bfloat16)
    torch.testing.assert_close(decoded, expected_decoded, rtol=0, atol=0)


def test_minwm_reference_latent_does_not_get_generic_i2v_mask_channels():
    config = MinWMCausalDMDConfig()
    latent = torch.randn(1, 48, 1, 30, 52)
    assert config.postprocess_image_latent(latent, None) is latent
    with pytest.raises(ValueError, match="48 channels"):
        config.postprocess_image_latent(torch.randn(1, 52, 1, 30, 52), None)
