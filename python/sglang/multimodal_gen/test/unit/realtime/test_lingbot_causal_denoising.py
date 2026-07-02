# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
)
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.models.dits import (
    lingbot_world as lingbot_world_module,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformer3DModel,
    CausalLingBotWorldTransformerBlock,
    LingBotWorldCamConditioner,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world import (
    LingBotWorldCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.realtime.states import RealtimeCausalDiTState

LINGBOT_INTERACTIVE_KV_WINDOW_ENV = "SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW"


def test_lingbot_denoising_stage_does_not_own_realtime_cache_refs():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.causal_kv_cache = [object()]
    stage.crossattn_cache = [object()]

    stage._clear_stage_causal_cache_refs()

    assert stage.causal_kv_cache is None
    assert stage.crossattn_cache is None


def test_lingbot_realtime_attention_cache_uses_bounded_sink_window():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.sliding_window_num_frames = 18
    stage.num_frames_per_block = 3
    stage.num_token_per_frame = 10

    assert stage._get_causal_sink_tokens() == 9 * 10
    assert stage._get_causal_kv_cache_size(sequence_shard_enabled=False) == 18 * 10
    assert stage._get_causal_kv_cache_size(sequence_shard_enabled=True) == 18 * 10


def test_lingbot_realtime_cache_config_overrides_checkpoint_defaults():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.sliding_window_num_frames = 18
    stage.num_token_per_frame = 10

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=3,
            realtime_causal_kv_cache_num_frames=45,
        )
    )
    stage._apply_causal_cache_overrides(SimpleNamespace(), server_args)

    assert stage._get_causal_sink_tokens() == 3 * 10
    assert stage._get_causal_kv_cache_size(sequence_shard_enabled=False) == 45 * 10


def test_lingbot_realtime_cache_config_uses_request_overrides():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.sliding_window_num_frames = 18
    stage.num_token_per_frame = 10

    batch = SimpleNamespace(
        realtime_causal_sink_size=4,
        realtime_causal_kv_cache_num_frames=12,
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=3,
            realtime_causal_kv_cache_num_frames=45,
        )
    )
    stage._apply_causal_cache_overrides(batch, server_args)

    assert stage._get_causal_sink_tokens() == 4 * 10
    assert stage._get_causal_kv_cache_size(sequence_shard_enabled=False) == 12 * 10


def test_lingbot_realtime_attention_cache_rolls_with_sink_window():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.num_transformer_blocks = 1
    stage.local_attn_size = -1
    stage.sink_size = 2
    stage.num_token_per_frame = 1
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 6
    stage.transformer = SimpleNamespace(
        num_attention_heads=1,
        attention_head_dim=1,
        config=SimpleNamespace(
            arch_config=SimpleNamespace(
                sink_size=2,
            )
        ),
    )

    stage._initialize_kv_cache(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert stage.causal_kv_cache is not None
    cache = stage.causal_kv_cache[0]
    assert cache.allow_growth is False
    assert cache.cache_size == 6
    assert cache.sink_tokens == 2

    cache.update_and_get_attention_kv(
        key=torch.ones(1, 3, 1, 1),
        value=torch.ones(1, 3, 1, 1),
        current_chunk_start=0,
    )

    assert cache.local_end_index_int == 3
    assert cache.global_end_index_int == 3

    cache.update_and_get_attention_kv(
        key=torch.full((1, 3, 1, 1), 2.0),
        value=torch.full((1, 3, 1, 1), 2.0),
        current_chunk_start=3,
    )
    third_view = cache.update_and_get_attention_kv(
        key=torch.full((1, 3, 1, 1), 3.0),
        value=torch.full((1, 3, 1, 1), 3.0),
        current_chunk_start=6,
    )

    assert cache.cache_size == 6
    assert cache.local_end_index_int == 6
    assert cache.global_end_index_int == 9
    assert torch.equal(cache.k[:, :2], torch.ones(1, 2, 1, 1))
    assert torch.equal(cache.k[:, 2:3], torch.full((1, 1, 1, 1), 2.0))
    assert torch.equal(cache.k[:, 3:6], torch.full((1, 3, 1, 1), 3.0))
    assert third_view.k.flatten().tolist() == [
        1.0,
        1.0,
        2.0,
        3.0,
        3.0,
        3.0,
    ]

    fourth_view = cache.update_and_get_attention_kv(
        key=torch.full((1, 3, 1, 1), 4.0),
        value=torch.full((1, 3, 1, 1), 4.0),
        current_chunk_start=9,
    )

    assert cache.local_end_index_int == 6
    assert cache.global_end_index_int == 12
    assert torch.equal(cache.k[:, :2], torch.ones(1, 2, 1, 1))
    assert torch.equal(cache.k[:, 2:3], torch.full((1, 1, 1, 1), 3.0))
    assert torch.equal(cache.k[:, 3:6], torch.full((1, 3, 1, 1), 4.0))
    assert fourth_view.k.flatten().tolist() == [
        1.0,
        1.0,
        3.0,
        4.0,
        4.0,
        4.0,
    ]


def test_lingbot_realtime_attention_cache_samples_sink_and_recent_window():
    cache = CausalSelfAttentionKVCache(
        k=torch.zeros(1, 8, 1, 1),
        v=torch.zeros(1, 8, 1, 1),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        global_end_index_int=0,
        local_end_index_int=0,
        cache_size=8,
        sink_tokens=2,
        attention_window_size=8,
    )

    cache.update_and_get_attention_kv(
        key=torch.ones(1, 3, 1, 1),
        value=torch.ones(1, 3, 1, 1),
        current_chunk_start=0,
    )
    cache.update_and_get_attention_kv(
        key=torch.full((1, 3, 1, 1), 2.0),
        value=torch.full((1, 3, 1, 1), 2.0),
        current_chunk_start=3,
    )
    sampled_view = cache.update_and_get_attention_kv(
        key=torch.full((1, 3, 1, 1), 3.0),
        value=torch.full((1, 3, 1, 1), 3.0),
        current_chunk_start=6,
        recent_window_tokens=1,
    )

    assert sampled_view.k.flatten().tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    assert sampled_view.v.flatten().tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]


def test_lingbot_interactive_kv_window_config_default_disabled():
    field = LingBotWorldCausalDMDConfig.__dataclass_fields__[
        "interactive_kv_window_enable"
    ]

    assert field.default is False


def test_lingbot_interactive_kv_window_samples_base_moving_and_still(monkeypatch):
    monkeypatch.delenv(LINGBOT_INTERACTIVE_KV_WINDOW_ENV, raising=False)
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.num_token_per_frame = 10
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 18
    stage.transformer = SimpleNamespace(num_attention_heads=1)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            interactive_kv_window_enable=True,
            interactive_kv_moving_window=12,
            interactive_kv_still_window=3,
            interactive_kv_still_chunks=2,
        )
    )
    cache_state = RealtimeCausalDiTState()

    batch = SimpleNamespace(condition_inputs={})
    previous = stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    assert previous is None
    assert stage.sliding_window_num_frames == 24
    assert batch.realtime_causal_kv_sample_tokens == 120
    policy = stage._build_realtime_causal_cache_policy(batch, server_args)
    assert policy.expected_cache_tokens == 240

    batch.condition_inputs = {"camera_actions": [["w"], [], []]}
    cache_state.chunk_idx = 0
    stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    assert batch.realtime_causal_kv_sample_tokens == 120

    batch.condition_inputs = {"camera_actions": [[], [], []]}
    cache_state.chunk_idx = 1
    stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    assert batch.realtime_causal_kv_sample_tokens == 120

    cache_state.chunk_idx = 2
    stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    assert batch.realtime_causal_kv_sample_tokens == 30


def test_lingbot_interactive_kv_window_none_uses_base_moving_window(monkeypatch):
    monkeypatch.delenv(LINGBOT_INTERACTIVE_KV_WINDOW_ENV, raising=False)
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.num_token_per_frame = 10
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 18
    stage.transformer = SimpleNamespace(num_attention_heads=1)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=9,
            realtime_causal_kv_cache_num_frames=18,
            interactive_kv_window_enable=True,
            interactive_kv_moving_window=None,
            interactive_kv_still_window=3,
            interactive_kv_still_chunks=2,
        )
    )
    cache_state = RealtimeCausalDiTState()
    batch = SimpleNamespace(condition_inputs={"camera_actions": [["w"], [], []]})

    stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    assert batch.realtime_causal_kv_sample_tokens == 60
    policy = stage._build_realtime_causal_cache_policy(batch, server_args)
    assert policy.expected_cache_tokens == 180


def test_lingbot_interactive_kv_window_updates_total_window_for_moving_default(
    monkeypatch,
):
    monkeypatch.delenv(LINGBOT_INTERACTIVE_KV_WINDOW_ENV, raising=False)
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.num_token_per_frame = 10
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 18
    stage.transformer = SimpleNamespace(num_attention_heads=1)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=9,
            realtime_causal_kv_cache_num_frames=18,
            interactive_kv_window_enable=True,
            interactive_kv_moving_window=12,
            interactive_kv_still_window=3,
            interactive_kv_still_chunks=2,
        )
    )
    batch = SimpleNamespace(condition_inputs={"camera_actions": [["w"], [], []]})

    policy = stage._build_realtime_causal_cache_policy(batch, server_args)

    assert stage.sliding_window_num_frames == 24
    assert policy.expected_cache_tokens == 240


def test_lingbot_interactive_kv_window_default_disabled(monkeypatch):
    monkeypatch.delenv(LINGBOT_INTERACTIVE_KV_WINDOW_ENV, raising=False)
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.num_token_per_frame = 10
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 18
    stage.transformer = SimpleNamespace(num_attention_heads=1)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=9,
            realtime_causal_kv_cache_num_frames=18,
            interactive_kv_window_enable=False,
            interactive_kv_moving_window=12,
            interactive_kv_still_window=3,
            interactive_kv_still_chunks=2,
        )
    )
    cache_state = RealtimeCausalDiTState()
    batch = SimpleNamespace(condition_inputs={"camera_actions": [["w"], [], []]})

    stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    policy = stage._build_realtime_causal_cache_policy(batch, server_args)

    assert stage.sliding_window_num_frames == 18
    assert batch.realtime_causal_kv_sample_tokens is None
    assert policy.expected_cache_tokens == 180


def test_lingbot_interactive_kv_window_env_can_enable_default(monkeypatch):
    monkeypatch.setenv(LINGBOT_INTERACTIVE_KV_WINDOW_ENV, "1")
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.num_token_per_frame = 10
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 18
    stage.transformer = SimpleNamespace(num_attention_heads=1)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            realtime_causal_sink_size=9,
            realtime_causal_kv_cache_num_frames=18,
            interactive_kv_window_enable=False,
            interactive_kv_moving_window=12,
            interactive_kv_still_window=3,
            interactive_kv_still_chunks=2,
        )
    )
    cache_state = RealtimeCausalDiTState()
    batch = SimpleNamespace(condition_inputs={"camera_actions": [["w"], [], []]})

    stage._set_lingbot_kv_sample_tokens(cache_state, batch, server_args)
    policy = stage._build_realtime_causal_cache_policy(batch, server_args)

    assert stage.sliding_window_num_frames == 24
    assert batch.realtime_causal_kv_sample_tokens == 120
    assert policy.expected_cache_tokens == 240


def test_lingbot_interactive_kv_window_allocates_expected_cache_size():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.num_transformer_blocks = 1
    stage.local_attn_size = -1
    stage.sink_size = 9
    stage.num_token_per_frame = 10
    stage.num_frames_per_block = 3
    stage.sliding_window_num_frames = 18
    stage.transformer = SimpleNamespace(num_attention_heads=1, attention_head_dim=1)
    policy = CausalDMDCachePolicy(
        sequence_shard_enabled=False,
        num_attention_heads=1,
        expected_cache_tokens=240,
        expected_sink_tokens=90,
        kv_cache_kwargs={},
    )

    stage._initialize_kv_cache(
        batch_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        **stage._causal_kv_cache_kwargs(policy),
    )

    assert stage.causal_kv_cache is not None
    cache = stage.causal_kv_cache[0]
    assert cache.cache_size == 240
    assert cache.k.shape[1] == 240
    assert cache.sink_tokens == 90
    assert cache.attention_window_size == 240


def test_lingbot_dynamic_condition_cache_clear_removes_chunk_entries():
    cache_state = SimpleNamespace(
        runtime_cache={
            "lingbot_c2ws_plucker_emb": object(),
            "lingbot_cam_conditioner": object(),
            "lingbot_rope": object(),
        }
    )

    LingBotWorldCausalDMDDenoisingStage._clear_lingbot_dynamic_condition_cache(
        cache_state
    )

    assert "lingbot_c2ws_plucker_emb" not in cache_state.runtime_cache
    assert "lingbot_cam_conditioner" not in cache_state.runtime_cache
    assert "lingbot_rope" in cache_state.runtime_cache


def test_lingbot_i2v_model_input_writer_reuses_buffer():
    latents = torch.ones(1, 16, 3, 2, 2)
    condition = torch.full((1, 20, 3, 2, 2), 2.0)

    write = LingBotWorldCausalDMDDenoisingStage._build_i2v_model_input_writer(
        latents=latents,
        condition=condition,
        target_dtype=torch.float32,
        device=latents.device,
    )
    first = write(latents)
    first_ptr = first.data_ptr()
    second = write(latents + 3.0)

    assert first_ptr == second.data_ptr()
    assert second.shape == (1, 36, 3, 2, 2)
    assert torch.equal(second[:, :16], latents + 3.0)
    assert torch.equal(second[:, 16:], condition)


def test_lingbot_cam_conditioner_scale_shift_matches_forward():
    hidden_states = torch.randn(2, 3, 4)
    c2ws_plucker_emb = torch.randn(2, 3, 4)
    conditioner = SimpleNamespace(
        compute_scale_shift=lambda c2ws: (c2ws * 0.25, c2ws - 0.5)
    )

    expected = LingBotWorldCamConditioner.forward(
        conditioner,
        hidden_states,
        c2ws_plucker_emb,
    )
    actual = LingBotWorldCamConditioner.forward(
        conditioner,
        hidden_states,
        c2ws_plucker_emb,
        conditioner.compute_scale_shift(c2ws_plucker_emb),
    )

    assert torch.allclose(actual, expected)


def test_lingbot_cam_conditioner_cache_reuses_source_tensor(monkeypatch):
    class _CamConditioner:
        def __init__(self):
            self.calls = 0

        def compute_scale_shift(self, c2ws_plucker_emb):
            self.calls += 1
            return c2ws_plucker_emb + 1, c2ws_plucker_emb + 2

    block = CausalLingBotWorldTransformerBlock.__new__(
        CausalLingBotWorldTransformerBlock
    )
    block.cam_conditioner = _CamConditioner()
    forward_batch = SimpleNamespace(extra={}, enable_sequence_shard=True)
    monkeypatch.setattr(
        lingbot_world_module, "get_ulysses_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        lingbot_world_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch, current_timestep=7),
    )

    c2ws_plucker_emb = torch.ones(1, 2, 3)
    first = block._cam_conditioner_scale_shift(c2ws_plucker_emb)
    second = block._cam_conditioner_scale_shift(c2ws_plucker_emb)
    next_source = c2ws_plucker_emb.clone()
    third = block._cam_conditioner_scale_shift(next_source)

    assert first is second
    assert third is not first
    assert block.cam_conditioner.calls == 2
    cache = forward_batch.extra["lingbot_cam_conditioner"]
    assert cache["source_key"][0] == next_source.data_ptr()
    assert len(cache["entries"]) == 1


def test_lingbot_cam_conditioner_cache_skips_non_sequence_shard(monkeypatch):
    class _CamConditioner:
        def __init__(self):
            self.calls = 0

        def compute_scale_shift(self, c2ws_plucker_emb):
            self.calls += 1
            return c2ws_plucker_emb + 1, c2ws_plucker_emb + 2

    block = CausalLingBotWorldTransformerBlock.__new__(
        CausalLingBotWorldTransformerBlock
    )
    block.cam_conditioner = _CamConditioner()
    forward_batch = SimpleNamespace(extra={}, enable_sequence_shard=False)
    monkeypatch.setattr(
        lingbot_world_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch, current_timestep=7),
    )

    c2ws_plucker_emb = torch.ones(1, 2, 3)
    first = block._cam_conditioner_scale_shift(c2ws_plucker_emb)
    second = block._cam_conditioner_scale_shift(c2ws_plucker_emb)

    assert first is not second
    assert first[0] is not second[0]
    assert block.cam_conditioner.calls == 2
    assert "lingbot_cam_conditioner" not in forward_batch.extra


def test_lingbot_cam_conditioner_cache_skips_single_ulysses_world(monkeypatch):
    class _CamConditioner:
        def __init__(self):
            self.calls = 0

        def compute_scale_shift(self, c2ws_plucker_emb):
            self.calls += 1
            return c2ws_plucker_emb + 1, c2ws_plucker_emb + 2

    block = CausalLingBotWorldTransformerBlock.__new__(
        CausalLingBotWorldTransformerBlock
    )
    block.cam_conditioner = _CamConditioner()
    forward_batch = SimpleNamespace(extra={}, enable_sequence_shard=True)
    monkeypatch.setattr(
        lingbot_world_module, "get_ulysses_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        lingbot_world_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch, current_timestep=7),
    )

    c2ws_plucker_emb = torch.ones(1, 2, 3)
    first = block._cam_conditioner_scale_shift(c2ws_plucker_emb)
    second = block._cam_conditioner_scale_shift(c2ws_plucker_emb)

    assert first is not second
    assert block.cam_conditioner.calls == 2
    assert "lingbot_cam_conditioner" not in forward_batch.extra


def test_lingbot_cam_conditioner_cache_reuses_context_update(monkeypatch):
    class _CamConditioner:
        def __init__(self):
            self.calls = 0

        def compute_scale_shift(self, c2ws_plucker_emb):
            self.calls += 1
            return c2ws_plucker_emb + 1, c2ws_plucker_emb + 2

    block = CausalLingBotWorldTransformerBlock.__new__(
        CausalLingBotWorldTransformerBlock
    )
    block.cam_conditioner = _CamConditioner()
    forward_batch = SimpleNamespace(extra={}, enable_sequence_shard=True)
    monkeypatch.setattr(
        lingbot_world_module, "get_ulysses_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        lingbot_world_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch, current_timestep=-1),
    )

    c2ws_plucker_emb = torch.ones(1, 2, 3)
    first = block._cam_conditioner_scale_shift(c2ws_plucker_emb)
    second = block._cam_conditioner_scale_shift(c2ws_plucker_emb)

    assert first is second
    assert block.cam_conditioner.calls == 1
    assert "lingbot_cam_conditioner" in forward_batch.extra


def test_lingbot_model_prepares_cam_conditioner_scale_shifts(monkeypatch):
    class _CamConditioner:
        def __init__(self, offset):
            self.offset = offset
            self.calls = 0

        def compute_scale_shift(self, c2ws_plucker_emb):
            self.calls += 1
            return (
                c2ws_plucker_emb + self.offset,
                c2ws_plucker_emb + self.offset + 10,
            )

    model = CausalLingBotWorldTransformer3DModel.__new__(
        CausalLingBotWorldTransformer3DModel
    )
    model.blocks = [
        SimpleNamespace(cam_conditioner=_CamConditioner(1)),
        SimpleNamespace(cam_conditioner=_CamConditioner(2)),
    ]
    forward_batch = SimpleNamespace(extra={}, enable_sequence_shard=True)
    monkeypatch.setattr(
        lingbot_world_module, "get_ulysses_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        lingbot_world_module,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch, current_timestep=7),
    )

    c2ws_plucker_emb = torch.ones(1, 2, 3)
    first = model._prepare_cam_conditioner_scale_shifts(c2ws_plucker_emb, forward_batch)
    second = model._prepare_cam_conditioner_scale_shifts(
        c2ws_plucker_emb, forward_batch
    )
    next_source = c2ws_plucker_emb.clone()
    third = model._prepare_cam_conditioner_scale_shifts(next_source, forward_batch)

    assert first is not None
    assert second is not None
    assert third is not None
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert third[0] is not first[0]
    assert [block.cam_conditioner.calls for block in model.blocks] == [2, 2]


def test_lingbot_condition_embedding_skips_text_when_crossattn_cache_ready(monkeypatch):
    class _ConditionEmbedder:
        def __init__(self):
            self.full_calls = 0
            self.time_calls = 0

        def time_embedder(self, timestep):
            self.time_calls += 1
            return timestep.float().unsqueeze(-1)

        def time_modulation(self, temb):
            return temb + 1.0

        def __call__(
            self, timestep, encoder_hidden_states, encoder_hidden_states_image
        ):
            self.full_calls += 1
            return timestep.float(), timestep.float(), encoder_hidden_states, None

    model = CausalLingBotWorldTransformer3DModel.__new__(
        CausalLingBotWorldTransformer3DModel
    )
    model.condition_embedder = _ConditionEmbedder()
    monkeypatch.setattr(
        lingbot_world_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            forward_batch=SimpleNamespace(extra={}),
            current_timestep=7,
        ),
    )
    crossattn_cache = [
        CrossAttentionKVCache(
            k=torch.empty(1, 1, 1, 1),
            v=torch.empty(1, 1, 1, 1),
            is_init=True,
        )
    ]

    temb, timestep_proj, _, image_states = model._prepare_condition_embeddings(
        timestep=torch.tensor([[7]]),
        encoder_hidden_states=torch.ones(1, 2, 3),
        encoder_hidden_states_image=torch.ones(1, 2, 3),
        crossattn_cache=crossattn_cache,
    )

    assert model.condition_embedder.time_calls == 1
    assert model.condition_embedder.full_calls == 0
    assert torch.equal(temb, torch.tensor([[7.0]]))
    assert torch.equal(timestep_proj, torch.tensor([[8.0]]))
    assert image_states is None


def test_lingbot_context_cache_update_skips_unused_projection(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world import (
        lingbot_world_causal_denoising as lingbot_denoising,
    )

    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    calls = []

    class _Transformer:
        def __call__(self, latent_input, prompt_embeds, timestep, **kwargs):
            calls.append((latent_input, prompt_embeds, timestep, kwargs))
            return latent_input

    stage.transformer = _Transformer()
    monkeypatch.setattr(
        lingbot_denoising,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )

    context_input = torch.ones(2, 3, 4, 5, 6)
    batch = SimpleNamespace()
    stage._update_causal_context_cache(
        batch,
        SimpleNamespace(pipeline_config=SimpleNamespace(context_noise=7)),
        context_input=context_input,
        prompt_embeds="prompt",
        kv_cache="kv",
        crossattn_cache="cross",
        current_start_tokens=12,
        start_frame=4,
        image_kwargs={"encoder_hidden_states_image": "image"},
        pos_cond_kwargs={"c2ws_plucker_emb": "pose"},
        attn_metadata="metadata",
        target_dtype=torch.float32,
        autocast_enabled=False,
    )

    assert len(calls) == 1
    latent_input, prompt_embeds, timestep, kwargs = calls[0]
    assert latent_input is context_input
    assert prompt_embeds == "prompt"
    assert timestep.shape == (2, 1)
    assert timestep.dtype == torch.long
    assert timestep.tolist() == [[7], [7]]
    assert kwargs["kv_cache"] == "kv"
    assert kwargs["crossattn_cache"] == "cross"
    assert kwargs["current_start"] == 12
    assert kwargs["start_frame"] == 4
    assert kwargs["encoder_hidden_states_image"] == "image"
    assert kwargs["c2ws_plucker_emb"] == "pose"
    assert kwargs["skip_final_projection"] is True
