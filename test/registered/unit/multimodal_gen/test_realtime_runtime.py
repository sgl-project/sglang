# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import numpy as np
import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    lingbot_world_realtime_adapter as lingbot_realtime,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
    ControlSignal,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSessionCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    _actions_to_c2ws,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world import (
    LingBotWorldCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime_vae import (
    RealtimeVAEDecodeState,
)
from sglang.multimodal_gen.runtime.models.dits.causal_attention_cache import (
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformer3DModel,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
    build_raw_rgb_frame_batches,
    build_delta_gzip_raw_rgb_payload,
    restore_delta_gzip_raw_rgb_payload,
)


class _Req(SimpleNamespace):
    realtime_session_id: str | None = None
    block_idx: int = 0
    session = None


class _State(BaseRealtimeState):
    def __init__(self):
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def test_realtime_session_cache_reuses_and_releases_state():
    cache = RealtimeSessionCache(max_sessions=1)
    first = _Req(realtime_session_id="session-a", block_idx=0, session=None)
    cache.attach(first)
    state = first.session.get_or_create_state(_State)

    second = _Req(realtime_session_id="session-a", block_idx=1, session=None)
    cache.attach(second)

    assert second.session is first.session
    assert second.session.get_state(_State) is state
    assert cache.release("session-a")
    assert state.disposed
    assert not cache.release("session-a")


def test_realtime_session_cache_rejects_missing_nonzero_chunk():
    cache = RealtimeSessionCache(max_sessions=1)
    req = _Req(realtime_session_id="missing", block_idx=1, session=None)

    try:
        cache.attach(req)
    except ValueError as exc:
        assert "Missing realtime session state" in str(exc)
    else:
        raise AssertionError("expected missing realtime session to fail")


def test_condition_event_queue_samples_chunk_and_repeats_last_item():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"], ["d"]]))

    chunk = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=4, default_item=[]),
    )

    assert chunk == [["w"], ["d"], ["d"], ["d"]]


def test_condition_event_contains_multiple_same_kind_control_signals():
    event = ConditionEvent(
        kind="camera_actions",
        payload=[
            ControlSignal(kind="camera_actions", payload=["w"]),
            ControlSignal(kind="camera_actions", payload=["d"]),
        ],
    )
    signals = list(event.iter_signals())

    assert [signal.kind for signal in signals] == [
        "camera_actions",
        "camera_actions",
    ]
    assert [signal.payload for signal in signals] == [["w"], ["d"]]


def test_condition_event_queue_samples_control_signal_payloads():
    queue = ConditionEventQueue()
    queue.push(
        ConditionEvent(
            kind="camera_actions",
            payload=[
                ControlSignal(kind="camera_actions", payload=["w"]),
                ControlSignal(kind="camera_actions", payload=["a"]),
                ControlSignal(kind="camera_actions", payload=["s"]),
            ],
        )
    )

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["a"]]
    assert second == [["s"], ["s"]]


def test_condition_event_queue_preserves_event_remainder_across_chunks():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"], ["a"], ["s"]]))

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["a"]]
    assert second == [["s"], ["s"]]


def test_condition_event_queue_does_not_persist_last_signal_across_empty_chunks():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"]]))

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["w"]]
    assert second == [[], []]


def test_condition_event_queue_replace_clears_pending_signals():
    queue = ConditionEventQueue()
    queue.push(
        ConditionEvent(kind="camera_actions", payload=[["w"], ["w"], ["w"], ["w"]])
    )

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    queue.replace(ConditionEvent(kind="camera_actions", payload=[["d"]]))
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=3, default_item=[]),
    )
    third = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=3, default_item=[]),
    )

    assert first == [["w"], ["w"]]
    assert second == [["d"], ["d"], ["d"]]
    assert third == [[], [], []]


def test_condition_event_queue_returns_none_without_default_item():
    queue = ConditionEventQueue()

    chunk = queue.sample_chunk("audio", ConditionSamplingParams(chunk_size=2))

    assert chunk is None


def test_condition_event_queue_empty_event_switches_to_default_item():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[]))

    chunk = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=3, default_item=[]),
    )

    assert chunk == [[], [], []]


def test_lingbot_realtime_state_uses_generic_condition_queue():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    assert state.sample_camera_actions(3) == [[], [], []]
    state.receive_camera_actions([["w"], ["a"], ["s"], ["d"]])
    assert state.sample_camera_actions(3) == [["w"], ["a"], ["s"]]
    assert state.sample_camera_actions(3) == [["d"], ["d"], ["d"]]

    state.receive_prompt("turn left")
    assert state.has_prompt()
    assert state.sample_prompt() == "turn left"
    assert not state.has_prompt()


def test_lingbot_realtime_camera_event_replaces_pending_script():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    state.receive_camera_actions([["w"], ["w"], ["w"], ["w"], ["w"], ["w"]])
    assert state.sample_camera_actions(3) == [["w"], ["w"], ["w"]]
    state.receive_camera_actions([["d"]], replace_pending=True)

    assert state.sample_camera_actions(3) == [["d"], ["d"], ["d"]]
    assert state.sample_camera_actions(3) == [[], [], []]


def test_lingbot_realtime_adapter_ingests_generic_events():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)

    camera_event = RealtimeEvent(
        type="event",
        kind="camera_actions",
        payload=[["w"], ["d"]],
        event_id=7,
    )
    prompt_event = RealtimeEvent(
        type="event",
        kind="prompt",
        payload="turn left",
        event_id=8,
    )

    assert (
        adapter.ingest_event(session, camera_event)
        == "kind=camera_actions, frames=2"
    )
    assert adapter.ingest_event(session, prompt_event) == "kind=prompt, prompt_len=9"
    state = adapter._state(session)
    assert state.sample_camera_actions(3) == [["w"], ["d"], ["d"]]
    assert state.sample_prompt() == "turn left"
    assert state.latest_event_id == 8


def test_generate_session_tracks_active_chunk_context():
    session = GenerateSession()

    chunk = session.new_chunk()

    assert chunk.session_id == session.id
    assert chunk.index == 0
    assert chunk.request_id.startswith(f"{session.id}_")
    try:
        session.new_chunk()
    except RuntimeError as exc:
        assert "previous realtime chunk" in str(exc)
    else:
        raise AssertionError("expected active chunk to block new chunk")

    session.generate_chunk_completed()
    next_chunk = session.new_chunk()

    assert next_chunk.index == 1
    assert next_chunk.request_id.startswith(f"{session.id}_")
    assert next_chunk.request_id != chunk.request_id


def test_generate_session_respects_max_chunks():
    session = GenerateSession()
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
            max_chunks=1,
        )
    )

    assert not session.reached_max_chunks()
    session.new_chunk()
    session.generate_chunk_completed()

    assert session.reached_max_chunks()


def test_generate_loop_overlaps_previous_send_with_next_generation(monkeypatch):
    events = []

    class _Adapter:
        def prepare_next_request(self, session, server_args, chunk):
            del session, server_args
            return SimpleNamespace(
                block_idx=chunk.index,
                request_id=chunk.request_id,
                condition_inputs={},
            )

        async def send_output(self, ws, session, result, batch):
            del ws, session, result
            events.append(f"send_start_{batch.block_idx}")
            await asyncio.sleep(0.05)
            events.append(f"send_end_{batch.block_idx}")
            return empty_frame_send_stats("test")

        def on_chunk_complete(self, session, result):
            del result
            session.generate_chunk_completed()

    async def fake_process_generation_batch(client, batch):
        del client
        events.append(f"generate_start_{batch.block_idx}")
        await asyncio.sleep(0.01)
        events.append(f"generate_end_{batch.block_idx}")
        return None, SimpleNamespace()

    monkeypatch.setattr(
        realtime_video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        realtime_video_api,
        "process_generation_batch",
        fake_process_generation_batch,
    )

    session = GenerateSession()
    session.adapter = _Adapter()
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
            max_chunks=2,
        )
    )

    asyncio.run(realtime_video_api._generate_loop(SimpleNamespace(), session))

    assert events.index("generate_start_1") < events.index("send_end_0")
    assert events[-1] == "send_end_1"


def test_lingbot_realtime_adapter_prepares_chunk_request(monkeypatch):
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
            num_frames=3,
            fps=24,
        )
    )
    state = adapter._state(session)
    state.receive_camera_actions([["w"], ["d"]], event_id=11)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(num_frames_per_block=3)
            )
        )
    )
    seen = {}

    def fake_build_sampling_params(request_id, **kwargs):
        seen["request_id"] = request_id
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            request_id=request_id,
            prompt=kwargs["prompt"],
            condition_inputs=kwargs["condition_inputs"],
            realtime_chunk_size=kwargs["realtime_chunk_size"],
        )

    def fake_prepare_backend_request(server_args, sampling_params):
        seen["server_args"] = server_args
        return SimpleNamespace(
            request_id=sampling_params.request_id,
            prompt=sampling_params.prompt,
            condition_inputs=dict(sampling_params.condition_inputs),
            realtime_chunk_size=sampling_params.realtime_chunk_size,
        )

    monkeypatch.setattr(
        lingbot_realtime,
        "build_sampling_params",
        fake_build_sampling_params,
    )
    monkeypatch.setattr(
        lingbot_realtime,
        "prepare_request",
        fake_prepare_backend_request,
    )
    chunk = session.new_chunk()

    batch = adapter.prepare_next_request(session, server_args, chunk)

    assert seen["request_id"] == chunk.request_id
    assert seen["kwargs"]["prompt"] == "walk forward"
    assert seen["kwargs"]["condition_inputs"] == {
        "camera_actions": [["w"], ["d"], ["d"]]
    }
    assert batch.request_id == chunk.request_id
    assert batch.condition_inputs == {"camera_actions": [["w"], ["d"], ["d"]]}
    assert batch.realtime_chunk_size == 3
    assert batch.session is session.realtime_session
    assert batch.realtime_session_id == session.id
    assert batch.block_idx == 0
    assert batch.return_raw_frames is True
    assert batch.realtime_event_id == 11


def test_lingbot_realtime_adapter_ingests_initial_condition_inputs():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="walk forward",
        condition_inputs={"camera_actions": [["w"], ["d"], ["a"], ["s"]]},
    )

    state = adapter._state(session)

    asyncio.run(adapter.on_init(session, request))

    assert state.sample_camera_actions(3) == [["w"], ["d"], ["a"]]
    assert state.sample_camera_actions(3) == [["s"], ["s"], ["s"]]


def test_lingbot_realtime_adapter_skips_stale_output_after_new_event():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    state = adapter._state(session)
    state.receive_camera_actions([["d"]], event_id=7)
    batch = SimpleNamespace(block_idx=3, realtime_event_id=6)
    result = OutputBatch(
        raw_frame_batches=[[b"stale"]],
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
    )

    stats = asyncio.run(adapter.send_output(SimpleNamespace(), session, result, batch))

    assert stats == empty_frame_send_stats("skipped-stale")


def test_lingbot_i2v_condition_does_not_repeat_reference_chunk():
    condition_full = torch.ones(1, 20, 3, 2, 2)

    first = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=0,
        chunk_size=3,
    )
    second = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=1,
        chunk_size=3,
    )

    assert torch.equal(first, condition_full)
    assert torch.count_nonzero(second) == 0
    assert second.shape == condition_full.shape


def test_lingbot_i2v_condition_pads_tail_without_reusing_reference():
    condition_full = torch.ones(1, 20, 1, 2, 2)

    first = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=0,
        chunk_size=3,
    )
    second = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=1,
        chunk_size=3,
    )

    assert first.shape == (1, 20, 3, 2, 2)
    assert torch.equal(first[:, :, :1], condition_full)
    assert torch.count_nonzero(first[:, :, 1:]) == 0
    assert second.shape == first.shape
    assert torch.count_nonzero(second) == 0


def test_lingbot_denoising_stage_does_not_own_realtime_cache_refs():
    stage = LingBotWorldCausalDMDDenoisingStage.__new__(
        LingBotWorldCausalDMDDenoisingStage
    )
    stage.causal_kv_cache = [object()]
    stage.crossattn_cache = [object()]

    stage._clear_stage_causal_cache_refs()

    assert stage.causal_kv_cache is None
    assert stage.crossattn_cache is None


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


def test_lingbot_condition_embedding_skips_text_when_crossattn_cache_ready():
    class _ConditionEmbedder:
        def __init__(self):
            self.full_calls = 0
            self.time_calls = 0

        def time_embedder(self, timestep):
            self.time_calls += 1
            return timestep.float().unsqueeze(-1)

        def time_modulation(self, temb):
            return temb + 1.0

        def __call__(self, timestep, encoder_hidden_states, encoder_hidden_states_image):
            self.full_calls += 1
            return timestep.float(), timestep.float(), encoder_hidden_states, None

    model = CausalLingBotWorldTransformer3DModel.__new__(
        CausalLingBotWorldTransformer3DModel
    )
    model.condition_embedder = _ConditionEmbedder()
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


def test_realtime_vae_decode_state_clears_model_cache_on_dispose():
    calls = []
    state = RealtimeVAEDecodeState()
    state.reset_causal_decode_state = lambda: calls.append("reset")

    state.dispose()

    assert calls == ["reset"]
    assert state.reset_causal_decode_state is None


def test_realtime_registry_resolves_lingbot_adapter():
    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
        LingBotWorldCausalDMDConfig,
    )

    server_args = SimpleNamespace(pipeline_config=LingBotWorldCausalDMDConfig())

    adapter = get_realtime_model_adapter(server_args)

    assert isinstance(adapter, lingbot_realtime.LingBotWorldRealtimeAdapter)


def test_sampling_params_apply_condition_inputs_to_req():
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        prompt="test",
        num_inference_steps=1,
        condition_inputs={"camera_actions": [["w"]]},
        realtime_chunk_size=3,
    )
    req = SimpleNamespace(extra={}, condition_inputs={}, realtime_chunk_size=None)

    sampling_params.apply_request_extra(req)

    assert req.condition_inputs == {"camera_actions": [["w"]]}
    assert req.realtime_chunk_size == 3


def test_realtime_chunk_latent_preparation_uses_chunk_spec():
    import torch
    from unittest.mock import patch

    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        RealtimeChunkLatentPreparationStage,
    )

    transformer = SimpleNamespace(
        config=SimpleNamespace(
            arch_config=SimpleNamespace(out_channels=16, num_frames_per_block=3)
        )
    )
    stage = RealtimeChunkLatentPreparationStage.__new__(
        RealtimeChunkLatentPreparationStage
    )
    stage.scheduler = SimpleNamespace(init_noise_sigma=10.0)
    stage.transformer = transformer
    batch = SimpleNamespace(
        batch_size=1,
        generator=None,
        height=None,
        width=None,
        image_latent=torch.zeros(2, 20, 6, 4, 5, dtype=torch.float32),
        latents=None,
        realtime_chunk_size=2,
    )

    def fake_randn_tensor(shape, generator, device, dtype):
        del generator
        return torch.ones(shape, device=device, dtype=dtype)

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation.get_local_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation.randn_tensor",
            side_effect=fake_randn_tensor,
        ),
    ):
        result = stage.forward(batch, SimpleNamespace())

    assert tuple(result.latents.shape) == (2, 16, 2, 4, 5)
    assert result.latents.dtype == torch.float32
    assert torch.all(result.latents == 1)
    assert result.raw_latent_shape == result.latents.shape


def test_lingbot_camera_actions_have_deterministic_pose_precision():
    poses = _actions_to_c2ws([["w"], ["d"]])

    np.testing.assert_allclose(poses[1][:3, 3], [0.0, 0.0, 0.05], atol=1e-6)
    np.testing.assert_allclose(poses[2][:3, 3], [0.05, 0.0, 0.05], atol=1e-6)

    yaw_pose = _actions_to_c2ws([["l"]])[1]
    expected_yaw = np.array(
        [
            [np.cos(np.deg2rad(6.0)), 0.0, np.sin(np.deg2rad(6.0))],
            [0.0, 1.0, 0.0],
            [-np.sin(np.deg2rad(6.0)), 0.0, np.cos(np.deg2rad(6.0))],
        ]
    )
    np.testing.assert_allclose(yaw_pose[:3, :3], expected_yaw, atol=1e-6)


def test_lingbot_camera_condition_uses_condition_inputs_without_session():
    import torch

    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
        LingBotWorldCausalDMDConfig,
    )

    batch = SimpleNamespace(
        c2ws_plucker_emb=None,
        condition_inputs={"camera_actions": [["w"]]},
        session=None,
        realtime_chunk_size=3,
        width=16,
        height=16,
        block_idx=0,
        realtime_session_id=None,
    )
    pipeline_config = LingBotWorldCausalDMDConfig()

    condition = pipeline_config.prepare_world_condition(
        batch=batch,
        device="cpu",
        dtype=torch.float32,
    )

    spatial_scale = pipeline_config.vae_config.arch_config.spatial_compression_ratio
    expected_shape = (
        1,
        6 * spatial_scale * spatial_scale,
        batch.realtime_chunk_size,
        batch.height // spatial_scale,
        batch.width // spatial_scale,
    )
    assert tuple(condition["c2ws_plucker_emb"].shape) == expected_shape
    assert condition["c2ws_plucker_emb"].dtype == torch.float32


def test_raw_rgb_frame_batches_preserve_frame_bytes_and_metadata():
    req = SimpleNamespace(
        request_id="req-1",
        block_idx=2,
        data_type="video",
        fps=24,
        output_compression=None,
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=1,
    )
    output_batch = OutputBatch(audio_sample_rate=None)
    grayscale = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    rgba = np.array(
        [
            [[5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20]],
        ],
        dtype=np.uint8,
    )

    def post_process_sample(*_args, **_kwargs):
        return [grayscale, rgba]

    frame_batches, metadata = build_raw_rgb_frame_batches(
        object(),
        req,
        output_batch,
        post_process_sample,
    )

    assert metadata == {
        "format": "rgb24",
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }
    assert len(frame_batches) == 1
    assert frame_batches[0][0] == bytes([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    assert frame_batches[0][1] == bytes(
        [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    )
    assert RAW_RGB_CONTENT_TYPE == "application/x-raw-rgb"


def test_raw_rgb_frame_batches_use_tensor_fast_path_without_postprocess():
    req = SimpleNamespace(
        request_id="req-1",
        block_idx=2,
        data_type="video",
        fps=24,
        output_compression=None,
        enable_frame_interpolation=False,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=None,
        enable_upscaling=False,
        upscaling_model_path=None,
        upscaling_scale=1,
    )
    output_batch = OutputBatch(audio_sample_rate=None)
    output = torch.tensor(
        [[[[[0.0]], [[0.25]]], [[[0.5]], [[0.75]]], [[[1.0]], [[1.0]]]]]
    )

    def post_process_sample(*_args, **_kwargs):
        raise AssertionError("tensor realtime output should not use postprocess")

    frame_batches, metadata = build_raw_rgb_frame_batches(
        output,
        req,
        output_batch,
        post_process_sample,
    )

    assert metadata == {
        "format": "rgb24",
        "width": 1,
        "height": 1,
        "channels": 3,
        "bytes_per_frame": 3,
    }
    assert frame_batches == [[bytes([0, 127, 255]), bytes([63, 191, 255])]]


def test_output_batch_uses_raw_frame_transport_names():
    output_batch = OutputBatch(
        raw_frame_batches=[[b"rgb"]],
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
        raw_frame_metadata={"format": "rgb24"},
    )

    assert output_batch.raw_frame_batches == [[b"rgb"]]
    assert output_batch.raw_frame_content_type == RAW_RGB_CONTENT_TYPE
    assert output_batch.raw_frame_metadata == {"format": "rgb24"}


def test_delta_gzip_raw_rgb_payload_roundtrips_exactly():
    frames = [
        bytes([1, 2, 3, 4, 5, 6]),
        bytes([1, 2, 4, 4, 6, 6]),
        bytes([2, 2, 4, 5, 6, 7]),
    ]

    payload = build_delta_gzip_raw_rgb_payload(frames)
    restored = restore_delta_gzip_raw_rgb_payload(
        payload,
        bytes_per_frame=6,
        num_frames=3,
    )

    assert restored == b"".join(frames)


def test_raw_rgb_realtime_output_adapter_uses_lossless_compressed_payload():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frame0 = bytes([1, 2, 3]) * 1000
        frame1 = bytes([1, 2, 4]) * 1000
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-1",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
        )
        result = OutputBatch(
            raw_frame_batches=[
                [frame0, frame1]
            ],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats, frame0 + frame1

    payloads, stats, expected_frames = asyncio.run(run())

    from msgpack import unpackb

    header = unpackb(payloads[0], raw=False)
    assert header["content_type"] == RAW_RGB_DELTA_GZIP_CONTENT_TYPE
    assert header["encoding"] == "delta-gzip"
    assert header["event_id"] == 3
    assert header["format"] == "rgb24"
    assert header["channels"] == 3
    assert header["bytes_per_frame"] == 3000
    assert header["raw_size"] == 6000
    assert header["total_size"] == len(payloads[1])
    assert stats["raw_bytes"] == 6000
    assert stats["ws_payload_bytes"] < 6000 + len(payloads[0])
    restored = restore_delta_gzip_raw_rgb_payload(
        payloads[1],
        bytes_per_frame=3000,
        num_frames=2,
    )
    assert restored == expected_frames


def test_raw_rgb_realtime_output_adapter_uses_previous_frame_reference():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        base_batch = SimpleNamespace(
            block_idx=0,
            request_id="req-1",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=9,
        )
        next_batch = SimpleNamespace(
            block_idx=1,
            request_id="req-2",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=9,
        )
        metadata = {
            "format": "rgb24",
            "width": 2,
            "height": 1,
            "channels": 3,
            "bytes_per_frame": 6,
        }
        first = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 3, 4, 5, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )
        second = OutputBatch(
            raw_frame_batches=[[bytes([1, 2, 4, 4, 6, 6])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata=metadata,
        )

        await adapter.send(ws, SimpleNamespace(), first, base_batch)
        await adapter.send(ws, SimpleNamespace(), second, next_batch)
        return ws.payloads

    payloads = asyncio.run(run())

    from msgpack import unpackb

    first_header = unpackb(payloads[0], raw=False)
    second_header = unpackb(payloads[2], raw=False)
    assert "delta_reference" not in first_header
    assert second_header["delta_reference"] == "previous-frame"
    first_frame = restore_delta_gzip_raw_rgb_payload(
        payloads[1],
        bytes_per_frame=6,
        num_frames=1,
    )
    second_frame = restore_delta_gzip_raw_rgb_payload(
        payloads[3],
        bytes_per_frame=6,
        num_frames=1,
        reference_frame=first_frame,
    )
    assert first_frame == bytes([1, 2, 3, 4, 5, 6])
    assert second_frame == bytes([1, 2, 4, 4, 6, 6])


def test_raw_rgb_realtime_output_adapter_splits_large_frame_batches():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        frames = [bytes([idx, idx + 1, idx + 2]) for idx in range(7)]
        batch = SimpleNamespace(
            block_idx=4,
            request_id="req-split",
            width=1,
            height=1,
            enable_upscaling=False,
            realtime_event_id=12,
        )
        result = OutputBatch(
            raw_frame_batches=[frames],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    from msgpack import unpackb

    headers = [unpackb(payloads[idx], raw=False) for idx in range(0, len(payloads), 2)]
    assert len(headers) == 3
    assert [header["chunk_index"] for header in headers] == [4, 4, 4]
    assert [header["frame_batch_index"] for header in headers] == [0, 1, 2]
    assert [header["num_frame_batches"] for header in headers] == [3, 3, 3]
    assert [header["num_frames"] for header in headers] == [3, 3, 1]
    assert [header["is_final_frame_batch"] for header in headers] == [
        False,
        False,
        True,
    ]
    assert "delta_reference" not in headers[0]
    assert headers[1]["delta_reference"] == "previous-frame"
    assert headers[2]["delta_reference"] == "previous-frame"
    assert stats["num_batches"] == 3
    assert stats["num_frames"] == 7


def test_raw_rgb_realtime_output_adapter_can_send_webp_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-webp",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="webp",
            output_compression=90,
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([255, 0, 0, 0, 255, 0])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    from msgpack import unpackb

    header = unpackb(payloads[0], raw=False)
    assert header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert header["format"] == "webp"
    assert header["encoding"] == "webp"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert payloads[1].startswith(b"RIFF")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_raw_rgb_realtime_output_adapter_can_send_jpeg_preview_frames():
    class _WebSocket:
        def __init__(self):
            self.payloads = []

        async def send_bytes(self, payload):
            self.payloads.append(payload)

    async def run():
        ws = _WebSocket()
        adapter = RawRGBRealtimeOutputAdapter()
        batch = SimpleNamespace(
            block_idx=0,
            request_id="req-jpeg",
            width=2,
            height=1,
            enable_upscaling=False,
            realtime_event_id=5,
            realtime_output_format="jpeg",
            output_compression=85,
        )
        result = OutputBatch(
            raw_frame_batches=[[bytes([255, 0, 0, 0, 255, 0])]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 2,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 6,
            },
        )

        stats = await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, stats

    payloads, stats = asyncio.run(run())

    from msgpack import unpackb

    header = unpackb(payloads[0], raw=False)
    assert header["content_type"] == JPEG_FRAME_CONTENT_TYPE
    assert header["format"] == "jpeg"
    assert header["encoding"] == "jpeg"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert payloads[1].startswith(b"\xff\xd8")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1
