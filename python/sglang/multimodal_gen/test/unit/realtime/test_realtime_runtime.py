# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from types import SimpleNamespace

import msgspec.msgpack
import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
    _actions_to_c2ws,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_adapter,
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    lingbot_world_realtime_adapter as lingbot_realtime,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    sana_wm_realtime_adapter as sana_wm_realtime,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world import (
    LingBotWorldCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.base import (
    RealtimeDiffusionStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.input_validation import (
    RealtimeInputValidationStage,
    RealtimeInputValidationState,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlStateTransition,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSessionCache,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCausalDecodeState,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
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


class _TestRealtimeDiffusionStage(RealtimeDiffusionStage):
    def forward(self, batch, component_manager=None):
        del batch, component_manager
        raise NotImplementedError


def test_realtime_diffusion_stage_declares_long_lived_components():
    stage = _TestRealtimeDiffusionStage()
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_precision="bf16", vae_precision="fp32")
    )

    uses = stage.component_uses(server_args, stage_name="realtime")

    assert [use.component_name for use in uses] == ["transformer", "vae"]
    assert [use.stage_name for use in uses] == ["realtime", "realtime"]
    assert uses[0].target_dtype == torch.bfloat16
    assert uses[0].memory_intensive
    assert uses[0].keep_ready_after_warmup
    assert uses[1].target_dtype == torch.float32
    assert not uses[1].memory_intensive
    assert uses[1].keep_ready_after_warmup


def test_realtime_diffusion_stage_requires_session():
    stage = _TestRealtimeDiffusionStage(default_height=480, default_width=832)
    req = _Req(session=None)

    try:
        stage.require_session(req, context="test realtime")
    except ValueError as exc:
        assert "test realtime requires a realtime session" in str(exc)
    else:
        raise AssertionError("expected missing realtime session to fail")

    session = object()
    req.session = session
    assert stage.require_session(req) is session


def test_realtime_causal_decode_state_dispose_resets_frontier():
    state = RealtimeCausalDecodeState()
    state.conv_cache = {"cache": object()}
    state.next_dec_idx = 7

    state.dispose()

    assert state.conv_cache is None
    assert state.next_dec_idx == 0


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


def test_lingbot_realtime_state_uses_control_script_and_prompt_queues():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    assert state.sample_camera_actions(3) == [[], [], []]
    state.receive_camera_action_script([["w"], ["a"], ["s"], ["d"]])
    assert state.sample_camera_actions(3) == [["w"], ["a"], ["s"]]
    assert state.sample_camera_actions(3) == [["d"], [], []]
    assert state.sample_camera_actions(3) == [[], [], []]

    state.receive_prompt("turn left")
    assert state.has_prompt()
    assert state.sample_prompt() == "turn left"
    assert not state.has_prompt()


def test_lingbot_realtime_camera_script_replaces_state_queue():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    state.receive_camera_state(["w"], event_id=7)
    state.receive_camera_action_script([["d"]], event_id=8)

    assert state.sample_camera_actions(3) == [["d"], [], []]
    assert state.latest_sampled_event_id == 8
    assert state.sample_camera_actions(3) == [[], [], []]


def test_lingbot_realtime_camera_events_preserve_short_presses():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    state.receive_camera_state(["w"], event_id=7)
    state.receive_camera_state([], event_id=8)

    assert state.sample_camera_actions(3) == [["w"], [], []]
    assert state.latest_sampled_event_id == 8
    assert state.sample_camera_actions(3) == [[], [], []]


def test_lingbot_realtime_camera_state_holds_until_release():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    state.receive_camera_state(["w"], event_id=7)

    assert state.sample_camera_actions(3) == [["w"], ["w"], ["w"]]
    assert state.latest_sampled_event_id == 7
    assert state.sample_camera_actions(3) == [["w"], ["w"], ["w"]]

    state.receive_camera_state([], event_id=8)

    assert state.sample_camera_actions(3) == [[], [], []]
    assert state.latest_sampled_event_id == 8


def test_lingbot_realtime_camera_state_compacts_multiple_pending_updates():
    state = lingbot_realtime.LingBotWorldRealtimeState()
    state.receive_camera_state_transitions(
        [
            ControlStateTransition(payload=["w"], seq_id=7),
            ControlStateTransition(payload=["w", "d"], seq_id=8),
            ControlStateTransition(payload=["d"], seq_id=9),
        ]
    )

    assert state.sample_camera_actions(3) == [["d"], ["d"], ["d"]]
    assert state.latest_sampled_event_id == 9


def test_sana_wm_realtime_camera_state_uses_sana_normalizer():
    state = sana_wm_realtime.SanaWMRealtimeAdapterState()
    result = state.receive_camera_control_event_payload(
        {
            "mode": "state",
            "transitions": [
                {"actions": ["W"], "client_ts_ms": 100},
                {"actions": [], "client_ts_ms": 120},
            ],
        },
        event_id=11,
    )

    assert result == "kind=camera_actions, mode=state, transitions=2"
    assert state.sample_camera_actions(10) == [["w"]] * 8 + [[], []]
    assert state.latest_sampled_event_id == 11


def test_sana_wm_realtime_adapter_preserves_requested_size():
    async def fake_save_image_to_path(image, target_path):
        return target_path

    old_save_image_to_path = realtime_adapter.save_image_to_path
    old_get_global_server_args = realtime_adapter.get_global_server_args
    realtime_adapter.save_image_to_path = fake_save_image_to_path
    realtime_adapter.get_global_server_args = lambda: SimpleNamespace(
        input_save_path=None
    )
    try:
        adapter = sana_wm_realtime.SanaWMRealtimeAdapter()
        session = GenerateSession()
        session.set_adapter(adapter)
        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
            first_frame=b"fake-image",
            size="832x480",
        )

        asyncio.run(adapter.on_init(session, request))

        assert request.size == "832x480"
    finally:
        realtime_adapter.save_image_to_path = old_save_image_to_path
        realtime_adapter.get_global_server_args = old_get_global_server_args


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
        == "kind=camera_actions, mode=script, frames=2"
    )
    assert adapter.ingest_event(session, prompt_event) == "kind=prompt, prompt_len=9"
    state = adapter._state(session)
    assert state.sample_camera_actions(3) == [["w"], ["d"], []]
    assert state.sample_prompt() == "turn left"
    assert state.latest_sampled_event_id == 8


def test_lingbot_realtime_adapter_ingests_composite_input_event():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
        )
    )

    composite_event = RealtimeEvent(
        type="event",
        kind="composite_input",
        payload={
            "input_types": ["prompt", "camera_actions"],
            "prompt": "turn left",
            "camera_actions": [["w"], ["d"]],
        },
        event_id=9,
    )

    event_log = adapter.ingest_event(session, composite_event)

    assert "kind=composite_input" in event_log
    chunk_inputs = adapter.sample_chunk_inputs(
        session,
        server_args=SimpleNamespace(),
        chunk=SimpleNamespace(index=1),
        chunk_size=3,
    )
    assert chunk_inputs.prompt == "turn left"
    assert chunk_inputs.condition_inputs[
        lingbot_realtime.LINGBOT_PROMPT_UPDATED_CONDITION
    ]
    assert chunk_inputs.condition_inputs[
        lingbot_realtime.LINGBOT_CAMERA_ACTIONS_CONDITION
    ] == [["w"], ["d"], []]
    assert adapter.get_realtime_event_id(session) == 9


def test_lingbot_realtime_adapter_rejects_composite_input_atomically():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
        )
    )

    composite_event = RealtimeEvent(
        type="event",
        kind="composite_input",
        payload={
            "input_types": ["prompt", "camera_actions"],
            "prompt": "turn left",
            "camera_actions": ["w"],
        },
        event_id=10,
    )

    with pytest.raises(ValueError, match="camera_actions"):
        adapter.ingest_event(session, composite_event)

    state = adapter._state(session)
    assert not state.has_prompt()
    assert state.sample_camera_actions(3) is None


def test_lingbot_realtime_prompt_event_marks_crossattn_reset():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
        )
    )
    state = adapter._state(session)
    state.receive_prompt("turn left", event_id=8)

    chunk_inputs = adapter.sample_chunk_inputs(
        session,
        server_args=SimpleNamespace(),
        chunk=SimpleNamespace(index=1),
        chunk_size=3,
    )

    assert chunk_inputs.prompt == "turn left"
    assert session.request.prompt == "turn left"
    assert chunk_inputs.condition_inputs[
        lingbot_realtime.LINGBOT_PROMPT_UPDATED_CONDITION
    ]


def test_lingbot_realtime_adapter_ingests_state_camera_events():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)

    camera_event = RealtimeEvent(
        type="event",
        kind="camera_actions",
        payload={
            "mode": "state",
            "transitions": [
                {"actions": ["w"], "client_ts_ms": 100},
                {"actions": [], "client_ts_ms": 120},
            ],
        },
        event_id=11,
    )

    assert (
        adapter.ingest_event(session, camera_event)
        == "kind=camera_actions, mode=state, transitions=2"
    )
    state = adapter._state(session)
    assert state.sample_camera_actions(3) == [["w"], [], []]
    assert state.latest_sampled_event_id == 11


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


def test_generate_loop_overlaps_send_with_next_generation(monkeypatch):
    events = []

    class _Adapter:
        async def wait_for_next_chunk(self, session):
            del session

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

    class _Ws:
        async def send_bytes(self, _message):
            pass

    asyncio.run(realtime_video_api._generate_loop(_Ws(), session))

    assert events.index("generate_start_1") < events.index("send_end_0")
    assert events.index("send_end_0") < events.index("send_start_1")
    assert events[-1] == "send_end_1"


def test_send_output_emits_chunk_stats_message():
    sent_messages = []

    class _Ws:
        async def send_bytes(self, message):
            sent_messages.append(message)

    class _Adapter:
        async def send_output(self, ws, session, result, batch):
            del ws, session, result, batch
            return {
                "header_pack_ms": 0.1,
                "header_write_ms": 0.2,
                "raw_payload_build_ms": 3.0,
                "raw_write_ms": 42.0,
                "ws_write_ms": 42.2,
                "raw_bytes": 1200,
                "ws_payload_bytes": 450,
                "num_frames": 3,
                "num_batches": 3,
                "frame_shape": (1, 2, 3),
                "content_type": "image/webp",
            }

    session = GenerateSession()
    session.adapter = _Adapter()
    chunk = session.new_chunk()
    batch = SimpleNamespace(
        block_idx=7,
        realtime_event_id=11,
        condition_inputs={"camera_actions": [["w"]]},
    )

    stats = asyncio.run(
        realtime_video_api._send_output_and_log(
            _Ws(),
            session,
            chunk,
            batch,
            SimpleNamespace(),
            request_prepare_ms=1.0,
            scheduler_forward_ms=2.0,
            chunk_started=time.perf_counter(),
        )
    )

    message = msgspec.msgpack.decode(sent_messages[-1])
    assert stats["raw_write_ms"] == 42.0
    assert message["type"] == "chunk_stats"
    assert message["chunk_index"] == 7
    assert message["event_id"] == 11
    assert message["raw_write_ms"] == 42
    assert message["ws_write_ms"] == 42
    assert message["ws_payload_bytes"] == 450
    assert message["content_type"] == "image/webp"
    assert bytes([0xCB]) not in sent_messages[-1]


def test_listen_generate_request_propagates_disconnect_without_error_write():
    sent_messages = []

    class _Ws:
        async def receive_bytes(self):
            raise realtime_video_api.WebSocketDisconnect(1000, "client close")

        async def send_bytes(self, message):
            sent_messages.append(message)

    try:
        asyncio.run(
            realtime_video_api._listen_generate_request(_Ws(), GenerateSession())
        )
    except realtime_video_api.WebSocketDisconnect:
        pass
    else:
        raise AssertionError("expected websocket disconnect to propagate")

    assert sent_messages == []


def test_wait_for_active_session_slot_observes_release(monkeypatch):
    async def run():
        realtime_video_api._ACTIVE_SESSION_IDS.clear()
        realtime_video_api._ACTIVE_SESSION_IDS.add("old-session")

        async def fake_sleep(_seconds):
            realtime_video_api._ACTIVE_SESSION_IDS.clear()

        monkeypatch.setattr(realtime_video_api.asyncio, "sleep", fake_sleep)
        try:
            return await realtime_video_api._wait_for_active_session_slot(
                timeout_s=1.0,
                interval_s=0.1,
            )
        finally:
            realtime_video_api._ACTIVE_SESSION_IDS.clear()

    assert asyncio.run(run())


def test_cleanup_realtime_session_keeps_active_slot_during_scheduler_release(
    monkeypatch,
):
    async def run():
        session = GenerateSession()
        realtime_video_api._ACTIVE_SESSION_IDS.clear()
        realtime_video_api._ACTIVE_SESSION_IDS.add(session.id)
        release_seen = []

        async def fake_forward(req):
            release_seen.append(
                (req.session_id, session.id in realtime_video_api._ACTIVE_SESSION_IDS)
            )

        monkeypatch.setattr(
            realtime_video_api.async_scheduler_client,
            "forward",
            fake_forward,
        )
        try:
            await realtime_video_api._cleanup_realtime_session(session, None, None)
            still_active_after_cleanup = (
                session.id in realtime_video_api._ACTIVE_SESSION_IDS
            )
        finally:
            realtime_video_api._ACTIVE_SESSION_IDS.clear()

        return session.id, release_seen, still_active_after_cleanup

    session_id, release_seen, still_active_after_cleanup = asyncio.run(run())

    assert release_seen == [(session_id, True)]
    assert still_active_after_cleanup


def test_lingbot_realtime_adapter_prepares_chunk_request(monkeypatch):
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    session.set_request(
        RealtimeVideoGenerationsRequest(
            type="init",
            prompt="walk forward",
            num_frames=9,
            fps=24,
            realtime_causal_sink_size=3,
            realtime_causal_kv_cache_num_frames=45,
        )
    )
    state = adapter._state(session)
    state.receive_camera_state(["w"], event_id=11)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(num_frames_per_block=3)
            ),
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(temporal_compression_ratio=4)
            ),
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
            session=None,
        )

    monkeypatch.setattr(
        realtime_adapter,
        "build_sampling_params",
        fake_build_sampling_params,
    )
    monkeypatch.setattr(
        realtime_adapter,
        "prepare_request",
        fake_prepare_backend_request,
    )
    chunk = session.new_chunk()

    batch = adapter.prepare_next_request(session, server_args, chunk)

    assert seen["request_id"] == chunk.request_id
    assert seen["kwargs"]["prompt"] == "walk forward"
    assert seen["kwargs"]["num_frames"] == 21
    assert seen["kwargs"]["num_inference_steps"] == 4
    assert seen["kwargs"]["guidance_scale"] == 1.0
    assert seen["kwargs"]["condition_inputs"] == {
        "camera_actions": [["w"], ["w"], ["w"]]
    }
    assert batch.request_id == chunk.request_id
    assert batch.condition_inputs == {"camera_actions": [["w"], ["w"], ["w"]]}
    assert batch.realtime_chunk_size == 3
    assert batch.session is None
    assert batch.realtime_session_id == session.id
    assert batch.block_idx == 0
    assert batch.return_raw_frames is True
    assert batch.realtime_event_id == 11
    assert batch.realtime_causal_sink_size == 3
    assert batch.realtime_causal_kv_cache_num_frames == 45


def test_lingbot_realtime_condition_horizon_repeats_blank_tail_chunk():
    config = LingBotWorldCausalDMDConfig()
    chunk_size = config.dit_config.arch_config.num_frames_per_block
    latent_channels = config.vae_config.arch_config.z_dim
    temporal_ratio = config.vae_config.arch_config.temporal_compression_ratio
    spatial_ratio = config.vae_config.arch_config.spatial_compression_ratio
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="walk forward",
        num_frames=9,
    )
    server_args = SimpleNamespace(pipeline_config=config)

    num_frames = lingbot_realtime.LingBotWorldRealtimeAdapter._condition_num_frames(
        request=request,
        server_args=server_args,
        chunk_size=chunk_size,
    )
    latent_frames = (num_frames - 1) // temporal_ratio + 1
    raw_request_latent_frames = (request.num_frames - 1) // temporal_ratio + 1

    assert raw_request_latent_frames == chunk_size
    assert latent_frames == chunk_size * 2

    latent_condition = torch.ones(1, latent_channels, latent_frames, 2, 2)
    batch = SimpleNamespace(
        height=2 * spatial_ratio,
        num_frames=num_frames,
        width=2 * spatial_ratio,
    )
    condition_full = config.postprocess_image_latent(latent_condition, batch)
    first_chunk = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=0,
        chunk_size=chunk_size,
    )
    tail_chunk = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=1,
        chunk_size=chunk_size,
    )
    repeated_tail = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=2,
        chunk_size=chunk_size,
    )

    assert torch.count_nonzero(first_chunk[:, :temporal_ratio]) > 0
    assert torch.count_nonzero(tail_chunk[:, :temporal_ratio]) == 0
    assert not torch.equal(tail_chunk, first_chunk)
    assert torch.equal(repeated_tail, tail_chunk)

    long_request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="walk forward",
        num_frames=45,
    )
    assert (
        lingbot_realtime.LingBotWorldRealtimeAdapter._condition_num_frames(
            request=long_request,
            server_args=server_args,
            chunk_size=chunk_size,
        )
        == 45
    )


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
    assert state.sample_camera_actions(3) == [["s"], [], []]
    assert state.sample_camera_actions(3) == [[], [], []]


def test_realtime_video_request_accepts_raw_lossless_output_format():
    request = RealtimeVideoGenerationsRequest(
        type="init",
        prompt="walk forward",
        realtime_output_format="raw",
        realtime_causal_sink_size=3,
        realtime_causal_kv_cache_num_frames=45,
    )

    assert request.realtime_output_format == "raw"
    assert request.realtime_causal_sink_size == 3
    assert request.realtime_causal_kv_cache_num_frames == 45


def test_lingbot_realtime_adapter_does_not_wait_for_idle_chunks():
    async def run():
        adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
        session = GenerateSession()
        session.set_adapter(adapter)
        await adapter.wait_for_next_chunk(session)

        session.generate_chunk_cnt = 1
        await asyncio.wait_for(adapter.wait_for_next_chunk(session), timeout=1)

    asyncio.run(run())


def test_lingbot_realtime_adapter_sends_stale_output_for_client_cutover():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)
    state = adapter._state(session)
    state.receive_camera_action_script([["d"]], event_id=7)
    calls = []

    async def fake_send(ws, session_arg, result_arg, batch_arg):
        calls.append((ws, session_arg, result_arg, batch_arg))
        return empty_frame_send_stats("sent")

    adapter.output_adapter = SimpleNamespace(send=fake_send)
    batch = SimpleNamespace(block_idx=3, realtime_event_id=6)
    result = OutputBatch(
        raw_frame_batches=[[b"stale"]],
        raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
    )

    stats = asyncio.run(adapter.send_output(SimpleNamespace(), session, result, batch))

    assert stats == empty_frame_send_stats("sent")
    assert calls[0][1] is session
    assert calls[0][2] is result
    assert calls[0][3] is batch


def test_lingbot_i2v_condition_repeats_last_chunk():
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
    assert second.shape == condition_full.shape
    assert torch.equal(second, condition_full)


def test_lingbot_i2v_condition_pads_tail_then_repeats_it():
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
    assert torch.equal(second, first)


def test_lingbot_i2v_condition_uses_available_non_initial_chunks():
    first_chunk = torch.ones(1, 20, 3, 2, 2)
    second_chunk = first_chunk * 2
    condition_full = torch.cat([first_chunk, second_chunk], dim=2)

    second = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=1,
        chunk_size=3,
    )
    third = LingBotWorldCausalDMDDenoisingStage._select_i2v_condition_chunk(
        condition_full,
        chunk_idx=2,
        chunk_size=3,
    )

    assert torch.equal(second, second_chunk)
    assert third.shape == second_chunk.shape
    assert torch.equal(third, second_chunk)


def test_realtime_input_validation_reuses_generator_across_chunks():
    stage = RealtimeInputValidationStage.__new__(RealtimeInputValidationStage)
    state = RealtimeInputValidationState()
    generator = torch.Generator(device="cpu").manual_seed(123)

    first = SimpleNamespace(
        block_idx=0,
        generator=generator,
        seeds=None,
        seed=123,
        generator_device="cpu",
        num_outputs_per_prompt=1,
    )
    second = SimpleNamespace(
        block_idx=1,
        generator=torch.Generator(device="cpu").manual_seed(123),
        seeds=None,
        seed=123,
        generator_device="cpu",
        num_outputs_per_prompt=1,
    )

    stage._cache_generator(first, state)
    stage._reuse_or_cache_generator(second, state)

    assert second.generator is generator


def test_realtime_registry_resolves_lingbot_adapter():
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
    from unittest.mock import patch

    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
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
