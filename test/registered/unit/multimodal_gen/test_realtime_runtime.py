# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from pathlib import Path
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
    realtime_output_adapter,
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.realtime.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
    ControlSignal,
    ControlStateSamplingQueue,
    ControlStateTransition,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSessionCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
    _actions_to_c2ws,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world import (
    LingBotWorldCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime_input_validation import (
    RealtimeInputValidationStage,
    RealtimeInputValidationState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime_vae import (
    CausalVaeDecodingStage,
    RealtimeVAEDecodeState,
)
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
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


def _unpack_frame_batch_messages(payloads):
    from msgpack import unpackb

    messages = []
    for payload in payloads:
        message = unpackb(payload, raw=False)
        assert message.pop("type") == "frame_batch"
        frame_payload = message.pop("payload")
        messages.append((message, frame_payload))
    return messages


def test_realtime_webui_presets_do_not_emit_camera_scripts():
    repo_root = Path(__file__).parents[4]
    app_js = (
        repo_root
        / "python/sglang/multimodal_gen/apps/realtime_webui/app.js"
    ).read_text()
    index_html = (
        repo_root
        / "python/sglang/multimodal_gen/apps/realtime_webui/index.html"
    ).read_text()
    styles_css = (
        repo_root
        / "python/sglang/multimodal_gen/apps/realtime_webui/styles.css"
    ).read_text()

    assert "preset.actions" not in app_js
    assert "repeatActions" not in app_js
    assert 'id="eventFrames"' not in index_html
    assert "ControlStateController" in app_js
    assert 'const DEFAULT_PREVIEW_OUTPUT_FORMAT = "webp";' in app_js
    assert 'id="transportFormat"' in index_html
    assert (
        'id="serverUrl" value="ws://127.0.0.1:30000/v1/realtime_video/generate"'
        in index_html
    )
    assert '<option value="webp" selected>WebP preview</option>' in index_html
    assert 'id="serverSendText"' in index_html
    assert 'id="theoreticalFpsText"' in index_html
    assert 'id="renderFps"' in index_html
    assert 'id="stageRenderFps"' not in index_html
    assert "sglang-diffusion Realtime Studio" in index_html
    assert "SGLD" not in index_html
    assert 'class="tabs"' not in index_html
    assert "Recordings" not in index_html
    assert "API" not in index_html
    assert "Info" not in index_html
    assert 'id="steps" type="number" value="4"' in index_html
    assert 'id="guidance" type="number" value="1"' in index_html
    assert "styles.css?v=realtime-fixes-v25" in index_html
    assert "app.js?v=realtime-fixes-v25" in index_html
    assert 'const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v6";' in app_js
    assert 'const REACTOR_PRESET_BASE_URL = "https://www.reactor.inc/lingbot-world-fast-v1";' in app_js
    assert "Dragon Dolly" in app_js
    assert "no creature morphing" in app_js
    assert app_js.index("Dragon Ride") < app_js.index("Dragon Dolly")
    assert app_js.index("Dragon Dolly") < app_js.index("Kid A")
    assert "dragon-ride.jpg" in app_js
    assert "stageRenderFps" not in app_js
    assert 'setStatus("Receiving"' not in app_js
    assert "decodeChain = decodeChain" in app_js
    assert "receiveChain" not in app_js
    assert 'message.type === "chunk_stats"' in app_js
    assert "chunkTotal > 0 ? numFrames / chunkTotal" in app_js
    assert "encodedImageElementFallback" in app_js
    assert "handleEncodedPreviewDecodeError" in app_js
    launch_server_py = (
        repo_root
        / "python/sglang/multimodal_gen/runtime/launch_server.py"
    ).read_text()
    assert "ws_per_message_deflate=False" in launch_server_py
    assert "#statusText" in styles_css
    assert "min-width: 92px" in styles_css
    assert "if (b === 0xca)" in app_js
    assert "if (b === 0xcb)" in app_js
    assert "if (b === 0xc4)" in app_js
    assert 'message.type === "frame_batch"' in app_js
    assert "RAW_RGB_FRAMES_PER_WS_MESSAGE = 1" in (
        repo_root
        / "python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_output_adapter.py"
    ).read_text()
    assert ".stage-controls .camera-pad button.is-key-active:not(:disabled)" in styles_css
    assert "stage-telemetry" in index_html
    assert "max-height: min(54vh, 560px);" in styles_css
    assert ".stage-telemetry" in styles_css
    assert "--pressed: #8c9288;" in styles_css
    assert "--pressed-ring: rgba(238, 241, 236, 0.2);" in styles_css


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


def test_condition_event_queue_can_repeat_last_signal_across_empty_chunks():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"]]))

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    queue.push(ConditionEvent(kind="camera_actions", payload=[[]]))
    third = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )

    assert first == [["w"], ["w"]]
    assert second == [["w"], ["w"]]
    assert third == [[], []]


def test_condition_event_queue_tracks_sampled_signal_seq_id():
    queue = ConditionEventQueue()
    queue.push(
        ConditionEvent(
            kind="camera_actions",
            payload=[
                ControlSignal(kind="camera_actions", payload=["w"], seq_id=7),
                ControlSignal(kind="camera_actions", payload=[], seq_id=8),
            ],
        )
    )

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=1,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    first_seq_id = queue.last_sampled_seq_id("camera_actions")
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=1,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    second_seq_id = queue.last_sampled_seq_id("camera_actions")

    assert first == [["w"]]
    assert first_seq_id == 7
    assert second == [[]]
    assert second_seq_id == 8


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


def test_control_state_sampling_queue_preserves_short_pulse():
    queue = ControlStateSamplingQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))
    queue.push(ControlStateTransition(payload=[], seq_id=8))

    chunk = queue.sample_chunk(3)

    assert chunk == [["w"], [], []]
    assert queue.latest_sampled_seq_id() == 8
    assert queue.sample_chunk(3) == [[], [], []]


def test_control_state_sampling_queue_holds_current_state_without_backlog():
    queue = ControlStateSamplingQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))

    assert queue.sample_chunk(3) == [["w"], ["w"], ["w"]]
    assert queue.latest_sampled_seq_id() == 7
    assert queue.sample_chunk(3) == [["w"], ["w"], ["w"]]
    assert queue.latest_sampled_seq_id() == 7


def test_control_state_sampling_queue_compacts_many_transitions():
    queue = ControlStateSamplingQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))
    queue.push(ControlStateTransition(payload=["w", "d"], seq_id=8))
    queue.push(ControlStateTransition(payload=["d"], seq_id=9))

    chunk = queue.sample_chunk(3)

    assert chunk == [["d"], ["d"], ["d"]]
    assert queue.latest_sampled_seq_id() == 9


def test_lingbot_realtime_state_uses_generic_condition_queue():
    state = lingbot_realtime.LingBotWorldRealtimeState()

    assert state.sample_camera_actions(3) == [[], [], []]
    state.receive_camera_actions([["w"], ["a"], ["s"], ["d"]])
    assert state.sample_camera_actions(3) == [["w"], ["a"], ["s"]]
    assert state.sample_camera_actions(3) == [["d"], [], []]
    assert state.sample_camera_actions(3) == [[], [], []]

    state.receive_prompt("turn left")
    assert state.has_prompt()
    assert state.sample_prompt() == "turn left"
    assert not state.has_prompt()


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

    from msgpack import unpackb

    message = unpackb(sent_messages[-1], raw=False)
    assert stats["raw_write_ms"] == 42.0
    assert message["type"] == "chunk_stats"
    assert message["chunk_index"] == 7
    assert message["event_id"] == 11
    assert message["raw_write_ms"] == 42.0
    assert message["ws_write_ms"] == 42.2
    assert message["ws_payload_bytes"] == 450
    assert message["content_type"] == "image/webp"


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
    assert seen["kwargs"]["num_frames"] == 21
    assert seen["kwargs"]["num_inference_steps"] == 4
    assert seen["kwargs"]["guidance_scale"] == 1.0
    assert seen["kwargs"]["condition_inputs"] == {
        "camera_actions": [["w"], ["w"], ["w"]]
    }
    assert batch.request_id == chunk.request_id
    assert batch.condition_inputs == {"camera_actions": [["w"], ["w"], ["w"]]}
    assert batch.realtime_chunk_size == 3
    assert batch.session is session.realtime_session
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
    state.receive_camera_actions([["d"]], event_id=7)
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
    assert (
        stage._get_lingbot_causal_kv_cache_size(sequence_shard_enabled=False)
        == 18 * 10
    )
    assert (
        stage._get_lingbot_causal_kv_cache_size(sequence_shard_enabled=True)
        == 18 * 10
    )


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
    stage._apply_realtime_causal_cache_config(SimpleNamespace(), server_args)

    assert stage._get_causal_sink_tokens() == 3 * 10
    assert stage._get_lingbot_causal_kv_cache_size(
        sequence_shard_enabled=False
    ) == 45 * 10


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
    stage._apply_realtime_causal_cache_config(batch, server_args)

    assert stage._get_causal_sink_tokens() == 4 * 10
    assert stage._get_lingbot_causal_kv_cache_size(
        sequence_shard_enabled=False
    ) == 12 * 10


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


def test_causal_vae_decoding_stage_keeps_wan_decoder_cache(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import realtime_vae

    class _WanVAE:
        def __init__(self):
            self.config = SimpleNamespace(patch_size=None)
            self.clear_calls = 0
            self.decoder_first_chunk_flags = []
            self._feat_map = []
            self._conv_idx = [0]

        def to(self, device=None, dtype=None):
            del device, dtype
            return self

        def clear_cache(self):
            self.clear_calls += 1
            self._feat_map = [None]
            self._conv_idx = [0]

        def post_quant_conv(self, latents):
            return latents

        def decoder(self, x, *, feat_cache, feat_idx, first_chunk=False):
            self.decoder_first_chunk_flags.append(first_chunk)
            if feat_cache[0] is None:
                feat_cache[0] = x.detach().clone()
            else:
                feat_cache[0] = torch.cat([feat_cache[0], x.detach().clone()], dim=2)
            feat_idx[0] += 1
            return x

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_tiling = False

        def get_decode_scale_and_shift(self, device, dtype, vae):
            del device, dtype, vae
            return 1.0, None

        def preprocess_decoding(self, latents, server_args, vae=None):
            del server_args, vae
            return latents

        def post_decoding(self, frames, server_args):
            del server_args
            return frames

    monkeypatch.setattr(
        realtime_vae,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    vae = _WanVAE()
    vae.clear_cache()
    vae.clear_calls = 0
    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    stage.vae = vae
    server_args = SimpleNamespace(
        pipeline_config=_PipelineConfig(),
        disable_autocast=True,
    )

    first = stage.decode_causal(
        torch.zeros(1, 1, 2, 1, 1),
        server_args,
        first_chunk=True,
    )
    second = stage.decode_causal(
        torch.ones(1, 1, 1, 1, 1),
        server_args,
        first_chunk=False,
    )

    assert tuple(first.shape) == (1, 1, 2, 1, 1)
    assert tuple(second.shape) == (1, 1, 1, 1, 1)
    assert vae.clear_calls == 0
    assert vae.decoder_first_chunk_flags == [True, False, False]
    assert tuple(vae._feat_map[0].shape) == (1, 1, 3, 1, 1)


def test_causal_vae_decoding_stage_prefers_native_causal_decode(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import realtime_vae

    class _NativeCausalVAE:
        def __init__(self):
            self.config = SimpleNamespace(patch_size=None)
            self.calls = []
            self._feat_map = [None]
            self._conv_idx = [0]

        def to(self, device=None, dtype=None):
            del device, dtype
            return self

        def clear_cache(self):
            self.calls.append("clear_cache")

        def reset_causal_decode_state(self):
            self.calls.append("reset")

        def post_quant_conv(self, latents):
            self.calls.append("post_quant_conv")
            return latents

        def decoder(self, x, *, feat_cache, feat_idx, first_chunk=False):
            del x, feat_cache, feat_idx, first_chunk
            self.calls.append("decoder")

        def causal_decode(self, latents):
            self.calls.append("causal_decode")
            return latents

    class _PipelineConfig:
        vae_precision = "fp32"
        vae_tiling = False

        def get_decode_scale_and_shift(self, device, dtype, vae):
            del device, dtype, vae
            return 1.0, None

        def preprocess_decoding(self, latents, server_args, vae=None):
            del server_args, vae
            return latents

    monkeypatch.setattr(
        realtime_vae,
        "get_local_torch_device",
        lambda: torch.device("cpu"),
    )

    vae = _NativeCausalVAE()
    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    stage.vae = vae
    server_args = SimpleNamespace(
        pipeline_config=_PipelineConfig(),
        disable_autocast=True,
    )

    frames = stage.decode_causal(
        torch.zeros(1, 1, 1, 1, 1),
        server_args,
        first_chunk=True,
    )

    assert tuple(frames.shape) == (1, 1, 1, 1, 1)
    assert vae.calls == ["causal_decode"]


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

    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["content_type"] == RAW_RGB_DELTA_GZIP_CONTENT_TYPE
    assert first_header["encoding"] == "delta-gzip"
    assert first_header["event_id"] == 3
    assert first_header["format"] == "rgb24"
    assert first_header["channels"] == 3
    assert first_header["bytes_per_frame"] == 3000
    assert first_header["raw_size"] == 3000
    assert first_header["total_size"] == len(first_payload)
    assert first_header["num_frames"] == 1
    assert first_header["num_frame_batches"] == 2
    assert first_header["frame_batch_index"] == 0
    assert "delta_reference" not in first_header
    assert second_header["raw_size"] == 3000
    assert second_header["num_frames"] == 1
    assert second_header["num_frame_batches"] == 2
    assert second_header["frame_batch_index"] == 1
    assert second_header["delta_reference"] == "previous-frame"
    assert stats["raw_bytes"] == 6000
    assert stats["num_batches"] == 2
    assert stats["num_frames"] == 2
    first_frame = restore_delta_gzip_raw_rgb_payload(
        first_payload,
        bytes_per_frame=3000,
        num_frames=1,
    )
    second_frame = restore_delta_gzip_raw_rgb_payload(
        second_payload,
        bytes_per_frame=3000,
        num_frames=1,
        reference_frame=first_frame,
    )
    assert first_frame + second_frame == expected_frames


def test_raw_rgb_realtime_output_adapter_offloads_delta_payload_build(
    monkeypatch,
):
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        realtime_output_adapter.asyncio,
        "to_thread",
        fake_to_thread,
    )

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
            request_id="req-offload-delta",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
            raw_frame_content_type=RAW_RGB_CONTENT_TYPE,
            raw_frame_metadata={
                "format": "rgb24",
                "width": 1000,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3000,
            },
        )

        await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads, frame0 + frame1

    payloads, expected_frames = asyncio.run(run())

    assert [call[0] for call in calls] == [
        realtime_output_adapter._build_transport_payload,
        realtime_output_adapter._build_transport_payload,
    ]
    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["encoding"] == "delta-gzip"
    assert "delta_reference" not in first_header
    assert second_header["delta_reference"] == "previous-frame"
    first_frame = restore_delta_gzip_raw_rgb_payload(
        first_payload,
        bytes_per_frame=3000,
        num_frames=1,
    )
    second_frame = restore_delta_gzip_raw_rgb_payload(
        second_payload,
        bytes_per_frame=3000,
        num_frames=1,
        reference_frame=first_frame,
    )
    assert first_frame + second_frame == expected_frames


def test_raw_rgb_realtime_output_adapter_can_send_uncompressed_raw_frames():
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
            request_id="req-raw",
            width=1000,
            height=1,
            enable_upscaling=False,
            realtime_event_id=3,
            realtime_output_format="raw",
        )
        result = OutputBatch(
            raw_frame_batches=[[frame0, frame1]],
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

    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert first_header["content_type"] == RAW_RGB_CONTENT_TYPE
    assert first_header["encoding"] == "raw"
    assert first_header["raw_size"] == 3000
    assert first_header["total_size"] == 3000
    assert first_header["num_frames"] == 1
    assert first_header["num_frame_batches"] == 2
    assert first_header["frame_batch_index"] == 0
    assert second_header["raw_size"] == 3000
    assert second_header["total_size"] == 3000
    assert second_header["num_frames"] == 1
    assert second_header["num_frame_batches"] == 2
    assert second_header["frame_batch_index"] == 1
    assert first_payload + second_payload == expected_frames
    assert stats["raw_bytes"] == 6000
    assert stats["num_batches"] == 2
    assert stats["num_frames"] == 2
    assert stats["ws_payload_bytes"] == sum(len(payload) for payload in payloads)


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

    (first_header, first_payload), (second_header, second_payload) = (
        _unpack_frame_batch_messages(payloads)
    )
    assert "delta_reference" not in first_header
    assert second_header["delta_reference"] == "previous-frame"
    first_frame = restore_delta_gzip_raw_rgb_payload(
        first_payload,
        bytes_per_frame=6,
        num_frames=1,
    )
    second_frame = restore_delta_gzip_raw_rgb_payload(
        second_payload,
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

    headers = [header for header, _ in _unpack_frame_batch_messages(payloads)]
    assert len(headers) == 7
    assert [header["chunk_index"] for header in headers] == [4] * 7
    assert [header["frame_batch_index"] for header in headers] == list(range(7))
    assert [header["num_frame_batches"] for header in headers] == [7] * 7
    assert [header["num_frames"] for header in headers] == [1] * 7
    assert [header["is_final_frame_batch"] for header in headers] == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert "delta_reference" not in headers[0]
    for header in headers[1:]:
        assert header["delta_reference"] == "previous-frame"
    assert stats["num_batches"] == 7
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

    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert header["format"] == "webp"
    assert header["encoding"] == "webp"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert frame_payload.startswith(b"RIFF")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1


def test_raw_rgb_realtime_output_adapter_offloads_preview_encoding(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        realtime_output_adapter.asyncio,
        "to_thread",
        fake_to_thread,
    )

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
            request_id="req-webp-offload",
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

        await adapter.send(ws, SimpleNamespace(), result, batch)
        return ws.payloads

    payloads = asyncio.run(run())

    assert [call[0] for call in calls] == [
        realtime_output_adapter._build_transport_payload,
    ]
    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == WEBP_FRAME_CONTENT_TYPE
    assert header["encoding"] == "webp"
    assert frame_payload.startswith(b"RIFF")


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

    [(header, frame_payload)] = _unpack_frame_batch_messages(payloads)
    assert header["content_type"] == JPEG_FRAME_CONTENT_TYPE
    assert header["format"] == "jpeg"
    assert header["encoding"] == "jpeg"
    assert header["num_frames"] == 1
    assert header["is_final_frame_batch"] is True
    assert frame_payload.startswith(b"\xff\xd8")
    assert stats["num_batches"] == 1
    assert stats["num_frames"] == 1
