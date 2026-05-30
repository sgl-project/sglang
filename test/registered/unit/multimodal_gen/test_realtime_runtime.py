# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    lingbot_world_realtime_adapter as lingbot_realtime,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import RealtimeEvent
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSessionCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.utils.lingbot_world_camera import (
    actions_to_c2ws,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
    build_raw_rgb_frame_batches,
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
    state.append_camera_actions([["w"], ["a"], ["s"], ["d"]])
    assert state.sample_camera_actions(3) == [["w"], ["a"], ["s"]]
    assert state.sample_camera_actions(3) == [["d"], ["d"], ["d"]]

    state.append_prompt("turn left")
    assert state.has_prompt()
    assert state.sample_prompt() == "turn left"
    assert not state.has_prompt()


def test_lingbot_realtime_adapter_ingests_generic_events():
    adapter = lingbot_realtime.LingBotWorldRealtimeAdapter()
    session = GenerateSession()
    session.set_adapter(adapter)

    camera_event = RealtimeEvent(
        type="event",
        kind="camera_actions",
        payload=[["w"], ["d"]],
    )
    prompt_event = RealtimeEvent(type="event", kind="prompt", payload="turn left")

    assert (
        adapter.ingest_event(session, camera_event)
        == "kind=camera_actions, frames=2"
    )
    assert adapter.ingest_event(session, prompt_event) == "kind=prompt, prompt_len=9"
    state = adapter._state(session)
    assert state.sample_camera_actions(3) == [["w"], ["d"], ["d"]]
    assert state.sample_prompt() == "turn left"


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


def test_lingbot_camera_actions_have_deterministic_pose_precision():
    poses = actions_to_c2ws([["w"], ["d"]])

    np.testing.assert_allclose(poses[1][:3, 3], [0.0, 0.0, 0.05], atol=1e-6)
    np.testing.assert_allclose(poses[2][:3, 3], [0.05, 0.0, 0.05], atol=1e-6)

    yaw_pose = actions_to_c2ws([["l"]])[1]
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
