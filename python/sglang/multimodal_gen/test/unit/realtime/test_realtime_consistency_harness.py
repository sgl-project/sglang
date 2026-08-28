# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import SimpleNamespace

import msgspec.msgpack
import numpy as np
import pytest

from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    build_delta_gzip_raw_rgb_payload,
)
from sglang.multimodal_gen.test.server.realtime_consistency import (
    build_realtime_event_payload,
    build_realtime_init_payload,
    collect_realtime_output,
    decode_realtime_raw_rgb_frames,
    parse_realtime_chunk_stats,
    pop_realtime_key_frames,
    prepare_realtime_first_frame,
    realtime_ws_url,
    record_realtime_key_frames,
    select_realtime_key_frames,
    summarize_realtime_perf_stats,
    validate_realtime_perf_stats,
)
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    LONGLIVE2_I2V_CI_sampling_params,
    LONGLIVE2_T2V_CI_sampling_params,
    REALTIME_MODEL_sampling_params,
)

# Request construction


def test_realtime_ws_url_uses_existing_openai_base_url():
    client = SimpleNamespace(base_url="http://127.0.0.1:30000/v1")

    assert realtime_ws_url(client) == "ws://127.0.0.1:30000/v1/realtime_video/generate"


def test_realtime_init_payload_uses_sampling_params_and_extras():
    params = DiffusionSamplingParams(
        prompt="turn camera left",
        seconds=1,
        fps=8,
        num_frames=6,
        extras={"seed": 7, "num_inference_steps": 4},
        realtime_num_chunks=2,
    )

    payload = build_realtime_init_payload(
        model_path="robbyant/lingbot-world-fast-diffusers",
        sampling_params=params,
        output_size="832x480",
        first_frame="https://example.com/first.png",
    )

    assert payload == {
        "type": "init",
        "model": "robbyant/lingbot-world-fast-diffusers",
        "prompt": "turn camera left",
        "size": "832x480",
        "seconds": 1,
        "first_frame": "https://example.com/first.png",
        "fps": 8,
        "num_frames": 6,
        "seed": 7,
        "num_inference_steps": 4,
    }


def test_realtime_init_payload_can_request_preview_transport():
    params = DiffusionSamplingParams(
        prompt="preview transport",
        realtime_num_chunks=1,
        realtime_output_format="webp",
    )

    payload = build_realtime_init_payload(
        model_path="robbyant/lingbot-world-fast-diffusers",
        sampling_params=params,
        output_size="832x480",
        first_frame=None,
    )

    assert payload["realtime_output_format"] == "webp"


def test_realtime_first_frame_accepts_url_or_file(tmp_path):
    frame_path = tmp_path / "first.png"
    frame_path.write_bytes(b"png-bytes")

    assert prepare_realtime_first_frame("https://example.com/first.png") == (
        "https://example.com/first.png"
    )
    assert prepare_realtime_first_frame(frame_path) == b"png-bytes"


def test_realtime_key_frames_are_selected_from_raw_websocket_frames():
    frames = [np.full((2, 2, 3), idx, dtype=np.uint8) for idx in range(5)]

    selected = select_realtime_key_frames(frames)
    assert [int(frame[0, 0, 0]) for frame in selected] == [0, 2, 4]

    record_realtime_key_frames("unit-raw-frames", frames)
    frames[2][:] = 99
    popped = pop_realtime_key_frames("unit-raw-frames")

    assert popped is not None
    assert [int(frame[0, 0, 0]) for frame in popped] == [0, 2, 4]
    assert pop_realtime_key_frames("unit-raw-frames") is None


def test_realtime_event_payload_strips_test_schedule_metadata():
    payload = build_realtime_event_payload(
        {
            "after_chunk": 0,
            "kind": "camera_actions",
            "payload": [["w"], ["d"]],
        }
    )

    assert payload == {
        "type": "event",
        "kind": "camera_actions",
        "payload": [["w"], ["d"]],
    }


# Raw RGB frame decoding


def test_decode_realtime_raw_rgb_frames_splits_payload_by_header_metadata():
    first = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    second = (np.arange(12, dtype=np.uint8) + 20).reshape(2, 2, 3)
    header = {
        "type": "frame_batch_header",
        "content_type": RAW_RGB_CONTENT_TYPE,
        "num_frames": 2,
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }

    frames = decode_realtime_raw_rgb_frames(
        header,
        first.tobytes() + second.tobytes(),
    )

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], first)
    np.testing.assert_array_equal(frames[1], second)


def test_decode_realtime_raw_rgb_frames_rejects_truncated_payload():
    header = {
        "type": "frame_batch_header",
        "content_type": RAW_RGB_CONTENT_TYPE,
        "num_frames": 1,
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }

    with pytest.raises(ValueError, match="payload size mismatch"):
        decode_realtime_raw_rgb_frames(header, b"too-short")


def test_decode_realtime_delta_gzip_raw_rgb_frames_roundtrips():
    first = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    second = (np.arange(12, dtype=np.uint8) + 1).reshape(2, 2, 3)
    header = {
        "type": "frame_batch_header",
        "content_type": RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
        "num_frames": 2,
        "width": 2,
        "height": 2,
        "channels": 3,
        "bytes_per_frame": 12,
    }

    frames = decode_realtime_raw_rgb_frames(
        header,
        build_delta_gzip_raw_rgb_payload([first.tobytes(), second.tobytes()]),
    )

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], first)
    np.testing.assert_array_equal(frames[1], second)


def test_decode_realtime_rgba_delta_gzip_strips_alpha():
    first = np.array(
        [[[1, 2, 3, 255], [4, 5, 6, 255]]],
        dtype=np.uint8,
    )
    second = np.array(
        [[[1, 2, 4, 255], [4, 6, 6, 255]]],
        dtype=np.uint8,
    )
    header = {
        "type": "frame_batch_header",
        "content_type": RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
        "num_frames": 2,
        "width": 2,
        "height": 1,
        "channels": 4,
        "bytes_per_frame": 8,
    }

    frames = decode_realtime_raw_rgb_frames(
        header,
        build_delta_gzip_raw_rgb_payload([first.tobytes(), second.tobytes()]),
    )

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], first[:, :, :3])
    np.testing.assert_array_equal(frames[1], second[:, :, :3])


# Stream collection and realtime performance stats


class _FakeRealtimeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, payload):
        self.sent.append(msgspec.msgpack.decode(payload))

    async def recv(self):
        if not self.messages:
            raise AssertionError("fake websocket received too many recv calls")
        return self.messages.pop(0)


def _packed_realtime_frame_message(chunk_index: int, frame: np.ndarray):
    header = {
        "type": "frame_batch_header",
        "content_type": RAW_RGB_CONTENT_TYPE,
        "chunk_index": chunk_index,
        "is_final_frame_batch": True,
        "num_frames": 1,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "channels": frame.shape[2],
        "bytes_per_frame": frame.nbytes,
    }
    return msgspec.msgpack.encode(header), frame.tobytes()


def _packed_realtime_combined_frame_message(chunk_index: int, frame: np.ndarray):
    header, payload = _packed_realtime_frame_message(chunk_index, frame)
    message = msgspec.msgpack.decode(header)
    message["type"] = "frame_batch"
    message["payload"] = payload
    return msgspec.msgpack.encode(message)


def _packed_realtime_chunk_stats(chunk_index: int, **overrides):
    payload = {
        "type": "chunk_stats",
        "request_id": f"req-{chunk_index}",
        "chunk_index": chunk_index,
        "content_type": RAW_RGB_CONTENT_TYPE,
        "num_frames": 1,
        "raw_bytes": 12,
        "ws_payload_bytes": 128,
        "request_prepare_ms": 1,
        "scheduler_forward_ms": 20,
        "raw_payload_build_ms": 2,
        "raw_write_ms": 3,
        "ws_write_ms": 4,
        "chunk_total_ms": 30,
    }
    payload.update(overrides)
    for key, value in list(payload.items()):
        if key.endswith("_ms"):
            payload[key] = max(0, int(value + 0.5))
    packed = msgspec.msgpack.encode(payload)
    assert bytes([0xCB]) not in packed
    return packed


def test_collect_realtime_output_skips_and_records_chunk_stats(monkeypatch):
    first = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    second = first + 1
    chunk0_header, chunk0_payload = _packed_realtime_frame_message(0, first)
    chunk1_header, chunk1_payload = _packed_realtime_frame_message(1, second)
    websocket = _FakeRealtimeWebSocket(
        [
            chunk0_header,
            chunk0_payload,
            _packed_realtime_chunk_stats(0, chunk_total_ms=31),
            chunk1_header,
            chunk1_payload,
            _packed_realtime_chunk_stats(1, chunk_total_ms=32),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda *args, **kwargs: websocket),
    )

    result = asyncio.run(
        collect_realtime_output(
            ws_url="ws://example.test/v1/realtime_video/generate",
            init_payload={"type": "init", "prompt": "test"},
            events=[
                {
                    "after_chunk": 0,
                    "kind": "camera_actions",
                    "payload": [["w"]],
                }
            ],
            num_chunks=2,
            require_chunk_stats=True,
        )
    )

    assert len(result.frames) == 2
    np.testing.assert_array_equal(result.frames[0], first)
    np.testing.assert_array_equal(result.frames[1], second)
    assert [stat.chunk_index for stat in result.chunk_stats] == [0, 1]
    assert [stat.chunk_total_ms for stat in result.chunk_stats] == [31.0, 32.0]
    assert websocket.sent == [
        {"type": "init", "prompt": "test"},
        {
            "type": "event",
            "kind": "camera_actions",
            "payload": [["w"]],
        },
    ]


def test_collect_realtime_output_accepts_combined_frame_batch(monkeypatch):
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    websocket = _FakeRealtimeWebSocket(
        [
            _packed_realtime_combined_frame_message(0, frame),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda *args, **kwargs: websocket),
    )

    result = asyncio.run(
        collect_realtime_output(
            ws_url="ws://example.test/v1/realtime_video/generate",
            init_payload={"type": "init", "prompt": "test"},
            events=[],
            num_chunks=1,
        )
    )

    assert len(result.frames) == 1
    np.testing.assert_array_equal(result.frames[0], frame)


def test_realtime_perf_stats_summary_and_thresholds():
    stats = [
        parse_realtime_chunk_stats(
            msgspec.msgpack.decode(
                _packed_realtime_chunk_stats(
                    0,
                    scheduler_forward_ms=10.0,
                    raw_write_ms=2.0,
                    ws_write_ms=3.0,
                    ws_payload_bytes=1024 * 1024,
                    chunk_total_ms=20.0,
                )
            )
        ),
        parse_realtime_chunk_stats(
            msgspec.msgpack.decode(
                _packed_realtime_chunk_stats(
                    1,
                    scheduler_forward_ms=30.0,
                    raw_write_ms=4.0,
                    ws_write_ms=5.0,
                    ws_payload_bytes=2 * 1024 * 1024,
                    chunk_total_ms=40.0,
                )
            )
        ),
    ]

    summary = summarize_realtime_perf_stats(stats)

    assert summary["num_chunks"] == 2
    assert summary["total_frames"] == 2
    assert summary["guarded_chunks"] == 2
    assert summary["avg_scheduler_forward_ms"] == 20.0
    assert summary["p95_chunk_total_ms"] == 40.0
    assert summary["avg_ws_payload_mb"] == 1.5
    validate_realtime_perf_stats(
        "case",
        stats,
        {
            "avg_scheduler_forward_ms": 25.0,
            "p95_chunk_total_ms": 45.0,
            "avg_ws_payload_mb": 2.0,
        },
    )
    with pytest.raises(pytest.fail.Exception, match="p95_chunk_total_ms"):
        validate_realtime_perf_stats(
            "case",
            stats,
            {"p95_chunk_total_ms": 35.0},
        )


def test_realtime_perf_stats_can_ignore_startup_chunks():
    stats = [
        parse_realtime_chunk_stats(
            msgspec.msgpack.decode(
                _packed_realtime_chunk_stats(
                    0,
                    scheduler_forward_ms=20000.0,
                    chunk_total_ms=21000.0,
                )
            )
        ),
        parse_realtime_chunk_stats(
            msgspec.msgpack.decode(
                _packed_realtime_chunk_stats(
                    1,
                    scheduler_forward_ms=7000.0,
                    chunk_total_ms=7200.0,
                )
            )
        ),
        parse_realtime_chunk_stats(
            msgspec.msgpack.decode(
                _packed_realtime_chunk_stats(
                    2,
                    scheduler_forward_ms=2300.0,
                    chunk_total_ms=2800.0,
                )
            )
        ),
    ]

    summary = summarize_realtime_perf_stats(stats, ignore_initial_chunks=2)

    assert summary["num_chunks"] == 3
    assert summary["ignored_initial_chunks"] == 2
    assert summary["guarded_chunks"] == 1
    assert summary["ignored_max_chunk_total_ms"] == 21000.0
    assert summary["p95_chunk_total_ms"] == 2800.0
    validate_realtime_perf_stats(
        "case",
        stats,
        {"p95_chunk_total_ms": 5000.0, "p95_scheduler_forward_ms": 4500.0},
        ignore_initial_chunks=2,
    )
    with pytest.raises(ValueError, match="leave at least one"):
        summarize_realtime_perf_stats(stats, ignore_initial_chunks=3)


# Generate function routing


def test_realtime_sampling_params_route_to_realtime_video_generator():
    params = DiffusionSamplingParams(
        prompt="turn camera left",
        realtime_num_chunks=2,
    )

    generate_fn = get_generate_fn(
        "robbyant/lingbot-world-fast-diffusers",
        "video",
        params,
    )

    assert generate_fn.__name__ == "generate_realtime_video"


def test_realtime_model_params_are_lossless_gt_ready():
    params = REALTIME_MODEL_sampling_params

    assert "floating island hotel" in params.prompt
    assert "825646291038" in str(params.image_path)
    assert params.output_size == "832x480"
    assert params.realtime_num_chunks == 4
    assert params.realtime_output_format is None
    assert params.realtime_perf_ignore_initial_chunks == 2
    assert params.realtime_perf_thresholds["p95_chunk_total_ms"] == 5000.0
    assert params.realtime_perf_thresholds["p95_scheduler_forward_ms"] == 4500.0
    assert params.realtime_events == []
    assert params.extras["condition_inputs"]["camera_actions"] == [
        ["w"],
        ["w"],
        ["w"],
        ["w"],
        ["w"],
        ["w"],
        [],
        [],
        [],
        [],
        [],
        [],
    ]


def test_longlive2_cases_share_realtime_model_sampling_profile():
    for params in (
        LONGLIVE2_T2V_CI_sampling_params,
        LONGLIVE2_I2V_CI_sampling_params,
    ):
        assert params.prompt == REALTIME_MODEL_sampling_params.prompt
        assert params.fps == REALTIME_MODEL_sampling_params.fps
        assert params.extras == {
            "seed": 42,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
        }
        assert params.realtime_num_chunks is None
        assert params.realtime_perf_thresholds == {}

    assert LONGLIVE2_T2V_CI_sampling_params.image_path is None
    assert (
        LONGLIVE2_T2V_CI_sampling_params.output_size
        == REALTIME_MODEL_sampling_params.output_size
    )
    assert (
        LONGLIVE2_I2V_CI_sampling_params.image_path
        == REALTIME_MODEL_sampling_params.image_path
    )
    assert LONGLIVE2_I2V_CI_sampling_params.output_size == "960x928"


def test_lingbot_realtime_case_is_registered_by_default():
    from sglang.multimodal_gen.test.server.gpu_cases import ONE_GPU_CASES

    case = next(
        item
        for item in ONE_GPU_CASES
        if item.id == "lingbot_world_realtime_plastic_beach"
    )
    assert case.id == "lingbot_world_realtime_plastic_beach"
    assert case.run_consistency_check is True
    assert case.run_perf_check is True
    assert case.sampling_params.realtime_output_format is None
