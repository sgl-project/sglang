# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    build_delta_gzip_raw_rgb_payload,
)
from sglang.multimodal_gen.test.server.realtime_consistency import (
    build_realtime_event_payload,
    build_realtime_init_payload,
    decode_realtime_raw_rgb_frames,
    prepare_realtime_first_frame,
    realtime_ws_url,
)
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
)


# Request construction


def test_realtime_ws_url_uses_existing_openai_base_url():
    client = SimpleNamespace(base_url="http://127.0.0.1:30000/v1")

    assert (
        realtime_ws_url(client)
        == "ws://127.0.0.1:30000/v1/realtime_video/generate"
    )


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


def test_realtime_first_frame_accepts_url_or_file(tmp_path):
    frame_path = tmp_path / "first.png"
    frame_path.write_bytes(b"png-bytes")

    assert prepare_realtime_first_frame("https://example.com/first.png") == (
        "https://example.com/first.png"
    )
    assert prepare_realtime_first_frame(frame_path) == b"png-bytes"


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
