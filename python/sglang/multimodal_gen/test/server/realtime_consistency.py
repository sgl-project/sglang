# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import pytest
from msgpack import packb, unpackb
from openai import Client

from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    restore_delta_gzip_raw_rgb_payload,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionSamplingParams
from sglang.multimodal_gen.test.test_utils import is_image_url

_REALTIME_WS_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_REALTIME_WS_TIMEOUT_SECS", "1200")
)


def realtime_ws_url(client: Client) -> str:
    base_url = str(client.base_url).rstrip("/")
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :] + "/realtime_video/generate"
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :] + "/realtime_video/generate"
    raise ValueError(f"Unsupported realtime client base_url: {base_url}")


def prepare_realtime_first_frame(
    image_path: Path | str | list[Path | str] | None,
) -> bytes | str | None:
    if image_path is None:
        return None
    if isinstance(image_path, list):
        image_path = image_path[0]
    if isinstance(image_path, str) and is_image_url(image_path):
        return image_path
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Realtime first frame file missing: {path}")
    return path.read_bytes()


def build_realtime_init_payload(
    *,
    model_path: str,
    sampling_params: DiffusionSamplingParams,
    output_size: str,
    first_frame: bytes | str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "init",
        "model": model_path,
        "prompt": sampling_params.prompt,
        "size": output_size,
        "seconds": sampling_params.seconds,
        "first_frame": first_frame,
    }
    optional_fields = {
        "fps": sampling_params.fps,
        "num_frames": sampling_params.num_frames,
    }
    payload.update({k: v for k, v in optional_fields.items() if v is not None})
    payload.update(dict(sampling_params.extras))
    return {k: v for k, v in payload.items() if v is not None}


def build_realtime_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.pop("after_chunk", None)
    payload.setdefault("type", "event")
    if "kind" not in payload:
        raise ValueError("realtime event config must include kind")
    return payload


def decode_realtime_raw_rgb_frames(
    header: dict[str, Any],
    payload: bytes,
    previous_frame: bytes | None = None,
) -> list[np.ndarray]:
    content_type = header.get("content_type")
    if content_type not in (
        RAW_RGB_CONTENT_TYPE,
        RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
        RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    ):
        raise ValueError(
            f"Unsupported realtime frame content type: {content_type}"
        )

    width = int(header["width"])
    height = int(header["height"])
    channels = int(header["channels"])
    num_frames = int(header["num_frames"])
    bytes_per_frame = int(header["bytes_per_frame"])
    expected_size = num_frames * bytes_per_frame
    if content_type in (
        RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
        RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    ):
        if header.get("delta_reference") != "previous-frame":
            previous_frame = None
        payload = restore_delta_gzip_raw_rgb_payload(
            payload,
            bytes_per_frame=bytes_per_frame,
            num_frames=num_frames,
            reference_frame=previous_frame,
        )
    if len(payload) != expected_size:
        raise ValueError(
            f"Realtime payload size mismatch: expected {expected_size}, got {len(payload)}"
        )

    frames = []
    for frame_idx in range(num_frames):
        offset = frame_idx * bytes_per_frame
        frame = np.frombuffer(
            payload[offset : offset + bytes_per_frame], dtype=np.uint8
        )
        frame = frame.reshape(height, width, channels)
        if channels > 3:
            frame = frame[:, :, :3]
        frames.append(frame.copy())
    return frames


def encode_realtime_frames_to_mp4(frames: list[np.ndarray], fps: int) -> bytes:
    if not frames:
        raise ValueError("Cannot encode empty realtime frame list")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name
    try:
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            format="mp4",
            codec="libx264",
            quality=5,
        )
        return Path(output_path).read_bytes()
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass


async def collect_realtime_frames(
    *,
    ws_url: str,
    init_payload: dict[str, Any],
    events: list[dict[str, Any]],
    num_chunks: int,
) -> list[np.ndarray]:
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets is required for realtime consistency checks")

    frames: list[np.ndarray] = []
    sent_event_indices: set[int] = set()

    async def send_events_for_boundary(ws, completed_chunk: int) -> None:
        for event_idx, event in enumerate(events):
            if event_idx in sent_event_indices:
                continue
            if int(event.get("after_chunk", 0)) != completed_chunk:
                continue
            await ws.send(packb(build_realtime_event_payload(event), use_bin_type=True))
            sent_event_indices.add(event_idx)

    async with websockets.connect(ws_url, max_size=None, ping_interval=None) as ws:
        await ws.send(packb(init_payload, use_bin_type=True))
        await send_events_for_boundary(ws, -1)

        received_chunks: set[int] = set()
        previous_frame: bytes | None = None
        while len(received_chunks) < num_chunks:
            header_payload = await asyncio.wait_for(
                ws.recv(), timeout=_REALTIME_WS_TIMEOUT_SECS
            )
            header = unpackb(header_payload, raw=False)
            if header.get("type") == "error":
                pytest.fail(f"Realtime generation failed: {header.get('content')}")
            if header.get("type") != "frame_batch_header":
                raise ValueError(f"Unexpected realtime message: {header}")

            raw_payload = await asyncio.wait_for(
                ws.recv(), timeout=_REALTIME_WS_TIMEOUT_SECS
            )
            if not isinstance(raw_payload, bytes):
                raise ValueError("Realtime frame payload must be bytes")

            chunk_frames = decode_realtime_raw_rgb_frames(
                header,
                raw_payload,
                previous_frame,
            )
            frames.extend(chunk_frames)
            if chunk_frames:
                previous_frame = chunk_frames[-1].tobytes()
            chunk_index = int(header["chunk_index"])
            if header.get("is_final_frame_batch", True):
                received_chunks.add(chunk_index)
                await send_events_for_boundary(ws, chunk_index)

    return frames
