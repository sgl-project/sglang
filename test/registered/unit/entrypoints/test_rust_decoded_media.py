"""CPU regressions for decoded ingress and preservation of processor results."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # Must precede modules that import sgl_kernel.

from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.media_io import DecodedAudio, load_audio, load_video
from sglang.srt.multimodal.modality import Modality
from sglang.srt.rust_server.decoded_media import (
    NativeDecodedMediaProcessor,
    materialize_media,
)
from sglang.srt.rust_server.multimodal import settle_shm_buffers
from sglang.srt.rust_server.server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def image(width=2, height=1, stride=6):
    return {
        "width": width,
        "height": height,
        "row_stride_bytes": stride,
        "pixel_format": "rgb8",
    }


def test_decoded_rgb_padding_and_video_order_are_preserved():
    descriptor = {
        "images": [image(height=2, stride=8)],
        "videos": [
            {
                "source_fps": 30.0,
                "source_frame_count": 60,
                "frames": [
                    {"image": image(), "timestamp_us": 0, "source_frame_index": 0},
                    {
                        "image": image(),
                        "timestamp_us": 500000,
                        "source_frame_index": 15,
                    },
                ],
            }
        ],
    }
    pixels = np.array([1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12], dtype=np.uint8)
    first, second = np.zeros(6, dtype=np.uint8), np.full(6, 255, dtype=np.uint8)
    images, videos, audios = materialize_media(descriptor, [pixels, first, second])
    np.testing.assert_array_equal(np.asarray(images[0]).reshape(-1), np.arange(1, 13))
    video = videos[0]
    assert load_video(video) is video
    frames, metadata = video.as_processor_input()
    assert tuple(frames.shape) == (2, 3, 1, 2)
    assert frames[0].sum() == 0 and frames[1].min() == 255
    assert metadata["frames_indices"] == [0, 15]
    assert metadata["fps"] == 30.0
    assert video.timestamps_us == [0, 500000]
    assert not audios


def test_decoded_pcm_uses_common_audio_loader_and_rejects_nonfinite_samples():
    descriptor = {
        "audios": [
            {
                "sample_format": "f32le",
                "sample_rate_hz": 16000,
                "sample_frames": 2,
                "channels": 2,
            }
        ]
    }
    samples = np.array([[0.5, -0.5], [1.0, 0.0]], dtype="<f4")
    _, _, audio = materialize_media(descriptor, [samples.view(np.uint8)])
    assert isinstance(audio[0], DecodedAudio)
    np.testing.assert_allclose(load_audio(audio[0], sr=16000), [0, 0.5])
    for value in (np.nan, np.inf, 1.01):
        bad = samples.copy()
        bad[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            materialize_media(descriptor, [bad.view(np.uint8)])


def output():
    return MultimodalProcessorOutput(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.zeros(2, 3),
                identity=bytes(32),
                offsets=[(1, 2)],
                model_specific_data={
                    "auxiliary": torch.tensor([4, 5]),
                    "scale": 0.25,
                },
            )
        ],
        input_ids=[1, 2, 3],
        padded_input_ids=[1, 2, 3],
        slice_start_id=9,
        vision_position_ids=torch.tensor([1, 2, 3]),
        token_type_ids=torch.tensor([0, 1, 1]),
    )


def test_bridge_preserves_the_full_existing_processor_output():
    result = output()
    bridge = object.__new__(NativeDecodedMediaProcessor)
    bridge.max_input_len = 100
    bridge.processor = SimpleNamespace(
        process_mm_data_async=AsyncMock(return_value=result),
        finalize_cache_identities=Mock(),
    )
    request = SimpleNamespace(image_data=[object()], audio_data=None)
    ids, actual = asyncio.run(bridge._process([7], request))
    assert ids == [1, 2, 3] and actual is result
    bridge.processor.finalize_cache_identities.assert_called_once_with(result)
    # The drain must pass the object itself, not narrow/rebuild it into the
    # Qwen-only native metadata shape; model-specific tensors remain intact.
    server = object.__new__(RustServer)
    server._multimodal_enabled = True
    obj = SimpleNamespace(mm_inputs=None, token_type_ids=None)
    buffers = {"mm.python_output": result}
    server._attach_buffers(obj, buffers)
    assert obj.mm_inputs is result
    assert obj.mm_inputs.slice_start_id == 9
    assert obj.mm_inputs.mm_items[0].model_specific_data["scale"] == 0.25
    assert obj.token_type_ids == [0, 1, 1]
    settle_shm_buffers(buffers, admitted=True)
    settle_shm_buffers(buffers, admitted=False)


def test_bridge_rejects_bad_processor_spans_before_queueing():
    bridge = object.__new__(NativeDecodedMediaProcessor)
    bridge.max_input_len = 100
    result = output()
    result.mm_items[0].offsets = [(0, 20)]
    bridge.processor = SimpleNamespace(
        process_mm_data_async=AsyncMock(return_value=result),
        finalize_cache_identities=Mock(),
    )
    request = SimpleNamespace(image_data=[object()], audio_data=None)
    with pytest.raises(RuntimeError, match="spans"):
        asyncio.run(bridge._process([7], request))
