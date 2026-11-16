"""Processors must read loaded videos through VideoDecoderWrapper's API.

``load_video`` returns a ``VideoDecoderWrapper``. Ernie4.5-VL and NVILA still
called the decord reader API on it (``get_avg_fps``, ``get_batch``,
``asnumpy``), which the wrapper does not have, so every video request to those
models raised ``AttributeError``.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from sglang.srt.multimodal.processors import ernie45_vl
from sglang.srt.multimodal.processors.nvila import NVILAMultimodalProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _InMemoryVideo(VideoDecoderWrapper):
    """The wrapper's public API over in-memory NHWC uint8 frames."""

    def __init__(self, frames: np.ndarray, fps: float):
        self._frames = frames
        self._fps = fps

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    @property
    def avg_fps(self) -> float:
        return self._fps

    def get_frames_at(self, indices):
        return self._frames[indices]

    def get_frames_as_tensor(self, indices):
        return torch.from_numpy(self._frames[indices])


def _frames(num_frames: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


class TestVideoDecoderWrapperCallers(CustomTestCase):
    def test_ernie45_vl_preprocess_video(self):
        video = _InMemoryVideo(_frames(16, 28, 56), fps=8.0)

        # The CPU engine skips pinning, which needs an accelerator.
        with patch.object(ernie45_vl, "is_cpu", return_value=True):
            frames, metadata = asyncio.run(ernie45_vl.preprocess_video(video))

        num_frames, height, width, channels = frames.shape
        self.assertEqual((num_frames, channels), (16, 3))
        self.assertEqual((height % 28, width % 28), (0, 0))
        self.assertEqual(metadata["fps"], 8.0)
        self.assertEqual(metadata["total_num_frames"], 16)

    def test_nvila_passes_decoded_frames_to_the_hf_processor(self):
        frames = _frames(4, 8, 8)
        processor = NVILAMultimodalProcessor.__new__(NVILAMultimodalProcessor)
        processor.mm_tokens = SimpleNamespace(image_token_id=1, video_token_id=2)
        processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(videos=[_InMemoryVideo(frames, fps=4.0)])
        )
        processor.process_and_combine_mm_data_async = AsyncMock(
            return_value=([], torch.tensor([2, 2]), None)
        )

        asyncio.run(
            processor.process_mm_data_async(
                image_data=None,
                audio_data=None,
                input_text="<video>",
                request_obj=SimpleNamespace(image_data=None, video_data=["clip"]),
            )
        )

        base_output = processor.process_and_combine_mm_data_async.call_args.args[0]
        (sent,) = base_output.videos
        self.assertEqual(len(sent), 4)
        for frame, expected in zip(sent, frames):
            self.assertIsInstance(frame, np.ndarray)
            np.testing.assert_array_equal(frame, expected)


if __name__ == "__main__":
    unittest.main()
