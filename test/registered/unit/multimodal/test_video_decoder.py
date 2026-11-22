"""CPU video decoding and preprocessing must not pin host memory.

Pinning decoded CPU video frames allocates large variable-sized buffers through
PyTorch's CachingHostAllocator (which never returns freed blocks to the OS) and
implicitly initializes a CUDA primary context on GPU 0 from the tokenizer
process even when video decoding/preprocessing runs on CPU.
"""

import asyncio
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.multimodal.processors import ernie45_vl
from sglang.srt.multimodal.qwen_vl_media_processing import (
    preprocess_qwen3_vl_video_sync,
    preprocess_video_sync,
)
from sglang.srt.utils import video_decoder
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _InMemoryVideo(VideoDecoderWrapper):
    def __init__(self, frames: np.ndarray, fps: float = 8.0):
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


def _make_frames(num_frames: int = 8, height: int = 56, width: int = 84) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


class _FakeTorchCodecDecoder:
    def __init__(self, frames: torch.Tensor):
        self._frames = frames

    def get_frames_at(self, indices):
        return SimpleNamespace(data=self._frames[indices].clone())


class _FakeDecordBatch:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class _FakeDecordReader:
    def __init__(self, frames: np.ndarray):
        self._frames = frames

    def get_batch(self, indices):
        return _FakeDecordBatch(self._frames[indices].copy())


class TestVideoDecoderNoPinMemory(CustomTestCase):
    def _forbid_pin_memory(self):
        return patch.object(
            torch.Tensor,
            "pin_memory",
            side_effect=AssertionError(
                "CPU video decode/preprocess must not call pin_memory()"
            ),
        )

    def test_get_frames_as_tensor_torchcodec_single_thread_no_pin(self):
        frames = torch.from_numpy(_make_frames(6, 32, 48))
        wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
        wrapper._decoder = _FakeTorchCodecDecoder(frames)
        wrapper._num_decode_threads = 1
        wrapper._source = None
        wrapper._tc_kwargs = {"dimension_order": "NHWC"}

        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            self._forbid_pin_memory(),
        ):
            out = wrapper.get_frames_as_tensor([0, 2, 4])

        self.assertEqual(out.device.type, "cpu")
        self.assertFalse(out.is_pinned())
        self.assertTrue(torch.equal(out, frames[[0, 2, 4]]))

    def test_get_frames_as_tensor_torchcodec_parallel_no_pin(self):
        frames = torch.from_numpy(_make_frames(8, 32, 48))
        wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
        wrapper._decoder = _FakeTorchCodecDecoder(frames)
        wrapper._num_decode_threads = 2
        wrapper._source = b"fake"
        wrapper._tc_kwargs = {"dimension_order": "NHWC"}

        def _fake_ctor(source, **kwargs):
            return _FakeTorchCodecDecoder(frames)

        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            patch.object(video_decoder, "VideoDecoder", _fake_ctor, create=True),
            self._forbid_pin_memory(),
        ):
            out = wrapper.get_frames_as_tensor([0, 1, 2, 3, 4, 5])

        self.assertEqual(out.device.type, "cpu")
        self.assertFalse(out.is_pinned())
        self.assertTrue(torch.equal(out, frames[[0, 1, 2, 3, 4, 5]]))

    def test_get_frames_as_tensor_decord_no_pin(self):
        frames = _make_frames(6, 32, 48)
        wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
        wrapper._decoder = _FakeDecordReader(frames)
        wrapper._num_decode_threads = 1

        with (
            patch.object(video_decoder, "_BACKEND", "decord"),
            self._forbid_pin_memory(),
        ):
            out = wrapper.get_frames_as_tensor([1, 3, 5])

        self.assertEqual(out.device.type, "cpu")
        self.assertFalse(out.is_pinned())
        np.testing.assert_array_equal(out.numpy(), frames[[1, 3, 5]])

    def test_preprocess_video_sync_no_pin(self):
        video = _InMemoryVideo(_make_frames(8, 56, 84), fps=4.0)
        with self._forbid_pin_memory():
            out, metadata = preprocess_video_sync(video, video_config={"nframes": 4})

        self.assertEqual(out.device.type, "cpu")
        self.assertFalse(out.is_pinned())
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(metadata["total_num_frames"], 8)

    def test_preprocess_qwen3_vl_video_sync_no_pin(self):
        video = _InMemoryVideo(_make_frames(8, 64, 96), fps=4.0)
        with self._forbid_pin_memory():
            out, metadata = preprocess_qwen3_vl_video_sync(
                video, image_factor=32, video_config={"nframes": 4}
            )

        self.assertEqual(out.device.type, "cpu")
        self.assertFalse(out.is_pinned())
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(metadata["total_num_frames"], 8)

    def test_ernie45_vl_preprocess_video_no_pin(self):
        video = _InMemoryVideo(_make_frames(8, 56, 84), fps=4.0)
        with self._forbid_pin_memory():
            out, metadata = asyncio.run(ernie45_vl.preprocess_video(video))

        self.assertEqual(out.device.type, "cpu")
        self.assertFalse(out.is_pinned())
        self.assertEqual(out.shape[0], 8)
        self.assertEqual(metadata["total_num_frames"], 8)

    def test_real_cpu_video_decode_no_pin(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = str(Path(tmpdir) / "clip.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=1:size=128x96:rate=8",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    mp4_path,
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._forbid_pin_memory():
                for num_threads in (1, 2):
                    with VideoDecoderWrapper(
                        mp4_path, device="cpu", num_decode_threads=num_threads
                    ) as decoder:
                        out = decoder.get_frames_as_tensor([0, 2, 4, 6])
                        self.assertEqual(out.shape, (4, 96, 128, 3))
                        self.assertEqual(out.device.type, "cpu")
                        self.assertFalse(out.is_pinned())


if __name__ == "__main__":
    unittest.main()
