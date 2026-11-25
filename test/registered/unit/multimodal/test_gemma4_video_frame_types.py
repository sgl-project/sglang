"""Gemma 4 video decoding must tolerate NumPy and Torch frame batches."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _InMemoryVideo:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def get_frames_at(self, indices):
        return self.frames[indices]


def _frames():
    return np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)


def _processor():
    processor = Gemma4SGLangProcessor.__new__(Gemma4SGLangProcessor)
    processor._processor = SimpleNamespace(
        video_processor=SimpleNamespace(num_frames=32)
    )
    return processor


class TestGemma4VideoFrameTypes(CustomTestCase):
    def assert_matches(self, actual, expected):
        self.assertEqual(actual.shape, (4, 3, 2, 3))
        self.assertEqual(actual.dtype, torch.uint8)
        torch.testing.assert_close(
            actual.cpu(), expected.permute(0, 3, 1, 2).contiguous().cpu()
        )

    def test_numpy_frames(self):
        frames = _frames()
        result = _processor()._video_decoder_to_tensor(_InMemoryVideo(frames))

        self.assert_matches(result, torch.from_numpy(frames))

    def test_cpu_tensor_frames(self):
        frames = torch.from_numpy(_frames())
        result = _processor()._video_decoder_to_tensor(_InMemoryVideo(frames))

        self.assert_matches(result, frames)

    def test_cuda_tensor_frames(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to test TorchCodec GPU frame output")

        frames = torch.from_numpy(_frames()).cuda()
        result = _processor()._video_decoder_to_tensor(_InMemoryVideo(frames))

        self.assert_matches(result, frames)


if __name__ == "__main__":
    unittest.main()
