"""CPU tests for model-specific Qwen3-VL video preprocessing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from sglang.srt.multimodal.processors import qwen3_vl, qwen_vl
from sglang.test.test_utils import CustomTestCase


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def permute(self, *dims):
        self.shape = tuple(self.shape[dim] for dim in dims)
        return self

    def pin_memory(self):
        return self


class _FakeVideoDecoder:
    def __init__(self, *, total_frames, fps, height=5, width=7):
        self.total_frames = total_frames
        self.avg_fps = fps
        self.height = height
        self.width = width
        self.requested_indices = None

    def __len__(self):
        return self.total_frames

    def get_frames_as_tensor(self, indices):
        self.requested_indices = indices
        return _FakeTensor((len(indices), self.height, self.width, 3))


class TestQwen3VLVideoPreprocess(CustomTestCase):
    def test_qwen3_preserves_source_geometry_for_model_processor(self):
        decoder = _FakeVideoDecoder(total_frames=10, fps=2.0)
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, metadata = asyncio.run(
                qwen3_vl.preprocess_video(decoder, video_config={"nframes": 4})
            )

        self.assertEqual(video.shape, (4, 3, 5, 7))
        self.assertEqual(decoder.requested_indices, [0, 3, 6, 9])
        np.testing.assert_array_equal(metadata["frames_indices"], [0, 3, 6, 9])

    def test_qwen3_max_frames_still_spans_full_video(self):
        decoder = _FakeVideoDecoder(total_frames=3000, fps=30.0)
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, _ = asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_config={"fps": 2, "max_frames": 10},
                )
            )

        self.assertEqual(video.shape, (10, 3, 5, 7))
        self.assertEqual(decoder.requested_indices[0], 0)
        self.assertEqual(decoder.requested_indices[-1], 2999)
        self.assertTrue(np.all(np.diff(np.asarray(decoder.requested_indices)) > 0))

    def test_qwen2_keeps_legacy_factor_28_resize(self):
        decoder = _FakeVideoDecoder(total_frames=10, fps=2.0, height=56, width=84)
        resized = _FakeTensor((4, 3, 56, 84))
        with (
            patch.object(qwen_vl, "VideoDecoderWrapper", _FakeVideoDecoder),
            patch.object(
                qwen_vl.torchvision.transforms.functional,
                "resize",
                return_value=resized,
            ) as resize,
        ):
            video, _ = asyncio.run(
                qwen_vl.preprocess_video(decoder, video_config={"nframes": 4})
            )

        resize.assert_called_once()
        self.assertEqual(resize.call_args.args[1], [280, 392])
        self.assertIs(video, resized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
