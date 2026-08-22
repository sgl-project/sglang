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
        self.pinned = False

    def permute(self, *dims):
        self.shape = tuple(self.shape[dim] for dim in dims)
        return self

    def pin_memory(self):
        self.pinned = True
        return self


class _FakeVideoDecoder:
    def __init__(self, *, total_frames=4304, fps=30.0, height=5, width=7):
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
    def _assert_uses_existing_evenly_spaced_selection(self, fps):
        decoder = _FakeVideoDecoder()
        video_config = {"fps": fps}
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, metadata = asyncio.run(
                qwen3_vl.preprocess_video(decoder, video_config=video_config)
            )

        nframes = qwen_vl.smart_nframes(
            video_config,
            total_frames=decoder.total_frames,
            video_fps=decoder.avg_fps,
        )
        expected_indices = np.unique(
            np.linspace(
                0,
                decoder.total_frames - 1,
                num=nframes,
                dtype=np.int64,
            )
        )
        self.assertEqual(video.shape, (nframes, 3, 5, 7))
        self.assertTrue(video.pinned)
        np.testing.assert_array_equal(decoder.requested_indices, expected_indices)
        np.testing.assert_array_equal(metadata["frames_indices"], expected_indices)

    def test_qwen3_keeps_existing_evenly_spaced_selection_at_one_fps(self):
        self._assert_uses_existing_evenly_spaced_selection(fps=1)

    def test_qwen3_keeps_existing_evenly_spaced_selection_at_two_fps(self):
        self._assert_uses_existing_evenly_spaced_selection(fps=2)

    def test_qwen3_preserves_source_geometry_for_model_processor(self):
        decoder = _FakeVideoDecoder(total_frames=10, fps=2.0)
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, metadata = asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_config={"nframes": 4},
                )
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
        decoder = _FakeVideoDecoder(
            total_frames=10,
            fps=2.0,
            height=56,
            width=84,
        )
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
                qwen_vl.preprocess_video(
                    decoder,
                    video_config={"nframes": 4},
                )
            )

        resize.assert_called_once()
        self.assertEqual(resize.call_args.args[1], [280, 392])
        self.assertIs(video, resized)
        self.assertTrue(video.pinned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
