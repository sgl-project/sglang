"""Unit tests for video frame-index sampling in qwen_vl ``preprocess_video``.

Frame indices must be rounded (not truncated) so the sampled frames match
qwen-vl-utils, which uses ``torch.linspace(...).round().long()``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.test_utils import CustomTestCase


def _fake_video(total_frames, fps=2.0):
    vr = MagicMock(spec=VideoDecoderWrapper)
    vr.__len__.return_value = total_frames
    vr.avg_fps = fps
    vr.get_frames_as_tensor.side_effect = lambda idx: torch.zeros(
        len(idx), 28, 28, 3, dtype=torch.uint8
    )
    return vr


class TestQwenVLFrameIndices(CustomTestCase):
    def _sampled_indices(self, total_frames, nframes):
        vr = _fake_video(total_frames)
        # is_cpu=True skips pin_memory, which needs CUDA.
        with patch(
            "sglang.srt.multimodal.processors.qwen_vl.is_cpu", return_value=True
        ):
            _, metadata = asyncio.run(
                preprocess_video(vr, video_config={"nframes": nframes})
            )
        requested = vr.get_frames_as_tensor.call_args.args[0]
        self.assertEqual(requested, metadata["frames_indices"].tolist())
        return requested

    def test_indices_are_rounded(self):
        # linspace(0, 11, 4) = [0, 3.67, 7.33, 11]; truncation would give 3 for 3.67.
        self.assertEqual(self._sampled_indices(12, 4), [0, 4, 7, 11])

    def test_matches_qwen_vl_utils(self):
        for total_frames, nframes in [(12, 4), (100, 6), (37, 8), (301, 16)]:
            with self.subTest(total_frames=total_frames, nframes=nframes):
                expected = (
                    torch.linspace(0, total_frames - 1, nframes).round().long()
                ).tolist()
                self.assertEqual(
                    self._sampled_indices(total_frames, nframes), expected
                )


if __name__ == "__main__":
    unittest.main()
