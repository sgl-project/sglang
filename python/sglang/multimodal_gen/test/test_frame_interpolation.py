# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FrameInterpolator and interpolate_video_frames.

These tests exercise the full interpolation logic (tensor conversion, recursive
frame generation, frame count arithmetic) without network access or GPU by
patching FrameInterpolator._ensure_model_loaded with a lightweight mock model.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.postprocess import (
    FrameInterpolator,
    interpolate_video_frames,
)


def _make_mock_model():
    """Return a mock Model whose inference returns the midpoint of I0 and I1."""
    mock_model = MagicMock()
    mock_model.device.return_value = torch.device("cpu")
    mock_model.inference.side_effect = (
        lambda I0, I1, scale=1.0, timestep=0.5: (I0 + I1) / 2.0
    )
    return mock_model


def _make_frames(n, h=8, w=8):
    """Create n random uint8 frames of shape [h, w, 3]."""
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


class TestFrameInterpolatorUnit(unittest.TestCase):
    def setUp(self):
        self.mock_model = _make_mock_model()
        self.patcher = patch.object(
            FrameInterpolator, "_ensure_model_loaded", return_value=self.mock_model
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_interpolate_2x_frame_count(self):
        # exp=1: each adjacent pair gains 1 intermediate → 3 frames become 5
        frames = _make_frames(3)
        result, multiplier = FrameInterpolator().interpolate(frames, exp=1)
        self.assertEqual(len(result), 5)
        self.assertEqual(multiplier, 2)

    def test_interpolate_4x_frame_count(self):
        # exp=2: each adjacent pair gains 3 intermediates → 3 frames become 9
        frames = _make_frames(3)
        result, multiplier = FrameInterpolator().interpolate(frames, exp=2)
        self.assertEqual(len(result), 9)
        self.assertEqual(multiplier, 4)

    def test_interpolate_returns_correct_multiplier(self):
        frames = _make_frames(3)
        _, mult1 = FrameInterpolator().interpolate(frames, exp=1)
        self.assertEqual(mult1, 2**1)
        _, mult2 = FrameInterpolator().interpolate(frames, exp=2)
        self.assertEqual(mult2, 2**2)

    def test_interpolate_single_frame_passthrough(self):
        # fewer than 2 frames → returned unchanged, multiplier=1
        frames = _make_frames(1)
        result, multiplier = FrameInterpolator().interpolate(frames, exp=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(multiplier, 1)
        np.testing.assert_array_equal(result[0], frames[0])

    def test_interpolate_two_frames_2x(self):
        # 2 input frames with exp=1 → 3 output frames
        frames = _make_frames(2)
        result, multiplier = FrameInterpolator().interpolate(frames, exp=1)
        self.assertEqual(len(result), 3)
        self.assertEqual(multiplier, 2)

    def test_interpolate_preserves_frame_shape(self):
        h, w = 16, 32
        frames = _make_frames(2, h=h, w=w)
        result, _ = FrameInterpolator().interpolate(frames, exp=1)
        for frame in result:
            self.assertEqual(frame.shape, (h, w, 3))

    def test_interpolate_output_dtype_uint8(self):
        frames = _make_frames(2)
        result, _ = FrameInterpolator().interpolate(frames, exp=1)
        for frame in result:
            self.assertEqual(frame.dtype, np.uint8)

    def test_interpolate_scale_param_forwarded(self):
        frames = _make_frames(2)
        FrameInterpolator().interpolate(frames, exp=1, scale=0.5)
        self.assertTrue(self.mock_model.inference.called)
        for call in self.mock_model.inference.call_args_list:
            self.assertEqual(call.kwargs.get("scale"), 0.5)


class TestInterpolateVideoFrames(unittest.TestCase):
    def setUp(self):
        self.mock_model = _make_mock_model()
        self.patcher = patch.object(
            FrameInterpolator, "_ensure_model_loaded", return_value=self.mock_model
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_wrapper_delegates_correctly(self):
        # The convenience wrapper must produce the same frame count and multiplier
        # as calling FrameInterpolator directly.
        frames = _make_frames(3)
        direct_result, direct_mult = FrameInterpolator().interpolate(frames, exp=1)
        wrapper_result, wrapper_mult = interpolate_video_frames(frames, exp=1)
        self.assertEqual(len(wrapper_result), len(direct_result))
        self.assertEqual(wrapper_mult, direct_mult)


class TestSamplingParamsFrameInterpolation(unittest.TestCase):
    def test_default_disabled(self):
        sp = SamplingParams()
        self.assertFalse(sp.enable_frame_interpolation)

    def test_default_exp(self):
        sp = SamplingParams()
        self.assertEqual(sp.frame_interpolation_exp, 1)

    def test_default_scale(self):
        sp = SamplingParams()
        self.assertEqual(sp.frame_interpolation_scale, 1.0)

    def test_can_enable(self):
        sp = SamplingParams(enable_frame_interpolation=True)
        self.assertTrue(sp.enable_frame_interpolation)


if __name__ == "__main__":
    unittest.main()
