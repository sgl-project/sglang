"""Regression tests for Qwen3-VL multimodal feature materialization."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingVisual:
    device = torch.device("meta")
    dtype = torch.bfloat16

    def __init__(self):
        self.pixel_values = None
        self.grid_thw = None

    def __call__(self, pixel_values, *, grid_thw):
        self.pixel_values = pixel_values
        self.grid_thw = grid_thw
        return pixel_values


class TestQwen3VLFeatureMaterialization(CustomTestCase):
    def test_image_features_are_packed_on_the_visual_device(self):
        visual = _RecordingVisual()
        model = SimpleNamespace(visual=visual, use_data_parallel=False)
        items = [
            SimpleNamespace(
                feature=torch.ones(2, 3),
                image_grid_thw=torch.tensor([[1, 1, 2]]),
            ),
            SimpleNamespace(
                feature=torch.ones(1, 3),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
            ),
        ]
        output = Qwen3VLForConditionalGeneration.get_image_feature(model, items)

        self.assertIs(visual.pixel_values, output)
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(output.device, visual.device)
        self.assertEqual(output.dtype, visual.dtype)

    def test_video_features_are_packed_on_the_visual_device(self):
        visual = _RecordingVisual()
        model = SimpleNamespace(visual=visual, use_data_parallel=False)
        items = [
            SimpleNamespace(
                feature=torch.ones(3, 4),
                video_grid_thw=torch.tensor([[1, 1, 3]]),
            ),
            SimpleNamespace(
                feature=torch.ones(2, 4),
                video_grid_thw=torch.tensor([[1, 1, 2]]),
            ),
        ]
        output = Qwen3VLForConditionalGeneration.get_video_feature(model, items)

        self.assertIs(visual.pixel_values, output)
        self.assertEqual(output.shape, (5, 4))
        self.assertEqual(output.device, visual.device)
        self.assertEqual(output.dtype, visual.dtype)
        self.assertTrue(
            torch.equal(visual.grid_thw, torch.tensor([[1, 1, 3], [1, 1, 2]]))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
