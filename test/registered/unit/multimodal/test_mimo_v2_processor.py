"""Unit tests for MiMoProcessor configuration and partial-segment video handling."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.multimodal.processors.mimo_v2 import MiMoProcessor, VideoInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiMoV2Processor(CustomTestCase):
    def test_from_hf_config_preserves_explicit_zero_values(self):
        hf_config = SimpleNamespace(
            vision_config={"patch_size": 14, "spatial_merge_size": 2},
            processor_config={
                "image_min_pixels": 0,
                "video_min_pixels": 0,
                "min_frames": 0,
                "use_video_timestamps": True,
            },
            audio_config={"sampling_rate": 24000},
        )
        processor = MiMoProcessor.from_hf_config(hf_config)
        self.assertEqual(processor.default_image_processor_kwargs["min_pixels"], 0)
        self.assertEqual(processor.default_video_processor_kwargs["min_pixels"], 0)
        self.assertEqual(processor.default_video_processor_kwargs["min_frames"], 0)

    def test_process_video_partial_segment_single_frame_scalar_fps(self):
        hf_config = SimpleNamespace(
            vision_config={"patch_size": 14, "spatial_merge_size": 2},
            processor_config={"use_video_timestamps": True},
            audio_config={"sampling_rate": 24000},
        )
        processor = MiMoProcessor.from_hf_config(hf_config)

        # Single sampled frame triggers scalar fps_sampled = 1 (int), where
        # (1 / fps_sampled) is a Python float rather than a torch.Tensor.
        frames = torch.zeros((1, 3, 28, 28), dtype=torch.float32)
        timestamps = torch.tensor([0.5], dtype=torch.float32)
        video_input = VideoInput(
            video=(frames, timestamps),
            segment_type="partial",
            start_time=0.0,
            end_time=2.0,
        )

        visual_patches, thw_grid, aligned_ts, video_meta = processor.process_video(
            video_input
        )
        self.assertEqual(video_meta["fps_sampled"], 1)
        self.assertAlmostEqual(video_meta["segment_start_time"], 0.5)
        self.assertAlmostEqual(video_meta["segment_end_time"], 1.5)
        self.assertEqual(thw_grid.shape[0], 3)
        self.assertGreater(visual_patches.shape[0], 0)
        self.assertEqual(aligned_ts.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
