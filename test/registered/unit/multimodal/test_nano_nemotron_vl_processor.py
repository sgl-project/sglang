"""Unit tests for the Nano Nemotron VL processor."""

import unittest

import torch

from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
from sglang.srt.multimodal.evs import EVSConfig, EVSProcessor
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
    _create_visual_items,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNanoNemotronVLProcessor(CustomTestCase):
    def test_supports_nemotron_h_omni(self):
        self.assertIn(
            NemotronH_Omni_Reasoning_V3,
            NanoNemotronVLImageProcessor.models,
        )

    def test_tubelet_label_matches_hf_reference(self):
        """Multi-frame tubelet labels must match the HF processor's prompt format."""
        processor = object.__new__(NanoNemotronVLImageProcessor)
        processor.PLACEHOLDER = "<unk>"
        processor.IMG_CONTEXT_TOKEN = "<image>"
        processor.IMG_END_TOKEN = "</img>"

        rendered = processor.render_tubelet(
            1, frame_indices=[2, 3], timestamps=[1.0, 1.5], num_tokens=2
        )

        self.assertEqual(
            rendered,
            "Frame 3 sampled at 1.00 seconds and frame 4 sampled at 1.50 seconds: "
            "<unk><image><image></img>",
        )

    def test_evs_video_next_to_dynamic_images_keeps_evs_metadata(self):
        """A video sent with dynamic-resolution images must still carry EVS metadata."""
        evs = EVSProcessor.__new__(EVSProcessor)
        evs.evs_config = EVSConfig(video_pruning_rate=0.5)
        create_data_items, _ = evs.static_size_data_items(
            frames_per_video=[2], num_images=1, rows=2, cols=2
        )
        input_ids = [1, 2, 3]

        image, video = _create_visual_items(
            create_data_items=create_data_items,
            image_feature=[torch.zeros(3, 32, 48)],
            image_offsets=[(0, 0)],
            num_tokens_per_image=[6],
            video_feature=torch.zeros(3, 3, 32, 32),
            video_offsets=[(1, 1), (2, 2)],
            frames_per_video=[3],
            input_ids_list=input_ids,
        )

        self.assertTrue(image.is_dynamic)
        self.assertEqual(image.num_tokens, 6)
        self.assertEqual(video.thw_grids, [(2, 2, 2)])
        self.assertEqual(video.pre_chunked_input_ids, input_ids)
        self.assertEqual(video.frames_per_video, [3])


if __name__ == "__main__":
    unittest.main()
