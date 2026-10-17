"""Unit tests for the Nano Nemotron VL processor."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.multimodal.evs import EVSConfig, EVSProcessor
from sglang.srt.multimodal.processors import nano_nemotron_vl
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
    _create_visual_items,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNanoNemotronVLProcessor(CustomTestCase):
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


def _fake_base_init(self, hf_config, server_args, _processor, *args, **kwargs):
    self._processor = _processor


class TestNanoNemotronVLImageProcessor(CustomTestCase):
    def test_init_keeps_pil_decompression_bomb_guard(self):
        hf_config = SimpleNamespace(
            image_size=512,
            patch_size=16,
            downsample_ratio=0.5,
            video_context_token="<video>",
            img_context_token="<image>",
            img_start_token="<img>",
            img_end_token="</img>",
            norm_mean=(0.5, 0.5, 0.5),
            norm_std=(0.5, 0.5, 0.5),
            use_thumbnail=True,
        )
        tokenizer = SimpleNamespace(
            unk_token="<unk>", convert_tokens_to_ids=lambda token: 0
        )
        original = Image.MAX_IMAGE_PIXELS
        try:
            with (
                patch.object(
                    nano_nemotron_vl.BaseMultimodalProcessor,
                    "__init__",
                    _fake_base_init,
                ),
                patch.object(nano_nemotron_vl, "EVSProcessor"),
                patch.object(
                    nano_nemotron_vl,
                    "get_model",
                    return_value=SimpleNamespace(context_length=8192),
                ),
                patch.object(
                    nano_nemotron_vl.MultimodalSpecialTokens,
                    "build",
                    lambda self, processor: self,
                ),
            ):
                NanoNemotronVLImageProcessor(hf_config, None, tokenizer)

            # Pillow's limit is process-global, so the processor must not change it
            self.assertEqual(Image.MAX_IMAGE_PIXELS, original)
            self.assertIsNotNone(Image.MAX_IMAGE_PIXELS)
        finally:
            Image.MAX_IMAGE_PIXELS = original


if __name__ == "__main__":
    unittest.main()
