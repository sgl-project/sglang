"""Unit tests for the NanoNemotron VL multimodal processor."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from sglang.srt.multimodal.processors import nano_nemotron_vl
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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
            with patch.object(
                nano_nemotron_vl.BaseMultimodalProcessor, "__init__", _fake_base_init
            ), patch.object(nano_nemotron_vl, "EVSProcessor"), patch.object(
                nano_nemotron_vl,
                "get_model",
                return_value=SimpleNamespace(context_length=8192),
            ), patch.object(
                nano_nemotron_vl.MultimodalSpecialTokens,
                "build",
                lambda self, processor: self,
            ):
                NanoNemotronVLImageProcessor(hf_config, None, tokenizer)

            # Pillow's limit is process-global, so the processor must not change it
            self.assertEqual(Image.MAX_IMAGE_PIXELS, original)
            self.assertIsNotNone(Image.MAX_IMAGE_PIXELS)
        finally:
            Image.MAX_IMAGE_PIXELS = original


if __name__ == "__main__":
    unittest.main()
