"""DeepSeek-OCR/OCR-2 local-crop geometry unit tests (no weights).

OCR-1 runs 640px local crops and OCR-2 runs 768px crops. Both checkpoints
declare candidate_resolutions=[[1024, 1024]] (the global base), so the local
crop size is selected from the vision encoder identity, not the config. At
768px a crop yields (768//16//4)**2 = 144 visual tokens, which is exactly the
length of the tuned query_768 embedding table in Qwen2Decoder2Encoder; at 640px
it yields 100 tokens, which falls back to an interpolated query table (off
design). These tests pin that mapping so a rewrite cannot silently put OCR-2
back on 640 crops or decouple the crop pixel size from the token budget.
"""

import math
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.srt.configs.deepseek_ocr import dynamic_preprocess
from sglang.srt.models.deepseek_ocr import _is_ocr2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

# Patch/downsample are copied from the checkpoints' processor_config.json
# (patch_size 16, downsample_ratio 4 for both DeepSeek-OCR and DeepSeek-OCR-2).
OCR_PATCH = 16
OCR_DOWNSAMPLE = 4


def _vis_tokens_per_crop(image_size: int) -> int:
    """Visual tokens one local crop expands to, mirroring the processor."""
    per_side = math.ceil((image_size // OCR_PATCH) / OCR_DOWNSAMPLE)
    return per_side * per_side


def _hf_config(vision_model_name: str, projector_input_dim) -> SimpleNamespace:
    return SimpleNamespace(
        vision_config=SimpleNamespace(model_name=vision_model_name),
        projector_config=SimpleNamespace(input_dim=projector_input_dim),
    )


class TestDeepseekOcrGeometry(CustomTestCase):
    def test_ocr2_selection_and_ocr1_untouched(self):
        # OCR-2 checkpoints match either signature; OCR-1 must stay 640.
        self.assertTrue(
            _is_ocr2(_hf_config("deepencoderv2", 896))
        )  # DeepSeek-OCR-2 (real config)
        self.assertFalse(
            _is_ocr2(_hf_config("deeplip_b_l", 2048))
        )  # DeepSeek-OCR (real config)
        self.assertFalse(_is_ocr2(_hf_config("", None)))

    def test_768_crop_tokens_are_144_and_640_are_100(self):
        # 768px local crop must hit the tuned query_768 table (144) and 1024px
        # the query_1024 table (256); 640px is the off-design 100-token case.
        self.assertEqual(_vis_tokens_per_crop(768), 144)
        self.assertEqual(_vis_tokens_per_crop(1024), 256)
        self.assertEqual(_vis_tokens_per_crop(640), 100)

    def test_dynamic_preprocess_crop_px_follows_image_size(self):
        wide = Image.new("RGB", (3072, 1024))  # aspect 3:1 -> 3x1 tile grid
        for image_size, expected_size in ((768, 768), (640, 640)):
            crops, ratio = dynamic_preprocess(
                image=wide, image_size=image_size, min_num=2, max_num=6
            )
            self.assertEqual(ratio, (3, 1))
            self.assertEqual(len(crops), 3)
            self.assertTrue(all(crop.size == (expected_size, expected_size) for crop in crops))


if __name__ == "__main__":
    unittest.main()
