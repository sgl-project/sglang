"""DeepSeek-OCR / OCR-2 local-crop geometry unit tests (no weights).

DeepSeek-OCR-2 must run its 768px local crops, not the 640px ones inherited from
the DeepSeek-OCR processor: at 768px a crop expands to (768 // 16 // 4) ** 2 ==
144 visual tokens, exactly the length of the tuned query_768 embedding table in
Qwen2Decoder2Encoder, while 640px yields 100 and falls back to an interpolated
table.

The crop size is not in the checkpoints -- both ship identical processor configs
with no `image_size` and candidate_resolutions=[[1024, 1024]] (the *global*
base) -- so `local_crop_size` selects it from the model config and
`apply_ocr_geometry` patches it onto the processor. These tests fail if OCR-2 is
put back on 640px crops, at either the policy or the wiring.

Not covered here: the in-processor thresholds that consume `image_size` (the
`img_w <= image_size` early-out and the global-view resize) need a real
tokenizer, so they are only observable end to end.
"""

import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.srt.configs.deepseek_ocr import (
    IMAGE_SIZE,
    OCR2_IMAGE_SIZE,
    dynamic_preprocess,
    local_crop_size,
)
from sglang.srt.multimodal.processors.deepseek_ocr import apply_ocr_geometry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _hf_config(vision_model_name: str, projector_input_dim) -> SimpleNamespace:
    return SimpleNamespace(
        vision_config=SimpleNamespace(model_name=vision_model_name),
        projector_config=SimpleNamespace(input_dim=projector_input_dim),
    )


def _ocr2_config() -> SimpleNamespace:
    """DeepSeek-OCR-2: DeepEncoder V2 vision encoder, 896-dim projector."""
    return _hf_config("deepencoderv2", 896)


def _ocr1_config() -> SimpleNamespace:
    """DeepSeek-OCR: deeplip_b_l vision encoder, 2048-dim projector."""
    return _hf_config("deeplip_b_l", 2048)


class TestDeepseekOcrGeometry(CustomTestCase):
    def test_local_crop_size_is_768_for_ocr2_and_640_for_ocr1(self):
        # Guards the processor's per-model selection: OCR-2 must not be served
        # with OCR-1's 640px crops, and OCR-1 must stay on 640.
        self.assertEqual(local_crop_size(_ocr2_config()), 768)
        self.assertEqual(local_crop_size(_ocr2_config()), OCR2_IMAGE_SIZE)
        self.assertEqual(local_crop_size(_ocr1_config()), 640)
        self.assertEqual(local_crop_size(_ocr1_config()), IMAGE_SIZE)

    def test_processor_geometry_wiring(self):
        # The processor patches both attributes onto the HF processor; assert the
        # patching itself so a regression in the call site is caught too.
        stub = SimpleNamespace()
        apply_ocr_geometry(stub, _ocr2_config())
        self.assertEqual(stub.image_size, 768)
        self.assertTrue(stub.ocr2_mode)
        apply_ocr_geometry(stub, _ocr1_config())
        self.assertEqual(stub.image_size, 640)
        self.assertFalse(stub.ocr2_mode)

    def test_dynamic_preprocess_crop_px_follows_image_size(self):
        # `tokenize_with_images` passes the processor's `image_size` through, so
        # the crop pixels follow whichever crop size was selected above.
        wide = Image.new("RGB", (3072, 1024))  # aspect 3:1 -> 3x1 tile grid
        for image_size, expected_size in ((768, 768), (640, 640)):
            crops, ratio = dynamic_preprocess(
                image=wide, image_size=image_size, min_num=2, max_num=6
            )
            self.assertEqual(ratio, (3, 1))
            self.assertEqual(len(crops), 3)
            self.assertTrue(
                all(crop.size == (expected_size, expected_size) for crop in crops)
            )


if __name__ == "__main__":
    unittest.main()
