"""Unit tests for Step3VL image preprocessing, patching, and token replacement."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.multimodal.processors.step3_vl import (
    GPUToTensor,
    ImagePatcher,
    Step3VisionProcessor,
    Step3VLProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestStep3VLProcessor(CustomTestCase):
    def test_gpu_to_tensor_handles_grayscale_pil_and_hwc1_numpy(self):
        to_tensor = GPUToTensor()
        vision_proc = Step3VisionProcessor(size=28, patch_size=14)

        pil_gray = Image.new("L", (16, 12), color=128)
        pil_out = to_tensor(pil_gray)
        self.assertEqual(tuple(pil_out.shape), (3, 12, 16))
        self.assertEqual(
            tuple(vision_proc(pil_gray)["pixel_values"].shape), (1, 3, 28, 28)
        )

        np_hwc1 = np.full((12, 16, 1), 64, dtype=np.uint8)
        np_out = to_tensor(np_hwc1)
        self.assertEqual(tuple(np_out.shape), (3, 12, 16))
        self.assertEqual(np_out.dtype, torch.float32)

        with self.assertRaises(TypeError):
            to_tensor(Image.new("RGBA", (8, 8)))

        with self.assertRaises(TypeError):
            to_tensor(np.zeros((8, 8, 4), dtype=np.uint8))

    def test_image_patcher_avoids_redundant_resizes(self):
        patcher = ImagePatcher()

        # Small square image (<= 3024, window_size == 0): no resize needed at all
        small_img = Image.new("RGB", (504, 504))
        with patch.object(patcher, "resize", wraps=patcher.resize) as mock_resize:
            out_img, patches, newline_mask = patcher(small_img)
            self.assertEqual(mock_resize.call_count, 0)
            self.assertIs(out_img, small_img)
            self.assertEqual(patches, [])
            self.assertIsNone(newline_mask)

        # Medium image (<= 3024, already multiple of window_size=504): no resize needed
        medium_img = Image.new("RGB", (1008, 1008))
        with patch.object(patcher, "resize", wraps=patcher.resize) as mock_resize:
            out_img, patches, newline_mask = patcher(medium_img)
            self.assertEqual(mock_resize.call_count, 0)
            self.assertEqual(len(patches), 4)
            self.assertEqual(newline_mask, [False, True, False, False])

        # Oversized image (> 3024) whose preprocessed size (3024x3024) equals crop size:
        # should be resized exactly once during preprocess, not a second time before crop
        large_img = torch.zeros((3, 4000, 4000), dtype=torch.float32)
        with patch.object(patcher, "resize", wraps=patcher.resize) as mock_resize:
            out_img, patches, newline_mask = patcher(large_img)
            self.assertEqual(mock_resize.call_count, 1)
            self.assertEqual(patcher.get_image_size(out_img), (3024, 3024))
            self.assertEqual(len(patches), 36)

    def test_image_token_id_caches_vocab_lookup(self):
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {"<im_patch>": 42}
        tokenizer.convert_tokens_to_ids.side_effect = lambda tok: {
            "<patch_start>": 10,
            "<patch_end>": 11,
            "<patch_newline>": 12,
            "<im_start>": 20,
            "<im_end>": 21,
        }[tok]

        processor = Step3VLProcessor(config=MagicMock(), tokenizer=tokenizer)
        _, token_ids = processor._get_image_repl_features(
            num_images=1,
            num_patches=4,
            patch_new_line_idx=[False, True, False, False],
        )

        self.assertEqual(tokenizer.get_vocab.call_count, 1)
        self.assertEqual(
            token_ids.count(42),
            4 * processor.num_patch_feature_size + processor.num_image_feature_size,
        )


if __name__ == "__main__":
    unittest.main()
