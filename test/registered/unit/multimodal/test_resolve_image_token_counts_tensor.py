"""Unit tests for image-size resolution on decoded image tensors.

``gpu_image_decode`` is on by default and returns a CHW ``uint8`` tensor for
JPEG payloads. ``resolve_image_token_counts`` must read the dimensions off such
a tensor, not only off PIL images, or ``SGLANG_MM_AVOID_RETOKENIZE`` silently
falls back to decode+retokenize for the most common image format.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    _image_height_width,
)
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # Only resolve_image_token_counts is exercised, so no model or GPU is
    # required -- but BaseMultimodalProcessor is an ABC, so the abstract method
    # still has to exist for the class to be instantiable.
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError

    def __init__(self, num_image_tokens):
        self._processor = MagicMock()
        self._processor._get_num_multimodal_tokens.return_value = SimpleNamespace(
            num_image_tokens=num_image_tokens
        )


class TestImageHeightWidth(CustomTestCase):
    def test_chw_tensor(self):
        self.assertEqual(_image_height_width(torch.zeros(3, 480, 640)), (480, 640))

    def test_nchw_tensor(self):
        self.assertEqual(_image_height_width(torch.zeros(1, 3, 480, 640)), (480, 640))

    def test_pil_like_object(self):
        self.assertEqual(
            _image_height_width(SimpleNamespace(height=480, width=640)), (480, 640)
        )

    def test_degenerate_tensor_rejected(self):
        with self.assertRaises(ValueError):
            _image_height_width(torch.zeros(7))


class TestResolveImageTokenCounts(CustomTestCase):
    def test_tensor_images_resolve_without_error(self):
        # Before the fix this raised AttributeError: 'Tensor' object has no
        # attribute 'height', which the caller swallowed into a retokenize
        # fallback.
        processor = _StubProcessor([12, 34])
        images = [torch.zeros(3, 480, 640), torch.zeros(3, 224, 224)]

        counts = processor.resolve_image_token_counts(images)

        self.assertEqual(counts, [12, 34])
        processor._processor._get_num_multimodal_tokens.assert_called_once_with(
            image_sizes=[(480, 640), (224, 224)]
        )

    def test_mixed_pil_and_tensor_images(self):
        processor = _StubProcessor([1, 2])
        images = [SimpleNamespace(height=32, width=64), torch.zeros(3, 480, 640)]

        counts = processor.resolve_image_token_counts(images)

        self.assertEqual(counts, [1, 2])
        processor._processor._get_num_multimodal_tokens.assert_called_once_with(
            image_sizes=[(32, 64), (480, 640)]
        )


if __name__ == "__main__":
    unittest.main()
