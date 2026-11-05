"""PIL-only consumers must opt out before device JPEG decoding occurs."""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.multimodal.cache import MediaSnapshot
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def encoded_image(mode="RGB", fmt="JPEG", size=(64, 48)):
    image = Image.new(mode, size, 100)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class PILPreprocessor:
    image_mean = (0.5, 0.5, 0.5)

    def __init__(self):
        self.size = {"shortest_edge": 32}
        self.crop_size = {"height": 32}

    def __call__(self, image):
        # These are real PIL operations, including the non-square pad/anyres path.
        image = image.convert("RGB").resize((32, 32))
        return {"pixel_values": [np.asarray(image).transpose(2, 0, 1)]}

    preprocess = __call__


class TestPILImageDecodePolicy(unittest.TestCase):
    def setUp(self):
        # Exercise the actual common loader on CPU CI. Only platform detection
        # and device decoding are replaced; a missed opt-out returns a Tensor.
        self.enterContext(patch.object(common, "is_cuda", return_value=False))
        self.enterContext(patch.object(common, "is_npu", return_value=True))
        self.decoded = torch.zeros((3, 48, 64), dtype=torch.uint8)
        self.decode = self.enterContext(
            patch.object(jpeg, "decode_jpeg", return_value=self.decoded)
        )

    def test_llava_cpu_worker_preserves_pil_preprocessing(self):
        data = encoded_image()
        processor = SimpleNamespace(image_processor=PILPreprocessor())
        for aspect in (None, "pad", "anyres"):
            with self.subTest(aspect=aspect):
                actual, image_hash, size = LlavaImageProcessor._preprocess_image_task(
                    data, 123, aspect, [[64, 64]], processor
                )
                # A CPU-only environment is the existing reference behavior.
                with patch.object(common, "is_npu", return_value=False):
                    expected, _, expected_size = (
                        LlavaImageProcessor._preprocess_image_task(
                            data, 123, aspect, [[64, 64]], processor
                        )
                    )
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float16)
                self.assertEqual(image_hash, 123)
                self.assertEqual(size, expected_size)
        self.decode.assert_not_called()

    def test_pil_policies_apply_to_ordinary_loading(self):
        data = encoded_image()
        for cls in (LlavaImageProcessor, PixtralProcessor):
            with self.subTest(processor=cls.__name__):
                ordinary = cls._load_single_item(data, Modality.IMAGE)
                self.assertIsInstance(ordinary, Image.Image)
                self.assertEqual(ordinary.mode, "RGB")
                self.assertEqual(ordinary.size, (64, 48))
        self.decode.assert_not_called()

    def test_pixtral_multi_image_rows_after_loading(self):
        processor = object.__new__(PixtralProcessor)
        processor.image_size = 1024
        processor._effective_patch_size = 28
        images = [
            processor._load_single_item(encoded_image(size=size), Modality.IMAGE)
            for size in ((56, 84), (84, 56))
        ]
        self.assertEqual(processor._get_image_nrows(images), [3, 2])
        self.decode.assert_not_called()

    def test_shared_tensor_path_remains_enabled(self):
        data = encoded_image()
        processor = object.__new__(QwenVLImageProcessor)
        ordinary = processor._load_single_item(data, Modality.IMAGE)
        snapshot = MediaSnapshot(data, "test-content", len(data), "test")
        cached = processor.decode_media_snapshot(snapshot, Modality.IMAGE)
        self.assertIs(ordinary, self.decoded)
        self.assertIs(cached, self.decoded)
        self.assertEqual(self.decode.call_count, 2)
        self.assertTrue(BaseMultimodalProcessor.npu_image_decode)
        self.assertTrue(BaseMultimodalProcessor.gpu_image_decode)

    def test_qwen_snapshot_honors_disabled_npu_policy(self):
        data = encoded_image()
        processor = object.__new__(QwenVLImageProcessor)
        processor.npu_image_decode = False
        snapshot = MediaSnapshot(data, "test-content", len(data), "test")
        image = processor.decode_media_snapshot(snapshot, Modality.IMAGE)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (64, 48))
        self.decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
