"""NPU JPEG routing through the shared loader and model policy."""

import io
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.multimodal import media_io
from sglang.srt.multimodal.cache import MediaSnapshot
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.llava import LlavaImageProcessor
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNPUImageDecodeModes(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(media_io, "is_cuda", return_value=False))
        self.enterContext(patch.object(media_io, "is_npu", return_value=True))
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (30, 80, 160)).save(buffer, format="JPEG")
        self.data = buffer.getvalue()
        self.decode = self.enterContext(patch.object(jpeg, "decode_jpeg"))

    def assert_pil_pixels(self, image):
        self.assertIsInstance(image, Image.Image)
        with Image.open(io.BytesIO(self.data)) as reference:
            self.assertEqual(image.tobytes(), reference.tobytes())

    def test_disabled_and_fancy_bypass_dvpp(self):
        for mode in (False, "nvjpeg_fancy"):
            with self.subTest(mode=mode):
                image, _ = media_io.load_image(self.data, gpu_image_decode=mode)
                self.assert_pil_pixels(image)
        self.decode.assert_not_called()

    def test_unsupported_image_uses_pil(self):
        self.decode.return_value = None
        image, _ = media_io.load_image(self.data, True)
        self.assertIsNone(image.fp)
        self.assert_pil_pixels(image)
        self.decode.assert_called_once_with(self.data)

    def test_execution_error_reaches_the_caller(self):
        error = RuntimeError("NPU execution failed")
        self.decode.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            media_io.load_image(self.data, True)
        self.assertIs(caught.exception, error)

    def test_model_policy_reaches_ordinary_and_snapshot_loading(self):
        image = LlavaImageProcessor._load_single_item(self.data, Modality.IMAGE)
        self.assert_pil_pixels(image)
        self.decode.assert_not_called()

        tensor = torch.empty((3, 48, 64), dtype=torch.uint8)
        self.decode.return_value = tensor
        processor = object.__new__(QwenVLImageProcessor)
        self.assertIs(processor._load_single_item(self.data, Modality.IMAGE), tensor)
        self.decode.assert_called_once_with(self.data)
        self.decode.reset_mock()
        snapshot = MediaSnapshot(self.data, "test-content", len(self.data), "test")
        self.assertIs(processor.decode_media_snapshot(snapshot, Modality.IMAGE), tensor)
        self.decode.assert_called_once_with(self.data)


if __name__ == "__main__":
    unittest.main()
