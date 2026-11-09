"""The shared image policy controls both CUDA and NPU JPEG dispatch."""

import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.multimodal.cache import MediaSnapshot
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.embedding_gemma2 import (
    EmbeddingGemma2SGLangProcessor,
)
from sglang.srt.multimodal.processors.kimi_common import unwrap_kimi_images
from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def encoded_image(fmt="JPEG"):
    pixels = np.random.RandomState(0).randint(0, 256, (48, 64, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=fmt)
    return buffer.getvalue()


class TestNPUImageDecodeModes(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(common, "is_cuda", return_value=False))
        self.enterContext(patch.object(common, "is_npu", return_value=True))
        self.decode = self.enterContext(
            patch.object(jpeg, "decode_jpeg", side_effect=AssertionError("DVPP called"))
        )

    def assert_pil_pixels(self, image, data):
        self.assertIsInstance(image, Image.Image)
        reference = Image.open(io.BytesIO(data))
        self.assertEqual(image.mode, reference.mode)
        self.assertEqual(image.size, reference.size)
        self.assertEqual(image.tobytes(), reference.tobytes())

    def test_fancy_preserves_pil_pixels_across_input_forms(self):
        data = encoded_image()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(data)
            sources = [
                data,
                "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
            ]
            # The existing loader accepts POSIX absolute paths.
            if str(path).startswith("/"):
                sources.append(str(path))
            for source in sources:
                with self.subTest(source_type=type(source).__name__):
                    image, _ = common.load_image(
                        source, gpu_image_decode="nvjpeg_fancy"
                    )
                    self.assert_pil_pixels(image, data)
        self.decode.assert_not_called()

    def test_true_still_selects_the_npu_decoder(self):
        data = encoded_image()
        expected = torch.zeros((3, 48, 64), dtype=torch.uint8)
        self.decode.side_effect = None
        self.decode.return_value = expected
        image, _ = common.load_image(data, True)
        self.assertIs(image, expected)
        args, kwargs = self.decode.call_args
        self.assertEqual(bytes(args[0].numpy()), data)
        self.assertEqual(kwargs, {"device": "npu"})
        self.decode.assert_called_once()

    def test_false_uses_pil_without_dvpp(self):
        data = encoded_image()
        image, _ = common.load_image(data, False)
        self.assert_pil_pixels(image, data)
        self.decode.assert_not_called()

    def test_unsupported_npu_input_falls_back_to_pil(self):
        data = encoded_image()
        self.decode.side_effect = None
        self.decode.return_value = None
        image, _ = common.load_image(data, True)
        self.assert_pil_pixels(image, data)
        self.decode.assert_called_once()

    def test_png_and_cpu_do_not_select_npu_fancy(self):
        with patch.object(common.logger, "warning") as warning:
            data = encoded_image("PNG")
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")
            self.assert_pil_pixels(image, data)
            with patch.object(common, "is_npu", return_value=False):
                data = encoded_image()
                image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")
                self.assert_pil_pixels(image, data)
        warning.assert_not_called()
        self.decode.assert_not_called()

    def test_gpu_fancy_keeps_its_existing_decoder(self):
        data = encoded_image()
        expected = torch.zeros((3, 48, 64), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.object(common, "is_npu", return_value=False),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                return_value=expected,
            ) as cuda_decode,
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")
            self.assertIs(image, expected)
            cuda_decode.assert_called_once_with(data)
        self.decode.assert_not_called()

    def test_cuda_true_and_false_keep_existing_dispatch(self):
        data = encoded_image()
        expected = torch.zeros((3, 48, 64), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.object(common, "is_npu", return_value=False),
            patch("torchvision.io.decode_jpeg", return_value=expected) as cuda_decode,
        ):
            image, _ = common.load_image(data, True)
            self.assertIs(image, expected)
            args, kwargs = cuda_decode.call_args
            self.assertEqual(bytes(args[0].numpy()), data)
            self.assertEqual(kwargs, {"device": "cuda"})
            image, _ = common.load_image(data, False)
            self.assert_pil_pixels(image, data)
            cuda_decode.assert_called_once()
        self.decode.assert_not_called()

    def test_cuda_fancy_failure_returns_pil_not_plain_decode(self):
        data = encoded_image()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.object(common, "is_npu", return_value=False),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                side_effect=RuntimeError("test decoder unavailable"),
            ),
            patch("torchvision.io.decode_jpeg") as plain_decode,
        ):
            image, _ = common.load_image(data, "nvjpeg_fancy")
            self.assert_pil_pixels(image, data)
            plain_decode.assert_not_called()
        self.decode.assert_not_called()

    def test_model_policies_reach_ordinary_loading(self):
        data = encoded_image()
        for cls in (KimiK3ImageProcessor, EmbeddingGemma2SGLangProcessor):
            with self.subTest(processor=cls.__name__):
                self.assertEqual(cls.gpu_image_decode, "nvjpeg_fancy")
                image = cls._load_single_item(data, Modality.IMAGE)
                image = unwrap_kimi_images([image])[0]
                self.assert_pil_pixels(image, data)
        self.decode.assert_not_called()

    def test_kimi_snapshot_retains_fancy_policy(self):
        data = encoded_image()
        processor = object.__new__(KimiK3ImageProcessor)
        snapshot = MediaSnapshot(data, "test-content", len(data), "test")
        image = processor.decode_media_snapshot(snapshot, Modality.IMAGE)
        self.assert_pil_pixels(image, data)
        self.decode.assert_not_called()

    def test_fingerprint_preserves_the_mode_value(self):
        processor = SimpleNamespace(
            _processor=object(),
            gpu_image_decode=True,
            image_processor_backend="pil",
            mm_feature_transport="cpu",
            image_config={},
            video_config={},
            audio_config={},
        )
        ordinary = BaseMultimodalProcessor.preprocess_fingerprint_payload(processor)
        processor.gpu_image_decode = "nvjpeg_fancy"
        fancy = BaseMultimodalProcessor.preprocess_fingerprint_payload(processor)
        self.assertIs(ordinary["gpu_image_decode"], True)
        self.assertEqual(fancy["gpu_image_decode"], "nvjpeg_fancy")
        self.assertNotEqual(ordinary, fancy)
        processor.gpu_image_decode = False
        disabled = BaseMultimodalProcessor.preprocess_fingerprint_payload(processor)
        self.assertNotEqual(disabled, ordinary)
        self.assertNotEqual(disabled, fancy)


if __name__ == "__main__":
    unittest.main()
