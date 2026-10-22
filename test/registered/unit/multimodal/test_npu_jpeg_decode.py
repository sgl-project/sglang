"""CPU contracts for opt-in NPU JPEG dispatch and decoder ownership."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import io
import os
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image, JpegImagePlugin

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.hardware_backend.npu.jpeg import dvpp
from sglang.srt.utils import common


def image_bytes(*, fmt="JPEG", mode="RGB", size=(64, 32), **options):
    output = io.BytesIO()
    Image.new(mode, size, 127).save(output, format=fmt, subsampling=2, **options)
    return output.getvalue()


class TestNpuJpegDispatch(unittest.TestCase):
    def setUp(self):
        self.data = image_bytes()

    def test_disabled_preserves_pil_and_never_calls_decoder(self):
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "0"}),
            patch.object(common, "is_npu", return_value=True),
            patch.object(common, "is_jpeg_with_cuda", return_value=False),
            patch.object(jpeg, "decode_jpeg") as decode,
        ):
            result = common._load_image(image_bytes=self.data)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (64, 32))
        decode.assert_not_called()

    def test_enabled_returns_decoder_tensor(self):
        output = torch.empty((3, 32, 64), dtype=torch.uint8)
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}),
            patch.object(common, "is_npu", return_value=True),
            patch.object(jpeg, "decode_jpeg", return_value=output) as decode,
        ):
            self.assertIs(common._load_image(image_bytes=self.data), output)
        decode.assert_called_once_with(self.data)

    def test_processor_disabled_and_fancy_keep_pil(self):
        for mode in [False, "nvjpeg_fancy"]:
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}),
                patch.object(common, "is_npu", return_value=True),
                patch.object(common, "is_jpeg_with_cuda", return_value=False),
                patch.object(jpeg, "decode_jpeg") as decode,
            ):
                result = common._load_image(
                    image_bytes=self.data, gpu_image_decode=mode
                )
                self.assertIsInstance(result, Image.Image)
                decode.assert_not_called()

    def test_non_npu_preserves_existing_loader(self):
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}),
            patch.object(common, "is_npu", return_value=False),
            patch.object(common, "is_jpeg_with_cuda", return_value=False),
            patch.object(jpeg, "decode_jpeg") as decode,
        ):
            self.assertIsInstance(
                common._load_image(image_bytes=self.data), Image.Image
            )
        decode.assert_not_called()

    def test_npu_runtime_failure_is_visible(self):
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}),
            patch.object(common, "is_npu", return_value=True),
            patch.object(
                jpeg, "decode_jpeg", side_effect=RuntimeError("device failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                common._load_image(image_bytes=self.data)

    def test_file_input_is_resolved_before_dispatch(self):
        output = object()
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}),
            patch.object(common, "is_npu", return_value=True),
            patch.object(common, "get_image_bytes", return_value=self.data) as read,
            patch.object(jpeg, "decode_jpeg", return_value=output) as decode,
        ):
            self.assertIs(common._load_image(image_file="image.jpg"), output)
        read.assert_called_once_with("image.jpg")
        decode.assert_called_once_with(self.data)


class TestNpuJpegEligibility(unittest.TestCase):
    def test_unsupported_headers_do_not_construct_decoder(self):
        exif = Image.Exif()
        exif[274] = 6
        cases = [
            image_bytes(fmt="PNG"),
            image_bytes(progressive=True),
            image_bytes(mode="L"),
            image_bytes(mode="CMYK"),
            image_bytes(size=(65, 32)),
            image_bytes(size=(16, 16)),
            image_bytes(exif=exif),
            image_bytes(icc_profile=b"profile"),
        ]
        with patch.object(jpeg, "_decoder", side_effect=AssertionError("device work")):
            for index, data in enumerate(cases):
                with self.subTest(index=index):
                    self.assertIsNone(jpeg.decode_jpeg(data))

    def test_valid_header_uses_current_device_without_cpu_pixel_decode(self):
        decoder = Mock()
        expected = object()
        decoder.decode.return_value = expected
        data = image_bytes()
        with (
            patch.object(jpeg, "_decoder", return_value=decoder) as get,
            patch.object(
                torch,
                "npu",
                types.SimpleNamespace(current_device=lambda: 5),
                create=True,
            ),
            patch.dict(sys.modules, {"torch_npu": types.ModuleType("torch_npu")}),
            patch.object(
                JpegImagePlugin.JpegImageFile,
                "load",
                side_effect=AssertionError("CPU pixel decode"),
            ),
        ):
            self.assertIs(jpeg.decode_jpeg(data), expected)
        get.assert_called_once_with(5)
        decoder.decode.assert_called_once_with(data)

    def test_malformed_complete_jpeg_remains_error(self):
        with self.assertRaises((OSError, ValueError)):
            jpeg.decode_jpeg(b"\xff\xd8broken\xff\xd9")

    def test_budget_failure_remains_error(self):
        with patch.object(jpeg, "inspect_jpeg", side_effect=ValueError("pixel budget")):
            with self.assertRaisesRegex(ValueError, "pixel budget"):
                jpeg.decode_jpeg(image_bytes())


class TestNpuJpegOwnership(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(jpeg, "_local", threading.local()),
            patch.object(jpeg, "_live", {}),
            patch.object(jpeg, "_closed", False),
            patch.object(jpeg, "_pid", os.getpid()),
            patch.object(dvpp, "TritonJPEGDecoder", side_effect=lambda device: Mock()),
        ]
        self.factory = None
        for item in self.patches:
            result = item.start()
            self.addCleanup(item.stop)
            if result is not None:
                self.factory = result
        self.addCleanup(jpeg.close_decoders)

    def test_same_worker_reuses_decoder_and_device_switch_closes_old(self):
        first = jpeg._decoder(3)
        self.assertIs(jpeg._decoder(3), first)
        second = jpeg._decoder(4)
        self.assertIsNot(second, first)
        first.close.assert_called_once()
        self.assertEqual(self.factory.call_count, 2)

    def test_terminated_worker_is_reclaimed_on_next_request(self):
        old = []
        worker = threading.Thread(target=lambda: old.append(jpeg._decoder(0)))
        worker.start()
        worker.join()
        old[0].close.assert_not_called()
        jpeg._decoder(0)
        old[0].close.assert_called_once()
        self.assertEqual(len(jpeg._live), 1)

    def test_close_failure_keeps_reference_for_retry(self):
        decoder = jpeg._decoder(0)
        decoder.close.side_effect = [RuntimeError("close failed"), None]
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            jpeg.close_decoders()
        self.assertEqual(len(jpeg._live), 1)
        jpeg.close_decoders()
        self.assertEqual(len(jpeg._live), 0)

    def test_terminal_close_rejects_reuse(self):
        decoder = jpeg._decoder(0)
        jpeg.close_decoders()
        jpeg.close_decoders()
        decoder.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            jpeg._decoder(0)

    def test_inherited_registry_is_not_used(self):
        with patch.object(jpeg, "_pid", -1):
            with self.assertRaisesRegex(RuntimeError, "spawn"):
                jpeg._decoder(0)


if __name__ == "__main__":
    unittest.main()
