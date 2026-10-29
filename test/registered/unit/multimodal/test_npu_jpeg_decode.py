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

import numpy as np
import torch
from PIL import Image, JpegImagePlugin

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.hardware_backend.npu.jpeg import dvpp
from sglang.srt.hardware_backend.npu.jpeg.normalize import (
    inspect_420_sof,
    normalize_420_sof,
)
from sglang.srt.hardware_backend.npu.jpeg.vpc import VPCJPEGDecoder
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
            image_bytes(size=(16, 16)),
            image_bytes(size=(31, 32)),
            image_bytes(size=(32, 31)),
            image_bytes(size=(4097, 32)),
            image_bytes(size=(32, 4097)),
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

    def test_device_limits_keep_real_pil_fallback(self):
        small = image_bytes()
        # Valid COM segments increase encoded size without changing the pixels.
        comment = b"\xff\xfe\xff\xff" + b"x" * 65533
        large_encoded = small[:2] + comment * 513 + small[2:]
        self.assertGreater(len(large_encoded), 32 * 1024 * 1024)
        cases = [
            image_bytes(size=(4097, 32)),
            image_bytes(size=(32, 4097)),
            image_bytes(size=(4097, 4097)),
            large_encoded,
        ]
        with (
            patch.dict(os.environ, {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}),
            patch.object(common, "is_npu", return_value=True),
            patch.object(common, "is_jpeg_with_cuda", return_value=False),
            patch.object(jpeg, "_decoder", side_effect=AssertionError("device work")),
        ):
            for index, data in enumerate(cases):
                with (
                    self.subTest(index=index),
                    Image.open(io.BytesIO(data)) as expected,
                ):
                    result = common._load_image(image_bytes=data)
                    self.assertIsInstance(result, Image.Image)
                    self.assertEqual(result.size, expected.size)
                    for pixel in [(0, 0), (expected.width - 1, expected.height - 1)]:
                        self.assertEqual(
                            result.getpixel(pixel), expected.getpixel(pixel)
                        )

    def test_odd_and_vpc_boundary_dimensions_are_admitted(self):
        for w, h in [
            (32, 32),
            (65, 32),
            (64, 33),
            (65, 33),
            (4095, 32),
            (4096, 32),
            (32, 4096),
        ]:
            with self.subTest(size=(w, h)):
                info = dvpp.inspect_jpeg(image_bytes(size=(w, h)))
                self.assertEqual((info.width, info.height), (w, h))
                self.assertEqual(
                    (info.decode_width, info.decode_height), (w + w % 2, h + h % 2)
                )
                self.assertLessEqual(max(info.width_stride, info.height_stride), 4096)
                if (w, h) == (32, 32):
                    self.assertEqual((info.width_stride, info.height_stride), (64, 32))

    def test_nonbaseline_frame_falls_back_for_even_and_odd_sizes(self):
        for size in [(64, 32), (65, 33)]:
            data = bytearray(image_bytes(size=size))
            offset = inspect_420_sof(data, *size)
            data[offset - 4] = 0xC1  # Valid extended sequential SOF1.
            with (
                self.subTest(size=size),
                patch.object(
                    jpeg, "_decoder", side_effect=AssertionError("device work")
                ),
            ):
                self.assertIsNone(jpeg.decode_jpeg(bytes(data)))

    def test_malformed_sof_segment_is_an_error(self):
        data = bytearray(image_bytes())
        offset = inspect_420_sof(data, 64, 32)
        data[offset - 3 : offset - 1] = b"\x00\x01"
        with self.assertRaisesRegex(ValueError, "segment length"):
            inspect_420_sof(bytes(data), 64, 32)


class TestOddJpegAdaptation(unittest.TestCase):
    def test_only_private_sof_dimensions_change_and_valid_pixels_are_preserved(self):
        for w, h in [(65, 32), (64, 33), (65, 33)]:
            with self.subTest(size=(w, h)):
                y, x = np.indices((h, w))
                pixels = np.stack((x * 7, y * 11, (x + y) * 5), axis=-1).astype(
                    np.uint8
                )
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="JPEG", subsampling=2)
                data = buffer.getvalue()
                offset = inspect_420_sof(data, w, h)
                normalized = normalize_420_sof(data, w, h)
                self.assertEqual(data, buffer.getvalue())
                self.assertEqual(data[:offset], normalized[:offset])
                self.assertEqual(data[offset + 4 :], normalized[offset + 4 :])
                with (
                    Image.open(io.BytesIO(data)) as original,
                    Image.open(io.BytesIO(normalized)) as adapted,
                ):
                    self.assertEqual(original.size, (w, h))
                    self.assertEqual(adapted.size, (w + w % 2, h + h % 2))
                    np.testing.assert_array_equal(
                        np.asarray(original), np.asarray(adapted)[:h, :w]
                    )

    def test_even_dimensions_keep_the_original_bytes(self):
        data = image_bytes()
        self.assertIs(normalize_420_sof(data, 64, 32), data)


class TestVpcCleanup(unittest.TestCase):
    def test_cleanup_error_keeps_original_failure_and_releases_remaining_resources(
        self,
    ):
        decoder = VPCJPEGDecoder.__new__(VPCJPEGDecoder)
        decoder.torch, decoder.device = Mock(), object()
        decoder.torch.empty.side_effect = RuntimeError("allocation failure")
        a = Mock()
        decoder.decoder = types.SimpleNamespace(acl=a, stream=1, poisoned=False)
        decoder.channel = 2
        decoder._description = Mock(side_effect=[101, 102])
        a.media.dvpp_malloc.return_value = (17, 0)
        a.media.dvpp_vpc_convert_color_async.return_value = 0
        a.rt.synchronize_stream.return_value = 0
        a.media.dvpp_get_pic_desc_ret_code.return_value = 0
        for name, value in [
            ("width", 64),
            ("height", 32),
            ("width_stride", 192),
            ("height_stride", 32),
        ]:
            getattr(a.media, "dvpp_get_pic_desc_" + name).return_value = value
        a.media.dvpp_destroy_pic_desc.__name__ = "destroy_pic_desc"
        a.media.dvpp_destroy_pic_desc.side_effect = [9, 0]
        a.media.dvpp_free.return_value = 0
        frame = types.SimpleNamespace(
            ptr=7, size=3072, info=dvpp.JPEGInfo(64, 32, 64, 32)
        )
        with (
            self.assertWarns(UserWarning),
            self.assertRaisesRegex(RuntimeError, "allocation failure"),
        ):
            decoder._convert_frame(frame)
        self.assertEqual(
            [c.args for c in a.media.dvpp_destroy_pic_desc.call_args_list],
            [(102,), (101,)],
        )
        a.media.dvpp_free.assert_called_once_with(17)


class TestDeviceFrameOwnership(unittest.TestCase):
    def setUp(self):
        self.decoder = dvpp.JPEGDecoder.__new__(dvpp.JPEGDecoder)
        d = self.decoder
        d.lock, d.pid, d.context = threading.RLock(), os.getpid(), 1
        d.poisoned, d.acl = False, Mock()
        d.acl.rt.set_context.return_value = 0
        d.acl.media.dvpp_free.return_value = 0
        self.frame = dvpp.DeviceFrame(d, 17, 3072, dvpp.JPEGInfo(64, 32, 64, 32))
        d.frames = {self.frame}

    def test_nonpoisoned_unwind_releases_frame_and_keeps_original_error(self):
        original = RuntimeError("conversion failed")
        with self.assertRaises(RuntimeError) as caught:
            with self.frame:
                raise original
        self.assertIs(caught.exception, original)
        self.decoder.acl.media.dvpp_free.assert_called_once_with(17)
        self.assertEqual(self.frame.ptr, 0)
        self.assertNotIn(self.frame, self.decoder.frames)

    def test_poisoned_unwind_preserves_sync_error_and_retains_dma_memory(self):
        original = dvpp.ACLFailure("synchronize_stream VPC: ACL error 9")
        with self.assertRaises(dvpp.ACLFailure) as caught:
            with self.frame:
                self.decoder.poisoned = True
                raise original
        self.assertIs(caught.exception, original)
        self.decoder.acl.media.dvpp_free.assert_not_called()
        self.assertEqual(self.frame.ptr, 17)
        self.assertIn(self.frame, self.decoder.frames)
        with self.assertRaisesRegex(dvpp.ACLFailure, "terminate"):
            self.frame.close()
        with self.assertRaisesRegex(dvpp.ACLFailure, "terminate"):
            self.decoder.close()
        with self.assertRaisesRegex(dvpp.ACLFailure, "terminate"):
            with self.frame:
                pass
        self.decoder.acl.media.dvpp_free.assert_not_called()


class TestNpuJpegOwnership(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(jpeg, "_local", threading.local()),
            patch.object(jpeg, "_live", {}),
            patch.object(jpeg, "_closed", False),
            patch.object(jpeg, "_pid", os.getpid()),
            patch(
                "sglang.srt.hardware_backend.npu.jpeg.vpc.VPCJPEGDecoder",
                side_effect=lambda device: Mock(),
            ),
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
