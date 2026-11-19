"""Public JPEG input, upload and small-image warning contracts on CPU CI."""

import ctypes
import io
import logging
import unittest
import weakref
from contextlib import contextmanager
from unittest.mock import patch

import torch
from _dvpp_test_utils import encoded_jpeg, load_dvpp_modules, simulated_npu
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _encoded_tensor(data):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


class TestNpuJpegInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header, cls.dvpp, cls.jpeg, cls.cache_module = load_dvpp_modules(cls)

    @contextmanager
    def cached_decoders(self):
        cache = self.cache_module._DvppCache()
        with patch.object(self.cache_module, "cached_decoders", cache):
            try:
                yield
            finally:
                cache.close()

    def test_upload_reads_correct_bytes_and_only_rewrites_odd_dimensions(self):
        for size in ((64, 48), (65, 49)):
            original = encoded_jpeg(size)
            storage = torch.zeros(len(original) + 7, dtype=torch.uint8)
            sliced = storage[7:]
            sliced.copy_(_encoded_tensor(original))
            for data in (original, sliced):
                with self.subTest(size=size, input=type(data)), simulated_npu() as rt:
                    pointer = (
                        rt.bytes_to_ptr(data)
                        if isinstance(data, bytes)
                        else data.data_ptr()
                    )
                    uploads = []

                    def observe_copy(dst, capacity, src, length, kind):
                        if kind == self.dvpp.ACL_MEMCPY_HOST_TO_DEVICE:
                            uploads.append((src, ctypes.string_at(src, length)))
                        return rt.memcpy(dst, capacity, src, length, kind)

                    with (
                        patch.object(rt.rt, "memcpy", observe_copy),
                        self.dvpp.Dvpp() as decoder,
                    ):
                        decoder.process_jpeg_dec(data)
                    self.assertEqual(len(uploads), 1)
                    address, uploaded = uploads[0]
                    if size == (64, 48):
                        self.assertEqual((address, uploaded), (pointer, original))
                    else:
                        start = self.header.inspect_jpeg(original).dimension_offset
                        expected = (
                            original[:start]
                            + bytes.fromhex("0032 0042")
                            + original[start + 4 :]
                        )
                        self.assertEqual(uploaded, expected)
                        self.assertNotEqual(address, pointer)
                    self.assertEqual(bytes(data), original)

    def test_temporary_tensor_owner_lives_through_jpeg_synchronization(self):
        def temporary_input():
            nonlocal owner
            tensor = _encoded_tensor(encoded_jpeg((64, 48)))
            owner = weakref.ref(tensor)
            return tensor

        owner = None
        with simulated_npu() as rt, self.dvpp.Dvpp() as decoder:

            def synchronize(stream):
                if rt.inflight[0] == "jpeg":
                    self.assertIsNotNone(owner())
                return rt.synchronize_stream(stream)

            with patch.object(rt.rt, "synchronize_stream", synchronize):
                decoder.process_jpeg_dec(temporary_input())
        self.assertIsNone(owner())

    def test_warning_uses_encoded_pixel_area(self):
        for size, should_warn in (
            ((105, 105), True),
            ((359, 361), True),
            ((360, 360), True),
            ((361, 361), False),
            ((180, 720), True),
            ((720, 180), True),
            ((180, 721), False),
        ):
            with (
                self.subTest(size=size),
                simulated_npu(),
                patch.object(self.cache_module.cached_decoders, "get") as factory,
                patch.object(self.jpeg.logger, "warning_once") as warning,
            ):
                data = encoded_jpeg(size)
                self.jpeg.decode_jpeg(data)
                self.assertEqual(warning.call_count, int(should_warn))
                factory.return_value._decode_prepared_jpeg.assert_called_once_with(
                    data, self.header.inspect_jpeg(data)
                )

    def test_warns_once_before_attempt_even_if_decode_fails(self):
        data = encoded_jpeg()
        logger = logging.Logger(__name__)
        with (
            simulated_npu(),
            patch.object(self.jpeg, "logger", logger),
            patch.object(logger, "warning") as warning,
            patch.object(self.cache_module.cached_decoders, "get") as factory,
        ):

            def fail_decode(data, info):
                warning.assert_called_once()
                raise RuntimeError("decoder failure")

            factory.return_value._decode_prepared_jpeg.side_effect = fail_decode
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                    self.jpeg.decode_jpeg(data)
            self.assertIn("129,600", warning.call_args.args[0])

    def test_noncontiguous_tensor_uses_requested_device_and_restores_caller(self):
        encoded = _encoded_tensor(encoded_jpeg())
        storage = torch.empty(2 * encoded.numel(), dtype=torch.uint8)
        storage[::2] = encoded
        with simulated_npu() as runtime, self.cached_decoders():
            image = self.jpeg.decode_jpeg(storage[::2], device="npu:1")
            self.assertEqual(image.shape, (3, 49, 65))
            self.assertIn(("architecture", 1), runtime.events)
            self.assertEqual(runtime.current, 3)

    def test_default_device_is_restored_on_failure(self):
        with simulated_npu((("copy_d2d", 1),)) as runtime, self.cached_decoders():
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "D2D"):
                self.jpeg.decode_jpeg(encoded_jpeg())
            self.assertIn(("architecture", 3), runtime.events)
            self.assertEqual(runtime.current, 3)

    def test_public_metadata_fallback_without_device_construction(self):
        png = io.BytesIO()
        Image.new("RGB", (64, 48)).save(png, format="PNG")
        exif = Image.Exif()
        exif[274] = 6
        cases = (
            png.getvalue(),
            encoded_jpeg(progressive=True),
            encoded_jpeg(exif=exif),
            encoded_jpeg(icc_profile=b"profile"),
            encoded_jpeg((16, 16)),
        )
        with (
            patch.object(self.jpeg.logger, "warning_once") as warning,
            patch.object(
                self.cache_module.cached_decoders,
                "get",
                side_effect=AssertionError("decoder constructed"),
            ),
        ):
            for encoded in cases:
                with self.subTest(length=len(encoded)):
                    self.assertIsNone(self.jpeg.decode_jpeg(_encoded_tensor(encoded)))
            warning.assert_not_called()

    def test_public_rejects_invalid_tensor_contract(self):
        cases = (
            torch.zeros(10, dtype=torch.float32),
            torch.zeros((2, 8), dtype=torch.uint8),
            torch.empty(8, dtype=torch.uint8, device="meta"),
        )
        with patch.object(
            self.cache_module.cached_decoders,
            "get",
            side_effect=AssertionError("decoder constructed"),
        ):
            for data in cases:
                with (
                    self.subTest(
                        dtype=data.dtype, shape=data.shape, device=data.device
                    ),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    self.jpeg.decode_jpeg(data)

    def test_unsupported_chip_and_invalid_target(self):
        data = encoded_jpeg()
        with simulated_npu(arch=3510) as runtime:
            self.assertIsNone(self.jpeg.decode_jpeg(data, device="npu:1"))
            with self.assertRaisesRegex(ValueError, "NPU device"):
                self.jpeg.decode_jpeg(data, device="cpu")
            self.assertEqual(runtime.events, [("architecture", 1)])

    def test_architecture_query_failure_precedes_allocation(self):
        with simulated_npu((("architecture", 1),)) as runtime:
            with self.assertRaisesRegex(self.dvpp.ACLFailure, "query NPU architecture"):
                self.jpeg.decode_jpeg(encoded_jpeg(), device="npu:1")
            self.assertEqual(runtime.events, [("architecture", 1)])


if __name__ == "__main__":
    unittest.main()
