"""Real JPEGD/VPC contracts through the ordinary image-loading path."""

import importlib.util
import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_npu_ci

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401 - registers the NPU backend

register_npu_ci(est_time=15, suite="base-b-test-1-npu-a3")


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires an Ascend NPU"
)
class TestNpuJpegDecode(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu", torch.npu.current_device())

    def test_patterns_keep_odd_edges_and_owned_rgb_pixels(self):
        from sglang.srt.utils.common import _load_image

        retained = []
        with patch.dict("os.environ", {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}):
            for w, h in [
                (64, 32),
                (65, 32),
                (64, 33),
                (65, 33),
                (4095, 32),
                (32, 4095),
            ]:
                with self.subTest(width=w, height=h):
                    y, x = np.indices((h, w))
                    pixels = np.stack((x * 7, y * 11, (x + y) * 5), axis=-1).astype(
                        np.uint8
                    )
                    buffer = io.BytesIO()
                    Image.fromarray(pixels).save(
                        buffer, format="JPEG", quality=95, subsampling=2
                    )
                    data = buffer.getvalue()
                    with Image.open(io.BytesIO(data)) as image:
                        expected = (
                            np.asarray(image.convert("RGB")).transpose(2, 0, 1).copy()
                        )
                    result = _load_image(image_bytes=data)
                    self.assertEqual(result.device, self.device)
                    self.assertEqual(result.dtype, torch.uint8)
                    self.assertEqual(tuple(result.shape), (3, h, w))
                    self.assertTrue(result.is_contiguous())
                    np.testing.assert_array_equal(result.cpu().numpy(), expected)
                    retained.append((result, expected))
        for result, expected in retained:
            np.testing.assert_array_equal(result.cpu().numpy(), expected)

    def test_close_is_idempotent_and_preserves_the_callers_context(self):
        from sglang.srt.hardware_backend.npu.jpeg.vpc import VPCJPEGDecoder

        for _ in range(2):
            decoder = VPCJPEGDecoder(self.device.index)
            try:
                buffer = io.BytesIO()
                Image.new("RGB", (65, 33), (31, 127, 235)).save(
                    buffer, format="JPEG", subsampling=2
                )
                result = decoder.decode(buffer.getvalue())
            finally:
                decoder.close()
            decoder.close()
            self.assertEqual(result.sum().device, self.device)
            self.assertEqual(torch.ones(4, device=self.device).sum().item(), 4)

    def test_real_decode_and_concurrent_worker_reclamation(self):
        from sglang.srt.hardware_backend.npu import jpeg
        from sglang.srt.utils.common import _load_image

        def call(color):
            # Establish this worker's device, rather than assuming logical zero.
            torch.npu.set_device(self.device)
            buffer = io.BytesIO()
            Image.new("RGB", (64, 32), color).save(
                buffer, format="JPEG", quality=95, subsampling=2
            )
            result = _load_image(image_bytes=buffer.getvalue())
            self.assertEqual(result.device, self.device)
            self.assertEqual(result.dtype, torch.uint8)
            self.assertEqual(tuple(result.shape), (3, 32, 64))
            self.assertTrue(result.is_contiguous())
            actual = result.cpu().numpy().astype(np.int16)
            expected = np.array(color, dtype=np.int16)[:, None, None]
            self.assertLessEqual(int(np.abs(actual - expected).max()), 2)

        with patch.dict("os.environ", {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(call, [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * 4))
            # A subsequent request reclaims decoders whose owner threads ended.
            call((255, 0, 0))
        self.assertEqual(len(jpeg._live), 1)

    def test_unsupported_format_retains_pil_pixels(self):
        from sglang.srt.utils.common import _load_image

        buffer = io.BytesIO()
        Image.new("RGB", (65, 33), (31, 127, 235)).save(buffer, format="PNG")
        data = buffer.getvalue()
        with patch.dict("os.environ", {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "0"}):
            baseline = _load_image(image_bytes=data)
        with patch.dict("os.environ", {"SGLANG_ENABLE_NPU_IMAGE_DECODE": "1"}):
            enabled = _load_image(image_bytes=data)
        self.assertIsInstance(enabled, Image.Image)
        np.testing.assert_array_equal(np.asarray(enabled), np.asarray(baseline))


if __name__ == "__main__":
    unittest.main()
