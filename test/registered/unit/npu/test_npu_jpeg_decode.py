"""Real DVPP/Triton image contracts; no model loading or CPU device substitute."""

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

    def test_padded_nv12_matches_independent_cpu_arithmetic(self):
        from sglang.srt.hardware_backend.npu.jpeg.dvpp import JPEGInfo
        from sglang.srt.hardware_backend.npu.jpeg.nv12 import nv12_to_rgb_triton

        random = np.random.default_rng(42)
        for w, h, ws, hs in [(32, 32, 64, 32), (66, 34, 128, 48), (128, 64, 128, 64)]:
            with self.subTest(width=w, height=h):
                raw = random.integers(0, 256, ws * hs * 3 // 2, dtype=np.uint8)
                y = raw[: ws * hs].reshape(hs, ws)[:h, :w].astype(np.float32)
                uv = raw[ws * hs :].reshape(hs // 2, ws)[: h // 2, :w]
                uv = uv.reshape(h // 2, w // 2, 2).astype(np.float32)
                uv = uv.repeat(2, axis=0).repeat(2, axis=1) - np.float32(128)
                u, v = uv[..., 0], uv[..., 1]
                expected = np.stack(
                    [
                        y + np.float32(1.402) * v,
                        (y - np.float32(0.344136) * u) - np.float32(0.714136) * v,
                        y + np.float32(1.772) * u,
                    ]
                )
                expected = np.rint(expected).clip(0, 255).astype(np.uint8)
                packed = torch.from_numpy(raw).to(self.device)
                result = nv12_to_rgb_triton(packed, JPEGInfo(w, h, ws, hs))
                torch.npu.synchronize(self.device)
                self.assertEqual(result.dtype, torch.uint8)
                self.assertEqual(result.device, self.device)
                self.assertTrue(result.is_contiguous())
                np.testing.assert_array_equal(result.cpu().numpy(), expected)
                np.testing.assert_array_equal(packed.cpu().numpy(), raw)

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
