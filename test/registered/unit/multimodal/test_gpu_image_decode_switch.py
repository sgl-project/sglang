"""Unit test for the SGLANG_MM_GPU_IMAGE_DECODE switch in ``is_jpeg_with_cuda``."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.utils import common
from sglang.test.test_utils import CustomTestCase

JPEG = b"\xff\xd8" + b"\x00" * 16 + b"\xff\xd9"


class TestGpuImageDecodeSwitch(CustomTestCase):
    def test_default_keeps_gpu_jpeg_decode(self):
        with patch.object(common, "is_cuda", return_value=True):
            self.assertTrue(common.is_jpeg_with_cuda(JPEG, True))

    def test_switch_routes_jpeg_to_cpu(self):
        with patch.object(common, "is_cuda", return_value=True):
            with envs.SGLANG_MM_GPU_IMAGE_DECODE.override(False):
                self.assertFalse(common.is_jpeg_with_cuda(JPEG, True))

    def test_non_jpeg_unaffected(self):
        with patch.object(common, "is_cuda", return_value=True):
            self.assertFalse(common.is_jpeg_with_cuda(b"\x89PNG\r\n\x1a\n", True))


if __name__ == "__main__":
    unittest.main()
