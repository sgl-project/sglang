"""Unit tests for the DSV4 platform guards in build_kv_cache.

Lifting the CUDA rejection of --disaggregation-decode-enable-radix-cache for
DeepSeek-V4 must not silently open the NPU (separate c4/c128 allocators plus a
tree-level C128 sidecar) or HIP (per-request unified-kv SWA ring) layouts.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.kv_cache_builder import (
    _validate_decode_radix_dsv4_platform,
)
from sglang.test.test_utils import CustomTestCase


class TestValidateDecodeRadixDSV4Platform(CustomTestCase):
    def test_npu_platform_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_decode_radix_dsv4_platform(
                is_npu_platform=True, is_hip_platform=False
            )
        self.assertIn("NPU", str(ctx.exception))

    def test_hip_platform_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_decode_radix_dsv4_platform(
                is_npu_platform=False, is_hip_platform=True
            )
        self.assertIn("HIP", str(ctx.exception))

    def test_cuda_platform_allowed(self):
        result = _validate_decode_radix_dsv4_platform(
            is_npu_platform=False, is_hip_platform=False
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
