"""Validate raw TileLang FP8 KV against incompatible backends and platforms."""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import _check_tilelang_dsa_fp8_kv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDsaTilelangFp8Validation(CustomTestCase):
    def test_cuda_fp8_tilelang_decode_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_prefill_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "trtllm", hip=False)

    @patch("torch.cuda.get_device_capability", return_value=(9, 0))
    def test_cuda_fp8_tilelang_allowed(self, _capability):
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_cuda_fp8_requires_fp8_tensor_cores(self, _capability):
        with self.assertRaisesRegex(ValueError, "SM89"):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_dcp_rejected(self):
        with self.assertRaisesRegex(ValueError, "dcp-size"):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3", "tilelang", "tilelang", hip=False, dcp_size=2
            )

    def test_hip_fp8_tilelang_allowed(self):
        # ROCm has a real fp8 tilelang kernel
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=True)

    def test_bf16_tilelang_allowed(self):
        _check_tilelang_dsa_fp8_kv("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_non_tilelang_allowed(self):
        # fp8-capable backends must pass
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "trtllm", hip=False)


if __name__ == "__main__":
    unittest.main()
