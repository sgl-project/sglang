"""Validation for the raw fp8_e4m3 TileLang DSA backend on CUDA.

Regression: the combination used to boot the server and crash at decode
CUDA-graph capture with ``kernel main input KV dtype expected bfloat16,
but got float8_e4m3fn``.
"""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import _check_tilelang_dsa_fp8_kv
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaTilelangFp8Validation(CustomTestCase):

    def test_cuda_fp8_tilelang_decode_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_prefill_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "trtllm", hip=False)

    def test_cuda_fp8_tilelang_pair_allowed_on_hopper(self):
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_rejects_dcp(self):
        with self.assertRaisesRegex(ValueError, "dcp-size > 1"):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3",
                "tilelang",
                "tilelang",
                hip=False,
                dcp_size=2,
            )

    def test_hip_fp8_tilelang_allowed(self):
        # ROCm has a real fp8 tilelang kernel
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=True)

    def test_bf16_tilelang_allowed(self):
        # what the CUDA kernel expects
        _check_tilelang_dsa_fp8_kv("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_non_tilelang_allowed(self):
        # fp8-capable backends must pass
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "trtllm", hip=False)


if __name__ == "__main__":
    unittest.main()
