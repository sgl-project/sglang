"""Deterministic mode refuses aiter prefill on an fp8 KV cache.

``--kv-cache-dtype auto`` passes the argument check, but becomes an fp8 KV cache at
load time when the checkpoint quantizes its KV cache to FP8; the backend checks the
resolved dtype.
"""

import unittest

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestAiterDeterministicFp8KvPrefillGuard(CustomTestCase):
    def test_raises_for_aiter_prefill_on_fp8_kv(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError) as ctx:
                    AiterAttnBackend._reject_deterministic_fp8_kv_prefill(
                        "aiter", dtype
                    )
                self.assertIn("bf16 KV cache", str(ctx.exception))

    def test_inert_for_bf16_kv_or_other_prefill_backend(self):
        AiterAttnBackend._reject_deterministic_fp8_kv_prefill("aiter", torch.bfloat16)
        AiterAttnBackend._reject_deterministic_fp8_kv_prefill(
            "triton", torch.float8_e4m3fn
        )
        AiterAttnBackend._reject_deterministic_fp8_kv_prefill(None, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
