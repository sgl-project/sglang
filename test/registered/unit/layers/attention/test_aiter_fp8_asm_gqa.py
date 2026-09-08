"""Tests for the AITER FP8 FMHA ASM GQA routing guard."""

import unittest

from sglang.srt.layers.attention.aiter_backend import _aiter_fp8_asm_supports_gqa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAiterFp8AsmSupportsGqa(unittest.TestCase):
    def test_supported_asm_gqa_ratios(self):
        for ratio in (1, 2, 4, 8, 16):
            with self.subTest(ratio=ratio):
                self.assertTrue(_aiter_fp8_asm_supports_gqa(ratio * 4, 4))

    def test_rejects_qwen38_27b_gqa6(self):
        self.assertFalse(_aiter_fp8_asm_supports_gqa(24, 4))

    def test_rejects_other_unsupported_ratios(self):
        for num_q_heads, num_kv_heads in ((12, 4), (32, 1), (24, 0), (23, 4), (0, 4)):
            with self.subTest(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads):
                self.assertFalse(_aiter_fp8_asm_supports_gqa(num_q_heads, num_kv_heads))


if __name__ == "__main__":
    unittest.main()
