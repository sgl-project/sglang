"""Lightweight (no-GPU) wiring guards for the MoRI blockwise-FP4 combine dtype.

Protects the sglang-side plumbing that exposes ``SGLANG_MORI_COMBINE_DTYPE=fp4`` and maps it to
MoRI's ``fp4_blockwise`` combine. The end-to-end kernel behaviour is covered by the GPU/eval tests.
"""

import unittest

from sglang.srt.layers.moe.token_dispatcher.moriep import CombineDtype


class TestMoriCombineDtypeFp4(unittest.TestCase):
    def test_fp4_enum_member_exists(self):
        self.assertTrue(hasattr(CombineDtype, "fp4"))
        self.assertEqual(CombineDtype.fp4.value, "fp4_blockwise")

    def test_fp4_value_roundtrip(self):
        self.assertIs(CombineDtype("fp4_blockwise"), CombineDtype.fp4)

    def test_existing_combine_dtypes_unchanged(self):
        # Guard against accidentally breaking the existing dtypes when adding fp4.
        self.assertEqual(CombineDtype.bf16.value, "bfloat16")
        self.assertEqual(CombineDtype.fp8.value, "float8_blockwise")
        self.assertEqual(CombineDtype.fp8_direct_cast.value, "float8_direct_cast")


if __name__ == "__main__":
    unittest.main()
