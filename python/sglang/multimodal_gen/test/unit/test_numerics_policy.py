# SPDX-License-Identifier: Apache-2.0
"""The worker states its approximate-numerics settings instead of inheriting them.

TF32 convolutions and bf16 reduced-precision reduction change what is computed,
so leaving them at whatever the installed PyTorch defaults to makes the exact
tier's contract depend on the build. The policy sets both explicitly.
"""

import sys
import unittest

import pytest
import torch

from sglang.multimodal_gen.runtime.utils.numerics_policy import apply_numerics_policy


class TestNumericsPolicy(unittest.TestCase):
    def setUp(self):
        self._cudnn_tf32 = torch.backends.cudnn.allow_tf32
        self._bf16_reduction = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )

    def tearDown(self):
        torch.backends.cudnn.allow_tf32 = self._cudnn_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            self._bf16_reduction
        )

    def test_both_switches_follow_the_policy(self):
        for tf32, bf16 in ((False, False), (True, True), (False, True)):
            with self.subTest(tf32=tf32, bf16=bf16):
                apply_numerics_policy(
                    allow_cudnn_tf32=tf32,
                    allow_bf16_reduced_precision_reduction=bf16,
                )
                self.assertEqual(torch.backends.cudnn.allow_tf32, tf32)
                self.assertEqual(
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    bf16,
                )

    def test_the_strict_setting_is_reachable_from_any_starting_state(self):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

        apply_numerics_policy(
            allow_cudnn_tf32=False,
            allow_bf16_reduced_precision_reduction=False,
        )

        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
