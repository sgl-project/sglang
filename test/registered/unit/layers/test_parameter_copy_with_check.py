"""CPU tests for the dtype rules of copy_with_check in srt/layers/parameter.py."""

import unittest

import torch

from sglang.srt.layers.parameter import copy_with_check
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCopyWithCheckE8M0(CustomTestCase):
    @staticmethod
    def _e8m0_scales():
        # Exponent bytes for 1.0, 2.0, 0.5, 8.0.
        exponent_bytes = torch.tensor([127, 128, 126, 130], dtype=torch.uint8)
        return exponent_bytes.view(torch.float8_e8m0fnu)

    def test_e8m0_scale_is_rejected_by_other_narrow_float_targets(self):
        """An E8M0 scale must not be value-cast into an fp8 or 16-bit float
        parameter; only an E8M0 or float32 target may receive it."""
        scales = self._e8m0_scales()
        for target_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.bfloat16,
            torch.float16,
        ):
            with self.subTest(target_dtype=target_dtype):
                target = torch.zeros(4, dtype=target_dtype)
                with self.assertRaises(ValueError):
                    copy_with_check(target, scales)

    def test_e8m0_scale_loads_into_e8m0_and_float32(self):
        scales = self._e8m0_scales()

        same_dtype = torch.zeros(4, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        copy_with_check(same_dtype, scales)
        self.assertTrue(
            torch.equal(same_dtype.view(torch.uint8), scales.view(torch.uint8))
        )

        as_float32 = torch.zeros(4, dtype=torch.float32)
        copy_with_check(as_float32, scales)
        self.assertEqual(as_float32.tolist(), [1.0, 2.0, 0.5, 8.0])


if __name__ == "__main__":
    unittest.main()
