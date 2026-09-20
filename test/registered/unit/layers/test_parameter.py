"""CPU coverage for parameter loading dtype checks."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.parameter import BlockQuantScaleParameter, copy_with_check
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCopyWithCheck(CustomTestCase):
    def setUp(self):
        # E8M0 represents both values exactly, but FP16 overflows/underflows.
        self.scales = torch.tensor([1.0, 2.0**20, 2.0**-30]).to(torch.float8_e8m0fnu)

    def test_e8m0_allowed_destinations(self):
        for dtype in (torch.float8_e8m0fnu, torch.float32):
            with self.subTest(dtype=dtype):
                target = torch.empty(self.scales.shape, dtype=dtype)
                copy_with_check(target, self.scales)
                torch.testing.assert_close(
                    target.float(), self.scales.float(), rtol=0, atol=0
                )

    def test_e8m0_rejects_other_destinations_before_copy(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ):
            for allow_downcasting in ("0", "1"):
                with self.subTest(dtype=dtype, allow_downcasting=allow_downcasting):
                    target = torch.ones(self.scales.shape).to(dtype)
                    with patch.dict(
                        os.environ,
                        {"SGLANG_QUANT_ALLOW_DOWNCASTING": allow_downcasting},
                    ):
                        with self.assertRaises(AssertionError):
                            copy_with_check(target, self.scales)
                    torch.testing.assert_close(
                        target.float(), torch.ones(self.scales.shape), rtol=0, atol=0
                    )

    def test_column_scale_loader_applies_e8m0_check(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                param = BlockQuantScaleParameter(
                    data=torch.ones(1, 3, dtype=dtype),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=lambda *args: None,
                )
                if dtype == torch.float16:
                    with self.assertRaises(AssertionError):
                        param.load_column_parallel_weight(
                            self.scales.reshape(1, 3),
                            tp_rank=0,
                            use_presharded_weights=True,
                        )
                    torch.testing.assert_close(param.data, torch.ones_like(param))
                else:
                    param.load_column_parallel_weight(
                        self.scales.reshape(1, 3),
                        tp_rank=0,
                        use_presharded_weights=True,
                    )
                    torch.testing.assert_close(
                        param.data, self.scales.float().reshape(1, 3), rtol=0, atol=0
                    )

    def test_other_dtypes_keep_rank_policy(self):
        source = torch.tensor([1.0, 2.0, 4.0])
        for src_dtype, dst_dtype in (
            (torch.float16, torch.float16),
            (torch.float16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.float32),
            (torch.float32, torch.float64),
        ):
            with self.subTest(src_dtype=src_dtype, dst_dtype=dst_dtype):
                target = torch.empty(source.shape, dtype=dst_dtype)
                copy_with_check(target, source.to(src_dtype))
                torch.testing.assert_close(target.float(), source, rtol=0, atol=0)

        target = torch.empty(source.shape, dtype=torch.float16)
        with patch.dict(os.environ, {"SGLANG_QUANT_ALLOW_DOWNCASTING": "0"}):
            with self.assertRaisesRegex(ValueError, "Downcasting not allowed"):
                copy_with_check(target, source)
        with patch.dict(os.environ, {"SGLANG_QUANT_ALLOW_DOWNCASTING": "1"}):
            copy_with_check(target, source)
        torch.testing.assert_close(target.float(), source, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
