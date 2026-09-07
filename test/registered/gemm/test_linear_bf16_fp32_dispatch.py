"""Dispatch tests for the bf16 x bf16 -> fp32 GEMM helper — no server, no model.

Covers sglang.kernels.ops.attention.dsv4.gemm's choice between the fused
``torch.mm(..., out_dtype=torch.float32)`` call and the fp32-materializing
fallback. ``aten::mm.dtype`` is a per-backend overload, so the choice has to be
probed rather than assumed from the device name; these run on CPU, where the
overload is deliberately absent.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4 import gemm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLinearBf16Fp32Dispatch(CustomTestCase):
    def setUp(self):
        # functools.cache is process-global; clear on the way in and out so
        # neither ordering nor a selective run can see a stale answer.
        gemm._mm_out_dtype_supported.cache_clear()

    def tearDown(self):
        gemm._mm_out_dtype_supported.cache_clear()

    def test_cpu_bf16_falls_back_and_matches_the_fp32_reference(self):
        # Regression guard for the device gate: CPU has no aten::mm.dtype
        # kernel, so treating it as supported would raise instead of returning
        # fp32. This is the case an XPU-only test can never reach.
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        y = torch.randn(6, 8, dtype=torch.bfloat16)

        out = gemm._linear_bf16_fp32_cublas(x, y)

        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, torch.mm(x.float(), y.float().t()))

    def test_probe_declines_backends_without_the_overload(self):
        self.assertFalse(gemm._mm_out_dtype_supported("cpu"))
        self.assertFalse(gemm._mm_out_dtype_supported("privateuseone"))
        self.assertTrue(gemm._mm_out_dtype_supported("cuda"))

    def test_probe_declines_when_the_fused_call_raises(self):
        # The exact failure a wheel without the XPU kernel produces.
        stub = torch.ones(1, 1, dtype=torch.bfloat16)
        with patch.object(torch, "ones", return_value=stub), patch.object(
            torch, "mm", side_effect=NotImplementedError("no kernel for this backend")
        ):
            self.assertFalse(gemm._mm_out_dtype_supported("xpu"))

    def test_probe_declines_when_the_keyword_is_unknown(self):
        stub = torch.ones(1, 1, dtype=torch.bfloat16)
        with patch.object(torch, "ones", return_value=stub), patch.object(
            torch, "mm", side_effect=TypeError("unexpected keyword argument")
        ):
            self.assertFalse(gemm._mm_out_dtype_supported("xpu"))

    def test_probe_runs_once_and_caches_its_answer(self):
        stub = torch.ones(1, 1, dtype=torch.bfloat16)
        with patch.object(torch, "ones", return_value=stub) as allocate, patch.object(
            torch, "mm", side_effect=NotImplementedError("no kernel")
        ):
            self.assertFalse(gemm._mm_out_dtype_supported("xpu"))
            self.assertFalse(gemm._mm_out_dtype_supported("xpu"))

        self.assertEqual(allocate.call_count, 1)

    def test_fused_path_is_taken_when_the_probe_accepts(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        y = torch.randn(6, 8, dtype=torch.bfloat16)
        expected = torch.mm(x.float(), y.float().t())

        with patch.object(
            gemm, "_mm_out_dtype_supported", return_value=True
        ), patch.object(torch, "mm", return_value=expected) as fused:
            out = gemm._linear_bf16_fp32_cublas(x, y)

        self.assertIs(out, expected)
        self.assertEqual(fused.call_args.kwargs["out_dtype"], torch.float32)

    def test_mixed_dtypes_use_the_fallback_without_probing(self):
        # bf16 activations against an fp32 weight must not reach the fused
        # branch, and must not pay for the probe either.
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        y = torch.randn(6, 8, dtype=torch.float32)

        with patch.object(gemm, "_mm_out_dtype_supported") as probe:
            out = gemm._linear_bf16_fp32_cublas(x, y)

        probe.assert_not_called()
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, torch.mm(x.float(), y.float().t()))


if __name__ == "__main__":
    unittest.main()
