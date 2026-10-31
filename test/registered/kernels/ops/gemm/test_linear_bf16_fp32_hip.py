"""Old ROCm Torch must return FP32 GEMM results without mixed-output BLAS support.

This uses a different CI contract from the Hopper-only HPC-Ops tests: exercise
the HIP compatibility path with real GPU arithmetic and graph replay.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm.bf16_fp32 import _linear_bf16_fp32_cublas
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "ROCm required")
class TestLinearBf16Fp32Hip(CustomTestCase):
    def _inputs(self):
        torch.manual_seed(0)
        # Binary fractions make the FP32 product exact while still exposing an
        # incorrect BF16 output followed by a cast to FP32.
        x = torch.randint(-32, 33, (17, 128), device="cuda").bfloat16() / 8
        weight = torch.randint(-32, 33, (64, 128), device="cuda").bfloat16() / 8
        reference = torch.mm(x.float(), weight.float().t())
        self.assertTrue((reference != reference.bfloat16().float()).any().item())
        return x, weight, reference

    def test_legacy_hip_eager_and_graph_preserve_fp32(self):
        x, weight, reference = self._inputs()
        real_mm = torch.mm

        def legacy_mm(*args, **kwargs):
            if "out_dtype" in kwargs:
                raise RuntimeError(
                    "BF16 input and FP32 output is not supported for ROCm"
                )
            return real_mm(*args, **kwargs)

        # Model the framework boundary, not the SGLang capability predicate, so
        # removing the version guard fails even on newer CI images.
        with (
            patch.object(torch, "__version__", "2.9.0a0+git7bcbafe"),
            patch.object(torch, "mm", side_effect=legacy_mm),
        ):
            eager = _linear_bf16_fp32_cublas(x, weight)
            self.assertEqual(eager.dtype, torch.float32)
            torch.testing.assert_close(eager, reference, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _linear_bf16_fp32_cublas(x, weight)
                _linear_bf16_fp32_cublas(x, weight)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _linear_bf16_fp32_cublas(x, weight)
            graph.replay()
            torch.testing.assert_close(captured, reference, rtol=0, atol=0)

    def test_supported_framework_keeps_bf16_inputs(self):
        x, weight, reference = self._inputs()
        real_mm = torch.mm
        try:
            real_mm(x, weight.t(), out_dtype=torch.float32)
        except RuntimeError as exc:
            if "not supported for ROCm" in str(exc):
                self.skipTest("This Torch build lacks native mixed-output GEMM")
            raise

        def native_mm(a, b, **kwargs):
            self.assertEqual(a.dtype, torch.bfloat16)
            self.assertEqual(b.dtype, torch.bfloat16)
            self.assertEqual(kwargs.get("out_dtype"), torch.float32)
            return real_mm(a, b, **kwargs)

        with patch.object(torch, "mm", side_effect=native_mm):
            actual = _linear_bf16_fp32_cublas(x, weight)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
