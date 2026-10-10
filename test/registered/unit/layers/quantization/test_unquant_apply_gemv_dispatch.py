"""
Unit tests for UnquantizedLinearMethod.apply's gemv-backend dispatch.

`--bf16-gemm-backend gemv` must route single-row CUDA BF16 linears through
`_bf16_gemm_dispatch_impl` the same way `cutedsl` does, keep the cuBLAS
fallbacks for the shapes the Hopper GEMV kernel does not serve, and be
rejected under --enable-deterministic-inference because it is batch-size
dependent.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.test_utils import CustomTestCase


def _gemv_backend():
    return patch.object(unquant, "_BF16_GEMM_BACKEND", unquant.Bf16GemmBackend.GEMV)


def _shape_gated_gemv():
    # The real predicate admits only single-row GEMMs.
    return patch.object(unquant, "_use_hopper_bf16_gemv", lambda m, n, k: m == 1)


def _recording_gemv(calls):
    def kernel(x, weight):
        calls.append(tuple(x.shape))
        return F.linear(x, weight)

    return kernel


class TestInitializeBf16GemmConfigGemvGuard(CustomTestCase):
    """gemv serves only M=1 GEMMs, so a request's result would depend on
    whether it was decoded alone; mirror cutedsl's deterministic rejection."""

    def test_gemv_rejected_under_deterministic_inference(self):
        exec_cfg = SimpleNamespace(
            kernel=SimpleNamespace(bf16_gemm_backend="gemv"),
            deterministic=SimpleNamespace(enable_deterministic_inference=True),
        )
        with (
            patch.object(unquant, "get_exec", return_value=exec_cfg),
            patch.object(
                unquant, "get_platform", return_value=SimpleNamespace(is_sm100=False)
            ),
        ):
            with self.assertRaisesRegex(ValueError, "deterministic-inference"):
                unquant.initialize_bf16_gemm_config()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestApplyGemvDispatch(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.projection = ReplicatedLinear(
            64, 128, bias=False, params_dtype=torch.bfloat16
        ).cuda()
        self.projection.weight.copy_(
            torch.randn_like(self.projection.weight) / 8.0  # 1 / sqrt(64)
        )
        self.method = self.projection.quant_method

    @torch.inference_mode()
    def test_single_row_dispatches_to_gemv(self):
        calls = []
        with (
            _gemv_backend(),
            _shape_gated_gemv(),
            patch.object(unquant, "_hopper_bf16_gemv", _recording_gemv(calls)),
        ):
            x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
            output = self.method.apply(self.projection, x)

        self.assertEqual(calls, [(1, 64)])
        torch.testing.assert_close(
            output, F.linear(x, self.projection.weight), rtol=0, atol=0
        )

    @torch.inference_mode()
    def test_larger_batches_fall_back_to_cublas(self):
        calls = []
        with (
            _gemv_backend(),
            _shape_gated_gemv(),
            patch.object(unquant, "_hopper_bf16_gemv", _recording_gemv(calls)),
        ):
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
            output = self.method.apply(self.projection, x)

        self.assertEqual(calls, [])
        torch.testing.assert_close(
            output, F.linear(x, self.projection.weight), rtol=0, atol=0
        )

    @torch.inference_mode()
    def test_biased_linear_falls_back_to_cublas(self):
        calls = []
        # A no-grad bias, as the gate requires; a live Parameter would
        # fail the requires_grad condition for the wrong reason.
        bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        with (
            _gemv_backend(),
            _shape_gated_gemv(),
            patch.object(unquant, "_hopper_bf16_gemv", _recording_gemv(calls)),
        ):
            x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
            output = self.method.apply(self.projection, x, bias)

        self.assertEqual(calls, [])
        torch.testing.assert_close(
            output, F.linear(x, self.projection.weight, bias), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
