"""
Unit tests for UnquantizedLinearMethod.apply_with_addend.

The cuBLAS route folds the addend into the GEMM beta input,
writing back into that buffer; every other route must add separately
and leave the caller's buffer intact.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.test_utils import CustomTestCase

# addmm rounds once in the accumulator where the separate add rounds twice;
# with 8 BF16 mantissa bits the paths agree only to this precision.
_BF16_RTOL = 1e-2
_BF16_ATOL = 3.125e-2


def _cublas_only_backend():
    # The TORCH backend leaves every custom-kernel global unpopulated.
    return patch.object(unquant, "_BF16_GEMM_BACKEND", unquant.Bf16GemmBackend.TORCH)


def _fake_kernel(calls, name):
    def kernel(x, weight, bias=None, *args):
        calls.append(name)
        return F.linear(x, weight, bias)

    return kernel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestApplyWithAddend(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.projection = ReplicatedLinear(
            64, 128, bias=False, params_dtype=torch.bfloat16
        ).cuda()
        self.projection.weight.copy_(
            torch.randn_like(self.projection.weight) / 8.0  # 1 / sqrt(64)
        )
        self.method = self.projection.quant_method

    def _reference(self, x, addend, bias=None):
        return F.linear(x, self.projection.weight, bias) + addend

    @torch.inference_mode()
    def test_cublas_route_accumulates_into_addend(self):
        with _cublas_only_backend():
            x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
            addend = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
            reference = self._reference(x, addend)

            output = self.method.apply_with_addend(self.projection, x, addend)

            self.assertEqual(output.data_ptr(), addend.data_ptr())
            torch.testing.assert_close(
                output, reference, rtol=_BF16_RTOL, atol=_BF16_ATOL
            )

    @torch.inference_mode()
    def test_cuda_graph_replay_reads_the_replayed_addend(self):
        """A graph replay must consume the addend written on that replay;
        addmm(out=addend) reads and writes the one buffer."""
        with _cublas_only_backend():
            x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
            produced = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)

            # Initialize the cuBLAS workspace before capture.
            self.method.apply_with_addend(self.projection, x, produced.clone())
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # clone() stands in for the shared expert,
                # which rewrites its output buffer on every replay.
                captured = self.method.apply_with_addend(
                    self.projection, x, produced.clone()
                )

            x.copy_(torch.randn_like(x))
            produced.copy_(torch.randn_like(produced))
            reference = self._reference(x, produced)
            graph.replay()
            torch.cuda.synchronize()

            torch.testing.assert_close(
                captured, reference, rtol=_BF16_RTOL, atol=_BF16_ATOL
            )

    @torch.inference_mode()
    def test_batch_invariant_mode_keeps_separate_add(self):
        """Deterministic inference must not reach the fused route;
        batch-invariant mode does not override aten::addmm.out."""
        with (
            _cublas_only_backend(),
            patch.object(unquant, "is_batch_invariant_mode_enabled", return_value=True),
        ):
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
            addend = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
            before = addend.clone()

            output = self.method.apply_with_addend(self.projection, x, addend)

            self.assertNotEqual(output.data_ptr(), addend.data_ptr())
            torch.testing.assert_close(addend, before, rtol=0, atol=0)
            torch.testing.assert_close(
                output, self._reference(x, before), rtol=_BF16_RTOL, atol=_BF16_ATOL
            )

    @torch.inference_mode()
    def test_unfused_routes_leave_the_addend_intact(self):
        """Every route that cannot use the GEMM beta input adds separately;
        consuming the addend there corrupts the caller's buffer."""
        for route in (
            "non_nvidia",
            "compiling",
            "cutedsl",
            "splitk",
            "hopper_gemv",
            "noncontiguous_addend",
            "bias",
        ):
            with self.subTest(route=route):
                self._assert_route_is_unfused(route)

    def _assert_route_is_unfused(self, route: str):
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        addend = (
            torch.randn(128, 4, device="cuda", dtype=torch.bfloat16).t()
            if route == "noncontiguous_addend"
            else torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        )
        bias = (
            torch.randn(128, device="cuda", dtype=torch.bfloat16)
            if route == "bias"
            else None
        )
        before = addend.clone()
        kernel_calls = []

        with ExitStack() as stack:
            enter = stack.enter_context
            enter(_cublas_only_backend())
            if route == "non_nvidia":
                enter(patch.object(unquant, "_is_cuda", False))
            elif route == "compiling":
                enter(patch.object(torch.compiler, "is_compiling", return_value=True))
            elif route == "cutedsl":
                enter(patch.object(unquant, "_use_cutedsl_bf16_gemm", lambda *a: True))
                enter(
                    patch.object(
                        unquant,
                        "_cutedsl_bf16_gemm",
                        _fake_kernel(kernel_calls, route),
                    )
                )
            elif route == "splitk":
                enter(patch.object(unquant, "_enable_bf16_splitk_gemm", True))
                enter(
                    patch.object(
                        unquant, "use_flashinfer_pr4266_bf16_gemm", lambda *a: True
                    )
                )
                enter(
                    patch.object(
                        unquant,
                        "_flashinfer_pr4266_bf16_gemm",
                        _fake_kernel(kernel_calls, route),
                    )
                )
            elif route == "hopper_gemv":
                enter(patch.object(unquant, "_use_hopper_bf16_gemv", lambda *a: True))
                enter(
                    patch.object(
                        unquant,
                        "_hopper_bf16_gemv",
                        _fake_kernel(kernel_calls, route),
                    )
                )

            output = self.method.apply_with_addend(self.projection, x, addend, bias)

        self.assertNotEqual(output.data_ptr(), addend.data_ptr())
        torch.testing.assert_close(addend, before, rtol=0, atol=0)
        torch.testing.assert_close(
            output, self._reference(x, before, bias), rtol=_BF16_RTOL, atol=_BF16_ATOL
        )
        if route in ("cutedsl", "splitk", "hopper_gemv"):
            self.assertEqual(kernel_calls, [route])


if __name__ == "__main__":
    unittest.main()
