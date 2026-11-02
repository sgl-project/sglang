"""The ROCm bf16 x bf16 -> fp32 GEMMs of DeepSeek-V4 keep fp32 accuracy."""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm import bf16_fp32
from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.gemm.router_gemv_hip import (
    ROCM_ROUTER_MAX_TOKENS,
    rocm_router_gemv_split_k,
    rocm_router_reduce_partials,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

NUM_EXPERTS, HIDDEN = 384, 5120


class TestAiterBf16Fp32Dispatch(unittest.TestCase):
    def test_enabled_aiter_uses_tgemm_fp32_linear_contract(self):
        x = object()
        weight = object()
        tgemm = mock.Mock()
        expected = object()
        tgemm.mm.return_value = expected

        with (
            mock.patch.object(envs.SGLANG_USE_AITER, "get", return_value=True),
            mock.patch.object(bf16_fp32, "_get_aiter_tgemm", return_value=tgemm),
            mock.patch.object(bf16_fp32, "_linear_bf16_fp32_cublas") as fallback,
        ):
            output = linear_bf16_fp32(x, weight)

        self.assertIs(output, expected)
        tgemm.mm.assert_called_once_with(x, weight, None, otype=torch.float32)
        fallback.assert_not_called()
        self.assertFalse(hasattr(bf16_fp32, "is_gfx1250_supported"))

    def test_disabled_or_unavailable_aiter_falls_back(self):
        x = object()
        weight = object()
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                mock.patch.object(envs.SGLANG_USE_AITER, "get", return_value=enabled),
                mock.patch.object(
                    bf16_fp32, "_get_aiter_tgemm", return_value=None
                ) as get_tgemm,
                mock.patch.object(
                    bf16_fp32, "_linear_bf16_fp32_cublas", return_value=enabled
                ) as fallback,
            ):
                self.assertEqual(linear_bf16_fp32(x, weight), enabled)
                if enabled:
                    get_tgemm.assert_called_once_with()
                else:
                    get_tgemm.assert_not_called()
                fallback.assert_called_once_with(x, weight)


@unittest.skipUnless(is_hip(), "ROCm only")
class TestRouterGemvHip(CustomTestCase):
    @unittest.skipUnless(is_gfx95_supported(), "ROCm gfx95 only")
    def test_gemv_accuracy_and_batch_invariance(self):
        """The split-K GEMV is within fp32 rounding of fp64, and every M runs the same
        16-row tile, so a row's result does not depend on the batch."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        x = torch.randn(
            ROCM_ROUTER_MAX_TOKENS, HIDDEN, device="cuda", generator=gen
        ).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(ROCM_ROUTER_MAX_TOKENS, NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        part = torch.empty(17, NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x[:17], weight), part)
        self.assertTrue(torch.equal(part, full[:17]))

    def test_linear_bf16_fp32_is_not_rounded_to_bf16(self):
        torch.manual_seed(0)
        for m, n, k, padding in (
            (37, 512, HIDDEN, 0),
            (1, 512, 7168, 0),
            (7, 1024, 7168, 128),
            (2048, 2048, 7168, 128),
        ):
            with self.subTest(m=m, n=n, k=k, padding=padding):
                x = torch.randn(m, k + padding, device="cuda", dtype=torch.bfloat16)
                x = x[:, :k]
                w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
                with envs.SGLANG_USE_AITER.override(True):
                    linear_bf16_fp32(x, w)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = linear_bf16_fp32(x, w)
                    for _ in range(2):
                        x.normal_()
                        graph.replay()
                        ref = x.double() @ w.double().t()
                        for out in (linear_bf16_fp32(x, w), captured):
                            self.assertEqual(out.dtype, torch.float32)
                            err = (out.double() - ref).abs().max()
                            bf16_err = (out.bfloat16().double() - ref).abs().max()
                            self.assertLess(err, bf16_err)
                            torch.testing.assert_close(
                                out.double(), ref, rtol=1e-4, atol=1e-4
                            )


if __name__ == "__main__":
    unittest.main()
