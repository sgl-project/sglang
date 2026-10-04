"""Numerical and graph-replay contract for full-precision router weights."""

import unittest

import torch

from sglang.kernels.ops.gemm.fp32_router import fp32_router_gemm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "SM90+ required",
)
class TestFp32Router(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_fp64_reference(self):
        w = torch.randn(256, 5120, device="cuda") / 5120**0.5
        for m in (1, 3, 8, 16, 32, 64):
            with self.subTest(m=m):
                x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
                result = fp32_router_gemm(x, w)
                self.assertEqual(result.dtype, torch.float32)
                reference = (x.double() @ w.double().T).float()
                torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-5)

    def test_weights_remain_fp32(self):
        # All these weights round to 1 in BF16. Preserving their low FP32 bits
        # is necessary to distinguish the two highest-scoring experts.
        x = torch.zeros(1, 5120, device="cuda", dtype=torch.bfloat16)
        x[:, 0] = 1
        w = torch.ones(256, 5120, device="cuda")
        w[:, 0] += torch.arange(256, device="cuda") * 2**-20
        result = fp32_router_gemm(x, w)
        torch.testing.assert_close(result[0], w[:, 0], atol=0, rtol=0)
        self.assertEqual(result.argmax().item(), 255)
        self.assertEqual(torch.unique(w[:, 0].bfloat16()).numel(), 1)

    def test_graph_observes_input_and_weight_updates(self):
        x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 5120, device="cuda") / 5120**0.5
        fp32_router_gemm(x, w)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fp32_router_gemm(x, w)
        for _ in range(3):
            x.normal_()
            w.normal_(std=5120**-0.5)
            graph.replay()
            torch.testing.assert_close(
                result, (x.double() @ w.double().T).float(), atol=2e-6, rtol=2e-5
            )

    def test_rejects_invalid_layout_and_dtype(self):
        x = torch.zeros(1, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.zeros(256, 5120, device="cuda")
        for bad_x, bad_w in (
            (x.float(), w),
            (x, w.bfloat16()),
            (torch.zeros(1, 10240, device="cuda", dtype=torch.bfloat16)[:, ::2], w),
            (x, w.T.contiguous().T),
            (
                torch.zeros(5121, device="cuda", dtype=torch.bfloat16)[1:].view(
                    1, 5120
                ),
                w,
            ),
        ):
            with self.assertRaises(Exception):
                fp32_router_gemm(bad_x, bad_w)


if __name__ == "__main__":
    unittest.main()
