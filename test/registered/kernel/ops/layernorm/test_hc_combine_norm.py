import unittest

import torch

from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
from sglang.kernels.ops.layernorm.mhc import hc_combine
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestHcCombineNorm(CustomTestCase):
    def _check(self, x, pre, norm, actual):
        expected = norm(hc_combine(x, pre, 4, torch.bfloat16))
        self.assertTrue(torch.isfinite(actual).all().item())
        # The same BF16 combine intermediate is preserved. RMS reductions may
        # round differently, so check the aggregate error as well as each value.
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
        relative_l2 = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-20)
        self.assertLess(relative_l2.item(), 5e-5)

    def test_prefill_and_small_batches(self):
        torch.manual_seed(514)
        norm = RMSNorm(5120, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        norm.weight.data.normal_(1, 0.1)
        for rows in (1, 6, 4096, 4097, 65536):
            with self.subTest(rows=rows):
                # Exercise row strides and offsets beyond 2 GiB in the 64K case.
                x = torch.randn(rows, 20488, device="cuda", dtype=torch.bfloat16)[
                    :, :20480
                ]
                pre = torch.rand(rows, 8, device="cuda")[:, :4]
                actual = hc_combine_norm(x, pre, norm.weight, norm.variance_epsilon)
                self._check(x, pre, norm, actual)

    def test_graph_replay_and_zero_input(self):
        torch.manual_seed(516)
        rows = 4097
        norm = RMSNorm(5120, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(rows, 20480, device="cuda", dtype=torch.bfloat16)
        pre = torch.rand(rows, 4, device="cuda")
        hc_combine_norm(x, pre, norm.weight, norm.variance_epsilon)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = hc_combine_norm(x, pre, norm.weight, norm.variance_epsilon)
        for scale in (1.0, 0.01, 0.0):
            with self.subTest(scale=scale):
                x.normal_().mul_(scale)
                pre.uniform_()
                graph.replay()
                self._check(x, pre, norm, actual)


if __name__ == "__main__":
    unittest.main()
