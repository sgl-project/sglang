"""Stress rounding and graph replay against the current TileLang post kernel."""

import unittest

import torch

from sglang.kernels.ops.layernorm.mhc import mhc_post
from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMhcPostSplitH(CustomTestCase):
    def test_exact_eager_and_graph(self):
        with get_parallel().override(attn_cp_size=1):
            for batch in (1, 4, 7, 8, 16, 32, 64):
                with self.subTest(batch_size=batch):
                    x = torch.empty(batch, 5120, device="cuda", dtype=torch.bfloat16)
                    residual = torch.empty(
                        batch, 4, 5120, device="cuda", dtype=torch.bfloat16
                    )
                    post = torch.empty(batch, 4, device="cuda")
                    comb = torch.empty(batch, 4, 4, device="cuda")
                    inputs = (x, residual, post, comb)
                    for value in inputs:
                        value.normal_()
                    mhc_post_split_h(*inputs)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = mhc_post_split_h(*inputs)
                    for seed in range(30):
                        torch.manual_seed(1000 + seed)
                        for value in inputs:
                            value.normal_()
                        expected = mhc_post(*inputs)
                        actual = mhc_post_split_h(*inputs)
                        graph.replay()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
