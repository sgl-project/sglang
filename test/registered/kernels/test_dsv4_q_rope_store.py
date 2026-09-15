import unittest

import torch

from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
from sglang.kernels.ops.attention.dsv4.q_rope_store import q_rope_store
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestQRopeStore(CustomTestCase):
    def test_exact_output_and_padding(self):
        torch.manual_seed(911)
        freqs = torch.polar(
            torch.ones(8192, 32, device="cuda"), torch.randn(8192, 32, device="cuda")
        )
        for rows in (1, 2, 5, 6, 8):
            for heads in (8, 16, 32):
                for dtype in (torch.int32, torch.int64):
                    q = torch.randn(
                        rows, heads + 1, 512, device="cuda", dtype=torch.bfloat16
                    )[:, :heads]
                    padding = torch.full(
                        (rows, 64, 512), 7.0, device="cuda", dtype=q.dtype
                    )
                    output = padding[:, :heads]
                    positions = torch.randint(
                        0, 8192, (rows,), device="cuda", dtype=dtype
                    )
                    original = q.clone()
                    expected = q.clone()
                    fused_rope_inplace(expected[..., 448:], None, freqs, positions)
                    q_rope_store(q, output, freqs, positions)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    torch.testing.assert_close(q, original, rtol=0, atol=0)
                    self.assertTrue((padding[:, heads:] == 7).all().item())

    def test_graph_replay(self):
        q = torch.randn(6, 16, 512, device="cuda", dtype=torch.bfloat16)
        output = torch.zeros(6, 64, 512, device="cuda", dtype=q.dtype)[:, :16]
        freqs = torch.polar(
            torch.ones(8192, 32, device="cuda"), torch.randn(8192, 32, device="cuda")
        )
        positions = torch.arange(4096, 4102, device="cuda")
        q_rope_store(q, output, freqs, positions)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q_rope_store(q, output, freqs, positions)
        for _ in range(3):
            q.normal_()
            positions.random_(0, 8192)
            graph.replay()
            expected = q.clone()
            fused_rope_inplace(expected[..., 448:], None, freqs, positions)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
