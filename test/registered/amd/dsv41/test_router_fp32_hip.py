"""V4 router scores retain distinctions below one BF16 output ULP."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "requires HIP")
class TestRouterFp32(unittest.TestCase):
    def setUp(self):
        from sglang.srt.models.deepseek_v2 import MoEGate
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.forward = MoEGate.forward

    def test_close_scores_and_mutable_graph(self):
        # Exact BF16 operands produce 16 distinct scores near 1; rounding the
        # GEMM output to BF16 would collapse them before expert selection.
        weight = torch.zeros(384, 5120, device="cuda", dtype=torch.bfloat16)
        weight[:, 0] = 1
        weight[:16, 1] = torch.arange(16, device="cuda") / 4096
        gate = SimpleNamespace(
            weight=weight, is_deepseek_v4=True, tiny_router_gemm_max_tokens=0
        )
        for rows in (64, 512):
            with self.subTest(rows=rows):
                x = torch.zeros(rows, 5120, device="cuda", dtype=torch.bfloat16)
                x[:, :2] = 1
                self.forward(gate, x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.forward(gate, x)
                for sign in (1, -1):
                    x[:, 1] = sign
                    graph.replay()
                    expected = (
                        1
                        + sign
                        * torch.arange(16, device="cuda", dtype=torch.float32)
                        / 4096
                    )
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(
                        output[:, :16], expected.expand(rows, -1), rtol=0, atol=0
                    )
                    self.assertEqual(torch.unique(output[0, :16]).numel(), 16)


if __name__ == "__main__":
    unittest.main()
