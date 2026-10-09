"""Single-token Beam GEMV against the deferred Triton MoE outputs it replaces.

The GEMV sums in a different FP32 order, so it keeps the screen it was merged
with rather than bit equality. A CUDA graph replay must read the expert IDs,
routing weights and activations present at replay, not at capture.
"""

import unittest

import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-gb300")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
    "SM103 only",
)
class TestBeamDecodeGemv(CustomTestCase):
    def test_matches_deferred_moe_under_graph_replay(self):
        from sglang.kernels.ops.moe.beam_decode_gemv import beam_decode_gemv
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )

        torch.manual_seed(0)
        w13 = torch.randn(256, 1280, 5120, dtype=torch.bfloat16, device="cuda")
        w13 /= 5120**0.5
        w2 = torch.randn(256, 5120, 640, dtype=torch.bfloat16, device="cuda")
        w2 /= 640**0.5

        def route(ids_dtype):
            x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda")
            ids = torch.randperm(256, device="cuda")[:8].view(1, 8).to(ids_dtype)
            weights = torch.rand(1, 8, device="cuda")
            return x, ids, weights / weights.sum()

        def check(actual, x, ids, weights):
            with (
                get_context().override_server_args(
                    enable_deterministic_inference=False
                ),
                get_parallel().override(tp_group=None),
            ):
                expected = fused_experts_impl(
                    x,
                    w13,
                    w2,
                    weights,
                    ids.to(torch.int32),
                    inplace=False,
                    filter_expert=False,
                    defer_sum=True,
                    fuse_swiglu_interleaved=True,
                )
            actual, expected = actual.float(), expected.float()
            torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.016)
            rel_rms = (actual - expected).square().mean() / expected.square().mean()
            self.assertLess(rel_rms.sqrt().item(), 0.005)

        for ids_dtype in (torch.int32, torch.int64):
            with self.subTest(ids_dtype=ids_dtype):
                x, ids, weights = route(ids_dtype)
                check(beam_decode_gemv(x, w13, w2, weights, ids), x, ids, weights)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = beam_decode_gemv(x, w13, w2, weights, ids)
                for _ in range(3):
                    new_inputs = route(ids_dtype)
                    for static, value in zip((x, ids, weights), new_inputs):
                        static.copy_(value)
                    graph.replay()
                    check(out, *new_inputs)


if __name__ == "__main__":
    unittest.main()
