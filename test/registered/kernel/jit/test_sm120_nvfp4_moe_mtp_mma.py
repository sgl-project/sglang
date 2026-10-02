"""Analytical small-M NVFP4 tests independent of the previous M1 implementation.

Power-of-two scales and FP4-representable activations give an exact closed-form
reference, including all three BF16 rounding boundaries. Real checkpoint
CUTLASS replay remains a separate numerical and quality integration gate.
"""

import unittest

import torch
from sglang.kernels.ops.gemm.sm120_nvfp4_moe_mtp_mma import nvfp4_moe_mtp_mma
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 required",
)
class TestNvfp4MoeMtpMma(CustomTestCase):
    def test_analytic_scales_routes_and_live_graph(self):
        torch.manual_seed(42)
        dev = "cuda"
        e, n, k, topk = 512, 2688, 1024, 22
        experts = torch.arange(e, device=dev)
        group = torch.arange(n // 16, device=dev)
        sf1_expert = 2.0 ** (experts % 3 - 1)
        sf1_group = torch.where(group % 2 == 0, 1.0, 0.5)
        sf2_expert = 2.0 ** (experts % 3 - 2)
        sf2_group = torch.where(group % 3 == 0, 0.5, 1.0)
        sf2_row = torch.where(torch.arange(k, device=dev) % 32 < 16, 1.0, 0.5)
        active = group % 3 != 0
        sign = torch.where(torch.arange(k, device=dev) % 16 < 8, 1.0, -1.0)
        w1 = torch.full((e, n, k // 2), 0x22, dtype=torch.uint8, device=dev)
        w1.view(e, n // 16, 16, k // 2)[:, ~active] = 0xAA
        w2 = torch.full((e, k, n // 2), 0x22, dtype=torch.uint8, device=dev)
        w2[:, sign < 0] = 0xAA
        sf1 = (
            (sf1_expert[:, None, None] * sf1_group.repeat_interleave(16)[None, :, None])
            .expand(e, n, k // 16)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        sf2 = (
            (
                sf2_expert[:, None, None]
                * sf2_row[None, :, None]
                * sf2_group[None, None, :]
            )
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        sw1, sw2 = swizzle_blockscale(sf1), swizzle_blockscale(sf2)
        a1 = torch.full((e,), 1 / 1024, device=dev)
        a2 = torch.full((e,), 1 / 2048, device=dev)
        global1 = torch.ones(1, device=dev)
        global2 = torch.ones(1, device=dev)
        # Exercise every supported M, including odd tails, with shared and
        # independent expert routes and changing input under each live graph.
        for m in range(1, 17):
            with self.subTest(m=m):
                x = torch.empty((m, k), dtype=torch.bfloat16, device=dev)
                ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
                routing = torch.empty((m, topk), device=dev)
                out = torch.empty_like(x)

                def reset(case):
                    values = (
                        torch.tensor([3.0, 6.0, 12.0, 0.0], device=dev)
                        .roll(case)
                        .repeat(4)[:m]
                    )
                    x.copy_(values[:, None].expand_as(x))
                    ids.copy_(
                        torch.stack(
                            [torch.randperm(e, device=dev)[:topk] for _ in range(m)]
                        )
                    )
                    if case == 2:
                        ids[1:].copy_(ids[:1].expand(m - 1, topk))
                    routing.copy_(
                        (2.0 ** ((torch.arange(m * topk, device=dev) + case) % 2)).view(
                            m, topk
                        )
                        / 32
                    )

                def reference():
                    # Each first-stage dot has K identical representable values.
                    selected = ids.long()
                    activation = (
                        x[:, 0].float()[:, None, None]
                        * sf1_expert[selected][:, :, None]
                        * sf1_group
                    ).square() * active
                    # Sixteen intermediate values share each scale, with half negative
                    # FC1 tiles suppressed by ReLU. All sums here are exact binary values.
                    dot = (activation * sf2_group).sum(-1) * 16
                    fc2 = (
                        dot[:, :, None]
                        * sf2_expert[selected][:, :, None]
                        * sf2_row
                        * sign
                        / 2048
                    )
                    terms = (fc2 * routing[:, :, None]).bfloat16()
                    return terms.float().sum(1).bfloat16()

                def candidate():
                    return nvfp4_moe_mtp_mma(
                        x,
                        w1,
                        sw1,
                        w2,
                        sw2,
                        a1,
                        a2,
                        global1,
                        global2,
                        ids,
                        routing,
                        out=out,
                    )

                reset(0)
                candidate()
                torch.testing.assert_close(out, reference(), rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    candidate()
                for case in range(4):
                    reset(case)
                    graph.replay()
                    torch.testing.assert_close(out, reference(), rtol=0, atol=0)
        for m in (0, 17):
            with self.subTest(unsupported_m=m):
                with self.assertRaisesRegex(Exception, "Expected 1 <= M <= 16"):
                    nvfp4_moe_mtp_mma(
                        torch.empty((m, k), dtype=torch.bfloat16, device=dev),
                        w1,
                        sw1,
                        w2,
                        sw2,
                        a1,
                        a2,
                        global1,
                        global2,
                        torch.zeros((m, topk), dtype=torch.int32, device=dev),
                        torch.zeros((m, topk), device=dev),
                    )


if __name__ == "__main__":
    unittest.main()
