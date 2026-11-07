"""gfx950 check for the decode-sized Triton MoE sort (sglang.kernels.ops.moe.moe_sorting_small) against the aiter
kernel it stands in for: the stage-1 MXFP4 quant it emits. Shapes are GLM-5.3-Flash's (288 routed experts, top-8,
plus the optional fused shared expert, hidden 4096) and Qwen3-30B-A3B's (128 experts, top-8, hidden 2048)."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")

# (num_experts, topk, hidden)
SHAPES = ((288, 8, 4096), (289, 9, 4096), (128, 8, 2048))


@unittest.skipUnless(is_gfx95_supported(), "the MXFP4 small sort is gfx950 only")
class TestMoeSortingSmallMxfp4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.moe import moe_sorting_small

        cls.mss = moe_sorting_small
        cls.dev = torch.device("cuda", 0)

    def setUp(self):
        torch.manual_seed(0)

    def _sort_buffers(self, m, topk, block_size, num_experts, hidden):
        max_pad = m * topk + num_experts * block_size - topk
        return (
            torch.empty(max_pad, dtype=torch.int32, device=self.dev),
            torch.empty(max_pad, dtype=torch.float32, device=self.dev),
            torch.empty(
                (max_pad + block_size - 1) // block_size,
                dtype=torch.int32,
                device=self.dev,
            ),
            torch.empty(2, dtype=torch.int32, device=self.dev),
            torch.empty(m, hidden, dtype=torch.bfloat16, device=self.dev),
        )

    def test_emitted_mxfp4_matches_aiter_bit_for_bit(self):
        from aiter.ops.quant import fused_dynamic_mxfp4_quant_moe_sort

        for num_experts, topk, hidden in SHAPES:
            scale_n = hidden // 32
            scale_n_pad = (scale_n + 7) // 8 * 8
            for block_size in (16, 32):
                for m in (1, 3, 4, 8, 16, 28, 32):
                    if m * topk > 256:
                        continue
                    with self.subTest(
                        num_experts=num_experts,
                        topk=topk,
                        hidden=hidden,
                        block_size=block_size,
                        m=m,
                    ):
                        ids = (
                            torch.rand(m, num_experts, device=self.dev)
                            .topk(topk, dim=1)
                            .indices.to(torch.int32)
                            .contiguous()
                        )
                        weights = torch.rand(m, topk, device=self.dev) + 0.01
                        x = (
                            torch.randn(m, hidden, device=self.dev)
                            * torch.logspace(-3, 2, hidden, device=self.dev)
                        ).to(torch.bfloat16)
                        x[0, :64] = 0
                        buffers = self._sort_buffers(
                            m, topk, block_size, num_experts, hidden
                        )
                        sorted_ids, sorted_weights, _, num_valid, _ = buffers
                        q, scale = self.mss._run_small_sort(
                            ids,
                            weights,
                            *buffers,
                            block_size,
                            x,
                            num_experts,
                            mx_fp4=True,
                        )
                        ref_q, ref_scale = fused_dynamic_mxfp4_quant_moe_sort(
                            x,
                            sorted_ids=sorted_ids,
                            num_valid_ids=num_valid,
                            token_num=m,
                            topk=topk,
                            block_size=block_size,
                            sorted_weights=sorted_weights,
                            num_experts_upper_bound=num_experts,
                        )
                        self.assertTrue(torch.equal(q, ref_q.view(torch.uint8)))
                        self.assertEqual(scale.shape, ref_scale.shape)

                        # aiter leaves the scales of padding rows and all-zero groups unspecified
                        valid = int(num_valid[0])
                        rows = torch.arange(valid, device=self.dev)
                        rows = rows[(sorted_ids[:valid] & 0xFFFFFF) < m][:, None]
                        k = torch.arange(scale_n, device=self.dev)
                        addr = (
                            (rows // 32) * (scale_n_pad * 32)
                            + (rows % 16) * 4
                            + (rows % 32) // 16
                            + (k // 8) * 256
                            + (k % 4) * 64
                            + ((k % 8) // 4) * 2
                        )
                        tokens = (sorted_ids[rows[:, 0]] & 0xFFFFFF).long()
                        nonzero = x[tokens].view(-1, scale_n, 32).abs().amax(-1) != 0
                        got = scale.view(torch.uint8).flatten()[addr]
                        want = ref_scale.view(torch.uint8).flatten()[addr]
                        self.assertTrue(torch.equal(got[nonzero], want[nonzero]))


if __name__ == "__main__":
    unittest.main()
