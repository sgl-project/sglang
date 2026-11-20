import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.models.nemotron_decision import PackedDecisionAttention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestPackedDecisionAttention(CustomTestCase):
    def test_paged_read_matches_independent_mask_with_mixed_request_lengths(self):
        layouts = [[5, 3, 4], None, [2, 5]]
        lengths = [12, 7, 7]
        count = sum(lengths)
        batch = SimpleNamespace(decision_layouts=layouts, extend_seq_lens_cpu=lengths)
        positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
        packed = PackedDecisionAttention()
        packed.prepare(batch, positions)
        torch.testing.assert_close(
            positions.cpu(),
            torch.tensor(
                list(range(5))
                + list(range(5, 8))
                + list(range(5, 9))
                + list(range(7))
                + list(range(2))
                + list(range(2, 7))
            ),
        )
        generator = torch.Generator(device="cuda").manual_seed(42)
        q = torch.randn(
            count, 4, 64, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(
            count, 2, 64, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        v = torch.randn(
            count, 2, 64, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        layer = SimpleNamespace(
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=64,
            v_head_dim=64,
            scaling=1 / math.sqrt(64),
        )
        actual = packed.run(q, k, v, layer).view_as(q)
        allowed = torch.zeros(count, count, dtype=torch.bool, device="cuda")
        offset = 0
        for length, layout in zip(lengths, layouts):
            if layout is None:
                allowed[offset : offset + length, offset : offset + length] = True
            else:
                segment = torch.cat(
                    [torch.full((n,), i, device="cuda") for i, n in enumerate(layout)]
                )
                allowed[offset : offset + length, offset : offset + length] = (
                    segment[None, :] == 0
                ) | (segment[:, None] == segment[None, :])
            offset += length
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.repeat_interleave(2, dim=1).transpose(0, 1)[None],
            v.repeat_interleave(2, dim=1).transpose(0, 1)[None],
            attn_mask=allowed[None, None],
        )[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        changed = v.clone()
        changed[8:12] += 10
        unaffected = packed.run(q, k, changed, layer).view_as(q)
        torch.testing.assert_close(unaffected[:8], actual[:8], atol=0, rtol=0)
        torch.testing.assert_close(unaffected[12:], actual[12:], atol=0, rtol=0)
        self.assertGreater((unaffected[8:12] - actual[8:12]).abs().max().item(), 1)


if __name__ == "__main__":
    unittest.main()
