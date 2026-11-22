import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.models.nemotron_decision import PackedDecisionAttention
from sglang.srt.utils.device import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestPackedDecisionAttention(CustomTestCase):
    def test_paged_read_matches_independent_mask_with_mixed_request_lengths(self):
        self._check_independent_mask(PackedDecisionAttention())

    @unittest.skipUnless(is_sm100_supported(), "FA4 packed reads require SM100/SM103")
    def test_fa4_matches_independent_mask_with_mixed_request_lengths(self):
        self._check_independent_mask(PackedDecisionAttention(use_fa4=True))

    def _check_independent_mask(self, packed):
        layouts = [[5, 3, 4], None, [2, 5]]
        lengths = [12, 7, 7]
        count = sum(lengths)
        batch = SimpleNamespace(decision_layouts=layouts, extend_seq_lens_cpu=lengths)
        positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
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

    @unittest.skipUnless(is_sm100_supported(), "FA4 packed reads require SM100/SM103")
    def test_fa4_bounds_and_reuses_copies_with_grouped_heads_and_padding(self):
        for query_heads, kv_heads in ((32, 8), (16, 4)):
            with self.subTest(query_heads=query_heads, kv_heads=kv_heads):
                layer = SimpleNamespace(
                    tp_q_head_num=query_heads,
                    tp_k_head_num=kv_heads,
                    tp_v_head_num=kv_heads,
                    head_dim=128,
                    v_head_dim=128,
                    scaling=128**-0.5,
                )
                batch = SimpleNamespace(
                    decision_layouts=[[16, 16, 16], [16, 16], None],
                    extend_seq_lens_cpu=[48, 32, 16],
                )
                q = torch.zeros(
                    112, query_heads, 128, device="cuda", dtype=torch.bfloat16
                )
                k = torch.zeros(112, kv_heads, 128, device="cuda", dtype=q.dtype)
                v = torch.empty_like(k)
                key_bias = (
                    16 * torch.arange(kv_heads, device="cuda", dtype=q.dtype)[:, None]
                )
                query_bias = (
                    16
                    * (torch.arange(query_heads, device="cuda") // 4).to(q.dtype)[
                        :, None
                    ]
                )
                for i, value in enumerate((1, 3, 7, 11, 13, 17, 1000)):
                    v[16 * i : 16 * (i + 1)] = value + key_bias
                # Zero Q/K gives exact means of the permitted state/question values.
                expected = torch.cat(
                    [
                        torch.full(
                            (16, query_heads, 128), value, device="cuda", dtype=q.dtype
                        )
                        + query_bias
                        for value in (1, 2, 4, 11, 12, 17)
                    ]
                )
                attention = PackedDecisionAttention(use_fa4=True)
                positions = torch.arange(112, device="cuda")
                copy_budget = 32 * 2 * kv_heads * 128 * q.element_size()
                with patch.object(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE,
                    "get",
                    return_value=copy_budget,
                ):
                    small_batch = SimpleNamespace(
                        decision_layouts=[[8, 8]], extend_seq_lens_cpu=[16]
                    )
                    attention.prepare(small_batch, positions)
                    small = attention.run(q, k, v, layer).view_as(expected[:16])
                    self.assertTrue(torch.equal(small, expected[:16]))
                    initial_rows = attention.fa4_keys.shape[0]
                    attention.prepare(batch, positions)
                    actual = attention.run(q, k, v, layer).view_as(expected)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertGreater(attention.fa4_keys.shape[0], initial_rows)
                    self.assertGreater(attention.fa4_values.shape[0], initial_rows)
                    self.assertGreater(len(attention.fa4_calls), 1)
                    self.assertLessEqual(
                        attention.fa4_keys.nbytes + attention.fa4_values.nbytes,
                        copy_budget,
                    )
                    key_pointer = attention.fa4_keys.data_ptr()
                    value_pointer = attention.fa4_values.data_ptr()
                    next_batch = SimpleNamespace(
                        decision_layouts=[[16, 16], [16, 16, 16], None],
                        extend_seq_lens_cpu=[32, 48, 16],
                    )
                    next_expected = expected.clone()
                    next_expected[32:48] = 7 + query_bias
                    next_expected[48:64] = 9 + query_bias
                    next_expected[64:80] = 10 + query_bias
                    attention.prepare(next_batch, positions)
                    repeated = attention.run(q, k, v, layer).view_as(expected)
                    self.assertTrue(torch.equal(repeated, next_expected))
                    self.assertEqual(attention.fa4_keys.data_ptr(), key_pointer)
                    self.assertEqual(attention.fa4_values.data_ptr(), value_pointer)
                    attention.prepare(batch, positions)
                    restored = attention.run(q, k, v, layer).view_as(expected)
                    self.assertTrue(torch.equal(restored, expected))
                if query_heads == 32:
                    with patch.object(
                        envs.SGLANG_FLASHINFER_WORKSPACE_SIZE,
                        "get",
                        return_value=copy_budget // 4,
                    ):
                        fresh = PackedDecisionAttention(use_fa4=True)
                        fresh.prepare(batch, positions)
                        oversized = fresh.run(q, k, v, layer).view_as(expected)
                        self.assertTrue(torch.equal(oversized, expected))
                        copied_bytes = fresh.fa4_keys.nbytes + fresh.fa4_values.nbytes
                        self.assertGreater(copied_bytes, copy_budget // 4)
                        self.assertLessEqual(copied_bytes, copy_budget)
                changed = v.clone()
                changed[32:48] = 99 + key_bias
                result = attention.run(q, k, changed, layer).view_as(expected)
                self.assertTrue(torch.equal(result[:32], actual[:32]))
                self.assertTrue(torch.equal(result[48:], actual[48:]))
                self.assertTrue(
                    torch.equal(
                        result[32:48], torch.full_like(result[32:48], 50) + query_bias
                    )
                )


if __name__ == "__main__":
    unittest.main()
