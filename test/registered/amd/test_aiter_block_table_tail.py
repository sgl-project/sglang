"""Block tables of the aiter backend carry a zeroed storage tail.

aiter's unified_attention loads a whole KV tile's block-table entries without a row
mask, so the last row of an exactly sized table is read past the tensor's end. The
tail keeps that load in bounds; the entries it reads there are masked out, so the
output does not depend on them.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm attention backend")
class TestAiterBlockTableTail(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import aiter_backend as ab

        cls.ab = ab

    def test_table_view_and_tail(self):
        table = self.ab._ua_block_table(3, 5, "cuda")
        self.assertEqual(table.shape, (3, 5))
        self.assertEqual(table.stride(), (5, 1))
        self.assertEqual(table.dtype, torch.int32)
        self.assertTrue(torch.all(table == 0))
        storage = torch.empty(0, dtype=torch.int32, device="cuda").set_(
            table.untyped_storage()
        )
        self.assertGreaterEqual(storage.numel(), 15 + self.ab._UA_BLOCK_TABLE_TAIL)
        self.assertTrue(torch.all(storage[15:] == 0))

    @torch.no_grad()
    def test_tail_values_do_not_reach_the_output(self):
        from aiter.ops.triton.attention.unified_attention import unified_attention

        torch.manual_seed(0)
        page, heads, kv_heads, dim, seqs = 16, 8, 1, 64, 2
        tokens = 40  # 3 pages; a 64-token KV tile reads a 4th block-table entry
        cols = (tokens + page - 1) // page
        table = self.ab._ua_block_table(seqs, cols, "cuda")
        table.copy_(torch.randperm(seqs * cols, device="cuda").view(seqs, cols))
        k = torch.randn(seqs * cols, page, kv_heads, dim, device="cuda").bfloat16()
        v = torch.randn_like(k)
        q = torch.randn(seqs * tokens, heads, dim, device="cuda").bfloat16()
        cu_seqlens_q = torch.arange(0, seqs + 1, device="cuda", dtype=torch.int32)
        seqused_k = torch.full((seqs,), tokens, device="cuda", dtype=torch.int32)

        def run():
            out = torch.empty_like(q)
            unified_attention(
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q * tokens,
                max_seqlen_q=tokens,
                seqused_k=seqused_k,
                max_seqlen_k=cols * page,
                softmax_scale=dim**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=table,
                softcap=0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
            )
            return out

        expected = run()
        storage = torch.empty(0, dtype=torch.int32, device="cuda").set_(
            table.untyped_storage()
        )
        storage[seqs * cols :] = 1 << 30
        self.assertTrue(torch.equal(run(), expected))


if __name__ == "__main__":
    unittest.main()
