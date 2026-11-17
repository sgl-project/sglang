"""gfx950 AITER paged-MQA logits: one next_n query per request in target verify."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HEADS, D, BLOCK_KV, NEXT_N = 32, 128, 64, 6


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "requires gfx950"
)
class TestAiterPagedMqaLogitsNextN(CustomTestCase):
    @torch.no_grad()
    def test_next_n_matches_one_row_per_query(self):
        torch.manual_seed(0)
        # pooled k lengths of each request's last draft row; earlier rows close fewer pools
        last_lens = [70, 130, 257]
        bs, rows = len(last_lens), len(last_lens) * NEXT_N
        width = (max(last_lens) + NEXT_N + BLOCK_KV - 1) // BLOCK_KV + 1
        num_blocks = bs * width + 1

        # one page: BLOCK_KV x D fp8 keys, then BLOCK_KV fp32 scales
        page = torch.empty(num_blocks, BLOCK_KV * (D + 4), dtype=torch.uint8)
        page[:, : BLOCK_KV * D] = (
            torch.randn(num_blocks, BLOCK_KV * D)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        page[:, BLOCK_KV * D :] = (
            torch.rand(num_blocks, BLOCK_KV).add(0.5).view(torch.uint8)
        )
        kv_cache = page.cuda().view(num_blocks, BLOCK_KV, 1, D + 4)

        q_fp8 = torch.randn(rows, HEADS, D, device="cuda").to(torch.float8_e4m3fn)
        weights = torch.randn(rows, HEADS, device="cuda")
        perm = torch.randperm(num_blocks - 1)[: bs * width] + 1
        req_tables = perm.view(bs, width).int().cuda()
        # the verify page table repeats each request's row for its draft rows
        row_tables = req_tables.repeat_interleave(NEXT_N, dim=0).contiguous()
        row_lens = torch.tensor(
            [n - (NEXT_N - 1 - j) // 4 for n in last_lens for j in range(NEXT_N)],
            dtype=torch.int32,
            device="cuda",
        )
        max_seq_len = width * BLOCK_KV

        want = aiter_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            row_lens,
            row_tables,
            max_seq_len,
            preshuffle=True,
            kv_block_size=BLOCK_KV,
        )
        ctx_lens = row_lens.view(bs, NEXT_N)[:, -1] + (NEXT_N - 1)
        got = aiter_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            ctx_lens,
            row_tables.view(bs, -1),
            max_seq_len,
            preshuffle=True,
            kv_block_size=BLOCK_KV,
            next_n=NEXT_N,
        )
        self.assertEqual(got.shape, want.shape)
        for r, n in enumerate(row_lens.tolist()):
            self.assertTrue(torch.equal(got[r, :n], want[r, :n]), f"row {r}")


if __name__ == "__main__":
    unittest.main()
