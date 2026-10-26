import math
import unittest

import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.srt.layers.attention.dsa.dsa_dcp import (
    dcp_trtllm_sparse_attention,
    trtllm_rows_per_call,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")

TOPK = 2048
SCALE = 1 / math.sqrt(192)


def _reference_page_table(page_table, dcp_size, rank):
    rows, counts = [], []
    for row in page_table.tolist():
        owned = [
            slot // dcp_size for slot in row if slot >= 0 and slot % dcp_size == rank
        ]
        rows.append(owned + [-1] * (len(row) - len(owned)))
        counts.append(len(owned))
    return torch.tensor(rows, dtype=page_table.dtype), torch.tensor(counts)


def _sparse_attention(query, kv, page_table, seq_lens, workspace_bytes=512 << 20):
    rows = query.shape[0]
    return dcp_trtllm_sparse_attention(
        query=query,
        block_tables=page_table.unsqueeze(1),
        seq_lens=seq_lens,
        cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
        kv_cache=kv.reshape(-1, 1, 64, 576),
        workspace_buffer=torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda"),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_seq_len=TOPK,
        sparse_mla_top_k=TOPK,
        bmm1_scale=SCALE,
        backend="trtllm-gen",
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDsaDcpAttention(CustomTestCase):
    def test_page_table_keeps_owned_slots_in_order(self):
        for width in (37, TOPK):
            page_table = torch.arange(3 * width, dtype=torch.int32).view(3, width)
            page_table = page_table.flip(1)
            page_table[0, ::3] = -1
            page_table[1] = -1
            for dcp_size in (2, 4):
                for rank in range(dcp_size):
                    with self.subTest(width=width, dcp=dcp_size, rank=rank):
                        expected, expected_counts = _reference_page_table(
                            page_table, dcp_size, rank
                        )
                        actual, counts = transform_index_page_table_dcp(
                            page_table.cuda(), dcp_size, rank
                        )
                        self.assertEqual(actual.cpu().tolist(), expected.tolist())
                        self.assertEqual(
                            counts.cpu().tolist(), expected_counts.tolist()
                        )

    def test_merged_shards_match_unsharded_attention(self):
        generator = torch.Generator().manual_seed(0)
        capacity, heads = 8192, 64
        page_table = torch.full((3, TOPK), -1, dtype=torch.int32)
        page_table[0, 0] = capacity - 4
        page_table[1, :127] = (
            torch.randperm(capacity // 4, generator=generator)[:127] * 4
        )
        page_table[2] = torch.randperm(capacity, generator=generator)[:TOPK]
        page_table = page_table.cuda()
        seq_lens = torch.tensor([1, 127, TOPK], dtype=torch.int32, device="cuda")
        query_cpu = torch.randn(3, 1, heads, 576, generator=generator) * 0.25
        kv_cpu = torch.randn(capacity, 576, generator=generator) * 0.25

        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            query, kv = query_cpu.cuda().to(dtype), kv_cpu.cuda().to(dtype)
            expected, expected_lse = _sparse_attention(query, kv, page_table, seq_lens)
            for dcp_size in (2, 4):
                with self.subTest(dtype=dtype, dcp_size=dcp_size):
                    outputs, lses = [], []
                    for rank in range(dcp_size):
                        local_page_table, counts = transform_index_page_table_dcp(
                            page_table, dcp_size, rank
                        )
                        output, lse = _sparse_attention(
                            query,
                            kv[rank::dcp_size].contiguous(),
                            local_page_table,
                            counts,
                        )
                        outputs.append(output.squeeze(1))
                        lses.append(lse)
                    merged, merged_lse = dcp_lse_combine_triton(
                        torch.stack(outputs),
                        torch.stack(lses),
                        is_lse_base_on_e=False,
                        return_lse=True,
                    )
                    atol = 0.008 if dtype == torch.float8_e4m3fn else 0.003
                    torch.testing.assert_close(
                        merged, expected.squeeze(1), atol=atol, rtol=0.02
                    )
                    torch.testing.assert_close(
                        merged_lse, expected_lse, atol=0.003, rtol=0.002
                    )

    def test_rows_split_to_fit_the_workspace(self):
        """A workspace too small for one call's softmax slab must split the rows, not overflow."""
        generator = torch.Generator().manual_seed(1)
        rows, heads, capacity, length = 2400, 32, 4096, 128
        page_table = torch.full((rows, TOPK), -1, dtype=torch.int32)
        for row in range(rows):
            page_table[row, :length] = torch.randperm(capacity, generator=generator)[
                :length
            ]
        page_table = page_table.cuda()
        seq_lens = torch.full((rows,), length, dtype=torch.int32, device="cuda")
        query = (torch.randn(rows, 1, heads, 576, generator=generator) * 0.25).cuda()
        kv = (torch.randn(capacity, 576, generator=generator) * 0.25).cuda()
        query, kv = query.to(torch.bfloat16), kv.to(torch.bfloat16)
        expected, expected_lse = _sparse_attention(query, kv, page_table, seq_lens)
        workspace_bytes = (128 << 20) + 64 * heads * 256 * 8
        self.assertEqual(
            trtllm_rows_per_call(workspace_bytes=workspace_bytes, heads=heads), 64
        )
        output, lse = _sparse_attention(
            query, kv, page_table, seq_lens, workspace_bytes=workspace_bytes
        )
        torch.testing.assert_close(output, expected, atol=0.003, rtol=0.02)
        torch.testing.assert_close(lse, expected_lse, atol=0.003, rtol=0.002)


if __name__ == "__main__":
    unittest.main()
