"""RoPE DSA sparse attention under DCP, with KV shards emulated on one GPU."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.srt.layers.attention.dsa.dsa_dcp import dsa_dcp_head_groups
from sglang.srt.layers.attention.dsa_backend import _dcp_trtllm_sparse_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The TRT-LLM sparse RoPE MLA kernels require SM100/SM103.
register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")

TOPK = 2048
SCALE = 1 / math.sqrt(192)


def _reference_page_table(page_table, dcp_size, rank, repeat_rows):
    rows, counts = [], []
    for row in page_table.tolist():
        owned = [
            slot // dcp_size for slot in row if slot >= 0 and slot % dcp_size == rank
        ]
        rows += [owned + [-1] * (len(row) - len(owned))] * repeat_rows
        counts += [len(owned)] * repeat_rows
    return torch.tensor(rows, dtype=page_table.dtype), torch.tensor(counts)


def _sparse_attention(query, kv, page_table, seq_lens, head_groups=1):
    rows, _, heads, _ = query.shape
    return _dcp_trtllm_sparse_attention(
        query=query,
        block_tables=page_table.unsqueeze(1),
        seq_lens=seq_lens,
        lse=torch.empty((rows, heads), dtype=torch.float32, device="cuda"),
        head_groups=head_groups,
        cu_seqlens_q=torch.arange(
            rows * head_groups + 1, dtype=torch.int32, device="cuda"
        ),
        kv_cache=kv.reshape(-1, 1, 64, 576),
        workspace_buffer=torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"),
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
                    for repeat_rows in (1, 2):
                        with self.subTest(
                            width=width, dcp=dcp_size, rank=rank, repeat=repeat_rows
                        ):
                            expected, expected_counts = _reference_page_table(
                                page_table, dcp_size, rank, repeat_rows
                            )
                            actual, counts = transform_index_page_table_dcp(
                                page_table.cuda(), dcp_size, rank, repeat_rows
                            )
                            self.assertEqual(actual.cpu().tolist(), expected.tolist())
                            self.assertEqual(
                                counts.cpu().tolist(), expected_counts.tolist()
                            )

    def test_folded_head_groups_match_single_group_attention(self):
        # FlashInfer 0.7.0.post1 returns a wrong LSE above 32 heads for short KV.
        generator = torch.Generator().manual_seed(0)
        heads, rows = 64, 4
        head_groups = dsa_dcp_head_groups(heads)
        page_table = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1, 63, 64, 0], dtype=torch.int32, device="cuda")
        for row, count in enumerate(seq_lens.tolist()):
            page_table[row, :count] = torch.randperm(64, generator=generator)[:count]
        query_cpu = torch.randn(rows, 1, heads, 576, generator=generator) * 0.25
        kv_cpu = torch.randn(64, 576, generator=generator) * 0.25

        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            with self.subTest(dtype=dtype):
                query, kv = query_cpu.cuda().to(dtype), kv_cpu.cuda().to(dtype)
                output, lse = _sparse_attention(
                    query,
                    kv,
                    page_table.repeat_interleave(head_groups, dim=0),
                    seq_lens.repeat_interleave(head_groups),
                    head_groups,
                )
                expected = [
                    _sparse_attention(
                        query[:, :, begin : begin + 32].contiguous(),
                        kv,
                        page_table,
                        seq_lens,
                    )
                    for begin in range(0, heads, 32)
                ]
                torch.testing.assert_close(
                    output, torch.cat([o for o, _ in expected], dim=1)
                )
                torch.testing.assert_close(
                    lse, torch.cat([l for _, l in expected], dim=1)
                )
                self.assertTrue((output[-1] == 0).all())
                self.assertTrue(torch.isneginf(lse[-1]).all())

    def test_merged_shards_match_unsharded_attention(self):
        generator = torch.Generator().manual_seed(0)
        capacity, heads = 8192, 32
        page_table = torch.full((3, TOPK), -1, dtype=torch.int32)
        # Rows 0 and 1 select KV that only rank 0 owns.
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
                        outputs.append(output)
                        lses.append(lse)
                    merged, merged_lse = dcp_lse_combine_triton(
                        torch.stack(outputs),
                        torch.stack(lses),
                        is_lse_base_on_e=False,
                        return_lse=True,
                    )
                    # Each shard quantizes FP8 softmax numerators against its
                    # own maximum, so FP8 merges are slightly less exact.
                    atol = 0.008 if dtype == torch.float8_e4m3fn else 0.003
                    torch.testing.assert_close(merged, expected, atol=atol, rtol=0.02)
                    torch.testing.assert_close(
                        merged_lse, expected_lse, atol=0.003, rtol=0.002
                    )


if __name__ == "__main__":
    unittest.main()
