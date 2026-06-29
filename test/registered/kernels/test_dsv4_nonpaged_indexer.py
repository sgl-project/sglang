from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.jit_kernel.dsv4 import topk_transform_512
from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


PAGE_SIZE = 64
HEAD_DIM = 128
SCALE_BYTES = 4
PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + SCALE_BYTES)


class TestDeepSeekV4NonPagedIndexer(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_batched_gather_matches_reference(self):
        num_pages = 4
        raw = torch.arange(num_pages * PAGE_BYTES, dtype=torch.int64, device="cuda")
        buffer = (raw % 251).to(torch.uint8).view(num_pages, PAGE_BYTES)
        pool = SimpleNamespace(
            index_k_with_scale_buffer=[buffer],
            page_size=PAGE_SIZE,
            index_head_dim=HEAD_DIM,
            quant_block_size=HEAD_DIM,
            device=torch.device("cuda"),
        )

        # Covers different lengths, partial final pages, out-of-order pages,
        # and the same physical page appearing twice.
        seq_lens = torch.tensor([65, 127], dtype=torch.int32, device="cuda")
        page_indices = torch.tensor([[3, 1], [2, 2]], dtype=torch.int32, device="cuda")

        actual_k, actual_scale = DeepSeekV4IndexerPool.get_index_k_scale_buffer(
            pool,
            layer_id=0,
            seq_len_tensor=seq_lens,
            page_indices=page_indices,
            seq_len_sum=192,
            max_seq_len=127,
        )

        expected_k = []
        expected_scale = []
        for request, seq_len in enumerate((65, 127)):
            for token in range(seq_len):
                page = int(page_indices[request, token // PAGE_SIZE].item())
                offset = token % PAGE_SIZE
                expected_k.append(
                    buffer[page, offset * HEAD_DIM : (offset + 1) * HEAD_DIM]
                )
                scale_start = PAGE_SIZE * HEAD_DIM + offset * SCALE_BYTES
                expected_scale.append(
                    buffer[page, scale_start : scale_start + SCALE_BYTES]
                )

        torch.testing.assert_close(actual_k, torch.stack(expected_k), rtol=0, atol=0)
        torch.testing.assert_close(
            actual_scale, torch.stack(expected_scale), rtol=0, atol=0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_paged_and_nonpaged_valid_logits_are_exact(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("requires Hopper or newer")
        # DeepGEMM is a hard dependency of the feature on its registered H100
        # runner. An import/ABI failure there must fail CI rather than skip it.
        import deep_gemm

        torch.manual_seed(7)
        query_rows = 256
        # Exercise a partial final page and the production-aligned output width.
        # 1021 also keeps every query above TOPK=512, so the final transform
        # performs a real score-based top-k rather than its short-sequence path.
        key_rows = 1021
        max_seqlen_k = 1024
        num_heads = 64
        num_physical_pages = 18
        page_order = torch.tensor(
            [17, 2, 15, 1, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8],
            dtype=torch.int32,
            device="cuda",
        )
        page_table = page_order.unsqueeze(0).repeat(query_rows, 1)

        k = torch.randn(
            num_physical_pages,
            PAGE_SIZE,
            1,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_scale = k.abs().float().amax(dim=3, keepdim=True).clamp(1.0e-4) / 448.0
        k_fp8 = (k.float() / k_scale).clamp(-448.0, 448.0).to(FP8_DTYPE)
        packed = torch.empty(
            (num_physical_pages, PAGE_BYTES), dtype=torch.uint8, device="cuda"
        )
        packed[:, : PAGE_SIZE * HEAD_DIM].copy_(
            k_fp8.view(num_physical_pages, PAGE_SIZE * HEAD_DIM).view(torch.uint8)
        )
        packed[:, PAGE_SIZE * HEAD_DIM :].copy_(
            k_scale.view(num_physical_pages, PAGE_SIZE).view(torch.uint8)
        )
        paged_cache = packed.view(
            num_physical_pages, PAGE_SIZE, 1, HEAD_DIM + SCALE_BYTES
        )

        q = torch.randn(
            query_rows,
            1,
            num_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_scale = q.abs().float().amax(dim=3, keepdim=True).clamp(1.0e-4) / 448.0
        q_fp8 = (q.float() / q_scale).clamp(-448.0, 448.0).to(FP8_DTYPE)
        base_weights = torch.randn(
            query_rows, num_heads, dtype=torch.float32, device="cuda"
        )
        weights = (base_weights[:, None, :, None] * q_scale).view(query_rows, num_heads)

        ke = torch.arange(
            key_rows - query_rows + 1,
            key_rows + 1,
            dtype=torch.int32,
            device="cuda",
        )
        context_lens = ke.unsqueeze(1)
        paged_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, PAGE_SIZE, deep_gemm.get_num_sms()
        )
        paged_logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            paged_cache,
            weights,
            context_lens,
            page_table,
            paged_schedule,
            max_seqlen_k,
            clean_logits=False,
        )

        pool = SimpleNamespace(
            index_k_with_scale_buffer=[packed],
            page_size=PAGE_SIZE,
            index_head_dim=HEAD_DIM,
            quant_block_size=HEAD_DIM,
            device=torch.device("cuda"),
        )
        gather_seq_lens = torch.tensor([key_rows], dtype=torch.int32, device="cuda")
        k_u8, scale_u8 = DeepSeekV4IndexerPool.get_index_k_scale_buffer(
            pool,
            layer_id=0,
            seq_len_tensor=gather_seq_lens,
            page_indices=page_table[:1],
            seq_len_sum=key_rows,
            max_seq_len=key_rows,
        )
        nonpaged_logits = deep_gemm.fp8_mqa_logits(
            q_fp8[:, 0],
            (k_u8.view(FP8_DTYPE), scale_u8.view(torch.float32).squeeze(-1)),
            weights,
            torch.zeros_like(ke),
            ke,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
        )

        valid = torch.arange(max_seqlen_k, device="cuda").unsqueeze(0) < ke.unsqueeze(1)
        torch.testing.assert_close(
            nonpaged_logits[valid], paged_logits[valid], rtol=0, atol=0
        )

        paged_page_indices = torch.empty(
            (query_rows, 512), dtype=torch.int32, device="cuda"
        )
        nonpaged_page_indices = torch.empty_like(paged_page_indices)
        paged_raw_indices = torch.empty_like(paged_page_indices)
        nonpaged_raw_indices = torch.empty_like(paged_page_indices)
        topk_transform_512(
            paged_logits,
            ke,
            page_table,
            paged_page_indices,
            PAGE_SIZE,
            paged_raw_indices,
        )
        topk_transform_512(
            nonpaged_logits,
            ke,
            page_table,
            nonpaged_page_indices,
            PAGE_SIZE,
            nonpaged_raw_indices,
        )
        torch.testing.assert_close(
            nonpaged_raw_indices, paged_raw_indices, rtol=0, atol=0
        )
        torch.testing.assert_close(
            nonpaged_page_indices, paged_page_indices, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
