from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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


if __name__ == "__main__":
    unittest.main()
