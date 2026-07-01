"""CPU unit tests for FlashAttentionAdaptor sparse metadata rewriting.

No GPU needed: builds a fake FA metadata object + forward batch and checks that
the adaptor rewrites page_table/cache_seqlens only for active requests, and that
a request which selected zero pages keeps its original (dense) metadata.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import FlashAttentionAdaptor


def _make_metadata(page_table, cache_seqlens):
    cu = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return SimpleNamespace(
        page_table=page_table.clone(),
        cache_seqlens_int32=cache_seqlens.clone(),
        cu_seqlens_k=cu,
        max_seq_len_k=int(cache_seqlens.max()),
    )


class TestFlashAttentionAdaptor(unittest.TestCase):
    def _run(self, selected_indices, valid_lengths, sparse_mask):
        dev = torch.device("cpu")
        adaptor = FlashAttentionAdaptor(dev)
        page_size = 1
        # request 0 tokens at slots 0..3, request 1 at slots 10..13
        req_to_token = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=torch.int64)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        seq_lens = torch.tensor([4, 4], dtype=torch.int64)
        page_table = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=torch.int32)
        cache_seqlens = torch.tensor([4, 4], dtype=torch.int32)

        metadata = _make_metadata(page_table, cache_seqlens)
        adaptor.save_original_metadata(metadata)
        forward_batch = SimpleNamespace(
            req_pool_indices=req_pool_indices, seq_lens=seq_lens
        )
        return adaptor.adapt_for_attn_metadata(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=page_size,
            layer_id=0,
        )

    def test_active_request_is_rewritten_and_zero_selected_keeps_dense(self):
        # req0 selects logical pages [0, 2] (2 valid); req1 selects nothing (0 valid)
        out = self._run(
            selected_indices=torch.tensor([[0, 2], [-1, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True]),
        )
        # req0: cache_seqlens shrinks to the 2 selected pages; page_table points at them
        self.assertEqual(int(out.cache_seqlens_int32[0]), 2)
        self.assertEqual(out.page_table[0, :2].tolist(), [0, 2])
        # req1: selected 0 pages -> must keep dense metadata (not 0/negative)
        self.assertEqual(int(out.cache_seqlens_int32[1]), 4)
        self.assertEqual(out.page_table[1].tolist(), [10, 11, 12, 13])

    def test_no_active_requests_returns_unchanged(self):
        out = self._run(
            selected_indices=torch.tensor([[-1, -1], [-1, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([0, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([False, False]),
        )
        self.assertEqual(out.cache_seqlens_int32.tolist(), [4, 4])
        self.assertEqual(out.page_table.tolist(), [[0, 1, 2, 3], [10, 11, 12, 13]])


if __name__ == "__main__":
    unittest.main()
