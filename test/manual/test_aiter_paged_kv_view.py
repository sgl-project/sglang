"""
Unit test for the aiter paged-prefill KV view conversion.

aiter_backend._build_paged_kv_view turns the token-level flashinfer KV table
into the page-level view aiter's fp8 hd256 page-64 paged-varlen asm prefill
expects. This checks the index arithmetic against a shuffled page allocation, so
a wrong page id or last-page length is caught. The tensors stay on CPU (no GPU
needed, though the import pulls in the usual sglang deps); the kernel path needs
ROCm/gfx950 and is covered by the evidence in the PR.

Run:
    python test/manual/test_aiter_paged_kv_view.py
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))


class TestAiterPagedKVView(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.aiter_backend import _build_paged_kv_view

        self.build = _build_paged_kv_view
        self.page_size = 64

    def _table(self, page_ids_per_req, seq_lens):
        """The token-level table create_flashinfer_kv_indices_triton produces:
        per request, the KV slots of its pages in page order."""
        offs = torch.arange(self.page_size, dtype=torch.int64)
        rows = []
        for pages, seq_len in zip(page_ids_per_req, seq_lens):
            slots = (torch.tensor(pages, dtype=torch.int64) * self.page_size).unsqueeze(
                -1
            ) + offs
            rows.append(slots.reshape(-1)[:seq_len])
        return torch.cat(rows).to(torch.int32)

    def test_shuffled_pages_and_ragged_lengths(self):
        # A real allocator hands out pages out of order, so page id != position
        # in the table; an identity table would pass even if this were wrong.
        page_ids_per_req = [[7, 3, 11], [5, 2]]
        seq_lens = [130, 128]  # 130 -> 3 pages, last one holding 2 tokens
        kv_indices = self._table(page_ids_per_req, seq_lens)

        page_indptr, page_ids, last_page_len = self.build(
            kv_indices, torch.tensor(seq_lens), self.page_size
        )

        self.assertEqual(page_indptr.tolist(), [0, 3, 5])
        self.assertEqual(page_ids.tolist(), [7, 3, 11, 5, 2])
        self.assertEqual(last_page_len.tolist(), [2, 64])
        self.assertEqual(page_indptr.dtype, torch.int32)
        self.assertEqual(page_ids.dtype, torch.int32)
        self.assertEqual(last_page_len.dtype, torch.int32)

    def test_exact_multiple_uses_full_last_page(self):
        kv_indices = self._table([[4, 9]], [128])
        _, page_ids, last_page_len = self.build(
            kv_indices, torch.tensor([128]), self.page_size
        )
        self.assertEqual(page_ids.tolist(), [4, 9])
        self.assertEqual(last_page_len.tolist(), [64])


if __name__ == "__main__":
    unittest.main()
