"""Index arithmetic for the aiter ASM context-chunk prefill KV gather.

kv_indptr/kv_indices are token-level for every page_size, so the gather must
resolve token t of sequence i to kv_indices[kv_indptr[i] + t]. Pure indexing,
so this runs on CPU.
"""

import unittest

import torch

from sglang.srt.layers.attention.aiter_backend import (
    _asm_context_prefill_gather_indices,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def _paged_pool(seq_lens, page_size, seed, headroom=2):
    """Lay each sequence out over shuffled pages, as a page allocator would.

    Returns (req_to_token, num_kv_slots): req_to_token[i][t] is the pool slot
    holding token t of sequence i. Pages are handed out in shuffled order, so a
    correct gather cannot rely on sequences being contiguous in the pool.
    """
    num_pages = headroom * sum((n + page_size - 1) // page_size for n in seq_lens)
    g = torch.Generator().manual_seed(seed)
    free_pages = torch.randperm(num_pages, generator=g).tolist()
    req_to_token = []
    for n in seq_lens:
        slots = []
        for _ in range((n + page_size - 1) // page_size):
            base = free_pages.pop() * page_size
            slots.extend(range(base, base + page_size))
        req_to_token.append(slots[:n])
    return req_to_token, num_pages * page_size


def _token_level_metadata(req_to_token):
    """Build kv_indptr/kv_indices the way AiterIndicesUpdaterPrefill does."""
    seq_lens = torch.tensor([len(s) for s in req_to_token], dtype=torch.long)
    kv_indptr = torch.zeros(len(req_to_token) + 1, dtype=torch.long)
    torch.cumsum(seq_lens, 0, out=kv_indptr[1:])
    kv_indices = torch.tensor(
        [slot for slots in req_to_token for slot in slots], dtype=torch.int32
    )
    return kv_indptr, kv_indices, seq_lens


class TestAsmPrefillGatherIndices(unittest.TestCase):
    def _check(self, seq_lens, page_size, seed=0xA17E4):
        req_to_token, num_kv_slots = _paged_pool(seq_lens, page_size, seed)
        kv_indptr, kv_indices, lens = _token_level_metadata(req_to_token)

        gathered = _asm_context_prefill_gather_indices(
            kv_indptr, kv_indices, lens, num_kv_slots
        )
        self.assertIsNotNone(gathered, "gather rejected valid metadata")
        tok_idx, cu_k = gathered

        expected = torch.tensor(
            [slot for slots in req_to_token for slot in slots], dtype=torch.long
        )
        self.assertEqual(tok_idx.tolist(), expected.tolist())
        self.assertEqual(cu_k.tolist(), kv_indptr.tolist())

    def test_page_sizes(self):
        # Chunked prefill shapes: several sequences with a long prefix.
        for page_size in (1, 16, 64):
            with self.subTest(page_size=page_size):
                self._check([43616, 1024, 512], page_size)

    def test_single_sequence(self):
        for page_size in (1, 16, 64):
            with self.subTest(page_size=page_size):
                self._check([2048], page_size)

    def test_unaligned_seq_lens(self):
        for page_size in (16, 64):
            with self.subTest(page_size=page_size):
                self._check([1000, 33, 1], page_size)

    def test_falls_back_when_metadata_is_short(self):
        # seq_lens longer than kv_indices has entries for (mixed/spec batches):
        # clamped, not an out-of-bounds gather.
        req_to_token, num_kv_slots = _paged_pool([512, 512], 64, seed=7)
        kv_indptr, kv_indices, lens = _token_level_metadata(req_to_token)
        gathered = _asm_context_prefill_gather_indices(
            kv_indptr, kv_indices, lens + 128, num_kv_slots
        )
        self.assertIsNotNone(gathered)
        tok_idx, cu_k = gathered
        self.assertEqual(cu_k.tolist(), kv_indptr.tolist())
        self.assertEqual(int(tok_idx.numel()), int(lens.sum()))

    def test_falls_back_when_slot_exceeds_pool(self):
        req_to_token, _ = _paged_pool([512, 512], 64, seed=11)
        kv_indptr, kv_indices, lens = _token_level_metadata(req_to_token)
        self.assertIsNone(
            _asm_context_prefill_gather_indices(kv_indptr, kv_indices, lens, 16)
        )


if __name__ == "__main__":
    unittest.main()
