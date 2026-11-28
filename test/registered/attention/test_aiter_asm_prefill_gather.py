"""Index and layout arithmetic for the aiter prefill paths.

kv_indptr/kv_indices are token-level for every page_size, so the gather must
resolve token t of sequence i to kv_indices[kv_indptr[i] + t]. The NHD batch
prefill hands q to its kernel as a view when q's layout allows. Pure indexing
and a stand-in kernel, so this runs on CPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.aiter_backend import (
    AiterAttnBackend,
    _asm_context_prefill_gather_indices,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd")


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


class TestBatchPrefillQLayout(unittest.TestCase):
    """What ``forward_extend`` hands ``mha_batch_prefill_func`` as q."""

    num_heads, head_dim, num_kv_heads, tokens = 4, 8, 2, 37

    def _run(self, q):
        """Run the NHD batch prefill on q with a stand-in kernel; return the q it got."""
        kv_cache = torch.zeros(self.tokens, self.num_kv_heads, self.head_dim)
        kv = SimpleNamespace(kv=lambda: (kv_cache, kv_cache))
        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = False
        backend.kv_cache_is_vectorized_5d = False
        backend.use_triton_unified_attention = False
        backend._use_unified_verify = False
        backend.kv_cache_dtype = torch.float32
        backend.input_dtype = torch.float32
        backend.page_size = 1
        backend.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda layer_id: kv)
        )
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.tensor([0, self.tokens], dtype=torch.int32),
            kv_indptr=torch.tensor([0, self.tokens], dtype=torch.int32),
            kv_indices=torch.arange(self.tokens, dtype=torch.int32),
            swa_page_table=None,
            paged_kv_view=None,
            max_q_len=self.tokens,
            max_kv_len=self.tokens,
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=self.num_heads,
            tp_k_head_num=self.num_kv_heads,
            tp_v_head_num=self.num_kv_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            logit_cap=0.0,
            sliding_window_size=None,
            is_cross_attention=False,
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            attn_attend_prefix_cache=False,
            extend_prefix_lens_cpu=[0],
            out_cache_loc=None,
        )
        seen = []

        def kernel(q_heads, *args, **kwargs):
            seen.append(q_heads)
            return torch.zeros(q_heads.shape)

        with (
            patch.object(aiter_backend, "mha_batch_prefill_func", kernel, create=True),
            patch.object(aiter_backend, "is_gfx95_supported", lambda: False),
        ):
            backend.forward_extend(
                q, None, None, layer, forward_batch, save_kv_cache=False
            )
        (q_heads,) = seen
        return q_heads

    def test_fused_qkv_slice_is_not_copied(self):
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        qkv = torch.randn(self.tokens, q_size + 2 * kv_size)
        q, _, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        expected = q.clone().view(-1, self.num_heads, self.head_dim)

        q_heads = self._run(q)
        self.assertEqual(q_heads.data_ptr(), qkv.data_ptr())
        self.assertEqual(q_heads.stride(), (qkv.stride(0), self.head_dim, 1))
        torch.testing.assert_close(q_heads, expected, rtol=0, atol=0)

    def test_strided_head_dim_is_still_copied(self):
        q = torch.randn(self.num_heads * self.head_dim, self.tokens).t()
        expected = q.clone().view(-1, self.num_heads, self.head_dim)

        q_heads = self._run(q)
        self.assertTrue(q_heads.is_contiguous())
        torch.testing.assert_close(q_heads, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
