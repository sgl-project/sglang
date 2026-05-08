"""DSFlashAttentionAdaptor correctness + CUDA-graph capture/replay.

Three properties pinned:
  1. Adaptor produces FA3-compatible metadata: page_table, cache_seqlens,
     cu_seqlens_k all consistent with the selected logical token sets.
  2. `max_seq_len_k` is set at construction and never modified per call
     (no host sync inside the captured graph).
  3. Capture+replay is allocation-free across replays; replays see updated
     selected sets via in-place writes into the captured graph buffers.
"""

import unittest
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.backend.ds_flash_attention_adaptor import (
    DSFlashAttentionAdaptor,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")


@dataclass
class _FA3MetadataLike:
    page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_k: int
    scheduler_metadata: Optional[object] = "DENSE_PLACEHOLDER"


@dataclass
class _Batch:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


def _make_metadata(bs, max_pages, max_seq_len_k, device):
    """Build a captured-graph-style metadata block."""
    return _FA3MetadataLike(
        page_table=torch.zeros(bs, max_pages, dtype=torch.int32, device=device),
        cache_seqlens_int32=torch.zeros(bs, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.zeros(bs + 1, dtype=torch.int32, device=device),
        max_seq_len_k=max_seq_len_k,
    )


def _make_req_to_token(num_reqs, max_ctx, T, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    r2t = torch.zeros(num_reqs, max_ctx, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        perm = torch.randperm(T, generator=g, device=device)[:max_ctx]
        r2t[r] = perm.to(torch.int32)
    return r2t


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDSAdaptor(CustomTestCase):
    BS = 2
    MAX_PAGES = 64  # captured FA3 page_table cap; must be >= any dense seq_len in tests
    MAX_CTX = 64
    T = 256
    MAX_SELECTED = 24

    def setUp(self):
        self.device = torch.device("cuda")
        self.adaptor = DSFlashAttentionAdaptor(
            self.device, max_selected_per_request=self.MAX_SELECTED
        )
        self.req_to_token = _make_req_to_token(
            self.BS, self.MAX_CTX, self.T, self.device
        )

    def _build_call(self, selected_logical, valid_lens, sparse_mask, dense_seq_lens):
        meta = _make_metadata(self.BS, self.MAX_PAGES, self.MAX_SELECTED, self.device)
        meta.cache_seqlens_int32.copy_(
            torch.tensor(dense_seq_lens, dtype=torch.int32, device=self.device)
        )
        # Initial dense page_table[i, :seq_lens[i]] = req_to_token row.
        for i in range(self.BS):
            sl = dense_seq_lens[i]
            meta.page_table[i, :sl] = self.req_to_token[i, :sl]
        torch.cumsum(
            meta.cache_seqlens_int32,
            dim=0,
            dtype=torch.int32,
            out=meta.cu_seqlens_k[1:],
        )
        # save_original_metadata snapshot
        self.adaptor.save_original_metadata(meta)
        forward_batch = _Batch(
            req_pool_indices=torch.arange(
                self.BS, dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor(
                dense_seq_lens, dtype=torch.int64, device=self.device
            ),
        )
        sel_t = torch.tensor(selected_logical, dtype=torch.int32, device=self.device)
        valid_t = torch.tensor(valid_lens, dtype=torch.int32, device=self.device)
        mask_t = torch.tensor(sparse_mask, dtype=torch.bool, device=self.device)
        return meta, sel_t, valid_t, mask_t, forward_batch

    def test_scheduler_metadata_set_to_none_on_save(self):
        meta = _make_metadata(self.BS, self.MAX_PAGES, self.MAX_SELECTED, self.device)
        self.adaptor.save_original_metadata(meta)
        self.assertIsNone(meta.scheduler_metadata)

    def test_max_seq_len_k_static_after_adapt(self):
        # The adaptor must not modify max_seq_len_k; it's set at construction.
        sel = [[0, 5, 10, -1] + [-1] * 20, [3, 7, 11, 15] + [-1] * 20]
        valid = [3, 4]
        mask = [True, True]
        seqs = [16, 32]
        meta, sel_t, valid_t, mask_t, fb = self._build_call(sel, valid, mask, seqs)
        before = meta.max_seq_len_k
        self.adaptor.adapt_for_attn_metadata(
            sel_t, valid_t, mask_t, meta, fb, self.req_to_token, page_size=1, layer_id=0
        )
        self.assertEqual(meta.max_seq_len_k, before)
        self.assertEqual(meta.max_seq_len_k, self.MAX_SELECTED)

    def test_page_table_logical_to_physical(self):
        # Sparsified row 0 selects logical [0, 3, 7]; dense row 1 untouched.
        sel = [[0, 3, 7] + [-1] * 21, [-1] * 24]
        valid = [3, 0]
        mask = [True, False]
        seqs = [10, 20]
        meta, sel_t, valid_t, mask_t, fb = self._build_call(sel, valid, mask, seqs)
        self.adaptor.adapt_for_attn_metadata(
            sel_t, valid_t, mask_t, meta, fb, self.req_to_token, page_size=1, layer_id=0
        )
        # First 3 entries of row 0 page_table = req_to_token[0, [0,3,7]]
        expected = self.req_to_token[0, [0, 3, 7]]
        self.assertTrue(torch.equal(meta.page_table[0, :3], expected))
        # Row 1 not sparsified: original layout preserved.
        self.assertTrue(torch.equal(meta.page_table[1, :20], self.req_to_token[1, :20]))

    def test_cache_seqlens_and_cu_seqlens_match_valid_lengths(self):
        sel = [[0, 1, 2, 3, 4, 5] + [-1] * 18, [10, 11, 12, 13] + [-1] * 20]
        valid = [6, 4]
        mask = [True, True]
        seqs = [40, 32]
        meta, sel_t, valid_t, mask_t, fb = self._build_call(sel, valid, mask, seqs)
        self.adaptor.adapt_for_attn_metadata(
            sel_t, valid_t, mask_t, meta, fb, self.req_to_token, page_size=1, layer_id=0
        )
        self.assertEqual(meta.cache_seqlens_int32.tolist(), valid)
        self.assertEqual(meta.cu_seqlens_k.tolist(), [0, 6, 10])

    def test_dense_fallback_row_keeps_original_seqlen(self):
        # mask=False on row 1 → cache_seqlens_int32[1] stays at original.
        sel = [[0, 5] + [-1] * 22, [-1] * 24]
        valid = [2, 0]
        mask = [True, False]
        seqs = [16, 32]
        meta, sel_t, valid_t, mask_t, fb = self._build_call(sel, valid, mask, seqs)
        self.adaptor.adapt_for_attn_metadata(
            sel_t, valid_t, mask_t, meta, fb, self.req_to_token, page_size=1, layer_id=0
        )
        self.assertEqual(meta.cache_seqlens_int32[1].item(), 32)
        # cu_seqlens reflects: [0, 2, 34]
        self.assertEqual(meta.cu_seqlens_k.tolist(), [0, 2, 34])

    def test_page_size_not_one_raises(self):
        sel = [[0] + [-1] * 23, [-1] * 24]
        valid = [1, 0]
        mask = [True, False]
        seqs = [16, 32]
        meta, sel_t, valid_t, mask_t, fb = self._build_call(sel, valid, mask, seqs)
        with self.assertRaisesRegex(ValueError, "page_size=1"):
            self.adaptor.adapt_for_attn_metadata(
                sel_t,
                valid_t,
                mask_t,
                meta,
                fb,
                self.req_to_token,
                page_size=4,
                layer_id=0,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCudaGraphCaptureReplay(CustomTestCase):
    """Capture the adaptor in a CUDA graph, replay 100x with mutated input
    tensors, verify outputs reflect each replay's input — and crucially,
    that no per-replay tensor allocation happens."""

    BS = 2
    MAX_PAGES = 64  # captured FA3 page_table cap
    MAX_CTX = 64
    T = 256
    MAX_SELECTED = 16

    def test_capture_replay_correctness_and_no_alloc(self):
        device = torch.device("cuda")
        adaptor = DSFlashAttentionAdaptor(
            device, max_selected_per_request=self.MAX_SELECTED
        )
        r2t = _make_req_to_token(self.BS, self.MAX_CTX, self.T, device, seed=11)

        # Build the captured-graph-style buffers ONCE. All future writes
        # are .copy_() / out= into these same buffers.
        meta = _make_metadata(self.BS, self.MAX_PAGES, self.MAX_SELECTED, device)

        # "Dense" page_table fill (one-time setup before capture).
        for i in range(self.BS):
            meta.page_table[i, : self.MAX_CTX] = r2t[i]
        meta.cache_seqlens_int32.copy_(
            torch.tensor([self.MAX_CTX, self.MAX_CTX], dtype=torch.int32, device=device)
        )
        torch.cumsum(
            meta.cache_seqlens_int32,
            dim=0,
            dtype=torch.int32,
            out=meta.cu_seqlens_k[1:],
        )
        adaptor.save_original_metadata(meta)

        # Persistent input buffers — these are what we mutate per replay.
        sel_buf = torch.full(
            (self.BS, self.MAX_SELECTED), -1, dtype=torch.int32, device=device
        )
        valid_buf = torch.zeros(self.BS, dtype=torch.int32, device=device)
        mask_buf = torch.ones(self.BS, dtype=torch.bool, device=device)
        fb = _Batch(
            req_pool_indices=torch.arange(self.BS, dtype=torch.int64, device=device),
            seq_lens=torch.tensor(
                [self.MAX_CTX, self.MAX_CTX], dtype=torch.int64, device=device
            ),
        )

        def _set_inputs(idx_lists, lengths):
            sel_buf.fill_(-1)
            for i, lst in enumerate(idx_lists):
                sel_buf[i, : len(lst)] = torch.tensor(
                    lst, dtype=torch.int32, device=device
                )
            valid_buf.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))

        # Warmup before graph capture (Triton autotune / cuBLAS workspace etc.).
        _set_inputs([[0, 1, 2, 3], [10, 11, 12, 13]], [4, 4])
        for _ in range(3):
            adaptor.adapt_for_attn_metadata(
                sel_buf, valid_buf, mask_buf, meta, fb, r2t, page_size=1, layer_id=0
            )
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            adaptor.adapt_for_attn_metadata(
                sel_buf, valid_buf, mask_buf, meta, fb, r2t, page_size=1, layer_id=0
            )

        # Replay with varied inputs; verify outputs match a fresh non-graph call.
        scenarios = [
            ([[0, 5, 9], [4, 8, 12]], [3, 3]),
            ([[1, 2, 3, 4, 5], [10, 11]], [5, 2]),
            ([[7, 8, 9, 10, 11, 12, 13, 14], [0, 1]], [8, 2]),
        ]
        for idx_lists, lens in scenarios:
            _set_inputs(idx_lists, lens)
            graph.replay()
            torch.cuda.synchronize()

            self.assertEqual(meta.cache_seqlens_int32.tolist(), lens)
            for i, lst in enumerate(idx_lists):
                expected = r2t[i, lst]
                self.assertTrue(torch.equal(meta.page_table[i, : len(lst)], expected))
            # cu_seqlens prefix sum
            expected_cu = [0] + [sum(lens[: i + 1]) for i in range(self.BS)]
            self.assertEqual(meta.cu_seqlens_k.tolist(), expected_cu)

        # Allocation count check: replay 50 more times, snapshot allocator
        # before/after to confirm no per-replay growth.
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(50):
            graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        self.assertEqual(
            after,
            before,
            f"replay allocated {after - before} bytes (should be 0)",
        )


if __name__ == "__main__":
    unittest.main()
