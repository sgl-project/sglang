import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.fused_decode_metadata import (
    fused_normal_decode_set_metadata,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


def _reference_decode_set_metadata(
    cache_seqlens_int32,
    cu_seqlens_k,
    page_table,
    req_to_token,
    req_pool_indices,
    strided_indices,
    max_seq_pages,
    seq_lens,
    seq_len_delta,
    page_size,
):
    """Pure-PyTorch reference matching normal_decode_set_metadata (no SWA)."""
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32))
    page_indices = req_to_token[
        req_pool_indices[:, None],
        strided_indices[:max_seq_pages][None, :],
    ]
    page_table[:, :max_seq_pages].copy_(page_indices // page_size)


class TestFusedDecodeMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    def _run_test(self, bs, max_batch, max_context_len, page_size, seq_len_delta):
        """Run both reference and fused, assert bit-exact match."""
        device = get_device()

        req_to_token = torch.randint(
            0,
            max_batch * max_context_len,
            (max_batch, max_context_len),
            dtype=torch.int64,
            device=device,
        )

        perm = torch.randperm(max_batch, device=device)[:bs]
        req_pool_indices = perm.to(torch.int64)

        max_seq = max_context_len // 2
        seq_lens = torch.randint(
            1, max(max_seq, 2), (bs,), dtype=torch.int32, device=device
        )

        max_len = seq_lens.max().item() + seq_len_delta
        max_num_pages = (max_context_len + page_size - 1) // page_size
        max_seq_pages = (max_len + page_size - 1) // page_size

        strided_indices = torch.arange(0, max_context_len, page_size, device=device)

        # --- Reference outputs ---
        ref_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        ref_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        ref_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        _reference_decode_set_metadata(
            ref_cache,
            ref_cu,
            ref_pt,
            req_to_token,
            req_pool_indices,
            strided_indices,
            max_seq_pages,
            seq_lens,
            seq_len_delta,
            page_size,
        )

        # --- Fused outputs ---
        fused_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        fused_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        fused_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        fused_normal_decode_set_metadata(
            fused_cache,
            fused_cu,
            fused_pt,
            req_to_token,
            req_pool_indices,
            strided_indices,
            max_seq_pages,
            seq_lens,
            seq_len_delta,
            page_size,
        )

        # --- Assertions ---
        self.assertTrue(
            torch.equal(ref_cache, fused_cache),
            f"cache_seqlens mismatch for bs={bs}, page_size={page_size}, delta={seq_len_delta}",
        )
        self.assertTrue(
            torch.equal(ref_cu, fused_cu),
            f"cu_seqlens_k mismatch for bs={bs}, page_size={page_size}, delta={seq_len_delta}",
        )
        self.assertTrue(
            torch.equal(ref_pt, fused_pt),
            f"page_table mismatch for bs={bs}, page_size={page_size}, delta={seq_len_delta}",
        )

    # -----------------------------------------------------------------------
    # Page size variations
    # -----------------------------------------------------------------------

    def test_page_size_1(self):
        for bs in [1, 32, 128, 512]:
            self._run_test(
                bs=bs,
                max_batch=1024,
                max_context_len=4096,
                page_size=1,
                seq_len_delta=0,
            )

    def test_page_size_16(self):
        for bs in [1, 32, 128, 512]:
            self._run_test(
                bs=bs,
                max_batch=1024,
                max_context_len=4096,
                page_size=16,
                seq_len_delta=0,
            )

    def test_page_size_64(self):
        for bs in [1, 32, 128, 512]:
            self._run_test(
                bs=bs,
                max_batch=1024,
                max_context_len=4096,
                page_size=64,
                seq_len_delta=0,
            )

    def test_page_size_128(self):
        for bs in [1, 64, 256]:
            self._run_test(
                bs=bs,
                max_batch=512,
                max_context_len=8192,
                page_size=128,
                seq_len_delta=0,
            )

    # -----------------------------------------------------------------------
    # Non-zero seq_len_delta
    # -----------------------------------------------------------------------

    def test_nonzero_delta(self):
        for delta in [1, 3, 5, 10]:
            for bs in [1, 64, 256]:
                self._run_test(
                    bs=bs,
                    max_batch=512,
                    max_context_len=4096,
                    page_size=16,
                    seq_len_delta=delta,
                )

    # -----------------------------------------------------------------------
    # Batch size sweep
    # -----------------------------------------------------------------------

    def test_single_request(self):
        self._run_test(
            bs=1, max_batch=16, max_context_len=512, page_size=1, seq_len_delta=0
        )

    def test_large_batch(self):
        self._run_test(
            bs=512, max_batch=1024, max_context_len=8192, page_size=64, seq_len_delta=0
        )

    def test_very_large_batch(self):
        self._run_test(
            bs=1024, max_batch=2048, max_context_len=4096, page_size=16, seq_len_delta=1
        )

    def test_odd_batch_sizes(self):
        """Non-power-of-2 batch sizes to stress irregular grid dimensions."""
        for bs in [3, 7, 37, 127, 513]:
            self._run_test(
                bs=bs,
                max_batch=1024,
                max_context_len=4096,
                page_size=16,
                seq_len_delta=0,
            )

    # -----------------------------------------------------------------------
    # max_context_len variations (affects max_seq_pages range)
    # -----------------------------------------------------------------------

    def test_small_context_len(self):
        self._run_test(
            bs=32, max_batch=64, max_context_len=128, page_size=1, seq_len_delta=0
        )

    def test_large_context_len(self):
        self._run_test(
            bs=64, max_batch=256, max_context_len=16384, page_size=64, seq_len_delta=0
        )

    def test_large_context_len_page_size_1(self):
        """Stress test: page_size=1 with long context forces many page iterations."""
        self._run_test(
            bs=16, max_batch=64, max_context_len=8192, page_size=1, seq_len_delta=0
        )

    # -----------------------------------------------------------------------
    # Boundary conditions
    # -----------------------------------------------------------------------

    def test_empty_batch(self):
        """bs=0 should be a no-op without errors."""
        device = get_device()
        cache = torch.zeros(0, dtype=torch.int32, device=device)
        cu = torch.zeros(1, dtype=torch.int32, device=device)
        pt = torch.zeros(0, 64, dtype=torch.int32, device=device)
        req_to_token = torch.zeros(1, 4096, dtype=torch.int64, device=device)
        req_pool = torch.zeros(0, dtype=torch.int64, device=device)
        strided = torch.arange(0, 4096, 1, device=device)
        seq = torch.zeros(0, dtype=torch.int32, device=device)

        fused_normal_decode_set_metadata(
            cache, cu, pt, req_to_token, req_pool, strided, 0, seq, 0, 1
        )

    def test_max_seq_pages_equals_capacity(self):
        """max_seq_pages exactly fills page_table — boundary of OOB check."""
        device = get_device()
        bs, max_batch, page_size = 16, 64, 16
        max_context_len = 1024
        max_num_pages = max_context_len // page_size

        req_to_token = torch.randint(
            0,
            max_batch * max_context_len,
            (max_batch, max_context_len),
            dtype=torch.int64,
            device=device,
        )
        req_pool_indices = torch.randperm(max_batch, device=device)[:bs].to(torch.int64)
        # Force seq_lens so max_seq_pages == max_num_pages
        seq_lens = torch.full(
            (bs,), max_context_len - 1, dtype=torch.int32, device=device
        )
        max_len = seq_lens.max().item()
        max_seq_pages = (max_len + page_size - 1) // page_size
        self.assertEqual(max_seq_pages, max_num_pages)

        strided_indices = torch.arange(0, max_context_len, page_size, device=device)

        ref_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        ref_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        ref_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)
        _reference_decode_set_metadata(
            ref_cache,
            ref_cu,
            ref_pt,
            req_to_token,
            req_pool_indices,
            strided_indices,
            max_seq_pages,
            seq_lens,
            0,
            page_size,
        )

        fused_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        fused_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        fused_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)
        fused_normal_decode_set_metadata(
            fused_cache,
            fused_cu,
            fused_pt,
            req_to_token,
            req_pool_indices,
            strided_indices,
            max_seq_pages,
            seq_lens,
            0,
            page_size,
        )

        self.assertTrue(torch.equal(ref_pt, fused_pt))

    def test_max_seq_pages_exceeds_capacity_raises(self):
        """max_seq_pages > page_table.shape[1] should raise ValueError."""
        device = get_device()
        cache = torch.zeros(4, dtype=torch.int32, device=device)
        cu = torch.zeros(5, dtype=torch.int32, device=device)
        pt = torch.zeros(4, 8, dtype=torch.int32, device=device)  # capacity=8
        req_to_token = torch.zeros(16, 4096, dtype=torch.int64, device=device)
        req_pool = torch.arange(4, dtype=torch.int64, device=device)
        strided = torch.arange(0, 4096, 1, device=device)
        seq = torch.full((4,), 100, dtype=torch.int32, device=device)

        with self.assertRaises(ValueError):
            fused_normal_decode_set_metadata(
                cache,
                cu,
                pt,
                req_to_token,
                req_pool,
                strided,
                max_seq_pages=10,  # > capacity 8
                seq_lens=seq,
                seq_len_delta=0,
                page_size=1,
            )

    # -----------------------------------------------------------------------
    # SWA fallback path
    # -----------------------------------------------------------------------

    def test_swa_fallback_path(self):
        """SWA path with mock token_to_kv_pool exercises the PyTorch fallback."""
        device = get_device()
        bs, max_batch, max_context_len, page_size = 32, 64, 4096, 16

        req_to_token = torch.randint(
            0,
            max_batch * max_context_len,
            (max_batch, max_context_len),
            dtype=torch.int64,
            device=device,
        )
        req_pool_indices = torch.randperm(max_batch, device=device)[:bs].to(torch.int64)
        seq_lens = torch.randint(
            1, max_context_len // 2, (bs,), dtype=torch.int32, device=device
        )
        max_len = seq_lens.max().item()
        max_num_pages = (max_context_len + page_size - 1) // page_size
        max_seq_pages = (max_len + page_size - 1) // page_size
        strided_indices = torch.arange(0, max_context_len, page_size, device=device)

        # Reference for the main page_table
        ref_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        ref_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        ref_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)
        _reference_decode_set_metadata(
            ref_cache,
            ref_cu,
            ref_pt,
            req_to_token,
            req_pool_indices,
            strided_indices,
            max_seq_pages,
            seq_lens,
            0,
            page_size,
        )

        # Expected SWA page indices (identity translation for test)
        page_indices = req_to_token[
            req_pool_indices[:, None],
            strided_indices[:max_seq_pages][None, :],
        ]
        expected_swa_pt = torch.zeros(
            bs, max_num_pages, dtype=torch.int32, device=device
        )
        # Mock translate_loc_from_full_to_swa as identity
        expected_swa_pt[:, :max_seq_pages].copy_(
            (page_indices // page_size).to(torch.int32)
        )

        # Mock SWAKVPool
        mock_pool = MagicMock()
        mock_pool.translate_loc_from_full_to_swa = lambda x: x  # identity

        fused_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        fused_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        fused_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)
        swa_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        fused_normal_decode_set_metadata(
            fused_cache,
            fused_cu,
            fused_pt,
            req_to_token,
            req_pool_indices,
            strided_indices,
            max_seq_pages,
            seq_lens,
            0,
            page_size,
            swa_page_table=swa_pt,
            token_to_kv_pool=mock_pool,
        )

        self.assertTrue(torch.equal(ref_cache, fused_cache))
        self.assertTrue(torch.equal(ref_pt, fused_pt))
        self.assertTrue(torch.equal(expected_swa_pt, swa_pt))

    def test_swa_partial_args_raises(self):
        """Providing only one of swa_page_table / token_to_kv_pool should raise."""
        device = get_device()
        cache = torch.zeros(4, dtype=torch.int32, device=device)
        cu = torch.zeros(5, dtype=torch.int32, device=device)
        pt = torch.zeros(4, 64, dtype=torch.int32, device=device)
        req_to_token = torch.zeros(16, 4096, dtype=torch.int64, device=device)
        req_pool = torch.arange(4, dtype=torch.int64, device=device)
        strided = torch.arange(0, 4096, 1, device=device)
        seq = torch.full((4,), 10, dtype=torch.int32, device=device)

        with self.assertRaises(ValueError):
            fused_normal_decode_set_metadata(
                cache,
                cu,
                pt,
                req_to_token,
                req_pool,
                strided,
                max_seq_pages=1,
                seq_lens=seq,
                seq_len_delta=0,
                page_size=1,
                swa_page_table=pt,
                token_to_kv_pool=None,
            )

    # -----------------------------------------------------------------------
    # Combined sweep (page_size x batch_size x delta)
    # -----------------------------------------------------------------------

    def test_combined_sweep(self):
        """Cross-product sweep similar to the parametrize style in jit_kernel tests."""
        for page_size in [1, 16, 64]:
            for bs in [1, 37, 128, 512]:
                for delta in [0, 1, 5]:
                    with self.subTest(page_size=page_size, bs=bs, delta=delta):
                        self._run_test(
                            bs=bs,
                            max_batch=1024,
                            max_context_len=4096,
                            page_size=page_size,
                            seq_len_delta=delta,
                        )


if __name__ == "__main__":
    unittest.main()
