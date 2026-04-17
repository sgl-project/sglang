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


# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFusedDecodeMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _make_inputs(bs, max_batch, max_context_len, page_size, seq_len_delta):
        """Build random inputs shaped the same way the backend would pass them."""
        device = get_device()

        req_to_token = torch.randint(
            0,
            max_batch * max_context_len,
            (max_batch, max_context_len),
            dtype=torch.int64,
            device=device,
        )
        req_pool_indices = torch.randperm(max_batch, device=device)[:bs].to(torch.int64)

        upper = max(max_context_len // 2, 2)
        seq_lens = torch.randint(1, upper, (bs,), dtype=torch.int32, device=device)

        strided_indices = torch.arange(0, max_context_len, page_size, device=device)
        max_num_pages = (max_context_len + page_size - 1) // page_size
        max_len = int(seq_lens.max().item()) + seq_len_delta
        max_seq_pages = (max_len + page_size - 1) // page_size

        return {
            "req_to_token": req_to_token,
            "req_pool_indices": req_pool_indices,
            "seq_lens": seq_lens,
            "strided_indices": strided_indices,
            "max_num_pages": max_num_pages,
            "max_seq_pages": max_seq_pages,
        }

    @staticmethod
    def _empty_outputs(bs, max_num_pages):
        device = get_device()
        return (
            torch.zeros(bs, dtype=torch.int32, device=device),
            torch.zeros(bs + 1, dtype=torch.int32, device=device),
            torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device),
        )

    def _assert_match(self, bs, max_batch, max_context_len, page_size, seq_len_delta):
        """Run both paths and assert bit-exact equality on all three outputs."""
        inp = self._make_inputs(
            bs, max_batch, max_context_len, page_size, seq_len_delta
        )
        max_seq_pages = inp["max_seq_pages"]

        ref_cache, ref_cu, ref_pt = self._empty_outputs(bs, inp["max_num_pages"])
        _reference_decode_set_metadata(
            ref_cache,
            ref_cu,
            ref_pt,
            inp["req_to_token"],
            inp["req_pool_indices"],
            inp["strided_indices"],
            max_seq_pages,
            inp["seq_lens"],
            seq_len_delta,
            page_size,
        )

        fused_cache, fused_cu, fused_pt = self._empty_outputs(bs, inp["max_num_pages"])
        fused_normal_decode_set_metadata(
            fused_cache,
            fused_cu,
            fused_pt,
            inp["req_to_token"],
            inp["req_pool_indices"],
            inp["strided_indices"],
            max_seq_pages,
            inp["seq_lens"],
            seq_len_delta,
            page_size,
        )

        ctx = f"bs={bs}, page_size={page_size}, delta={seq_len_delta}"
        self.assertTrue(
            torch.equal(ref_cache, fused_cache), f"cache_seqlens mismatch ({ctx})"
        )
        self.assertTrue(torch.equal(ref_cu, fused_cu), f"cu_seqlens_k mismatch ({ctx})")
        self.assertTrue(torch.equal(ref_pt, fused_pt), f"page_table mismatch ({ctx})")

    # -- parameterized correctness sweeps -------------------------------

    def test_combined_sweep(self):
        """page_size x batch_size x seq_len_delta cross-product."""
        for page_size in (1, 16, 64, 128):
            for bs in (1, 37, 128, 512):
                for delta in (0, 1, 5):
                    with self.subTest(page_size=page_size, bs=bs, delta=delta):
                        self._assert_match(
                            bs=bs,
                            max_batch=1024,
                            max_context_len=4096,
                            page_size=page_size,
                            seq_len_delta=delta,
                        )

    def test_odd_batch_sizes(self):
        """Non-power-of-2 batch sizes stress irregular grid dimensions."""
        for bs in (3, 7, 37, 127, 513):
            with self.subTest(bs=bs):
                self._assert_match(
                    bs=bs,
                    max_batch=1024,
                    max_context_len=4096,
                    page_size=16,
                    seq_len_delta=0,
                )

    def test_very_large_batch(self):
        self._assert_match(
            bs=1024,
            max_batch=2048,
            max_context_len=4096,
            page_size=16,
            seq_len_delta=1,
        )

    def test_context_length_extremes(self):
        """Tiny and long contexts, plus page_size=1 over long context."""
        cases = [
            dict(
                bs=32,
                max_batch=64,
                max_context_len=128,
                page_size=1,
                seq_len_delta=0,
            ),
            dict(
                bs=64,
                max_batch=256,
                max_context_len=16384,
                page_size=64,
                seq_len_delta=0,
            ),
            dict(
                bs=16,
                max_batch=64,
                max_context_len=8192,
                page_size=1,
                seq_len_delta=0,
            ),
        ]
        for case in cases:
            with self.subTest(**case):
                self._assert_match(**case)

    # -- boundary / validation ------------------------------------------

    def test_empty_batch_is_noop(self):
        """bs=0 should early-return without touching any tensor."""
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
        self.assertEqual(cu.sum().item(), 0)

    def test_max_seq_pages_equals_capacity(self):
        """max_seq_pages exactly fills page_table (OOB-check boundary)."""
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
        seq_lens = torch.full(
            (bs,), max_context_len - 1, dtype=torch.int32, device=device
        )
        max_seq_pages = (int(seq_lens.max().item()) + page_size - 1) // page_size
        self.assertEqual(max_seq_pages, max_num_pages)

        strided_indices = torch.arange(0, max_context_len, page_size, device=device)

        ref_cache, ref_cu, ref_pt = self._empty_outputs(bs, max_num_pages)
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

        fused_cache, fused_cu, fused_pt = self._empty_outputs(bs, max_num_pages)
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
        device = get_device()
        with self.assertRaises(ValueError):
            fused_normal_decode_set_metadata(
                cache_seqlens_int32=torch.zeros(4, dtype=torch.int32, device=device),
                cu_seqlens_k=torch.zeros(5, dtype=torch.int32, device=device),
                page_table=torch.zeros(4, 8, dtype=torch.int32, device=device),
                req_to_token=torch.zeros(16, 4096, dtype=torch.int64, device=device),
                req_pool_indices=torch.arange(4, dtype=torch.int64, device=device),
                strided_indices=torch.arange(0, 4096, 1, device=device),
                max_seq_pages=10,
                seq_lens=torch.full((4,), 100, dtype=torch.int32, device=device),
                seq_len_delta=0,
                page_size=1,
            )

    # -- SWA fallback ---------------------------------------------------

    def test_swa_fallback_path(self):
        """SWA path with mocked token_to_kv_pool exercises the PyTorch fallback."""
        device = get_device()
        bs, max_batch, max_context_len, page_size = 32, 64, 4096, 16
        inp = self._make_inputs(bs, max_batch, max_context_len, page_size, 0)
        max_num_pages = inp["max_num_pages"]
        max_seq_pages = inp["max_seq_pages"]

        ref_cache, ref_cu, ref_pt = self._empty_outputs(bs, max_num_pages)
        _reference_decode_set_metadata(
            ref_cache,
            ref_cu,
            ref_pt,
            inp["req_to_token"],
            inp["req_pool_indices"],
            inp["strided_indices"],
            max_seq_pages,
            inp["seq_lens"],
            0,
            page_size,
        )

        page_indices = inp["req_to_token"][
            inp["req_pool_indices"][:, None],
            inp["strided_indices"][:max_seq_pages][None, :],
        ]
        expected_swa_pt = torch.zeros(
            bs, max_num_pages, dtype=torch.int32, device=device
        )
        expected_swa_pt[:, :max_seq_pages].copy_(
            (page_indices // page_size).to(torch.int32)
        )

        mock_pool = MagicMock()
        mock_pool.translate_loc_from_full_to_swa = lambda x: x  # identity

        fused_cache, fused_cu, fused_pt = self._empty_outputs(bs, max_num_pages)
        swa_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        fused_normal_decode_set_metadata(
            fused_cache,
            fused_cu,
            fused_pt,
            inp["req_to_token"],
            inp["req_pool_indices"],
            inp["strided_indices"],
            max_seq_pages,
            inp["seq_lens"],
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
        with self.assertRaises(ValueError):
            fused_normal_decode_set_metadata(
                cache_seqlens_int32=torch.zeros(4, dtype=torch.int32, device=device),
                cu_seqlens_k=torch.zeros(5, dtype=torch.int32, device=device),
                page_table=torch.zeros(4, 64, dtype=torch.int32, device=device),
                req_to_token=torch.zeros(16, 4096, dtype=torch.int64, device=device),
                req_pool_indices=torch.arange(4, dtype=torch.int64, device=device),
                strided_indices=torch.arange(0, 4096, 1, device=device),
                max_seq_pages=1,
                seq_lens=torch.full((4,), 10, dtype=torch.int32, device=device),
                seq_len_delta=0,
                page_size=1,
                swa_page_table=torch.zeros(4, 64, dtype=torch.int32, device=device),
                token_to_kv_pool=None,
            )


if __name__ == "__main__":
    unittest.main()
