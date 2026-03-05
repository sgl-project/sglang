import unittest

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
    cache_seqlens_int32, cu_seqlens_k, page_table,
    req_to_token, req_pool_indices, strided_indices,
    max_seq_pages, seq_lens, seq_len_delta, page_size,
):
    """Pure-PyTorch reference matching normal_decode_set_metadata (no SWA)."""
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32)
    )
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
        seq_lens = torch.randint(1, max(max_seq, 2), (bs,), dtype=torch.int32, device=device)

        max_len = seq_lens.max().item() + seq_len_delta
        max_num_pages = (max_context_len + page_size - 1) // page_size
        max_seq_pages = (max_len + page_size - 1) // page_size

        strided_indices = torch.arange(
            0, max_context_len, page_size, device=device
        )

        # --- Reference outputs ---
        ref_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        ref_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        ref_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        _reference_decode_set_metadata(
            ref_cache, ref_cu, ref_pt,
            req_to_token, req_pool_indices, strided_indices,
            max_seq_pages, seq_lens, seq_len_delta, page_size,
        )

        # --- Fused outputs ---
        fused_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        fused_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        fused_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        fused_normal_decode_set_metadata(
            fused_cache, fused_cu, fused_pt,
            req_to_token, req_pool_indices, strided_indices,
            max_seq_pages, seq_lens, seq_len_delta, page_size,
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

    def test_page_size_1(self):
        for bs in [1, 32, 128]:
            self._run_test(bs=bs, max_batch=256, max_context_len=4096, page_size=1, seq_len_delta=0)

    def test_page_size_16(self):
        for bs in [1, 32, 128]:
            self._run_test(bs=bs, max_batch=256, max_context_len=4096, page_size=16, seq_len_delta=0)

    def test_page_size_64(self):
        for bs in [1, 32, 128]:
            self._run_test(bs=bs, max_batch=256, max_context_len=4096, page_size=64, seq_len_delta=0)

    def test_nonzero_delta(self):
        for delta in [1, 3, 5]:
            self._run_test(bs=64, max_batch=256, max_context_len=4096, page_size=16, seq_len_delta=delta)

    def test_large_batch(self):
        self._run_test(bs=512, max_batch=1024, max_context_len=8192, page_size=64, seq_len_delta=0)

    def test_single_request(self):
        self._run_test(bs=1, max_batch=16, max_context_len=512, page_size=1, seq_len_delta=0)

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


if __name__ == "__main__":
    unittest.main()
