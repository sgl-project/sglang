"""Unit test for the device-side page-table build used by trtllm_mha.

trtllm_mha builds its CUDA-graph block table on-device from ``seq_lens`` (via
``create_trtllm_mha_kv_indices_triton``) instead of a host-max PyTorch gather, so
it never reads a runtime max (no D2H sync). This test checks the device build is
bit-identical to the legacy gather for the columns each request uses, for both
the full page table and the SWA-translated page table, across context lengths,
page sizes, and batch sizes.
"""

import unittest

import torch

from sglang.srt.layers.attention.triton_ops.trtllm_mha_page_table import (
    create_trtllm_mha_kv_indices_triton,
    get_num_mha_kv_index_blocks,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Triton kernel unit test for the trtllm_mha device-side page-table build.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _reference_gather(req_to_token, req_pool_indices, page_size, max_seq_pages):
    """Legacy trtllm_mha build: strided gather of token slots, then // page_size."""
    strided = torch.arange(
        0, req_to_token.shape[1], page_size, device=req_to_token.device
    )[:max_seq_pages]
    page_indices = req_to_token[req_pool_indices[:, None], strided[None, :]]
    return page_indices  # raw token slots at page boundaries


def _kernel_build(
    req_to_token,
    req_pool_indices,
    cache_seqlens,
    page_size,
    max_num_pages,
    full_to_swa=None,
):
    dev = req_to_token.device
    bs = req_pool_indices.shape[0]
    page_table = torch.zeros((bs, max_num_pages), dtype=torch.int32, device=dev)
    has_swa = full_to_swa is not None
    swa_page_table = (
        torch.zeros((bs, max_num_pages), dtype=torch.int32, device=dev)
        if has_swa
        else None
    )
    create_trtllm_mha_kv_indices_triton[
        (bs, get_num_mha_kv_index_blocks(max_num_pages, page_size))
    ](
        req_to_token,
        req_pool_indices,
        cache_seqlens,
        full_to_swa,
        page_table,
        swa_page_table,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        HAS_SWA=has_swa,
    )
    return page_table, swa_page_table


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTrtllmMhaPageTable(CustomTestCase):
    def _run_case(self, max_context_len, page_size, num_reqs, bs, swa=False):
        torch.manual_seed(0)
        dev = "cuda"
        max_num_pages = (max_context_len + page_size - 1) // page_size
        n_slots = num_reqs * max_context_len
        req_to_token = torch.randint(
            0, n_slots, (num_reqs, max_context_len), dtype=torch.int32, device=dev
        )
        req_pool_indices = torch.randperm(num_reqs, device=dev)[:bs].to(torch.int32)
        cache_seqlens = torch.randint(
            1, max_context_len + 1, (bs,), dtype=torch.int32, device=dev
        )
        full_to_swa = None
        if swa:
            # Arbitrary full-slot -> SWA-slot lookup table.
            full_to_swa = torch.randint(
                0, n_slots, (n_slots,), dtype=torch.int32, device=dev
            )

        pt_kernel, swa_kernel = _kernel_build(
            req_to_token,
            req_pool_indices,
            cache_seqlens,
            page_size,
            max_num_pages,
            full_to_swa=full_to_swa,
        )

        max_len = int(cache_seqlens.max().item())
        max_seq_pages = (max_len + page_size - 1) // page_size
        slots_ref = _reference_gather(
            req_to_token, req_pool_indices, page_size, max_seq_pages
        )
        pt_ref = slots_ref // page_size
        swa_ref = (full_to_swa[slots_ref] // page_size) if swa else None

        for i in range(bs):
            npages = (int(cache_seqlens[i].item()) + page_size - 1) // page_size
            self.assertTrue(
                torch.equal(pt_kernel[i, :npages], pt_ref[i, :npages]),
                f"page_table mismatch req={i} max_ctx={max_context_len} "
                f"page_size={page_size} bs={bs} swa={swa}",
            )
            if swa:
                self.assertTrue(
                    torch.equal(swa_kernel[i, :npages], swa_ref[i, :npages]),
                    f"swa_page_table mismatch req={i} max_ctx={max_context_len} "
                    f"page_size={page_size} bs={bs}",
                )

    def test_matches_reference_gather(self):
        for max_ctx in (2048, 4096, 131072):
            for page_size in (1, 32, 64, 128):
                for bs in (1, 7, 32):
                    self._run_case(max_ctx, page_size, num_reqs=max(64, bs), bs=bs)

    def test_swa_matches_reference(self):
        for max_ctx in (2048, 4096):
            for page_size in (1, 64, 128):
                for bs in (1, 7, 32):
                    self._run_case(
                        max_ctx, page_size, num_reqs=max(64, bs), bs=bs, swa=True
                    )


if __name__ == "__main__":
    unittest.main()
