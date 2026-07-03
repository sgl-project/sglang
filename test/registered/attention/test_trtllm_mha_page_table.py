"""Unit test for the device-side page-table build used by trtllm_mha.

trtllm_mha builds its CUDA-graph block table on-device from ``seq_lens`` (via
``create_trtllm_mha_kv_indices_triton``) instead of a host-max PyTorch gather, so
it never reads a runtime max (no D2H sync). This test checks the device build is
bit-identical to the legacy gather for the columns each request uses, for both
the full page table and the SWA-translated page table, across context lengths,
page sizes, and batch sizes.

It also pins the invariant that lets the no-host-max (GPU-only) path hand the
kernel a static ``max_num_pages``-wide buffer: every column past a request's
page count must be left untouched, i.e. the kernel bounds its writes by the
device-side ``cache_seqlens`` alone.
"""

import unittest
from typing import Optional

import torch

from sglang.srt.layers.attention.triton_ops.trtllm_mha_page_table import (
    build_trtllm_mha_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Triton kernel unit test for the trtllm_mha device-side page-table build.
register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-small")


def _build_page_table_reference(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    full_to_swa: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference impl: host-side strided gather, then // page_size.

    Sized to the batch max (uses a host-side ``.max()``, which the kernel path
    avoids). Returns the same (page_table, swa_page_table) block-id tables as
    ``_build_page_table_kernel`` for the columns each request uses.
    """
    max_len = int(cache_seqlens.max().item())
    max_seq_pages = (max_len + page_size - 1) // page_size
    strided = torch.arange(
        0, req_to_token.shape[1], page_size, device=req_to_token.device
    )[:max_seq_pages]
    slots = req_to_token[req_pool_indices[:, None], strided[None, :]]  # token slots
    page_table = slots // page_size
    swa_page_table = (
        full_to_swa[slots] // page_size if full_to_swa is not None else None
    )
    return page_table, swa_page_table


def _build_page_table_kernel(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    max_num_pages: int,
    full_to_swa: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Device-side impl."""
    dev = req_to_token.device
    bs = req_pool_indices.shape[0]
    page_table = torch.zeros((bs, max_num_pages), dtype=torch.int32, device=dev)
    swa_page_table = (
        torch.zeros((bs, max_num_pages), dtype=torch.int32, device=dev)
        if full_to_swa is not None
        else None
    )
    build_trtllm_mha_page_table(
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        page_size=page_size,
        swa_page_table=swa_page_table,
        full_to_swa=full_to_swa,
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

        pt_kernel, swa_kernel = _build_page_table_kernel(
            req_to_token,
            req_pool_indices,
            cache_seqlens,
            page_size,
            max_num_pages,
            full_to_swa=full_to_swa,
        )
        pt_ref, swa_ref = _build_page_table_reference(
            req_to_token,
            req_pool_indices,
            cache_seqlens,
            page_size,
            full_to_swa=full_to_swa,
        )

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
            for page_size in (1, 32, 64, 128, 256):
                for bs in (1, 7, 32):
                    self._run_case(max_ctx, page_size, num_reqs=max(64, bs), bs=bs)

    def test_swa_matches_reference(self):
        for max_ctx in (2048, 4096):
            for page_size in (1, 64, 128):
                for bs in (1, 7, 32):
                    self._run_case(
                        max_ctx, page_size, num_reqs=max(64, bs), bs=bs, swa=True
                    )

    def _run_self_guard_case(self, max_context_len, page_size, bs, swa=False):
        """Short sequences against a full static buffer -- the GPU-only shape.

        Pre-fill the page table with a sentinel and run the kernel with the
        static ``max_num_pages`` width (no host max to tighten it). Used columns
        must hold the right block ids; every tail column must keep the sentinel,
        proving the kernel never writes past the device-side ``cache_seqlens``.
        """
        torch.manual_seed(0)
        dev = "cuda"
        num_reqs = max(64, bs)
        max_num_pages = (max_context_len + page_size - 1) // page_size
        n_slots = num_reqs * max_context_len
        req_to_token = torch.randint(
            0, n_slots, (num_reqs, max_context_len), dtype=torch.int32, device=dev
        )
        req_pool_indices = torch.randperm(num_reqs, device=dev)[:bs].to(torch.int32)
        # Cap lengths well below max_context_len so most tail columns stay unused.
        hi = max(2, max_context_len // 8)
        cache_seqlens = torch.randint(1, hi + 1, (bs,), dtype=torch.int32, device=dev)
        full_to_swa = (
            torch.randint(0, n_slots, (n_slots,), dtype=torch.int32, device=dev)
            if swa
            else None
        )

        SENTINEL = -1
        page_table = torch.full(
            (bs, max_num_pages), SENTINEL, dtype=torch.int32, device=dev
        )
        swa_page_table = (
            torch.full((bs, max_num_pages), SENTINEL, dtype=torch.int32, device=dev)
            if swa
            else None
        )
        build_trtllm_mha_page_table(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            page_size=page_size,
            swa_page_table=swa_page_table,
            full_to_swa=full_to_swa,
        )
        pt_ref, swa_ref = _build_page_table_reference(
            req_to_token, req_pool_indices, cache_seqlens, page_size, full_to_swa
        )

        tag = f"max_ctx={max_context_len} page_size={page_size} bs={bs} swa={swa}"
        for i in range(bs):
            npages = (int(cache_seqlens[i].item()) + page_size - 1) // page_size
            self.assertTrue(
                torch.equal(page_table[i, :npages], pt_ref[i, :npages]),
                f"used-column mismatch req={i} {tag}",
            )
            self.assertTrue(
                torch.all(page_table[i, npages:] == SENTINEL),
                f"kernel wrote past cache_seqlens req={i} npages={npages} {tag}",
            )
            if swa:
                self.assertTrue(
                    torch.equal(swa_page_table[i, :npages], swa_ref[i, :npages]),
                    f"swa used-column mismatch req={i} {tag}",
                )
                self.assertTrue(
                    torch.all(swa_page_table[i, npages:] == SENTINEL),
                    f"swa wrote past cache_seqlens req={i} npages={npages} {tag}",
                )

    def test_writes_bounded_by_cache_seqlens(self):
        for max_ctx in (4096, 131072):
            for page_size in (1, 64, 256):
                for bs in (1, 8):
                    self._run_self_guard_case(max_ctx, page_size, bs=bs)
                    self._run_self_guard_case(max_ctx, page_size, bs=bs, swa=True)


if __name__ == "__main__":
    unittest.main()
