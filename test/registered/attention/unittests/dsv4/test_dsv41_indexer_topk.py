"""Low-ratio indexer decode path: DeepGEMM fp4 paged logits and the paged top-k
transform vs a torch golden of the reference scoring,
(einsum(q, k).relu() * weights).sum(heads). Production layout: 64-slot indexer-K
pages behind the FULL 256-token page table, expanded by _expand_index_page_table.
GPU only (DeepGEMM fp8_fp4_paged_mqa_logits + topk_transform_paged).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")

FULL_PAGE_SIZE = 256
INDEX_PAGE_SIZE = 64
HEAD_DIM = 128
N_HEADS = 32
TOPK = 512


def golden_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """q [b, H, d], k [b, n, d] (per-request table), weights [b, H] -> [b, n]."""
    s = torch.einsum("bhd,bnd->bhn", q.float(), k.float())
    return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=1)


def golden_expand(page_table: torch.Tensor, blocks_per_page: int) -> torch.Tensor:
    """Naive loop form of the page-table expansion."""
    bs, n = page_table.shape
    out = torch.empty(bs, n * blocks_per_page, dtype=torch.int32)
    for b in range(bs):
        for p in range(n):
            for k in range(blocks_per_page):
                out[b, p * blocks_per_page + k] = (
                    int(page_table[b, p]) * blocks_per_page + k
                )
    return out.to(page_table.device)


def compressed_slot(
    full_page_table: torch.Tensor, ratio: int, j: torch.Tensor
) -> torch.Tensor:
    """Physical slot of compressed position j in a page-size // ratio pool whose
    pages line up with the FULL pool (the c1/c2 KV pool contract)."""
    slots_per_page = FULL_PAGE_SIZE // ratio
    phys_page = full_page_table.gather(1, (j // slots_per_page).to(torch.int64))
    return phys_page.to(torch.int64) * slots_per_page + j % slots_per_page


class TestIndexerTopk(CustomTestCase):
    def test_expand_index_page_table(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _expand_index_page_table,
        )

        torch.manual_seed(0)
        page_table = torch.randperm(64, dtype=torch.int32, device="cuda")[:24].view(
            3, 8
        )
        for ratio in (1, 2, 4):
            blocks_per_page = FULL_PAGE_SIZE // ratio // INDEX_PAGE_SIZE
            got = _expand_index_page_table(
                page_table,
                full_page_size=FULL_PAGE_SIZE,
                compress_ratio=ratio,
                index_page_size=INDEX_PAGE_SIZE,
            )
            self.assertEqual(got.dtype, torch.int32)
            self.assertTrue(
                torch.equal(got, golden_expand(page_table, blocks_per_page)),
                msg=f"expansion mismatch for {ratio = }",
            )

    def _run_decode(self, ratio: int):
        import deep_gemm

        from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
            store_fp4_index_k_cache,
        )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _expand_index_page_table,
            _fp4_paged_mqa_logits,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(ratio)
        bs, n_full_pages = 4, 8
        slots_per_page = FULL_PAGE_SIZE // ratio
        max_slots = n_full_pages * slots_per_page  # per request
        n_phys_pages = bs * n_full_pages + 3  # a few never-referenced pages
        total_slots = n_phys_pages * slots_per_page
        n_index_pages = total_slots // INDEX_PAGE_SIZE

        # FULL page table: a random permutation of physical pages per request.
        full_page_table = (
            torch.randperm(n_phys_pages, device="cuda")[: bs * n_full_pages]
            .view(bs, n_full_pages)
            .to(torch.int32)
        )
        # Visible compressed lengths, including page-boundary and tiny cases.
        seq_lens = torch.tensor(
            [slots_per_page, slots_per_page + 5, 37, max_slots],
            dtype=torch.int32,
            device="cuda",
        )

        # fp4-grid q, K and head weights (the production tensors are on the grid).
        q = fake_quant_fp4(
            torch.randn(bs, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        k_all = fake_quant_fp4(
            torch.randn(total_slots, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        weights = torch.rand(bs, N_HEADS, device="cuda", dtype=torch.float32)

        # Packed fp4 K pool paged at 64 slots, written at the physical slots.
        k_cache = torch.zeros(
            n_index_pages, INDEX_PAGE_SIZE * (64 + 4), dtype=torch.uint8, device="cuda"
        )
        loc = torch.arange(total_slots, dtype=torch.int32, device="cuda")
        store_fp4_index_k_cache(
            input=k_all, cache=k_cache, loc=loc, page_size=INDEX_PAGE_SIZE, rne=True
        )

        # Kernel path, exactly as _low_ratio_index_topk_decode does it.
        page_table = _expand_index_page_table(
            full_page_table,
            full_page_size=FULL_PAGE_SIZE,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_SIZE,
        )
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(bs, 1, N_HEADS, 64)
        q_sf = q_sf.view(bs, 1, N_HEADS)
        k_cache_view = k_cache.view(n_index_pages, INDEX_PAGE_SIZE, 1, 68)
        deep_gemm_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens.unsqueeze(-1), INDEX_PAGE_SIZE, deep_gemm.get_num_sms()
        )
        logits = _fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache_view,
            weights,
            seq_lens,
            page_table,
            deep_gemm_metadata,
            page_table.shape[1] * INDEX_PAGE_SIZE,
        )
        page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        topk_transform_paged(
            logits, seq_lens, page_table, page_indices, INDEX_PAGE_SIZE, raw_indices
        )

        # Golden: gather each request's K by compressed position through the
        # FULL page table, score, mask the invisible tail.
        j = torch.arange(max_slots, device="cuda")
        slots = compressed_slot(full_page_table, ratio, j.expand(bs, -1))  # [bs, n]
        ref = golden_scores(q, k_all[slots], weights)
        visible = j[None, :] < seq_lens[:, None]

        diff = (logits.float()[:, :max_slots] - ref).abs()
        rel = diff / ref.abs().clamp_min(1.0)
        self.assertLess(
            rel[visible].max().item(),
            2e-2,
            msg=f"kernel logits diverged: max rel diff {rel[visible].max()}",
        )

        # Top-k transform: -1 past the visible count, raw positions within it, and
        # each slot is the FULL-table resolution of its raw position.
        for b in range(bs):
            n_valid = min(TOPK, int(seq_lens[b]))
            sel_raw = raw_indices[b, :n_valid]
            sel_slot = page_indices[b, :n_valid]
            self.assertTrue(
                bool((sel_raw >= 0).all()) and bool((sel_raw < seq_lens[b]).all())
            )
            self.assertTrue(bool((raw_indices[b, n_valid:] == -1).all()))
            self.assertTrue(bool((page_indices[b, n_valid:] == -1).all()))
            expect = compressed_slot(
                full_page_table[b : b + 1], ratio, sel_raw[None].to(torch.int64)
            )[0]
            self.assertTrue(
                torch.equal(sel_slot.to(torch.int64), expect),
                msg=f"page-table transform mismatch, request {b}",
            )
            # Selection agrees with the golden top-k up to kernel rounding.
            ref_sel = ref[b].masked_fill(~visible[b], -torch.inf).topk(n_valid).indices
            overlap = len(set(ref_sel.tolist()) & set(sel_raw.tolist())) / n_valid
            self.assertGreaterEqual(
                overlap, 0.9, msg=f"top-k overlap {overlap:.3f} too low, request {b}"
            )

    def test_decode_ratio1_matches_golden(self):
        self._run_decode(ratio=1)

    def test_decode_ratio2_matches_golden(self):
        self._run_decode(ratio=2)


if __name__ == "__main__":
    unittest.main()
