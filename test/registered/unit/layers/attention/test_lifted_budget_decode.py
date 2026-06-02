"""Unit + GPU-smoke tests for the lifted-budget decode index core.

CPU tests pin the request-local physical->compact remap (no ``-1`` in the
flattened page table, request-local ordinals, within-row dedup keeping the
highest-rank occurrence, prefix-sharing isolation, pad/edge handling).

GPU smokes prove the kernel half of the opt-in path: ``flash_mla_sparse_fwd``
attends a wider-than-2048 (4096-budget) selection driven by the remap's compact
indices and matches a reference attention; and the full fp8 ->
``dequantize_k_cache_paged`` -> ``flash_mla_sparse_fwd`` pipe matches a reference
that attends the dequantized selected slots (incl. prefix sharing).
"""

import unittest

import torch

from sglang.srt.layers.attention.double_sparsity.lifted_budget import (
    build_compact_decode_index,
)

_HAS_CUDA = torch.cuda.is_available()


class TestCompactDecodeIndex(unittest.TestCase):
    """Pure-tensor remap correctness (CPU)."""

    def _build(self, rows, valid_lengths, width, pad_value=-1):
        bs = len(rows)
        sel = torch.full((bs, width), pad_value, dtype=torch.int64)
        for i, r in enumerate(rows):
            sel[i, : len(r)] = torch.tensor(r, dtype=torch.int64)
        vl = torch.tensor(valid_lengths, dtype=torch.int64)
        return build_compact_decode_index(sel, vl, pad_value=pad_value)

    def test_request_local_mapping(self):
        res = self._build([[10, 20, 30], [40, 50]], [3, 2], width=6)
        self.assertEqual(
            res.page_table_1_flattened.tolist(), [10, 20, 30, 40, 50]
        )
        self.assertEqual(res.valid_counts.tolist(), [3, 2])
        self.assertEqual(res.total_valid, 5)
        self.assertEqual(res.dropped_duplicates, 0)
        # request 0 -> compact ordinals 0,1,2 ; request 1 -> 3,4 (request_base=3).
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, 2, -1, -1, -1])
        self.assertEqual(res.compact_indices[1].tolist(), [3, 4, -1, -1, -1, -1])

    def test_no_minus1_in_flattened(self):
        res = self._build([[7, 8], [9]], [2, 1], width=5)
        ptf = res.page_table_1_flattened
        self.assertTrue(bool((ptf >= 0).all()))
        self.assertEqual(ptf.tolist(), [7, 8, 9])

    def test_prefix_sharing_request_local_spans(self):
        # Slots 7 and 9 are shared across both requests (a shared prefix).
        res = self._build([[5, 7, 9], [7, 9, 11]], [3, 3], width=4)
        # Each request contributes its OWN copy of the shared slot.
        self.assertEqual(
            res.page_table_1_flattened.tolist(), [5, 7, 9, 7, 9, 11]
        )
        # request 1's compact ordinals point only into its own span (3,4,5),
        # never into request 0's copies (1,2) of the same physical slots.
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, 2, -1])
        self.assertEqual(res.compact_indices[1].tolist(), [3, 4, 5, -1])
        self.assertEqual(res.dropped_duplicates, 0)

    def test_within_row_dedup_keeps_first(self):
        # Slot 10 appears at lane 0 and lane 2 of request 0 -> the lane-0 (higher
        # selection rank) copy is kept, the later one dropped.
        res = self._build([[10, 20, 10]], [3], width=5)
        self.assertEqual(res.page_table_1_flattened.tolist(), [10, 20])
        self.assertEqual(res.valid_counts.tolist(), [2])
        self.assertEqual(res.dropped_duplicates, 1)
        # lane 2 (the duplicate) is masked to -1; the kept ordinals are 0,1.
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, -1, -1, -1])

    def test_dedup_keeps_highest_rank_when_first_repeats_later(self):
        # 20 at lanes 0 and 2: keep lane 0 (rank 0), drop lane 2.
        res = self._build([[20, 10, 20]], [3], width=4)
        self.assertEqual(res.page_table_1_flattened.tolist(), [20, 10])
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, -1, -1])
        self.assertEqual(res.dropped_duplicates, 1)

    def test_zero_valid_row_keeps_base_accounting(self):
        # Middle request has zero valid entries.
        res = self._build([[3, 4], [], [5]], [2, 0, 1], width=4)
        self.assertEqual(res.page_table_1_flattened.tolist(), [3, 4, 5])
        self.assertEqual(res.valid_counts.tolist(), [2, 0, 1])
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, -1, -1])
        self.assertEqual(res.compact_indices[1].tolist(), [-1, -1, -1, -1])
        # request 2's base must skip the empty request and start at 2.
        self.assertEqual(res.compact_indices[2].tolist(), [2, -1, -1, -1])

    def test_valid_lengths_shorter_than_value_prefix(self):
        # Even if more non-pad values are present, only valid_lengths count.
        res = self._build([[1, 2, 3, 4]], [2], width=4)
        self.assertEqual(res.page_table_1_flattened.tolist(), [1, 2])
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, -1, -1])

    def test_order_preserved_within_request(self):
        # The remap must not reorder a request's selection (the selector's
        # deterministic score-desc/position-asc order is carried into the
        # compact ordinals).
        res = self._build([[30, 10, 20]], [3], width=3)
        self.assertEqual(res.page_table_1_flattened.tolist(), [30, 10, 20])
        self.assertEqual(res.compact_indices[0].tolist(), [0, 1, 2])


def _ref_sparse_attention(q, kv_compact, compact_indices, sm_scale):
    """Reference: per request, softmax-attend the valid (non-masked) compact
    rows. ``q``: [bs, h, 576]; ``kv_compact``: [S, 1, 576]; ``compact_indices``:
    [bs, topk] int (``-1``/``>=S`` are masked)."""
    bs, h, _ = q.shape
    S = kv_compact.shape[0]
    qf = q.float()
    kvf = kv_compact[:, 0, :].float()
    out = torch.zeros(bs, h, 512, dtype=torch.float32, device=q.device)
    for i in range(bs):
        idx = compact_indices[i].long()
        valid = (idx >= 0) & (idx < S)
        vidx = idx[valid]
        if vidx.numel() == 0:
            continue
        ksel = kvf[vidx]
        scores = (qf[i] @ ksel.t()) * sm_scale
        p = torch.softmax(scores, dim=-1)
        out[i] = p @ ksel[:, :512]
    return out


@unittest.skipUnless(_HAS_CUDA, "lifted-budget kernel smoke requires CUDA")
class TestLiftedBudgetKernelSmoke(unittest.TestCase):
    """GPU proof of the opt-in decode kernel half at a >2048 budget."""

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        major = torch.cuda.get_device_capability()[0]
        cls.h = 128 if major >= 10 else 64  # flash_mla_sparse_fwd head multiple
        torch.manual_seed(0xB12)

    def _sparse_fwd(self, q, kv, compact_indices, sm_scale):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        idx = compact_indices.unsqueeze(1).to(torch.int32)  # [bs, 1, topk]
        o, _, _ = flash_mla_sparse_fwd(
            q=q, kv=kv, indices=idx, sm_scale=sm_scale, d_v=512
        )
        return o

    def test_flash_mla_sparse_fwd_4k_budget_matches_reference(self):
        # Two requests select MORE than the 2048 cap (3000 / 1500 valid) inside a
        # 4096-wide padded budget; proves no 2048 cap + -1 pad masking +
        # request-local compact spans.
        width = 4096
        v0, v1 = 3000, 1500
        rows = [list(range(v0)), list(range(10000, 10000 + v1))]
        sel = torch.full((2, width), -1, dtype=torch.int64)
        sel[0, :v0] = torch.tensor(rows[0])
        sel[1, :v1] = torch.tensor(rows[1])
        res = build_compact_decode_index(
            sel.to(self.device), torch.tensor([v0, v1], device=self.device)
        )
        self.assertEqual(res.total_valid, v0 + v1)
        self.assertEqual(res.dropped_duplicates, 0)

        S = res.total_valid
        d_qk = 576
        sm_scale = 1.0 / (d_qk**0.5)
        q = (torch.randn(2, self.h, d_qk, device=self.device, dtype=torch.bfloat16) / 8)
        kv = (
            torch.randn(S, 1, d_qk, device=self.device, dtype=torch.bfloat16) / 8
        )
        q.clamp_(-6, 6)
        kv.clamp_(-6, 6)

        out = self._sparse_fwd(q, kv, res.compact_indices, sm_scale)
        ref = _ref_sparse_attention(q, kv, res.compact_indices, sm_scale)
        # Confirm request 0 actually attended > 2048 rows (no cap).
        self.assertGreater(int((res.compact_indices[0] >= 0).sum().item()), 2048)
        torch.testing.assert_close(
            out.float(), ref.to(out.dtype).float(), atol=3e-2, rtol=3e-2
        )

    def test_dequant_remap_end_to_end_with_prefix_sharing(self):
        from sglang.srt.layers.attention.dsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

        # Physical fp8 KV cache (P slots), built by round-tripping a bf16 cache.
        P, d_qk = 64, 576
        phys_bf16 = (
            torch.randn(P, 1, 1, d_qk, device=self.device, dtype=torch.bfloat16) / 8
        )
        phys_bf16.clamp_(-6, 6)
        quant = quantize_k_cache(phys_bf16)  # [P,1,1,656] fp8
        # Full dequant reference for the gather check.
        full_dequant = dequantize_k_cache_paged(
            quant, torch.arange(P, device=self.device, dtype=torch.int32)
        )  # [P,1,576]

        # Two requests share physical slots 7 and 9 (prefix sharing). The padded
        # index width must be a multiple of the kernel block (topk % (2*B_TOPK)
        # == 0, i.e. a multiple of 128) — the real budgets 4096/8192 satisfy this.
        rows = [[5, 7, 9, 13], [7, 9, 11]]
        width = 256
        sel = torch.full((2, width), -1, dtype=torch.int64, device=self.device)
        sel[0, : len(rows[0])] = torch.tensor(rows[0], device=self.device)
        sel[1, : len(rows[1])] = torch.tensor(rows[1], device=self.device)
        res = build_compact_decode_index(
            sel, torch.tensor([4, 3], device=self.device)
        )
        self.assertEqual(
            res.page_table_1_flattened.tolist(), [5, 7, 9, 13, 7, 9, 11]
        )

        # Compact dequant buffer keyed by the flattened page table.
        compact = dequantize_k_cache_paged(quant, res.page_table_1_flattened)
        # Each compact row equals the full-dequant row of its physical slot.
        torch.testing.assert_close(
            compact.float(),
            full_dequant[res.page_table_1_flattened.long()].float(),
            atol=0.0,
            rtol=0.0,
        )

        sm_scale = 1.0 / (d_qk**0.5)
        q = torch.randn(2, self.h, d_qk, device=self.device, dtype=torch.bfloat16) / 8
        q.clamp_(-6, 6)
        out = self._sparse_fwd(q, compact, res.compact_indices, sm_scale)
        ref = _ref_sparse_attention(q, compact, res.compact_indices, sm_scale)
        torch.testing.assert_close(
            out.float(), ref.to(out.dtype).float(), atol=3e-2, rtol=3e-2
        )

    def _quant_store(self, P, d_qk=576):
        from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

        phys = (
            torch.randn(P, 1, 1, d_qk, device=self.device, dtype=torch.bfloat16) / 8
        )
        phys.clamp_(-6, 6)
        return quantize_k_cache(phys)  # [P,1,1,656] fp8

    def test_wired_lifted_decode_4096(self):
        # The production decode helper at the 4096 lifted width: one request
        # selects 3000 > 2048 slots; matches a reference attention.
        from sglang.srt.layers.attention.double_sparsity.lifted_budget import (
            build_lifted_compact_kv,
        )

        d_qk = 576
        quant = self._quant_store(3200, d_qk)
        width, v0, v1 = 4096, 3000, 1500
        sel = torch.full((2, width), -1, dtype=torch.int64, device=self.device)
        sel[0, :v0] = torch.arange(v0, device=self.device)
        sel[1, :v1] = torch.arange(100, 100 + v1, device=self.device)

        compact_kv, compact_indices, vcounts = build_lifted_compact_kv(
            quant, sel, store_is_fp8=True
        )
        self.assertEqual(int(vcounts[0]), v0)
        self.assertGreater(int((compact_indices[0] >= 0).sum().item()), 2048)

        sm_scale = 1.0 / (d_qk**0.5)
        q = torch.randn(2, self.h, d_qk, device=self.device, dtype=torch.bfloat16) / 8
        q.clamp_(-6, 6)
        out = self._sparse_fwd(q, compact_kv, compact_indices, sm_scale)
        ref = _ref_sparse_attention(q, compact_kv, compact_indices, sm_scale)
        torch.testing.assert_close(
            out.float(), ref.to(out.dtype).float(), atol=3e-2, rtol=3e-2
        )

    def test_wired_lifted_decode_8192_dedup_and_prefix_share(self):
        # 8192 lifted width; request 1 has a within-row duplicate physical slot
        # (interior -1 after dedup) and shares slots with request 0 (prefix
        # sharing → its own compact span). Matches a reference attention.
        from sglang.srt.layers.attention.double_sparsity.lifted_budget import (
            build_lifted_compact_kv,
        )

        d_qk = 576
        quant = self._quant_store(3200, d_qk)
        width, v0 = 8192, 3000
        sel = torch.full((2, width), -1, dtype=torch.int64, device=self.device)
        sel[0, :v0] = torch.arange(v0, device=self.device)
        # slot 5 duplicated at lane 2; slots 5/7/9/11 shared with request 0.
        sel[1, :5] = torch.tensor([5, 7, 5, 9, 11], device=self.device)

        compact_kv, compact_indices, vcounts = build_lifted_compact_kv(
            quant, sel, store_is_fp8=True
        )
        # request 1 deduped to 4 valid; the duplicate (lane 2) is an interior -1.
        self.assertEqual(int(vcounts[1]), 4)
        self.assertEqual(int(compact_indices[1, 2].item()), -1)
        # prefix-sharing: request 1's slot 5 maps to ITS OWN compact span base,
        # not request 0's copy of slot 5.
        base1 = int(vcounts[0].item())
        self.assertEqual(int(compact_indices[1, 0].item()), base1)

        sm_scale = 1.0 / (d_qk**0.5)
        q = torch.randn(2, self.h, d_qk, device=self.device, dtype=torch.bfloat16) / 8
        q.clamp_(-6, 6)
        out = self._sparse_fwd(q, compact_kv, compact_indices, sm_scale)
        ref = _ref_sparse_attention(q, compact_kv, compact_indices, sm_scale)
        torch.testing.assert_close(
            out.float(), ref.to(out.dtype).float(), atol=3e-2, rtol=3e-2
        )


if __name__ == "__main__":
    unittest.main()
