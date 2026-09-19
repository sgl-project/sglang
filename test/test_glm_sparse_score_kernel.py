"""Unit tests for the GLM sparse Triton scoring kernel.

Validates ``glm_sparse_compute_scores`` against the original Python einsum
loop over sglang's flat KV pool, covering:
- correctness across (batch, kv_heads, head_dim) shapes
- correct handling of variable seq_lens (padding to -inf)
- empty-batch / zero-length corner cases
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "python")

import torch


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestGlmSparseScoreKernel(unittest.TestCase):
    def _reference_scores(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_score_len: int,
    ) -> torch.Tensor:
        """Per-query-head GEMM, then max-pool over each kv-head's GQA group.

        ``q`` is ``[batch, num_heads, head_dim]`` and ``k_cache`` has ``kv_heads``
        heads; ``group = num_heads // kv_heads`` query heads map to each kv head.
        """
        batch, num_heads, _ = q.shape
        kv_heads = k_cache.shape[1]
        group = num_heads // kv_heads
        ref = torch.full(
            (batch * kv_heads, max_score_len),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        for b in range(batch):
            sl = int(seq_lens[b].item())
            if sl <= 0:
                continue
            req_idx = int(req_pool_indices[b].item())
            phys = req_to_token[req_idx, :sl].to(torch.long)
            keys = k_cache[phys].float()  # [sl, kv_heads, head_dim]
            # Repeat each kv head across its query-head group, score per head,
            # then max-pool the group -> [kv_heads, sl].
            keys_g = keys.repeat_interleave(group, dim=1)  # [sl, num_heads, head_dim]
            per_head = torch.einsum("hd,shd->hs", q[b].float(), keys_g)  # [num_heads, sl]
            pooled = per_head.reshape(kv_heads, group, sl).amax(dim=1)  # [kv_heads, sl]
            for h in range(kv_heads):
                ref[b * kv_heads + h, :sl] = pooled[h]
        return ref

    def test_matches_python_loop_basic(self):
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )

        torch.manual_seed(0)
        batch, kv_heads, group, head_dim = 2, 4, 3, 128
        num_heads = kv_heads * group
        pool_size = 512
        max_seq = 200

        q = torch.randn(batch, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k_cache = torch.randn(pool_size, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        req_to_token = torch.randperm(pool_size, device="cuda", dtype=torch.int32)[: max_seq * batch]
        req_to_token = req_to_token.view(batch, max_seq).contiguous()
        req_pool_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([150, 200], dtype=torch.int32, device="cuda")

        scores = glm_sparse_compute_scores(
            q.float(),  # use fp32 q to match Python einsum precision
            k_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            max_score_len=max_seq,
        )
        ref = self._reference_scores(
            q.float(),
            k_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            max_score_len=max_seq,
        )

        # finite positions must match closely; padding positions both must be -inf
        finite_mask = torch.isfinite(ref)
        torch.testing.assert_close(
            scores[finite_mask], ref[finite_mask], atol=1e-1, rtol=1e-1
        )
        self.assertTrue(torch.all(scores[~finite_mask] == float("-inf")))

    def test_zero_seq_len_row_is_neg_inf(self):
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )

        batch, kv_heads, head_dim = 2, 2, 64
        pool_size, max_seq = 64, 32
        q_pooled = torch.randn(batch, kv_heads, head_dim, device="cuda")
        k_cache = torch.randn(pool_size, kv_heads, head_dim, device="cuda")
        req_to_token = torch.arange(batch * max_seq, dtype=torch.int32, device="cuda").view(batch, max_seq)
        req_pool_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([0, 10], dtype=torch.int32, device="cuda")

        scores = glm_sparse_compute_scores(
            q_pooled, k_cache, req_to_token, req_pool_indices, seq_lens, max_score_len=max_seq
        )
        # Row 0 (batch 0) corresponds to scores[:kv_heads], must be all -inf.
        self.assertTrue(torch.all(scores[:kv_heads] == float("-inf")))
        # Row 1 batch 1 should be finite for the first 10 columns and -inf after.
        for h in range(kv_heads):
            row = scores[kv_heads + h]
            self.assertTrue(torch.all(torch.isfinite(row[:10])))
            self.assertTrue(torch.all(row[10:] == float("-inf")))

    def test_topk_consistency_with_torch_topk(self):
        """Sanity-check: topk indices on kernel output match topk on the reference."""
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )

        batch, kv_heads, head_dim = 1, 2, 128
        pool_size, max_seq = 1024, 1024
        q_pooled = torch.randn(batch, kv_heads, head_dim, device="cuda")
        k_cache = torch.randn(pool_size, kv_heads, head_dim, device="cuda")
        req_to_token = torch.arange(max_seq, dtype=torch.int32, device="cuda").view(batch, max_seq)
        req_pool_indices = torch.zeros(batch, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([800], dtype=torch.int32, device="cuda")

        scores = glm_sparse_compute_scores(
            q_pooled, k_cache, req_to_token, req_pool_indices, seq_lens, max_score_len=max_seq
        )
        ref = self._reference_scores(
            q_pooled, k_cache, req_to_token, req_pool_indices, seq_lens, max_score_len=max_seq
        )

        for row in range(batch * kv_heads):
            top64_kernel = torch.topk(scores[row], k=64).indices.sort().values
            top64_ref = torch.topk(ref[row], k=64).indices.sort().values
            torch.testing.assert_close(top64_kernel, top64_ref)

    def test_force_select_boundaries_are_inf_and_selected(self):
        """Forced sink + local-window columns must be +inf and always in TopK."""
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )

        batch, kv_heads, head_dim = 2, 4, 128
        pool_size, max_seq = 4096, 4096
        force_left, force_right, topk = 64, 128, 2048
        q_pooled = torch.randn(batch, kv_heads, head_dim, device="cuda")
        k_cache = torch.randn(pool_size, kv_heads, head_dim, device="cuda")
        req_to_token = torch.arange(
            batch * max_seq, dtype=torch.int32, device="cuda"
        ).view(batch, max_seq)
        req_pool_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
        # One long row (> topk) and one short row (< topk) to cover both paths.
        seq_lens = torch.tensor([3000, 100], dtype=torch.int32, device="cuda")

        scores = glm_sparse_compute_scores(
            q_pooled,
            k_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            max_score_len=max_seq,
            force_left=force_left,
            force_right=force_right,
        )

        for b in range(batch):
            sl = int(seq_lens[b].item())
            left = set(range(min(force_left, sl)))
            right = set(range(max(sl - force_right, 0), sl))
            forced = left | right
            for h in range(kv_heads):
                row = scores[b * kv_heads + h]
                # Forced columns are exactly +inf.
                for c in forced:
                    self.assertEqual(row[c].item(), float("inf"), f"col {c}")
                # TopK must contain every forced column.
                sel = set(torch.topk(row, k=min(topk, sl)).indices.tolist())
                self.assertTrue(forced.issubset(sel))
                # Padding beyond seq_len stays -inf (never forced out of range).
                self.assertTrue(torch.all(row[sl:] == float("-inf")))

    def test_force_select_disabled_matches_plain(self):
        """force_left=force_right=0 must be identical to the no-force path."""
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )

        batch, kv_heads, head_dim = 1, 2, 128
        pool_size, max_seq = 512, 512
        q_pooled = torch.randn(batch, kv_heads, head_dim, device="cuda")
        k_cache = torch.randn(pool_size, kv_heads, head_dim, device="cuda")
        req_to_token = torch.arange(max_seq, dtype=torch.int32, device="cuda").view(
            batch, max_seq
        )
        req_pool_indices = torch.zeros(batch, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([300], dtype=torch.int32, device="cuda")

        common = dict(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            max_score_len=max_seq,
        )
        plain = glm_sparse_compute_scores(q_pooled, k_cache, **common)
        forced0 = glm_sparse_compute_scores(
            q_pooled, k_cache, force_left=0, force_right=0, **common
        )
        torch.testing.assert_close(plain, forced0)


if __name__ == "__main__":
    unittest.main()
