"""GLM sparse training-free indexer: max-pool + force-select, with/without
cross-layer TopK reuse.

Two scenarios, both built on the REAL Triton scoring kernel
(``glm_sparse_compute_scores``), which does per-query-head GEMM, max-pools over
each kv-head's GQA group, and applies the StreamingLLM-style force-select mask:

1. ``TestMaxPoolForceSelect`` -- single layer: max-pool + force-select.
2. ``TestMaxPoolForceSelectCrossLayer`` -- multi-layer: same, plus cross-layer
   reuse where "skip" layers reuse the previous compute layer's TopK.

The reuse decision mirrors ``Glm4MoeAttention._glm_sparse_compute_reuse_flags``
in ``python/sglang/srt/models/glm4_moe.py``; that module can't be imported
standalone here (heavy deps), so the ~10-line decision is replicated below. If
the model-side logic changes, update ``_reuse_flags`` to match.
"""

from __future__ import annotations

import sys
import types
import unittest

# Stub the sglang package hierarchy so the glm_sparse leaf modules import
# without running the full sglang __init__ (mirrors the other standalone tests).
for _name in [
    "sglang", "sglang.srt", "sglang.srt.layers",
    "sglang.srt.layers.attention", "sglang.srt.layers.attention.glm_sparse",
]:
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [f"python/{_name.replace('.', '/')}"]
    sys.modules[_name] = _pkg

sys.path.insert(0, "python")

import torch

PLACEHOLDER_REUSE = None


def _reuse_flags(layer_id, start_layer, freq=1, pattern=None):
    """Replica of Glm4MoeAttention._glm_sparse_compute_reuse_flags.

    Returns ``(skip_topk, next_skip_topk)``. ``pattern`` ('S'=skip/reuse) wins
    over ``freq``; the first sparse layer always computes.
    """
    if pattern is not None:
        skip = layer_id < len(pattern) and pattern[layer_id] == "S"
        nxt = (layer_id + 1) < len(pattern) and pattern[layer_id + 1] == "S"
    else:
        freq = max(int(freq), 1)
        rel = layer_id - start_layer
        skip = (rel % freq) != 0
        nxt = ((rel + 1) % freq) != 0
    if layer_id == start_layer:
        skip = False
    return skip, nxt


def _reference_maxpool_scores(q, k_cache, req_to_token, req_pool_indices, seq_lens,
                              max_score_len, force_left=0, force_right=0):
    """Pure-torch reference: per-head GEMM -> max-pool over GQA group ->
    force-select mask. Mirrors the Triton kernel semantics.

    ``q`` is ``[batch, num_heads, head_dim]``; ``k_cache`` has ``kv_heads``.
    Output ``[batch*kv_heads, max_score_len]`` fp32 with -inf padding.
    """
    batch, num_heads, _ = q.shape
    kv_heads = k_cache.shape[1]
    group = num_heads // kv_heads
    ref = torch.full((batch * kv_heads, max_score_len), float("-inf"),
                     dtype=torch.float32, device=q.device)
    for b in range(batch):
        sl = int(seq_lens[b].item())
        if sl <= 0:
            continue
        req_idx = int(req_pool_indices[b].item())
        phys = req_to_token[req_idx, :sl].to(torch.long)
        keys = k_cache[phys].float()                       # [sl, kv_heads, d]
        keys_g = keys.repeat_interleave(group, dim=1)      # [sl, num_heads, d]
        per_head = torch.einsum("hd,shd->hs", q[b].float(), keys_g)  # [num_heads, sl]
        pooled = per_head.reshape(kv_heads, group, sl).amax(dim=1)   # [kv_heads, sl]
        for h in range(kv_heads):
            row = pooled[h].clone()
            if force_left > 0:
                row[:min(force_left, sl)] = float("inf")
            if force_right > 0:
                row[max(sl - force_right, 0):sl] = float("inf")
            ref[b * kv_heads + h, :sl] = row
    return ref


def _select_topk(scores, seq_lens, kv_heads, topk):
    """torch.topk selection matching glm4_moe's capture-safe fallback:
    front-packed valid indices, -1 padding for columns >= seq_len."""
    batch = seq_lens.shape[0]
    max_score_len = scores.shape[1]
    seq_lens_i32 = seq_lens.to(torch.int32)
    out = torch.full((batch, kv_heads, topk), -1, dtype=torch.int32, device=scores.device)
    k = min(topk, max_score_len)
    _, top_idx = torch.topk(scores, k=k, dim=-1)
    top_idx = top_idx.to(torch.int32).view(batch, kv_heads, k)
    valid = top_idx < seq_lens_i32[:, None, None]
    top_idx = torch.where(valid, top_idx, torch.full_like(top_idx, -1))
    out[:, :, :k] = top_idx
    return out


def _make_inputs(batch, kv_heads, group, head_dim, seq_lens_list, max_seq,
                 pool_size, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    num_heads = kv_heads * group
    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device="cuda")
    k_cache = torch.randn(pool_size, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    # Non-overlapping physical slots per request.
    req_to_token = torch.zeros(batch, max_seq, dtype=torch.int32, device="cuda")
    offset = 0
    for b in range(batch):
        sl = seq_lens_list[b]
        req_to_token[b, :sl] = torch.arange(offset, offset + sl, dtype=torch.int32, device="cuda")
        offset += sl
    req_pool_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    return q, k_cache, req_to_token, req_pool_indices, seq_lens, num_heads


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestMaxPoolForceSelect(unittest.TestCase):
    """Scenario 1: max-pool + force-select on a single layer."""

    def _compute(self, q, k_cache, req_to_token, req_pool_indices, seq_lens,
                 max_seq, force_left, force_right):
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )
        return glm_sparse_compute_scores(
            q, k_cache, req_to_token, req_pool_indices, seq_lens,
            max_score_len=max_seq, force_left=force_left, force_right=force_right,
        )

    def test_scores_match_maxpool_reference(self):
        batch, kv_heads, group, head_dim = 2, 4, 3, 128
        seq_lens_list = [300, 150]
        max_seq, pool_size = 512, 1024
        force_left, force_right = 16, 32
        q, k_cache, r2t, rpi, sl, _ = _make_inputs(
            batch, kv_heads, group, head_dim, seq_lens_list, max_seq, pool_size, seed=1
        )
        scores = self._compute(q, k_cache, r2t, rpi, sl, max_seq, force_left, force_right)
        ref = _reference_maxpool_scores(
            q, k_cache, r2t, rpi, sl, max_seq, force_left, force_right
        )
        finite = torch.isfinite(ref)
        torch.testing.assert_close(scores[finite], ref[finite], atol=1e-1, rtol=1e-1)
        # +inf forced columns and -inf padding match exactly.
        self.assertTrue(torch.all((scores == float("inf")) == (ref == float("inf"))))
        self.assertTrue(torch.all((scores == float("-inf")) == (ref == float("-inf"))))

    def test_forced_columns_are_selected(self):
        batch, kv_heads, group, head_dim = 2, 4, 2, 128
        # One row > topk (forcing competes for budget), one row < topk.
        seq_lens_list = [3000, 100]
        max_seq, pool_size, topk = 4096, 8192, 2048
        force_left, force_right = 64, 128
        q, k_cache, r2t, rpi, sl, _ = _make_inputs(
            batch, kv_heads, group, head_dim, seq_lens_list, max_seq, pool_size, seed=2
        )
        scores = self._compute(q, k_cache, r2t, rpi, sl, max_seq, force_left, force_right)
        topk_indices = _select_topk(scores, sl, kv_heads, topk)
        for b in range(batch):
            s = seq_lens_list[b]
            forced = set(range(min(force_left, s))) | set(range(max(s - force_right, 0), s))
            for h in range(kv_heads):
                sel = set(topk_indices[b, h].tolist()) - {-1}
                self.assertTrue(forced.issubset(sel),
                                f"forced cols missing at batch {b} head {h}")
                # Selected indices stay in-range.
                self.assertTrue(all(0 <= i < s for i in sel))

    def test_maxpool_differs_from_meanpool(self):
        """max-pool selection should not be identical to the old mean-pool path
        (sanity that we actually changed behavior, given divergent heads)."""
        batch, kv_heads, group, head_dim = 1, 2, 8, 64
        seq_lens_list = [400]
        max_seq, pool_size, topk = 512, 1024, 64
        q, k_cache, r2t, rpi, sl, num_heads = _make_inputs(
            batch, kv_heads, group, head_dim, seq_lens_list, max_seq, pool_size, seed=3
        )
        scores_max = self._compute(q, k_cache, r2t, rpi, sl, max_seq, 0, 0)
        sel_max = _select_topk(scores_max, sl, kv_heads, topk)
        # Old behavior: mean-pool q over the group, then score.
        q_mean = q.reshape(batch, kv_heads, group, head_dim).mean(dim=2)
        q_mean_as_heads = q_mean.repeat_interleave(group, dim=1)  # feed kernel same API
        scores_mean = self._compute(q_mean_as_heads, k_cache, r2t, rpi, sl, max_seq, 0, 0)
        sel_mean = _select_topk(scores_mean, sl, kv_heads, topk)
        self.assertFalse(torch.equal(sel_max, sel_mean),
                         "max-pool and mean-pool produced identical selections")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestMaxPoolForceSelectCrossLayer(unittest.TestCase):
    """Scenario 2: max-pool + force-select + cross-layer TopK reuse.

    Simulates a stack of sparse layers: compute layers run the real
    score+topk; skip layers reuse the previous layer's TopK. Verifies the
    reuse plan and that reused selections are byte-identical to the source.
    """

    def _run_layer(self, q, k_cache, r2t, rpi, sl, max_seq, kv_heads, topk,
                   force_left, force_right):
        from sglang.srt.layers.attention.glm_sparse.score_kernel import (
            glm_sparse_compute_scores,
        )
        scores = glm_sparse_compute_scores(
            q, k_cache, r2t, rpi, sl, max_score_len=max_seq,
            force_left=force_left, force_right=force_right,
        )
        return _select_topk(scores, sl, kv_heads, topk)

    def _simulate_stack(self, num_layers, start_layer, force_left, force_right,
                        freq=1, pattern=None, seed=10):
        """Run the layer loop with cross-layer reuse, threading prev TopK.

        Returns ``(per_layer_topk, plan)`` where plan[i]=(skip, nxt).
        """
        batch, kv_heads, group, head_dim = 2, 4, 3, 128
        seq_lens_list = [800, 256]
        max_seq, pool_size, topk = 1024, 4096, 256
        per_layer = []
        plan = []
        prev = None
        for L in range(num_layers):
            layer_id = start_layer + L
            skip, nxt = _reuse_flags(layer_id, start_layer, freq=freq, pattern=pattern)
            plan.append((skip, nxt))
            # Each layer gets its own q (different projection) but shares the KV.
            q, k_cache, r2t, rpi, sl, _ = _make_inputs(
                batch, kv_heads, group, head_dim, seq_lens_list, max_seq,
                pool_size, seed=seed + layer_id
            )
            if skip and prev is not None:
                topk_indices = prev  # reuse: no score/topk this layer
            else:
                topk_indices = self._run_layer(
                    q, k_cache, r2t, rpi, sl, max_seq, kv_heads, topk,
                    force_left, force_right
                )
            per_layer.append(topk_indices)
            prev = topk_indices
        return per_layer, plan, seq_lens_list, kv_heads, force_left, force_right

    def test_freq2_reuse_plan_and_identity(self):
        per_layer, plan, seq_lens_list, kv_heads, fl, fr = self._simulate_stack(
            num_layers=6, start_layer=0, force_left=16, force_right=16, freq=2
        )
        # freq=2: layer 0 computes, 1 reuses 0, 2 computes, 3 reuses 2, ...
        expected_skip = [False, True, False, True, False, True]
        self.assertEqual([p[0] for p in plan], expected_skip)
        # Reused layers are byte-identical to the preceding compute layer.
        for L in range(1, 6):
            if plan[L][0]:
                self.assertTrue(torch.equal(per_layer[L], per_layer[L - 1]),
                                f"layer {L} reuse not identical to {L-1}")
        # Compute layers (even) generally differ from each other (different q).
        self.assertFalse(torch.equal(per_layer[0], per_layer[2]))

    def test_pattern_reuse_plan(self):
        # pattern 'CSCS': layers 0,2 compute; 1,3 skip/reuse.
        per_layer, plan, *_ = self._simulate_stack(
            num_layers=4, start_layer=0, force_left=8, force_right=8,
            pattern="CSCS"
        )
        self.assertEqual([p[0] for p in plan], [False, True, False, True])
        self.assertTrue(torch.equal(per_layer[1], per_layer[0]))
        self.assertTrue(torch.equal(per_layer[3], per_layer[2]))

    def test_first_layer_never_skips(self):
        # Even if pattern marks the start layer 'S', it must compute (no prev).
        _, plan, *_ = self._simulate_stack(
            num_layers=3, start_layer=5, force_left=4, force_right=4,
            pattern="SSS"  # indices 5,6,7 -> out of pattern range anyway
        )
        self.assertFalse(plan[0][0], "start layer must compute")

    def test_forced_columns_survive_reuse(self):
        """Reused TopK must still contain the force-selected columns (they were
        baked into the source compute layer)."""
        per_layer, plan, seq_lens_list, kv_heads, fl, fr = self._simulate_stack(
            num_layers=4, start_layer=0, force_left=32, force_right=32, freq=2
        )
        for L in range(4):
            for b in range(len(seq_lens_list)):
                s = seq_lens_list[b]
                forced = set(range(min(fl, s))) | set(range(max(s - fr, 0), s))
                for h in range(kv_heads):
                    sel = set(per_layer[L][b, h].tolist()) - {-1}
                    self.assertTrue(forced.issubset(sel),
                                    f"forced cols missing at layer {L} batch {b} head {h}")


if __name__ == "__main__":
    unittest.main()
