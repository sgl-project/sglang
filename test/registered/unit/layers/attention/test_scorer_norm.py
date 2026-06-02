"""Unit tests for the flag-gated cosine DS scorer (Loop-7 Tier-2.B candidate).

CPU-only. Pins: default off == raw channel-dot; cosine mode is magnitude-
invariant and promotes a direction-aligned needle over a high-magnitude
background token (the bias the M0 oracle implicated at 16K).
"""

from __future__ import annotations

import os
import unittest

import torch

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    compute_token_scores,
)


def _inputs():
    # bs=1, H=1, head_dim=label_dim=4, T=3.
    queries = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # [1,1,4]
    # token0 = needle (aligned, modest magnitude); token1 = background (high
    # magnitude, off-axis); token2 = zero.
    sigs = torch.tensor(
        [[[[3.0, 0.0, 0.0, 0.0]], [[4.0, 4.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]]
    )  # [L=1, T=3, H=1, D=4]
    written = torch.ones(1, 3, dtype=torch.bool)
    chan_sel = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)  # [L,H,label_dim]
    chan_w = torch.ones(1, 1, 4, dtype=torch.float32)
    return queries, sigs, written, chan_sel, chan_w


class TestScorerNorm(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("SGLANG_DS_SCORER_NORM")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SGLANG_DS_SCORER_NORM", None)
        else:
            os.environ["SGLANG_DS_SCORER_NORM"] = self._prev

    def _scores(self, mode):
        if mode is None:
            os.environ.pop("SGLANG_DS_SCORER_NORM", None)
        else:
            os.environ["SGLANG_DS_SCORER_NORM"] = mode
        q, s, w, cs, cw = _inputs()
        return compute_token_scores(q, s, w, cs, cw, layer_id=0)

    def test_default_is_raw_dot(self):
        scores = self._scores(None)  # unset -> off
        # raw dot of q=[1,0,0,0] with [3,..],[4,4,..],[0,..] = [3,4,0]
        self.assertEqual(scores.tolist(), [[3.0, 4.0, 0.0]])

    def test_off_equals_default(self):
        self.assertTrue(torch.equal(self._scores("off"), self._scores(None)))

    def test_raw_ranks_background_above_needle(self):
        scores = self._scores("off")[0]
        # background (token1) outranks needle (token0) under raw dot
        self.assertGreater(scores[1].item(), scores[0].item())

    def test_cosine_promotes_needle_over_background(self):
        scores = self._scores("cosine")[0]
        # cosine: needle aligned -> ~1.0; background off-axis -> ~0.707
        self.assertAlmostEqual(scores[0].item(), 1.0, places=4)
        self.assertLess(scores[1].item(), 0.99)
        self.assertGreater(scores[0].item(), scores[1].item())  # needle now wins

    def test_cosine_is_scale_invariant(self):
        # int8 path: cosine must ignore token_scales (it cancels under norm).
        q, s, w, cs, cw = _inputs()
        os.environ["SGLANG_DS_SCORER_NORM"] = "cosine"
        scales = torch.tensor([[[100.0], [0.01], [1.0]]]).squeeze(-1)  # [L,T,H]
        a = compute_token_scores(q, s, w, cs, cw, layer_id=0)
        b = compute_token_scores(q, s, w, cs, cw, layer_id=0, token_scales=scales)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))  # scale ignored

    def test_unwritten_tokens_are_neg_inf_both_modes(self):
        q, s, w, cs, cw = _inputs()
        w2 = w.clone()
        w2[0, 1] = False  # mark background unwritten
        for mode in ("off", "cosine"):
            os.environ["SGLANG_DS_SCORER_NORM"] = mode
            scores = compute_token_scores(q, s, w2, cs, cw, layer_id=0)
            self.assertTrue(torch.isneginf(scores[0, 1]))


if __name__ == "__main__":
    unittest.main()
