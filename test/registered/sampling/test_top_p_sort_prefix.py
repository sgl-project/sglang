"""Prefix-sort parity for the torch sampler fallbacks.

The prefix is a ROCm-only shortcut; elsewhere the full sort is kept. At vocab
154,880 a per-element atol exceeds the signal, so the checks are on mass moved
and support.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

VOCAB = 154880
TOP_P = 0.95


def _probs(batch, vocab, *, nucleus=None, scale=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=gen).cuda() * scale
    if nucleus is not None:
        logits[:, :nucleus] += 20.0
    return logits.softmax(-1).float()


def _exact_top_p_normalize(probs, top_ps):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs).scatter_(-1, probs_idx, probs_sort)


def _nucleus_width(probs, top_p):
    desc = torch.sort(probs, dim=-1, descending=True).values
    return int((torch.cumsum(desc, dim=-1) <= top_p).sum(dim=-1).max().item()) + 1


class TestTopPSortPrefix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("sampler fallbacks run on device")

    def _prefix(self):
        """Active prefix width, or None where the full sort is kept."""
        from sglang.srt.layers import sampler

        if not is_hip():
            self.skipTest("the prefix shortcut is the ROCm path")
        return sampler._TOP_P_SORT_PREFIX

    def test_prefix_width_prefers_exact_max_top_k(self):
        """max_top_k stays exact and is never replaced by the prefix."""
        from sglang.srt.layers.sampler import _descending_prefix_width

        self.assertEqual(_descending_prefix_width(vocab_size=VOCAB, max_top_k=32), 32)
        self.assertIsNone(_descending_prefix_width(vocab_size=64, max_top_k=None))
        self.assertEqual(
            _descending_prefix_width(vocab_size=VOCAB, max_top_k=None),
            self._prefix(),
        )

    def test_normalize_returns_full_vocabulary_width(self):
        """The scatter target is the full row, not the prefix-width sort."""
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        probs = _probs(8, VOCAB, nucleus=64, seed=1)
        top_ps = torch.full((8,), TOP_P, device=probs.device)
        got = top_p_normalize_probs_torch(probs=probs, top_ps=top_ps)
        self.assertEqual(got.shape, probs.shape)
        self.assertLess(self._prefix(), VOCAB)

    def test_normalize_covered_rows_match_full_sort(self):
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        for nucleus in (1, 15, 512):
            with self.subTest(nucleus=nucleus):
                probs = _probs(8, VOCAB, nucleus=nucleus, seed=nucleus)
                top_ps = torch.full((8,), TOP_P, device=probs.device)
                self.assertLess(_nucleus_width(probs, TOP_P), self._prefix())
                got = top_p_normalize_probs_torch(probs=probs, top_ps=top_ps)
                want = _exact_top_p_normalize(probs.clone(), top_ps)
                self.assertLess(got.sub(want).abs().sum(-1).max().item(), 1e-6)
                self.assertTrue(torch.equal(got > 0, want > 0))

    def test_normalize_uncovered_row_truncates_at_the_prefix(self):
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        probs = _probs(4, VOCAB, scale=0.1, seed=3)
        top_ps = torch.full((4,), TOP_P, device=probs.device)
        self.assertGreater(_nucleus_width(probs, TOP_P), self._prefix())
        got = top_p_normalize_probs_torch(probs=probs, top_ps=top_ps)
        self.assertLessEqual(int((got > 0).sum(-1).max().item()), self._prefix())
        want = _exact_top_p_normalize(probs.clone(), top_ps)
        self.assertGreater(int((want > 0).sum(-1).min().item()), self._prefix())

    def test_sampler_uses_max_top_k_ahead_of_the_prefix(self):
        """A real max_top_k must not be replaced by the fixed 2048 prefix."""
        from sglang.srt.layers import sampler

        probs = _probs(2, 4096, nucleus=8, seed=4)
        seen = {}
        real_topk = torch.topk

        def _record(values, k, dim=-1, sorted=True):
            seen["k"] = k
            return real_topk(values, k=k, dim=dim, sorted=sorted)

        with mock.patch.object(sampler.torch, "topk", _record):
            sampler.top_k_top_p_min_p_sampling_from_probs_torch(
                probs,
                torch.full((2,), 32, device=probs.device, dtype=torch.int64),
                torch.full((2,), TOP_P, device=probs.device),
                torch.zeros(2, device=probs.device),
                False,
                None,
                torch.arange(2, device=probs.device),
                max_top_k=32,
            )
        self.assertEqual(seen["k"], 32)

    def test_return_filtered_probs_uses_the_full_sort(self):
        from sglang.srt.layers.sampler import (
            top_k_top_p_min_p_sampling_from_probs_torch,
        )

        probs = _probs(4, VOCAB, nucleus=16, seed=8)
        out = top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            torch.full((4,), 16, device=probs.device, dtype=torch.int64),
            torch.full((4,), TOP_P, device=probs.device),
            torch.zeros(4, device=probs.device),
            False,
            None,
            torch.arange(4, device=probs.device),
            return_filtered_probs=True,
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[1].shape[-1], VOCAB)
        self.assertEqual(out[2].shape[-1], VOCAB)


if __name__ == "__main__":
    unittest.main()
