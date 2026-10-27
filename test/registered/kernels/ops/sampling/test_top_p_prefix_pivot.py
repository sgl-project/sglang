"""Top-p prefix-pivot parity for the ROCm renorm path.

Assertions are on mass moved and the retained support. A per-element atol
against a vocabulary-sized distribution hides a pivot that moves 20% of the mass.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

# GLM-5.3-Flash ships vocab 154,880 and top_p 0.95.
VOCAB = 154880
TOP_P = 0.95


def _exact_renorm(probs_fp32, top_ps):
    vocab_size = probs_fp32.shape[-1]
    sorted_probs = torch.sort(probs_fp32, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(
        1
    )
    cutoff.clamp_(max=vocab_size - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1))
    out = torch.where(probs_fp32 >= pivots, probs_fp32, torch.zeros_like(probs_fp32))
    return out / out.sum(dim=-1, keepdim=True)


def _probs(batch, vocab, *, nucleus=None, scale=1.0, seed=0, device="cuda:0"):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=gen).to(device) * scale
    if nucleus is not None:
        logits[:, :nucleus] += 20.0
    return logits.softmax(-1).float()


def _nucleus_width(probs_fp32, top_p):
    desc = torch.sort(probs_fp32, dim=-1, descending=True).values
    return int((torch.cumsum(desc, dim=-1) <= top_p).sum(dim=-1).max().item()) + 1


class TestTopPPrefixPivot(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_hip():
            raise unittest.SkipTest("renorm_triton top-p prefix is the ROCm path")

    def _prefix(self):
        from sglang.kernels.ops.sampling import renorm_triton

        return renorm_triton._TOP_P_PIVOT_PREFIX

    def test_covered_rows_match_the_full_sort(self):
        from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

        for batch_size, nucleus in ((1, 1), (48, 15), (8, 512)):
            with self.subTest(batch_size=batch_size, nucleus=nucleus):
                probs = _probs(batch_size, VOCAB, nucleus=nucleus, seed=nucleus)
                top_ps = torch.full((batch_size,), TOP_P, device=probs.device)
                self.assertLess(_nucleus_width(probs, TOP_P), self._prefix())
                got = top_p_renorm_probs_triton(probs=probs, top_p=top_ps)
                want = _exact_renorm(probs, top_ps)
                self.assertLess(got.sub(want).abs().sum(-1).max().item(), 1e-6)
                self.assertTrue(torch.equal(got > 0, want > 0))

    def test_uncovered_row_keeps_the_whole_row(self):
        """A nucleus wider than the prefix returns pivot 0 and does not truncate."""
        from sglang.kernels.ops.sampling.renorm_triton import (
            _top_p_pivots,
            top_p_renorm_probs_triton,
        )

        probs = _probs(4, VOCAB, scale=0.1, seed=3)
        top_ps = torch.full((4,), TOP_P, device=probs.device)
        self.assertGreater(_nucleus_width(probs, TOP_P), self._prefix())
        self.assertTrue(
            torch.equal(
                _top_p_pivots(probs_fp32=probs, top_ps=top_ps),
                torch.zeros(4, device=probs.device),
            )
        )
        got = top_p_renorm_probs_triton(probs=probs, top_p=top_ps)
        self.assertEqual(int((got > 0).sum(-1).min().item()), VOCAB)
        want = _exact_renorm(probs, top_ps)
        self.assertGreater(got.sub(want).abs().sum(-1).max().item(), 1e-2)

    def test_vocab_smaller_than_prefix_uses_the_exact_path(self):
        from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

        small = min(1024, self._prefix() // 2)
        probs = _probs(8, small, seed=2)
        top_ps = torch.full((8,), TOP_P, device=probs.device)
        got = top_p_renorm_probs_triton(probs=probs, top_p=top_ps)
        want = _exact_renorm(probs, top_ps)
        self.assertLess(got.sub(want).abs().sum(-1).max().item(), 1e-6)
        self.assertTrue(torch.equal(got > 0, want > 0))


if __name__ == "__main__":
    unittest.main()
