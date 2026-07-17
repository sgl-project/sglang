"""Fused Gumbel-max draw (gumbel_argmax_sample) vs the unfused torch tail it
replaces. With explicit noise the draw is deterministic, so the kernel must
reproduce argmax(probs.float() / clamp(noise)) and the winning-probability
gather exactly (also a race detector for the last-block reduction); the
default in-kernel Philox path is checked distributionally and for per-launch
/ per-graph-replay freshness.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.gumbel_sample import gumbel_argmax_sample
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")

DEV = "cuda"


def _ref_tail(probs, noise):
    q = noise.clone().clamp_min_(torch.finfo(torch.float32).tiny)
    scores = probs.float() / q
    sample_index = scores.argmax(dim=-1, keepdim=True)
    return probs.gather(1, sample_index), sample_index, scores


@unittest.skipUnless(torch.cuda.is_available(), "kernel needs CUDA")
class TestGumbelArgmaxSample(CustomTestCase):
    def test_matches_unfused_tail(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for bs, vocab in ((1, 152064), (8, 129280), (64, 4097), (3, 5)):
                with self.subTest(dtype=dtype, bs=bs, vocab=vocab):
                    probs = torch.softmax(
                        torch.randn(bs, vocab, device=DEV) * 3, dim=-1
                    ).to(dtype)
                    noise = torch.empty(
                        bs, vocab, dtype=torch.float32, device=DEV
                    ).exponential_(1.0)
                    ref_p, ref_i, scores = _ref_tail(probs, noise)
                    got_p, got_i = gumbel_argmax_sample(probs, noise)
                    self.assertEqual(got_i.shape, (bs, 1))
                    self.assertEqual(got_i.dtype, torch.int64)
                    self.assertEqual(got_p.dtype, probs.dtype)
                    # Continuous random noise makes ties measure-zero; the
                    # winning score must match exactly, and so must the index
                    # unless two scores are bitwise-equal (then either is
                    # correct).
                    got_scores = scores.gather(1, got_i)
                    ref_scores = scores.gather(1, ref_i)
                    self.assertTrue(torch.equal(got_scores, ref_scores))
                    self.assertTrue(torch.equal(got_p, probs.gather(1, got_i)))

    def test_last_block_reduction_race_stress(self):
        # The last-block pattern's failure mode is a torn/stale partial read;
        # any occurrence shows up as a wrong winner vs the reference.
        torch.manual_seed(2)
        bs, vocab = 16, 8192
        probs = torch.softmax(torch.randn(bs, vocab, device=DEV), dim=-1)
        for i in range(500):
            noise = torch.empty(
                bs, vocab, dtype=torch.float32, device=DEV
            ).exponential_(1.0)
            _, ref_i, scores = _ref_tail(probs, noise)
            _, got_i = gumbel_argmax_sample(probs, noise)
            self.assertTrue(
                torch.equal(scores.gather(1, got_i), scores.gather(1, ref_i)),
                f"race-stress mismatch at iteration {i}",
            )

    def test_zero_noise_clamped(self):
        probs = torch.softmax(torch.randn(4, 1000, device=DEV), dim=-1)
        noise = torch.empty(4, 1000, dtype=torch.float32, device=DEV).exponential_(1.0)
        noise[:, 7] = 0.0
        got_p, got_i = gumbel_argmax_sample(probs, noise)
        ref_p, ref_i, _ = _ref_tail(probs, noise)
        self.assertTrue(torch.equal(got_i, ref_i))
        self.assertTrue(torch.equal(got_p, ref_p))

    def test_philox_distribution(self):
        torch.manual_seed(1)
        n, vocab = 200_000, 500
        probs_row = torch.softmax(torch.randn(vocab, device=DEV) * 2, dim=-1)
        probs = probs_row.expand(n, vocab).contiguous()
        _, idx = gumbel_argmax_sample(probs)
        emp = torch.bincount(idx.flatten(), minlength=vocab).float() / n
        tv = 0.5 * (emp - probs_row).abs().sum().item()
        self.assertLess(tv, 0.02)

    def test_philox_fresh_across_launches_and_replays(self):
        torch.manual_seed(3)
        vocab = 4096
        probs = torch.softmax(torch.randn(4, vocab, device=DEV), dim=-1)
        _, a = gumbel_argmax_sample(probs)
        _, b = gumbel_argmax_sample(probs)
        # Same probs, fresh noise: 4 rows all colliding twice in a row is
        # ~impossible for a non-degenerate distribution.
        self.assertFalse(torch.equal(a, b))

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                gumbel_argmax_sample(probs)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            _, idx = gumbel_argmax_sample(probs)
        draws = []
        for _ in range(8):
            g.replay()
            torch.cuda.synchronize()
            draws.append(idx.clone())
        distinct = len({tuple(d.flatten().tolist()) for d in draws})
        self.assertGreater(distinct, 1, "graph replays reused identical noise")
        for d in draws:
            self.assertTrue((d >= 0).all().item() and (d < vocab).all().item())

    def test_empty_batch(self):
        probs = torch.zeros(0, 128, device=DEV)
        got_p, got_i = gumbel_argmax_sample(probs)
        self.assertEqual(got_p.shape, (0, 1))
        self.assertEqual(got_i.shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
