"""The one-pass Gumbel-max draft sample picks the same token as the three-op path."""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.sampling import argmax as sampling_argmax
from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.sampling.argmax_jit import can_argmax
from sglang.kernels.ops.speculative.row_argmax import div_argmax
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_utils import fast_sample, sample_draft_proposal
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_TINY = torch.finfo(torch.float32).tiny
_SPECIALS = [
    0.0,
    -0.0,
    float("nan"),
    float("inf"),
    -float("inf"),
    1e-45,
    1e-40,
    _TINY,
    3e38,
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _reference_fast_sample(probs: torch.Tensor):
    """fast_sample's unseeded path before this change: clamp_min_, division, argmax."""
    q = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
    q.clamp_min_(_TINY)
    index = sampling_argmax(probs.float() / q).unsqueeze(-1)
    return probs.gather(1, index), index


def _plant(x: torch.Tensor, frac: float, gen: torch.Generator) -> torch.Tensor:
    specials = torch.tensor(_SPECIALS, device=x.device)
    hit = torch.rand(x.shape, generator=gen, device=x.device) < frac
    pick = torch.randint(0, len(_SPECIALS), x.shape, generator=gen, device=x.device)
    return torch.where(hit, specials[pick], x)


def _edge_rows(vocab: int):
    p = torch.full((5, vocab), 0.5, device="cuda")
    q = torch.ones((5, vocab), device="cuda")
    p[0] = float("nan")  # no number at all
    p[1] = -float("inf")  # nothing above -inf
    p[2] = 0.0  # +0 / -0 tie
    p[2, 3] = -0.0
    p[3, vocab - 1] = 2.0  # duplicate maxima: the lowest index wins
    p[3, vocab // 2] = 2.0
    q[4, 7] = 0.0  # clamp_min makes this q tiny
    q[4, 9] = float("nan")
    return p, q


class TestGumbelArgmax(CustomTestCase):
    def _check(self, p, q):
        scores = p / q.clamp_min(_TINY)
        self.assertTrue(can_argmax(scores))
        # ignore_nan follows the CUDA sampling.argmax kernel; the default follows torch.argmax.
        fused = div_argmax(p, q, divisor_min=_TINY, ignore_nan=True)
        self.assertTrue(torch.equal(fused, sampling_argmax(scores)))
        fused_torch_rule = div_argmax(p, q, divisor_min=_TINY)
        self.assertTrue(torch.equal(fused_torch_rule, torch.argmax(scores, dim=-1)))

    def test_div_argmax_options_match_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        shapes = [(1, 4096), (7, 151936), (64, 151936), (5, 248320), (300, 32000)]
        for rows, vocab in shapes:
            logits = torch.randn((rows, vocab), generator=gen, device="cuda") * 4
            p = torch.softmax(logits, dim=-1)
            q = torch.empty_like(p).exponential_(1.0, generator=gen)
            self._check(p, q)
            self._check(_plant(p, 0.01, gen), _plant(q, 0.01, gen))
        for vocab in (4096, 248320):
            self._check(*_edge_rows(vocab))

    def test_div_argmax_default_unchanged(self):
        gen = torch.Generator(device="cuda").manual_seed(3)
        x = torch.randn((9, 151936), generator=gen, device="cuda")
        x = _plant(x, 0.001, gen)
        d = torch.rand((9, 151936), generator=gen, device="cuda") + 0.5
        self.assertTrue(torch.equal(div_argmax(x, d), torch.argmax(x / d, dim=-1)))

    def test_fast_sample_matches_reference(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        # Below 2**23 elements and for 4097 (not a whole number of vectors) both
        # sides take the torch path; 2049 x 4097 is above the size gate.
        cases = [(1, 32000, 0), (64, 151936, 1), (65, 248320, 1), (2049, 4097, 0)]
        for rows, vocab, fused_calls in cases:
            logits = torch.randn((rows, vocab), generator=gen, device="cuda") * 3
            probs = torch.softmax(logits, dim=-1)
            torch.cuda.manual_seed(1000 + rows)
            ref_p, ref_index = _reference_fast_sample(probs)
            torch.cuda.manual_seed(1000 + rows)
            with mock.patch.object(
                spec_utils, "div_argmax", wraps=spec_utils.div_argmax
            ) as fused:
                p, index = fast_sample(probs)
            self.assertEqual(fused.call_count, fused_calls)
            self.assertTrue(torch.equal(index, ref_index))
            self.assertTrue(torch.equal(_bits(p), _bits(ref_p)))

    def test_fast_sample_matches_reference_on_a_step_view(self):
        # The multi-layer EAGLE single-graph draft samples from
        # draft_probs[:bs, step] of a [max_bs, num_steps, vocab] buffer.
        gen = torch.Generator(device="cuda").manual_seed(3)
        rows, steps, vocab = 64, 3, 151936
        logits = torch.randn((rows, steps, vocab), generator=gen, device="cuda") * 3
        buffer = torch.softmax(logits, dim=-1)
        for step in range(steps):
            probs = buffer[:, step]
            self.assertFalse(probs.is_contiguous())
            torch.cuda.manual_seed(2000 + step)
            ref_p, ref_index = _reference_fast_sample(probs)
            torch.cuda.manual_seed(2000 + step)
            with mock.patch.object(
                spec_utils, "div_argmax", wraps=spec_utils.div_argmax
            ) as fused:
                p, index = fast_sample(probs)
            self.assertEqual(fused.call_count, 1)  # the fused path ran
            self.assertTrue(torch.equal(index, ref_index))
            self.assertTrue(torch.equal(_bits(p), _bits(ref_p)))

    def test_draft_proposal_cuda_graph_replay(self):
        rows, vocab = 64, 151936
        logits = torch.zeros((rows, vocab), device="cuda")
        temperatures = torch.ones((rows, 1), device="cuda")
        top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")

        def proposal():
            return sample_draft_proposal(logits, temperatures, top_ks)

        def reference_proposal():
            # No greedy rows (top_k = 20), so the proposal is the plain sample.
            probs = sampling_softmax(logits, temperatures=temperatures)
            return (probs, *_reference_fast_sample(probs))

        graphs = []
        for fn in (proposal, reference_proposal):
            fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            graphs.append((graph, out))

        gen = torch.Generator(device="cuda").manual_seed(2)
        for rep in range(3):
            logits.copy_(torch.randn((rows, vocab), generator=gen, device="cuda") * 4)
            temps = torch.rand((rows, 1), generator=gen, device="cuda") * 1.5 + 0.1
            temperatures.copy_(temps)
            results = []
            for graph, out in graphs:
                torch.cuda.manual_seed(77 + rep)
                graph.replay()
                results.append([t.clone() for t in out])
            (probs, p, index), (ref_probs, ref_p, ref_index) = results
            self.assertTrue(torch.equal(_bits(probs), _bits(ref_probs)))
            self.assertTrue(torch.equal(_bits(p), _bits(ref_p)))
            self.assertTrue(torch.equal(index, ref_index))


if __name__ == "__main__":
    unittest.main()
