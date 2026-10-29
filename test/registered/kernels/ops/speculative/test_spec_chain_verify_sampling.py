# SPDX-License-Identifier: Apache-2.0
"""Chain verify sampling: the lean target-only path and its two-launch form give the same bits.

For chains, ``tree_speculative_sampling_target_only`` is called three ways that must
return the same predicts / accept_index / accept_token_num:

* with a zero-filled ``draft_probs`` (the stock call),
* with ``draft_probs=None`` (``SGLANG_SPEC_LEAN_SAMPLED_VERIFY``),
* with ``draft_probs=None, chunked=True`` (``SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING``: the
  two vocabulary passes over one CTA per chunk),

over accept / reject at every position, edge coins (0 and the largest float below 1),
flat, tail-only and zero-mass rows (the last-valid-index fallback), every vector width
(vocab sizes with gcd(4, vocab) = 4 / 2 / 1) and a one-chunk vocabulary; the chunked
form also under CUDA-graph replay. Dividing the logits by each request's temperature
through a broadcast view gives the same probabilities as the expanded temperatures.
"""

import unittest

import torch

from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.speculative.sampling import (
    tree_speculative_sampling_target_only,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

VOCAB = 154880


def _run(probs, candidates, coins, final, draft_probs, chunked, out=None):
    bs, n, _ = probs.shape
    if out is None:
        out = (
            torch.empty(bs * n, dtype=torch.int32, device="cuda"),
            torch.empty((bs, n), dtype=torch.int32, device="cuda"),
            torch.empty((bs,), dtype=torch.int32, device="cuda"),
        )
    predicts, accept_index, accept_num = out
    predicts.fill_(-7)
    accept_index.fill_(-1)
    accept_num.fill_(-7)
    index = torch.arange(bs * n, device="cuda", dtype=torch.int64).view(bs, n)
    next_token = torch.arange(1, n + 1, device="cuda", dtype=torch.int64).repeat(bs, 1)
    next_token[:, -1] = -1
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        retrive_index=index,
        retrive_next_token=next_token,
        retrive_next_sibling=torch.full((bs, n), -1, device="cuda", dtype=torch.int64),
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final,
        target_probs=probs,
        draft_probs=draft_probs,
        deterministic=True,
        chunked=chunked,
    )
    return out


def _candidates(probs, gen, accept_rate=0.6):
    """Per position the target's top token (likely accepted) or a random one."""
    bs, n, vocab = probs.shape
    top = probs.argmax(-1)
    rand_tok = torch.randint(0, vocab, (bs, n), device="cuda", generator=gen)
    pick = torch.rand((bs, n), device="cuda", generator=gen) < accept_rate
    candidates = torch.where(pick, top, rand_tok).to(torch.int64)
    candidates[:, 1:] = torch.where(pick[:, :-1], top[:, :-1], rand_tok[:, 1:])
    return candidates


def _coins(bs, n, mode, gen):
    if mode == "rand":
        return (
            torch.rand((bs, n), device="cuda", generator=gen),
            torch.rand((bs,), device="cuda", generator=gen),
        )
    value = 0.0 if mode == "zero" else 1.0 - 2**-24
    return (
        torch.full((bs, n), value, device="cuda"),
        torch.full((bs,), value, device="cuda"),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestChainVerifySampling(CustomTestCase):
    def _check(self, probs, gen, coin_mode, **sub):
        candidates = _candidates(probs, gen)
        coins, final = _coins(probs.shape[0], probs.shape[1], coin_mode, gen)
        ref = _run(probs, candidates, coins, final, torch.zeros_like(probs), False)
        with self.subTest(coins=coin_mode, **sub):
            for chunked in (False, True):
                got = _run(probs, candidates, coins, final, None, chunked)
                for name, r, g in zip(("predicts", "index", "num"), ref, got):
                    self.assertTrue(torch.equal(r, g), f"{name} chunked={chunked}")
        return ref

    def test_grid(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for bs in (1, 8):
            for n in (2, 5):
                for sharp in (0.5, 12.0):
                    for coin_mode in ("rand", "zero", "high"):
                        logits = torch.randn(
                            (bs, n, VOCAB), device="cuda", generator=gen
                        )
                        probs = torch.softmax(logits * sharp, dim=-1)
                        self._check(probs, gen, coin_mode, bs=bs, n=n, sharp=sharp)

    def test_edge_distributions(self):
        gen = torch.Generator(device="cuda").manual_seed(13)
        bs, n = 4, 4
        flat = torch.full((bs, n, VOCAB), 1.0 / VOCAB, device="cuda")
        tail = torch.zeros((bs, n, VOCAB), device="cuda")
        tail[..., -3:] = torch.softmax(
            torch.randn((bs, n, 3), device="cuda", generator=gen), dim=-1
        )
        zero = torch.zeros((bs, n, VOCAB), device="cuda")
        for name, probs in (("flat", flat), ("tail", tail), ("zero", zero)):
            for coin_mode in ("rand", "zero", "high"):
                ref = self._check(probs, gen, coin_mode, dist=name)
                if name == "zero" and coin_mode == "high":  # nothing is accepted
                    self.assertEqual(ref[2].tolist(), [0] * bs)

    def test_vector_widths_and_small_vocab(self):
        gen = torch.Generator(device="cuda").manual_seed(2)
        for vocab in (154882, 154881, 4096, 4100, 3):
            logits = torch.randn((3, 4, vocab), device="cuda", generator=gen)
            probs = torch.softmax(logits * 3, dim=-1)
            for coin_mode in ("rand", "high"):
                self._check(probs, gen, coin_mode, vocab=vocab)

    def test_chunked_cuda_graph_replay(self):
        gen = torch.Generator(device="cuda").manual_seed(21)
        bs, n = 2, 4
        probs = torch.zeros((bs, n, VOCAB), device="cuda")
        candidates = torch.zeros((bs, n), device="cuda", dtype=torch.int64)
        coins = torch.zeros((bs, n), device="cuda")
        final = torch.zeros((bs,), device="cuda")
        got = _run(probs, candidates, coins, final, None, True)  # load the module
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run(probs, candidates, coins, final, None, True, out=got)
        for rep in range(4):
            logits = torch.randn((bs, n, VOCAB), device="cuda", generator=gen)
            probs.copy_(torch.softmax(logits * (1 + 3 * rep), dim=-1))
            candidates.copy_(_candidates(probs, gen))
            c, f = _coins(bs, n, "rand", gen)
            coins.copy_(c)
            final.copy_(f)
            graph.replay()
            torch.cuda.synchronize()
            ref = _run(probs, candidates, coins, final, torch.zeros_like(probs), False)
            with self.subTest(rep=rep):
                for r, g in zip(ref, got):
                    self.assertTrue(torch.equal(r, g))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTemperatureBroadcast(CustomTestCase):
    def test_same_softmax(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        for bs in (1, 2, 8):
            n = 4
            logits = torch.randn((bs * n, VOCAB), device="cuda", generator=gen) * 5
            temps = torch.rand((bs, 1), device="cuda", generator=gen) * 1.5 + 0.1
            ref = sampling_softmax(
                logits, temperatures=torch.repeat_interleave(temps, n, dim=0)
            )
            got = sampling_softmax(
                (logits.view(bs, n, -1) / temps.view(bs, 1, 1)).view(bs * n, -1)
            )
            with self.subTest(bs=bs):
                self.assertTrue(torch.equal(ref, got))


if __name__ == "__main__":
    unittest.main()
