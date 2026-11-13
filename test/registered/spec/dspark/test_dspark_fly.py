"""FLy verification for DSpark: the greedy accept it yields and the sampling draws it
zeroes on hand-built steps, the triton pre-passes against their torch references, the
split and vocab-shard sampling accepts against the chain rule on the reference draws,
and the settings validation."""

import math
import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_accept_split import (
    chain_speculative_sampling_split,
)
from sglang.kernels.ops.speculative.dspark.sharded_accept_hip import (
    _ACCEPT_STEP,
    _philox,
    accept_sampling_sharded,
)
from sglang.kernels.ops.speculative.fly import (
    FlyGreedyPrepass,
    FlySamplingPrepass,
    compute_fly_entropy,
    compute_fly_entropy_from_probs,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.dflash_utils import _get_or_create_chain_verify_buffers
from sglang.srt.speculative.fly_config import (
    MAX_FLY_ENTROPY_TOP_K,
    MAX_FLY_GAMMA,
    FlyConfig,
    resolve_fly_config,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")

VOCAB = 16
GAMMA = 5
NUM_ROWS = GAMMA + 1


def _step(targets, drafts, uncertain=(), masked=()):
    """One request's verify step: ``targets`` the target argmax at each of the
    NUM_ROWS rows, ``drafts`` the GAMMA draft tokens. An ``uncertain`` draft row
    puts its draft token second (top-3 entropy ~0.81); every other row is peaked
    (entropy ~0). A ``masked`` row gives its draft token a -inf target logit."""
    logits = torch.zeros(NUM_ROWS, VOCAB)
    for row, target in enumerate(targets):
        logits[row, target] = 20.0
    for row in uncertain:
        logits[row] = 0.0
        logits[row, targets[row]] = 2.0
        logits[row, drafts[row]] = 1.5
    for row in masked:
        logits[row, drafts[row]] = float("-inf")
    candidates = torch.tensor([[7] + list(drafts)])
    return candidates, logits


def _accept(candidates, logits, fly, verify_lens=None):
    correct_len, bonus, _ = AcceptGreedy.torch(
        candidates=candidates,
        target_logits=logits,
        verify_num_draft_tokens=NUM_ROWS,
        cutoff_verify_lens=verify_lens,
        fly=fly,
    )
    return int(correct_len[0]), int(bonus[0])


class TestFlyGreedyAccept(CustomTestCase):
    fly = FlyConfig(window_size=2, entropy_threshold=0.3, entropy_top_k=3)

    def test_defers_uncertain_rejection_with_full_window(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6], drafts=[1, 9, 3, 4, 8], uncertain=[1]
        )
        self.assertEqual(_accept(candidates, logits, None), (1, 2))
        # Row 1 is deferred (rows 2 and 3 match); the walk stops at row 4.
        self.assertEqual(_accept(candidates, logits, self.fly), (4, 5))

    def test_confident_rejection_is_kept(self):
        candidates, logits = _step(targets=[1, 2, 3, 4, 5, 6], drafts=[1, 9, 3, 4, 5])
        self.assertEqual(_accept(candidates, logits, self.fly), (1, 2))

    def test_threshold_gates_the_deferral(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6], drafts=[1, 9, 3, 4, 5], uncertain=[1]
        )
        entropy = float(compute_fly_entropy(logits[1:2], 3)[0])
        strict = FlyConfig(window_size=2, entropy_threshold=entropy + 1e-3)
        self.assertEqual(_accept(candidates, logits, strict), (1, 2))
        self.assertEqual(_accept(candidates, logits, self.fly), (5, 6))

    def test_rejection_inside_window_rejects_original(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6], drafts=[1, 9, 3, 8, 5], uncertain=[1, 3]
        )
        # Row 3 rejects inside row 1's window, so row 1's own target is the bonus.
        self.assertEqual(_accept(candidates, logits, self.fly), (1, 2))

    def test_short_trailing_window_is_not_deferred(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6], drafts=[1, 2, 3, 9, 5], uncertain=[3]
        )
        # Row 3 has one draft after it, fewer than the window of 2.
        self.assertEqual(_accept(candidates, logits, self.fly), (3, 4))
        wide = FlyConfig(window_size=1)
        self.assertEqual(_accept(candidates, logits, wide), (5, 6))

    def test_masked_draft_is_not_deferred(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6],
            drafts=[1, 9, 3, 4, 5],
            uncertain=[1],
            masked=[1],
        )
        self.assertEqual(_accept(candidates, logits, self.fly), (1, 2))

    def test_chained_deferrals(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6], drafts=[9, 2, 8, 4, 5], uncertain=[0, 2]
        )
        self.assertEqual(_accept(candidates, logits, FlyConfig(window_size=1)), (5, 6))

    def test_window_stays_inside_verified_drafts(self):
        candidates, logits = _step(
            targets=[1, 2, 3, 4, 5, 6], drafts=[9, 2, 3, 4, 5], uncertain=[0]
        )
        # Four verified rows (three drafts): rows 1 and 2 form row 0's window.
        self.assertEqual(
            _accept(candidates, logits, self.fly, verify_lens=torch.tensor([4])),
            (3, 4),
        )
        # Three verified rows: row 0 lacks a full window inside them.
        self.assertEqual(
            _accept(candidates, logits, self.fly, verify_lens=torch.tensor([3])),
            (0, 1),
        )


class TestFlyEntropy(CustomTestCase):
    def test_top_k_entropy_definition(self):
        logits = torch.tensor(
            [
                [2.0, 1.5, 0.0, 0.0],
                [0.0, float("-inf"), 0.0, 0.0],
                [5.0, -1e4, -1e4, -1e4],
            ]
        )
        expected = []
        for row in logits.softmax(dim=-1):
            top = sorted(row.tolist(), reverse=True)[:3]
            top = [p / sum(top) for p in top]
            expected.append(-sum(p * math.log(p) for p in top if p > 0) / math.log(3))
        expected = torch.tensor(expected)
        torch.testing.assert_close(
            compute_fly_entropy(logits, 3), expected, rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            compute_fly_entropy_from_probs(logits.softmax(dim=-1), 3),
            expected,
            rtol=0,
            atol=1e-6,
        )

    def test_renormalizes_the_top_k(self):
        # The top-3 values (0.8, 0.05, 0.01) and (0.06, 0.02, 0.01), as in vLLM main.
        probs = torch.tensor([[0.8, 0.05, 0.01, 0.0], [0.06, 0.02, 0.01, 0.0]])
        torch.testing.assert_close(
            compute_fly_entropy_from_probs(probs, 3),
            torch.tensor([0.2589, 0.7725]),
            rtol=0,
            atol=1e-4,
        )

    def test_top_1_is_zero(self):
        logits = torch.randn(4, 16)
        self.assertTrue(torch.equal(compute_fly_entropy(logits, 1), torch.zeros(4)))
        self.assertTrue(
            torch.equal(
                compute_fly_entropy_from_probs(logits.softmax(dim=-1), 1),
                torch.zeros(4),
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "triton pre-pass needs a GPU")
class TestFlyTritonParity(CustomTestCase):
    def _random_step(self, bs, num_rows, vocab, generator):
        device = "cuda"
        scale = torch.rand(bs * num_rows, 1, device=device, generator=generator) * 6
        logits = (
            torch.randn(bs * num_rows, vocab, device=device, generator=generator)
            * scale
        )
        top2 = logits.topk(2, dim=-1).indices.view(bs, num_rows, 2)
        drafts = top2[:, :-1, 0].clone()
        roll = torch.rand(bs, num_rows - 1, device=device, generator=generator)
        drafts = torch.where(roll < 0.3, top2[:, :-1, 1], drafts)
        masked = torch.rand(bs, num_rows - 1, device=device, generator=generator) < 0.1
        rows = torch.arange(bs * num_rows, device=device).view(bs, num_rows)[:, :-1]
        logits[rows[masked], drafts[masked]] = float("-inf")
        anchor = torch.zeros(bs, 1, dtype=drafts.dtype, device=device)
        return torch.cat([anchor, drafts], dim=1).contiguous(), logits.contiguous()

    def test_triton_matches_torch(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        cases = [
            (64, 6, 1000, 2, False),
            (64, 6, 1000, 1, True),
            (33, 8, 129280, 3, True),
            (256, 6, 1000, 4, False),
        ]
        for bs, num_rows, vocab, window, ragged in cases:
            with self.subTest(bs=bs, num_rows=num_rows, vocab=vocab, window=window):
                candidates, logits = self._random_step(bs, num_rows, vocab, generator)
                target_predict = logits.argmax(dim=-1)
                verify_lens = (
                    torch.randint(
                        0, num_rows + 1, (bs,), device="cuda", generator=generator
                    )
                    if ragged
                    else None
                )
                fly = FlyConfig(window_size=window)
                kwargs = dict(
                    candidates=candidates,
                    target_predict=target_predict,
                    target_logits=logits,
                    fly=fly,
                    verify_lens=verify_lens,
                )
                expected = FlyGreedyPrepass.torch(**kwargs)
                actual = FlyGreedyPrepass.triton(**kwargs)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(target_predict, logits.argmax(dim=-1)))
                self.assertTrue(bool((actual != target_predict).any()))

    def test_top_1_defers_only_at_threshold_zero(self):
        generator = torch.Generator(device="cuda").manual_seed(3)
        candidates, logits = self._random_step(64, 6, 1000, generator)
        target_predict = logits.argmax(dim=-1)
        for threshold, defers in ((0.0, True), (0.3, False)):
            with self.subTest(threshold=threshold):
                kwargs = dict(
                    candidates=candidates,
                    target_predict=target_predict,
                    target_logits=logits,
                    fly=FlyConfig(
                        window_size=1, entropy_threshold=threshold, entropy_top_k=1
                    ),
                )
                actual = FlyGreedyPrepass.triton(**kwargs)
                self.assertTrue(torch.equal(actual, FlyGreedyPrepass.torch(**kwargs)))
                self.assertEqual(bool((actual != target_predict).any()), defers)


def _chain_accepts(p, q, u):
    """Accepted drafts per row under the chain rule u * q < p, each [bs, gamma]."""
    ok = (q > 0) & (q <= 1) & (u * q < p)
    return torch.where(ok.all(dim=1), ok.shape[1], (~ok).int().argmax(dim=1))


def _sampling_step(targets, drafts, uncertain=(), unproposed=()):
    """``_step``'s target rows as probabilities and a one-hot draft (q = 1 on each
    draft token), except that an ``unproposed`` row's draft gives its token q = 0."""
    candidates, logits = _step(targets, drafts, uncertain)
    target_probs = logits.softmax(dim=-1)[None]
    draft_probs = torch.zeros(1, GAMMA, VOCAB)
    draft_probs[0, torch.arange(GAMMA), torch.tensor(drafts)] = 1.0
    for row in unproposed:
        draft_probs[0, row] = 0.0
        draft_probs[0, row, (drafts[row] + 1) % VOCAB] = 1.0
    return candidates, target_probs, draft_probs


class TestFlySamplingDraws(CustomTestCase):
    fly = FlyConfig(window_size=2, entropy_threshold=0.3, entropy_top_k=3)

    def _run(self, candidates, target_probs, draft_probs, draws, verify_lens=None):
        """(accepted drafts natively, the FLy draws, accepted drafts on them)."""
        rows = torch.arange(GAMMA)
        drafts = candidates[0, 1:]
        p = target_probs[0, rows, drafts][None]
        q = draft_probs[0, rows, drafts][None]
        fly_draws = FlySamplingPrepass.torch(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            uniform_samples=draws,
            fly=self.fly,
            verify_lens=verify_lens,
        )
        native, accepted = _chain_accepts(p, q, draws), _chain_accepts(p, q, fly_draws)
        return int(native[0]), fly_draws[0].tolist(), int(accepted[0])

    def test_defers_uncertain_rejection_with_full_window(self):
        step = _sampling_step([1, 2, 3, 4, 5, 6], [1, 9, 3, 4, 5], uncertain=[1])
        # p = 0.17 at row 1 fails the 0.5 draw; FLy zeroes it and the chain completes.
        self.assertEqual(
            self._run(*step, torch.full((1, GAMMA), 0.5)),
            (1, [0.5, 0.0, 0.5, 0.5, 0.5], 5),
        )

    def test_passing_draw_is_left_alone(self):
        step = _sampling_step([1, 2, 3, 4, 5, 6], [1, 9, 3, 4, 5], uncertain=[1])
        draws = torch.tensor([[0.5, 0.1, 0.5, 0.5, 0.5]])
        self.assertEqual(self._run(*step, draws), (5, draws[0].tolist(), 5))

    def test_confident_rejection_is_kept(self):
        step = _sampling_step([1, 2, 3, 4, 5, 6], [1, 9, 3, 4, 5])
        self.assertEqual(
            self._run(*step, torch.full((1, GAMMA), 0.5)), (1, [0.5] * GAMMA, 1)
        )

    def test_unproposed_draft_is_not_deferred(self):
        step = _sampling_step(
            [1, 2, 3, 4, 5, 6], [1, 9, 3, 4, 5], uncertain=[1], unproposed=[1]
        )
        self.assertEqual(
            self._run(*step, torch.full((1, GAMMA), 0.5)), (1, [0.5] * GAMMA, 1)
        )

    def test_rejection_inside_window_rejects_original(self):
        step = _sampling_step([1, 2, 3, 4, 5, 6], [1, 9, 3, 8, 5], uncertain=[1, 3])
        self.assertEqual(
            self._run(*step, torch.full((1, GAMMA), 0.5)), (1, [0.5] * GAMMA, 1)
        )

    def test_window_stays_inside_verified_drafts(self):
        step = _sampling_step([1, 2, 3, 4, 5, 6], [9, 2, 3, 4, 5], uncertain=[0])
        draws = torch.full((1, GAMMA), 0.5)
        self.assertEqual(
            self._run(*step, draws, verify_lens=torch.tensor([4]))[1],
            [0.0, 0.5, 0.5, 0.5, 0.5],
        )
        self.assertEqual(
            self._run(*step, draws, verify_lens=torch.tensor([3]))[1], [0.5] * GAMMA
        )


@triton.jit
def _philox_draws_kernel(key, offset, out, SLOTS_P2: tl.constexpr):
    """The draws accept_sampling_sharded's accept takes for request b."""
    b = tl.program_id(0)
    slot = tl.arange(0, SLOTS_P2)
    word, _, _, _ = _philox(key, slot, b, _ACCEPT_STEP, offset)
    tl.store(out + b * SLOTS_P2 + slot, tl.uint_to_uniform_float(word))


@unittest.skipUnless(torch.cuda.is_available(), "triton pre-pass needs a GPU")
class TestFlySamplingKernels(CustomTestCase):
    fly = FlyConfig(window_size=2, entropy_threshold=0.3, entropy_top_k=3)
    cases = [(64, 6, 1000, False), (64, 6, 1000, True), (32, 8, 129280, True)]

    def _random_step(self, bs, num_rows, vocab, ragged, generator):
        """A sampled verify step: target rows with three strong candidates over a weak
        background, draft logits tracking them, the drafts drawn from the draft
        probabilities at the requests' temperatures, every request verifying at
        least its anchor row."""
        device = "cuda"
        rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)
        randn = lambda *shape: torch.randn(*shape, device=device, generator=generator)
        temperatures = 0.5 + rand(bs)
        target_logits = randn(bs * num_rows, vocab)
        peaks = torch.randint(
            0, vocab, (bs * num_rows, 3), device=device, generator=generator
        )
        target_logits.scatter_(1, peaks, 20.0 + 2.0 * randn(bs * num_rows, 3))
        draft_logits = (
            target_logits.view(bs, num_rows, vocab)[:, :-1]
            + randn(bs, num_rows - 1, vocab) * 0.5
        ).contiguous()
        scale = temperatures[:, None, None]
        target_probs = torch.softmax(
            target_logits.view(bs, num_rows, vocab) / scale, -1
        )
        draft_probs = torch.softmax(draft_logits / scale, -1)
        drafts = torch.multinomial(
            draft_probs.reshape(-1, vocab), 1, generator=generator
        ).view(bs, num_rows - 1)
        anchor = torch.zeros(bs, 1, dtype=drafts.dtype, device=device)
        verify_lens = (
            torch.randint(1, num_rows + 1, (bs,), device=device, generator=generator)
            if ragged
            else None
        )
        return dict(
            temperatures=temperatures,
            target_logits=target_logits,
            draft_logits=draft_logits,
            target_probs=target_probs.contiguous(),
            draft_probs=draft_probs.contiguous(),
            candidates=torch.cat([anchor, drafts], dim=1).contiguous(),
            draws=rand(bs, num_rows - 1),
            verify_lens=verify_lens,
        )

    def _reference_draws(self, step, draws):
        return FlySamplingPrepass.torch(
            candidates=step["candidates"],
            target_probs=step["target_probs"],
            draft_probs=step["draft_probs"],
            uniform_samples=draws,
            fly=self.fly,
            verify_lens=step["verify_lens"],
        )

    @staticmethod
    def _capped(accepted, verify_lens):
        if verify_lens is None:
            return accepted.int()
        return torch.clamp(torch.minimum(accepted, verify_lens - 1), min=0).int()

    def test_triton_prepass_matches_torch(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        for bs, num_rows, vocab, ragged in self.cases:
            with self.subTest(bs=bs, num_rows=num_rows, vocab=vocab, ragged=ragged):
                step = self._random_step(bs, num_rows, vocab, ragged, generator)
                expected = self._reference_draws(step, step["draws"])
                actual = FlySamplingPrepass.triton(
                    candidates=step["candidates"],
                    target_probs=step["target_probs"],
                    draft_probs=step["draft_probs"],
                    uniform_samples=step["draws"],
                    fly=self.fly,
                    verify_lens=step["verify_lens"],
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(bool((expected == 0).any()))

    def test_split_accept_matches_chain_rule(self):
        generator = torch.Generator(device="cuda").manual_seed(1)
        for bs, num_rows, vocab, ragged in self.cases:
            with self.subTest(bs=bs, num_rows=num_rows, vocab=vocab, ragged=ragged):
                step = self._random_step(bs, num_rows, vocab, ragged, generator)
                final = torch.rand(bs, device="cuda", generator=generator)

                def accepted(sampler, **kwargs):
                    buffers = _get_or_create_chain_verify_buffers(
                        bs=bs, draft_token_num=num_rows, device="cuda"
                    )
                    retrive_index, next_token, next_sibling, predicts = buffers[:4]
                    accept_index, accept_token_num = buffers[4:]
                    if sampler is chain_speculative_sampling_triton:
                        kwargs.update(
                            retrive_next_token=next_token,
                            retrive_next_sibling=next_sibling,
                            target_probs=step["target_probs"],
                            threshold_single=1.0,
                            threshold_acc=1.0,
                            deterministic=True,
                        )
                    else:
                        kwargs.update(
                            target_logits=step["target_logits"],
                            temperatures=step["temperatures"],
                        )
                    sampler(
                        predicts=predicts,
                        accept_index=accept_index,
                        accept_token_num=accept_token_num,
                        candidates=step["candidates"],
                        retrive_index=retrive_index,
                        uniform_samples_for_final_sampling=final,
                        draft_probs=step["draft_probs"],
                        **kwargs,
                    )
                    return self._capped(accept_token_num.clone(), step["verify_lens"])

                expected = accepted(
                    chain_speculative_sampling_triton,
                    uniform_samples=self._reference_draws(step, step["draws"]),
                )
                actual = accepted(
                    chain_speculative_sampling_split,
                    uniform_samples=step["draws"],
                    fly=self.fly,
                    verify_lens=step["verify_lens"],
                )
                native = accepted(
                    chain_speculative_sampling_split, uniform_samples=step["draws"]
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertGreater(actual.sum().item(), native.sum().item())

    def test_shard_accept_matches_chain_rule(self):
        generator = torch.Generator(device="cuda").manual_seed(2)
        key, offset = 1234, torch.tensor([7], dtype=torch.int64, device="cuda")
        for bs, num_rows, vocab, ragged in self.cases:
            with self.subTest(bs=bs, num_rows=num_rows, vocab=vocab, ragged=ragged):
                step = self._random_step(bs, num_rows, vocab, ragged, generator)
                slots_p2 = triton.next_power_of_2(num_rows)
                draws = torch.empty(bs, slots_p2, device="cuda")
                _philox_draws_kernel[(bs,)](key, offset, draws, slots_p2)
                fly_draws = self._reference_draws(
                    step, draws[:, : num_rows - 1].contiguous()
                )
                drafts = step["candidates"][:, 1:, None]
                p = step["target_probs"][:, :-1].gather(-1, drafts)[..., 0]
                q = step["draft_probs"].gather(-1, drafts)[..., 0]
                expected = self._capped(
                    _chain_accepts(p, q, fly_draws), step["verify_lens"]
                )

                def accepted(fly):
                    correct_len, _, _ = accept_sampling_sharded(
                        candidates=step["candidates"],
                        target_logits=step["target_logits"],
                        draft_logits=step["draft_logits"],
                        temperatures=step["temperatures"],
                        vocab_start=0,
                        width=vocab,
                        gather_stacked=lambda stats: stats,
                        key=key,
                        offset=offset,
                        verify_lens=step["verify_lens"],
                        fly=fly,
                    )
                    return correct_len.int()

                self.assertTrue(torch.equal(accepted(self.fly), expected))
                self.assertGreater(
                    accepted(self.fly).sum().item(), accepted(None).sum().item()
                )


class TestResolveFlyConfig(CustomTestCase):
    def test_defaults_and_validation(self):
        resolve = lambda gamma, **kw: resolve_fly_config(
            gamma=gamma,
            window_size=kw.get("window_size"),
            entropy_threshold=kw.get("entropy_threshold", 0.3),
            entropy_top_k=kw.get("entropy_top_k", 3),
        )
        self.assertEqual(resolve(5).window_size, 4)
        self.assertEqual(resolve(15).window_size, 6)
        self.assertEqual(resolve(5, window_size=2).window_size, 2)
        self.assertEqual(resolve(MAX_FLY_GAMMA).window_size, 6)
        top_k = resolve(5, entropy_top_k=MAX_FLY_ENTROPY_TOP_K).entropy_top_k
        self.assertEqual(top_k, MAX_FLY_ENTROPY_TOP_K)
        self.assertEqual(resolve(5, entropy_threshold=1.5).entropy_threshold, 1.5)
        for gamma, kwargs in [
            (1, {}),
            (MAX_FLY_GAMMA + 1, {}),
            (5, {"window_size": 5}),
            (5, {"window_size": 0}),
            (5, {"entropy_top_k": 0}),
            (5, {"entropy_top_k": MAX_FLY_ENTROPY_TOP_K + 1}),
            (5, {"entropy_threshold": -0.1}),
            (5, {"entropy_threshold": float("nan")}),
        ]:
            with self.subTest(gamma=gamma, **kwargs):
                with self.assertRaises(ValueError):
                    resolve(gamma, **kwargs)


if __name__ == "__main__":
    unittest.main()
