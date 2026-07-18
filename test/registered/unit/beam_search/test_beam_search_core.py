"""Golden and differential tests for the beam search core (S2 layer).

Covers the pure selection functions (joint_select / select_final_topk), the
backpointer history DAG, and the BeamGroup lifecycle. The selection semantics
are pinned two ways:
- golden cases with hand-computed candidate walks
- randomized differential testing against a naive reference implementation of
  the walk-in-order loop (the semantics of the original beam expansion)
"""

import random
import unittest

import torch

from sglang.srt.beam_search import (
    BeamGroup,
    BeamNode,
    joint_select,
    materialize_tokens,
    select_final_topk,
    tail_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def T(data, dtype):
    return torch.tensor(data, dtype=dtype)


def run_select(cum, logprobs, tokens, stop_ids, k):
    return joint_select(
        T(cum, torch.float32),
        T(logprobs, torch.float32),
        T(tokens, torch.int64),
        T(sorted(stop_ids), torch.int64),
        k,
    )


def reference_select(cum, logprobs, tokens, stop_ids, k):
    """Naive port of the original walk-in-order expansion loop."""
    num_candidates = len(logprobs[0])
    cands = [
        (cum[r] + logprobs[r][c], r, tokens[r][c])
        for r in range(len(cum))
        for c in range(num_candidates)
    ]
    cands.sort(key=lambda x: -x[0])
    cands = cands[:num_candidates]

    survivors, finished = [], []
    for score, row, token in cands:
        if token in stop_ids:
            finished.append((score, row, token))
        else:
            survivors.append((score, row, token))
            if len(survivors) == k:
                break
    return survivors, finished


def unpack(sel):
    ns, nf = int(sel.num_survivors), int(sel.num_finished)
    survivors = list(
        zip(
            sel.new_cum_logprobs[:ns].tolist(),
            sel.parent_idx[:ns].tolist(),
            sel.next_tokens[:ns].tolist(),
        )
    )
    finished = list(
        zip(
            sel.fin_cum_logprobs[:nf].tolist(),
            sel.fin_parent_idx[:nf].tolist(),
            sel.fin_tokens[:nf].tolist(),
        )
    )
    return survivors, finished


class TestJointSelectGolden(CustomTestCase):
    CUM = [0.0, -1.0]
    LOGPROBS = [[-0.1, -0.2, -0.3, -0.4], [-0.05, -0.5, -0.6, -0.7]]
    TOKENS = [[10, 11, 12, 13], [20, 21, 22, 23]]

    def assert_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for (a_score, a_row, a_tok), (e_score, e_row, e_tok) in zip(actual, expected):
            self.assertAlmostEqual(a_score, e_score, places=5)
            self.assertEqual((a_row, a_tok), (e_row, e_tok))

    def test_no_stop_fast_path(self):
        sel = run_select(self.CUM, self.LOGPROBS, self.TOKENS, set(), 2)
        survivors, finished = unpack(sel)
        self.assert_close(survivors, [(-0.1, 0, 10), (-0.2, 0, 11)])
        self.assertEqual(finished, [])

    def test_stop_routing(self):
        # Sorted walk: 10 survives, 11 is a stop (examined), 12 survives -> stop.
        # 13 and every row-1 candidate fall outside the examined window.
        sel = run_select(self.CUM, self.LOGPROBS, self.TOKENS, {11, 20}, 2)
        survivors, finished = unpack(sel)
        self.assert_close(survivors, [(-0.1, 0, 10), (-0.3, 0, 12)])
        self.assert_close(finished, [(-0.2, 0, 11)])

    def test_insufficient_survivors(self):
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2
        )
        survivors, finished = unpack(sel)
        self.assert_close(survivors, [(-0.4, 0, 8)])
        self.assert_close(finished, [(-0.1, 0, 5), (-0.2, 0, 6), (-0.3, 0, 7)])

    def test_all_stop(self):
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7, 8}, 2
        )
        survivors, finished = unpack(sel)
        self.assertEqual(survivors, [])
        self.assertEqual(len(finished), 4)

    def test_select_final_topk(self):
        sel = select_final_topk(
            T(self.CUM, torch.float32),
            T(self.LOGPROBS, torch.float32),
            T(self.TOKENS, torch.int64),
            2,
        )
        self.assertEqual(sel.tokens.tolist(), [10, 11])
        self.assertEqual(sel.parent_idx.tolist(), [0, 0])
        for actual, expected in zip(sel.cum_logprobs.tolist(), [-0.1, -0.2]):
            self.assertAlmostEqual(actual, expected, places=5)


class TestJointSelectDifferential(CustomTestCase):
    def test_random_vs_reference(self):
        rng = random.Random(42)
        torch.manual_seed(42)
        for trial in range(200):
            k = rng.choice([1, 2, 3, 5])
            num_rows = rng.choice([1, k])
            num_candidates = 2 * k
            vocab = list(range(100))

            logprobs = torch.randn(num_rows, num_candidates)
            tokens = [rng.sample(vocab, num_candidates) for _ in range(num_rows)]
            cum = [rng.uniform(-5, 0) for _ in range(num_rows)]
            stop_density = rng.choice([0.0, 0.3, 0.9, 1.0])
            stop_ids = {t for t in vocab if rng.random() < stop_density}

            sel = joint_select(
                T(cum, torch.float32),
                logprobs,
                T(tokens, torch.int64),
                T(sorted(stop_ids), torch.int64),
                k,
            )
            survivors, finished = unpack(sel)
            ref_survivors, ref_finished = reference_select(
                cum, logprobs.tolist(), tokens, stop_ids, k
            )

            msg = f"trial={trial} k={k} rows={num_rows} density={stop_density}"
            self.assertEqual(len(survivors), len(ref_survivors), msg)
            self.assertEqual(len(finished), len(ref_finished), msg)
            for actual, expected in zip(
                survivors + finished, ref_survivors + ref_finished
            ):
                self.assertAlmostEqual(actual[0], expected[0], places=4, msg=msg)
                self.assertEqual(actual[1:], expected[1:], msg)


class TestHistory(CustomTestCase):
    def test_materialize_and_tail(self):
        a = BeamNode(1)
        b = BeamNode(2, a)
        c = BeamNode(3, a)  # reparent: sibling branch off the same prefix
        self.assertEqual(materialize_tokens(b), [1, 2])
        self.assertEqual(materialize_tokens(c), [1, 3])
        self.assertEqual(materialize_tokens(None), [])
        self.assertEqual(tail_tokens(c, 1), [3])
        self.assertEqual(tail_tokens(b, 5), [1, 2])


class TestBeamGroup(CustomTestCase):
    def _make_group(self, **kwargs):
        defaults = dict(beam_width=2, stop_token_ids=[99], max_new_tokens=3)
        defaults.update(kwargs)
        return BeamGroup(**defaults)

    def test_lifecycle_eos_and_length_finish(self):
        group = self._make_group()

        # Prefill selection: single pseudo-row frontier.
        sel = run_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], {99}, 2)
        self.assertFalse(group.advance(sel))
        self.assertEqual(
            [materialize_tokens(leaf) for leaf in group.leaves], [[1], [2]]
        )

        # Step 2: the best candidate is a stop token; two survivors remain.
        sel = run_select(
            [-0.1, -0.2],
            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[99, 5, 6, 7], [8, 9, 10, 11]],
            {99},
            2,
        )
        self.assertFalse(group.advance(sel))
        self.assertEqual(len(group.completed), 1)
        self.assertEqual(
            [materialize_tokens(leaf) for leaf in group.leaves], [[1, 5], [2, 8]]
        )

        # Step 3 hits max_new_tokens: host decides, final top-k all finish.
        self.assertTrue(group.next_step_is_final())
        fsel = select_final_topk(
            group.frontier_cum_logprobs,
            T([[-0.1, -0.9], [-0.05, -0.9]], torch.float32),
            T([[12, 13], [14, 15]], torch.int64),
            2,
        )
        self.assertTrue(group.advance_final(fsel))

        results = group.finalize()
        self.assertEqual(len(results), 2)
        # EOS beam: cum -0.15 over 2 tokens -> score -0.075, the best.
        self.assertEqual(results[0].tokens, [1, 99])
        self.assertEqual(results[0].matched_token, 99)
        self.assertAlmostEqual(results[0].beam_score, -0.075, places=5)
        # Runner-up: [2, 8, 14] with cum -0.36 over 3 tokens -> -0.12.
        self.assertEqual(results[1].tokens, [2, 8, 14])
        self.assertIsNone(results[1].matched_token)
        self.assertAlmostEqual(results[1].beam_score, -0.12, places=5)

    def test_insufficient_survivors_finishes_group(self):
        group = self._make_group(stop_token_ids=[5, 6, 7])
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2
        )
        self.assertTrue(group.advance(sel))
        results = group.finalize()
        # Pool: three stop-finished beams + the folded-in partial survivor.
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].tokens, [5])
        self.assertAlmostEqual(results[0].beam_score, -0.1, places=5)

    def test_length_penalty_ordering(self):
        group = self._make_group(length_penalty=0.0)
        # penalty 0: score == cum_logprob regardless of length.
        self.assertAlmostEqual(group.beam_score(-0.3, 2), -0.3, places=6)
        group2 = self._make_group(length_penalty=1.0)
        self.assertAlmostEqual(group2.beam_score(-0.3, 2), -0.15, places=6)


if __name__ == "__main__":
    unittest.main()
