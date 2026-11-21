import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=10, suite="extra-test-cpu-intel")

VOCAB_SIZE = 32
DEVICE = "cpu"


def _req(num, output_ids=(), **sampling_params):
    return SimpleNamespace(
        rid=f"r{num}",
        penalty_index=None,
        output_ids=list(output_ids),
        sampling_params=SamplingParams(**sampling_params),
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
    )


def _pool():
    return PenaltyStatusPool(VOCAB_SIZE, DEVICE)


def _admit(pool, req):
    pool.alloc([req])
    pool.reset_history([req])


def _expected_penalties(reqs, histories):
    additive = torch.zeros(len(reqs), VOCAB_SIZE)
    scaling = torch.ones(len(reqs), VOCAB_SIZE)
    for i, (req, history) in enumerate(zip(reqs, histories)):
        params = req.sampling_params
        counts = torch.bincount(
            torch.tensor(history, dtype=torch.long), minlength=VOCAB_SIZE
        ).float()
        additive[i] -= params.frequency_penalty * counts
        additive[i] -= params.presence_penalty * (counts > 0)
        if len(history) < params.min_new_tokens:
            additive[i, sorted(params.stop_token_ids or ())] = float("-inf")
        scaling[i] = torch.where(counts > 0, params.repetition_penalty, 1.0)
    return additive, scaling


def _stacked(additive, scaling, batch_size):
    if additive is None:
        additive = torch.zeros(batch_size, VOCAB_SIZE)
    if scaling is None:
        scaling = torch.ones(batch_size, VOCAB_SIZE)
    return torch.stack([additive, scaling])


class TestMinNewTokens(CustomTestCase):
    def test_ignores_stop_ids_outside_vocab(self):
        pool = _pool()
        reqs = [
            _req(
                1,
                min_new_tokens=3,
                stop_token_ids=[-1, 0, 5, VOCAB_SIZE - 1, VOCAB_SIZE, VOCAB_SIZE + 7]
                + [2**63],
            ),
            _req(2, stop_token_ids=[VOCAB_SIZE + 1]),
        ]
        for req in reqs:
            _admit(pool, req)

        additive, _ = StepPenalties.build(reqs, pool).logit_penalties()

        blocked = torch.isneginf(additive)
        self.assertEqual(
            blocked[0].nonzero().flatten().tolist(), [0, 5, VOCAB_SIZE - 1]
        )
        self.assertFalse(blocked[1].any())


class TestHistory(CustomTestCase):
    def _step(self, pool, reqs, histories, rng):
        step = StepPenalties.build(reqs, pool)
        torch.testing.assert_close(
            _stacked(*step.logit_penalties(), len(reqs)),
            torch.stack(_expected_penalties(reqs, histories)),
        )
        tokens = [rng.randrange(VOCAB_SIZE) for _ in reqs]
        step.commit(torch.tensor(tokens))
        for req, token in zip(reqs, tokens):
            req.output_ids.append(token)

    def test_history_follows_join_leave_and_retract(self):
        rng = random.Random(0)
        pool = _pool()
        a = _req(1, frequency_penalty=0.5, presence_penalty=0.25)
        b = _req(2)
        c = _req(3, repetition_penalty=1.5, min_new_tokens=3, stop_token_ids=[7, 9])
        d = _req(4, frequency_penalty=-0.3, repetition_penalty=0.8)

        running = [a, b]
        for req in running:
            _admit(pool, req)
        for _ in range(3):
            self._step(pool, running, [r.output_ids for r in running], rng)

        _admit(pool, c)
        running.append(c)
        for _ in range(4):
            self._step(pool, running, [r.output_ids for r in running], rng)

        # a is retracted (keeps its index).
        running.remove(a)
        _admit(pool, d)
        running.append(d)
        for _ in range(2):
            self._step(pool, running, [r.output_ids for r in running], rng)

        _admit(pool, a)
        running.append(a)
        for _ in range(3):
            self._step(pool, running, [r.output_ids for r in running], rng)

        pool.free(c)
        running.remove(c)
        self._step(pool, running, [r.output_ids for r in running], rng)
        self.assertEqual(pool.num_indices - len(pool.free_indices), 2)

    def test_commit_skips_ids_outside_the_vocab(self):
        pool = _pool()
        reqs = [_req(1, frequency_penalty=1.0), _req(2)]
        for req in reqs:
            _admit(pool, req)

        StepPenalties.build(reqs, pool).commit(torch.tensor([-1, VOCAB_SIZE]))
        StepPenalties.build(reqs, pool).commit(
            torch.tensor([[3, VOCAB_SIZE, -1], [VOCAB_SIZE + 5, 2, 2]]),
            accept_mask=torch.tensor([[True, True, False], [False, True, True]]),
        )

        additive, _ = StepPenalties.build(reqs[:1], pool).logit_penalties()
        expected, _ = _expected_penalties(reqs[:1], [[3]])
        torch.testing.assert_close(additive, expected)
        indices = torch.tensor([reqs[0].penalty_index])
        self.assertEqual(pool.gather_lens(indices).tolist(), [1])

    def test_commit_into_freed_index_is_reset_by_next_admission(self):
        pool = _pool()
        live = _req(3, frequency_penalty=1.0)
        a = _req(1, frequency_penalty=1.0)
        _admit(pool, live)
        _admit(pool, a)
        index = a.penalty_index
        step = StepPenalties.build([a], pool)

        pool.free(a)
        step.commit(torch.tensor([7]))
        b = _req(2, frequency_penalty=1.0, output_ids=[3])
        _admit(pool, b)
        self.assertEqual(b.penalty_index, index)

        additive, _ = StepPenalties.build([b], pool).logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([b], [[3]])[0])

    def test_step_after_the_last_free_reads_and_counts_nothing(self):
        pool = _pool()
        a = _req(
            1,
            output_ids=[4, 4],
            frequency_penalty=1.0,
            presence_penalty=0.5,
            repetition_penalty=1.5,
            min_new_tokens=3,
            stop_token_ids=[7],
        )
        b = _req(2, output_ids=[5], presence_penalty=1.0)
        _admit(pool, a)
        _admit(pool, b)
        read_at_launch = StepPenalties.build([a, b], pool)
        tables = read_at_launch.logit_penalties()
        step = StepPenalties.build([a, b], pool)

        pool.free(a)
        pool.free(b)
        self.assertEqual(pool.counts.shape[0], 0)
        # Tables built at launch stay; a first read after the drop sees no history.
        self.assertIs(read_at_launch.logit_penalties(), tables)
        additive, scaling = step.logit_penalties()
        expected = _expected_penalties([a, b], [[], []])
        torch.testing.assert_close(
            _stacked(additive, scaling, 2), torch.stack(expected)
        )
        step.commit(torch.tensor([1, 2]))
        self.assertEqual(pool.counts.shape[0], 0)

        c = _req(3, frequency_penalty=1.0, output_ids=[3])
        _admit(pool, c)
        self.assertEqual(c.penalty_index, 0)
        additive, _ = StepPenalties.build([c], pool).logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([c], [[3]])[0])

    def test_readmission_resets_a_discarded_chunk_sample(self):
        pool = _pool()
        a = _req(1, presence_penalty=1.0)
        _admit(pool, a)
        middle_chunk = StepPenalties.build([a], pool)
        middle_chunk.commit(torch.tensor([9]))

        _admit(pool, a)
        additive, _ = StepPenalties.build([a], pool).logit_penalties()
        torch.testing.assert_close(additive, torch.zeros(1, VOCAB_SIZE))


class TestBuild(CustomTestCase):
    def test_step_built_before_the_previous_sample_reads_it(self):
        """Under overlap a step is built while the previous one is in flight."""
        pool = _pool()
        req = _req(1, frequency_penalty=1.0, output_ids=[2])
        _admit(pool, req)
        previous = StepPenalties.build([req], pool)
        current = StepPenalties.build([req], pool)
        previous.commit(torch.tensor([7]))

        additive, _ = current.logit_penalties()
        expected, _ = _expected_penalties([req], [[2, 7]])
        torch.testing.assert_close(additive, expected)


if __name__ == "__main__":
    unittest.main()
