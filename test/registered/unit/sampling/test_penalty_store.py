"""Unit tests for penalty history rows and per-step penalty builds."""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.penalties import PenaltyStore, StepPenalties
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=10, suite="extra-test-cpu-intel")

VOCAB_SIZE = 32
NUM_SLOTS = 16
DEVICE = "cpu"


def _req(slot, rid=None, output_ids=(), **sampling_params):
    return SimpleNamespace(
        rid=rid if rid is not None else f"r{slot}",
        kv=SimpleNamespace(req_pool_idx=slot),
        penalty_row=None,
        output_ids=list(output_ids),
        sampling_params=SamplingParams(**sampling_params),
        uses_penalty=False,
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
    )


def _store():
    return PenaltyStore(VOCAB_SIZE, NUM_SLOTS, DEVICE)


def _slots(reqs):
    return torch.tensor([r.kv.req_pool_idx for r in reqs])


def _build(reqs, store):
    return StepPenalties.build(reqs, _slots(reqs), store)


def _held_rows(store):
    return store.num_rows - len(store.free_rows)


def _expected_penalties(reqs, histories):
    """Penalties of a token history, as the previous per-batch penalizers computed them."""
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


def _apply(additive, scaling, batch_size):
    """Combine both penalty parts into one comparable tensor."""
    if additive is None:
        additive = torch.zeros(batch_size, VOCAB_SIZE)
    if scaling is None:
        scaling = torch.ones(batch_size, VOCAB_SIZE)
    return torch.stack([additive, scaling])


class TestNoPenaltyPath(CustomTestCase):
    def test_no_penalty_request_allocates_nothing(self):
        store = _store()
        reqs = [_req(1), _req(2, temperature=0.5, top_k=4)]
        for req in reqs:
            store.admit(req)

        self.assertIsNone(_build(reqs, store))
        self.assertEqual(store.num_rows, 0)
        self.assertEqual(store.counts.numel(), 0)


class TestMinNewTokens(CustomTestCase):
    def test_ignores_stop_ids_outside_vocab(self):
        """Stop ids outside the vocab never match and must not break the batch."""
        store = _store()
        reqs = [
            _req(
                1,
                min_new_tokens=3,
                stop_token_ids=[-1, 0, 5, VOCAB_SIZE - 1, VOCAB_SIZE, VOCAB_SIZE + 7]
                + [2**63],
            ),
            # No min_new_tokens of its own; shares the batch with one that has it.
            _req(2, stop_token_ids=[VOCAB_SIZE + 1]),
        ]
        for req in reqs:
            store.admit(req)

        additive, _ = _build(reqs, store).logit_penalties()

        blocked = torch.isneginf(additive)
        self.assertEqual(
            blocked[0].nonzero().flatten().tolist(), [0, 5, VOCAB_SIZE - 1]
        )
        self.assertFalse(blocked[1].any())


class TestHistory(CustomTestCase):
    def _step(self, store, reqs, histories, rng):
        """Build, check penalties against the history, sample, commit."""
        step = _build(reqs, store)
        torch.testing.assert_close(
            _apply(*step.logit_penalties(), len(reqs)),
            torch.stack(_expected_penalties(reqs, histories)),
        )
        tokens = [rng.randrange(VOCAB_SIZE) for _ in reqs]
        step.commit(torch.tensor(tokens))
        for req, token in zip(reqs, tokens):
            req.output_ids.append(token)

    def test_history_follows_join_leave_and_retract(self):
        rng = random.Random(0)
        store = _store()
        a = _req(1, frequency_penalty=0.5, presence_penalty=0.25)
        b = _req(2)
        c = _req(3, repetition_penalty=1.5, min_new_tokens=3, stop_token_ids=[7, 9])
        d = _req(4, frequency_penalty=-0.3, repetition_penalty=0.8)

        running = [a, b]
        for req in running:
            store.admit(req)
        for _ in range(3):
            self._step(store, running, [r.output_ids for r in running], rng)

        # c joins.
        store.admit(c)
        running.append(c)
        for _ in range(4):
            self._step(store, running, [r.output_ids for r in running], rng)

        # a is retracted and keeps its row; d takes a's slot meanwhile.
        running.remove(a)
        d.kv.req_pool_idx = a.kv.req_pool_idx
        store.admit(d)
        running.append(d)
        for _ in range(2):
            self._step(store, running, [r.output_ids for r in running], rng)

        # a comes back on another slot; its history is replayed, not restarted.
        a.kv.req_pool_idx = 5
        store.admit(a)
        running.append(a)
        self.assertEqual(len(a.output_ids), 7)
        for _ in range(3):
            self._step(store, running, [r.output_ids for r in running], rng)

        # c finishes and leaves.
        store.free(c)
        running.remove(c)
        self._step(store, running, [r.output_ids for r in running], rng)
        self.assertEqual(_held_rows(store), 2)

    def test_growth_keeps_rows(self):
        rng = random.Random(1)
        store = _store()
        running = []
        for slot in range(1, 12):
            req = _req(slot, frequency_penalty=0.1 * slot)
            store.admit(req)
            running.append(req)
            self._step(store, running, [r.output_ids for r in running], rng)

    def test_spec_commit_counts_accepted_tokens_only(self):
        store = _store()
        reqs = [_req(1, frequency_penalty=1.0), _req(2), _req(3, presence_penalty=1.0)]
        for req in reqs:
            store.admit(req)

        _build(reqs, store).commit(
            torch.tensor([[4, 4, 6], [1, 2, 3], [5, 0, 0]]),
            accept_mask=torch.tensor(
                [[True, True, False], [True, True, True], [True, False, False]]
            ),
        )

        additive, _ = _build(reqs, store).logit_penalties()
        expected, _ = _expected_penalties(reqs, [[4, 4], [], [5]])
        torch.testing.assert_close(additive, expected)

    def test_commit_skips_ids_outside_the_vocab(self):
        store = _store()
        reqs = [_req(1, frequency_penalty=1.0), _req(2)]
        for req in reqs:
            store.admit(req)

        # One token per row, then a verify block whose masked slots hold
        # ids no sampler can emit.
        _build(reqs, store).commit(torch.tensor([-1, VOCAB_SIZE]))
        _build(reqs, store).commit(
            torch.tensor([[3, VOCAB_SIZE, -1], [VOCAB_SIZE + 5, 2, 2]]),
            accept_mask=torch.tensor([[True, True, False], [False, True, True]]),
        )

        additive, _ = _build(reqs[:1], store).logit_penalties()
        expected, _ = _expected_penalties(reqs[:1], [[3]])
        torch.testing.assert_close(additive, expected)
        rows = store.rows(_slots(reqs[:1]))
        self.assertEqual(store.gather_lens(rows).tolist(), [1])

    def test_commit_into_freed_row_is_reset_by_next_admit(self):
        store = _store()
        live = _req(3, frequency_penalty=1.0)
        a = _req(1, frequency_penalty=1.0)
        store.admit(live)
        store.admit(a)
        row = a.penalty_row
        step = _build([a], store)

        # a finishes before its last step samples; b then takes the freed row.
        store.free(a)
        step.commit(torch.tensor([7]))
        b = _req(2, frequency_penalty=1.0, output_ids=[3])
        store.admit(b)
        self.assertEqual(b.penalty_row, row)

        additive, _ = _build([b], store).logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([b], [[3]])[0])

    def test_tables_go_when_no_row_is_live(self):
        store = _store()
        a = _req(1, frequency_penalty=1.0, output_ids=[4])
        store.admit(a)
        step = _build([a], store)
        step.logit_penalties()
        self.assertEqual(store.counts.shape[0], 1)

        # a finishes before its last step commits: the free drops the tables,
        # and that commit counts nothing.
        store.free(a)
        self.assertEqual(store.counts.shape[0], 0)
        self.assertEqual(store.lens.shape[0], 0)
        step.commit(torch.tensor([7]))
        self.assertEqual(store.counts.shape[0], 0)
        self.assertEqual(store.lens.shape[0], 0)

        # The next penalized request starts the rows over.
        b = _req(2, frequency_penalty=1.0, output_ids=[3])
        store.admit(b)
        self.assertEqual(b.penalty_row, 0)
        additive, _ = _build([b], store).logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([b], [[3]])[0])

    def test_step_whose_rows_were_released_reads_no_history(self):
        """A step built before its rows' release reads zero counts and lengths."""
        store = _store()
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
        store.admit(a)
        store.admit(b)
        step = _build([a, b], store)

        store.free(a)
        store.free(b)
        additive, scaling = step.logit_penalties()
        expected = _expected_penalties([a, b], [[], []])
        torch.testing.assert_close(_apply(additive, scaling, 2), torch.stack(expected))
        step.commit(torch.tensor([1, 2]))
        self.assertEqual(store.num_rows, 0)

    def test_step_reads_the_row_its_slot_holds_at_sampling(self):
        """A request re-admitted into the same slot with another row is read
        from that row by a step built before the re-admission."""
        store = _store()
        live = _req(3, frequency_penalty=1.0)
        a = _req(1, frequency_penalty=1.0, output_ids=[4])
        b = _req(2, frequency_penalty=1.0, output_ids=[5])
        store.admit(live)
        store.admit(a)
        store.admit(b)
        shared = _build([a], store)

        # Both leave; a comes back to slot 1 with the row b freed last.
        store.free(a)
        store.free(b)
        store.admit(a)
        self.assertEqual(a.penalty_row, 2)

        additive, _ = shared.logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([a], [[4]])[0])

    def test_readmission_resets_a_discarded_chunk_sample(self):
        store = _store()
        a = _req(1, presence_penalty=1.0)
        store.admit(a)
        middle_chunk = _build([a], store)
        middle_chunk.commit(torch.tensor([9]))

        # The next chunk re-admits the request; the discarded sample is gone.
        store.admit(a)
        additive, _ = _build([a], store).logit_penalties()
        torch.testing.assert_close(additive, torch.zeros(1, VOCAB_SIZE))


class TestBuild(CustomTestCase):
    def test_build_twice_gives_same_penalties_and_no_side_effects(self):
        store = _store()
        reqs = [_req(1, frequency_penalty=0.5, output_ids=[2, 2, 5]), _req(2)]
        store.admit(reqs[0])
        first = _build(reqs, store)
        second = _build(reqs, store)
        torch.testing.assert_close(
            first.logit_penalties()[0], second.logit_penalties()[0]
        )
        self.assertEqual(reqs[0].output_ids, [2, 2, 5])
        self.assertEqual(_held_rows(store), 1)

    def test_step_built_before_the_previous_sample_reads_it(self):
        """Under overlap a step is built while the previous one is in flight;
        its penalties read the rows when it samples, so they include the
        previous step's token (stream order, not build order)."""
        store = _store()
        req = _req(1, frequency_penalty=1.0, output_ids=[2])
        store.admit(req)
        previous = _build([req], store)
        current = _build([req], store)
        previous.commit(torch.tensor([7]))

        additive, _ = current.logit_penalties()
        expected, _ = _expected_penalties([req], [[2, 7]])
        torch.testing.assert_close(additive, expected)

    def test_slots_past_the_requests_are_ignored(self):
        """A decode batch's slots can run past its requests (beam member rows)."""
        store = _store()
        req = _req(1, frequency_penalty=1.0, output_ids=[3])
        store.admit(req)
        step = StepPenalties.build([req], torch.tensor([1, 9, 10]), store)
        additive, _ = step.logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([req], [[3]])[0])


if __name__ == "__main__":
    unittest.main()
