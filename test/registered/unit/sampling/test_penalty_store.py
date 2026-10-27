"""Unit tests for penalty history rows and per-step sampling info builds."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.custom_logit_processor import (
    DisallowedTokensLogitsProcessor,
    Qwen3ThinkingBudgetLogitProcessor,
)
from sglang.srt.sampling.penalties import PenaltyStore, StepPenalties
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB_SIZE = 32
NUM_SLOTS = 16
DEVICE = "cpu"


def _req(slot, rid=None, output_ids=(), **sampling_params):
    return SimpleNamespace(
        rid=rid if rid is not None else f"r{slot}",
        kv=SimpleNamespace(req_pool_idx=slot, penalty_row=None),
        output_ids=list(output_ids),
        sampling_params=SamplingParams(**sampling_params),
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
        grammar=None,
        custom_logit_processor=None,
        return_sampling_mask=False,
        sampling_logprobs_mode=None,
    )


def _store():
    return PenaltyStore(VOCAB_SIZE, NUM_SLOTS, DEVICE)


def _slots(reqs):
    return torch.tensor([r.kv.req_pool_idx for r in reqs])


def _build(reqs, store):
    return SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE, store, _slots(reqs))


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


class _BuildTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.exec_ns = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
            features=SimpleNamespace(
                enable_custom_logit_processor=True, enable_watermark=False
            ),
        )
        exec_patch = patch(
            "sglang.srt.sampling.sampling_batch_info.get_exec",
            return_value=self.exec_ns,
        )
        exec_patch.start()
        self.addCleanup(exec_patch.stop)


class TestNoPenaltyPath(_BuildTestCase):
    def test_no_penalty_request_allocates_nothing(self):
        store = _store()
        reqs = [_req(1), _req(2, temperature=0.5, top_k=4)]
        for req in reqs:
            store.alloc(req)

        info = _build(reqs, store)
        info.commit_sampled_tokens(torch.tensor([3, 4]))

        self.assertIsNone(info.penalties)
        self.assertEqual(info.logit_penalties(), (None, None))
        self.assertEqual(store.num_rows, 0)
        self.assertEqual(store.counts.numel(), 0)

    def test_penalized_request_without_row_fails(self):
        req = _req(1, frequency_penalty=0.5)
        with self.assertRaises(RuntimeError):
            SamplingBatchInfo.build([req], VOCAB_SIZE, DEVICE, None)
        with self.assertRaises(RuntimeError):
            _build([req], _store())


class TestHistory(_BuildTestCase):
    def _step(self, store, reqs, histories, rng):
        """Build, check penalties against the history, sample, commit."""
        info = _build(reqs, store)
        torch.testing.assert_close(
            _apply(*info.logit_penalties(), len(reqs)),
            torch.stack(_expected_penalties(reqs, histories)),
        )
        tokens = [rng.randrange(VOCAB_SIZE) for _ in reqs]
        info.commit_sampled_tokens(torch.tensor(tokens))
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
            store.alloc(req)
        for _ in range(3):
            self._step(store, running, [r.output_ids for r in running], rng)

        # c joins.
        store.alloc(c)
        running.append(c)
        for _ in range(4):
            self._step(store, running, [r.output_ids for r in running], rng)

        # a is retracted: its slot is freed, d takes a new slot meanwhile.
        store.free(a.kv)
        running.remove(a)
        store.alloc(d)
        running.append(d)
        for _ in range(2):
            self._step(store, running, [r.output_ids for r in running], rng)

        # a comes back on another slot; its history is replayed, not restarted.
        a.kv.req_pool_idx = 5
        store.alloc(a)
        running.append(a)
        self.assertEqual(len(a.output_ids), 7)
        for _ in range(3):
            self._step(store, running, [r.output_ids for r in running], rng)

        # c finishes and leaves.
        store.free(c.kv)
        running.remove(c)
        self._step(store, running, [r.output_ids for r in running], rng)
        self.assertEqual(_held_rows(store), 2)

    def test_growth_keeps_rows(self):
        rng = random.Random(1)
        store = _store()
        running = []
        for slot in range(1, 12):
            req = _req(slot, frequency_penalty=0.1 * slot)
            store.alloc(req)
            running.append(req)
            self._step(store, running, [r.output_ids for r in running], rng)

    def test_spec_commit_counts_accepted_tokens_only(self):
        store = _store()
        reqs = [_req(1, frequency_penalty=1.0), _req(2), _req(3, presence_penalty=1.0)]
        for req in reqs:
            store.alloc(req)
        info = _build(reqs, store)

        info.commit_sampled_tokens(
            torch.tensor([[4, 4, 6], [1, 2, 3], [5, 0, 0]]),
            accept_mask=torch.tensor(
                [[True, True, False], [True, True, True], [True, False, False]]
            ),
        )

        additive, _ = _build(reqs, store).logit_penalties()
        expected, _ = _expected_penalties(reqs, [[4, 4], [], [5]])
        torch.testing.assert_close(additive, expected)

    def test_commit_into_freed_row_is_reset_by_next_alloc(self):
        store = _store()
        a = _req(1, frequency_penalty=1.0)
        store.alloc(a)
        step = _build([a], store)

        # a finishes before its last step samples; b then takes the freed row.
        store.free(a.kv)
        step.commit_sampled_tokens(torch.tensor([7]))
        b = _req(2, frequency_penalty=1.0, output_ids=[3])
        store.alloc(b)

        additive, _ = _build([b], store).logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([b], [[3]])[0])

    def test_step_reads_the_row_its_slot_holds_at_sampling(self):
        """A decode step's info is shared while its requests keep their slots;
        a request re-admitted into the same slot with another row is read from
        that row."""
        store = _store()
        a = _req(1, frequency_penalty=1.0, output_ids=[4])
        b = _req(2, frequency_penalty=1.0, output_ids=[5])
        store.alloc(a)
        store.alloc(b)
        shared = _build([a], store)

        # Both are retracted; a comes back to slot 1 with the row b freed last.
        store.free(a.kv)
        store.free(b.kv)
        store.alloc(a)
        self.assertEqual(a.kv.penalty_row.index, 1)

        additive, _ = shared.logit_penalties()
        torch.testing.assert_close(additive, _expected_penalties([a], [[4]])[0])

    def test_readmission_resets_a_discarded_chunk_sample(self):
        store = _store()
        a = _req(1, presence_penalty=1.0)
        store.alloc(a)
        middle_chunk = _build([a], store)
        middle_chunk.commit_sampled_tokens(torch.tensor([9]))

        # The next chunk re-admits the request; the discarded sample is gone.
        store.alloc(a)
        additive, _ = _build([a], store).logit_penalties()
        torch.testing.assert_close(additive, torch.zeros(1, VOCAB_SIZE))


class TestBuildFromRequests(_BuildTestCase):
    def test_random_requests_match_their_params(self):
        """Every field of a build is what the requests' params give, row by row."""
        rng = random.Random(2)
        self.exec_ns.deterministic.enable_deterministic_inference = True
        processors = [
            DisallowedTokensLogitsProcessor.to_str(),
            Qwen3ThinkingBudgetLogitProcessor.to_str(),
        ]
        grammar = object()
        for _ in range(30):
            reqs = []
            for slot in range(1, rng.randrange(2, 8)):
                req = _req(
                    slot,
                    temperature=rng.choice([0.0, 0.7, 1.0]),
                    top_p=rng.choice([1.0, 0.9]),
                    top_k=rng.choice([TOP_K_ALL, 1, 5, 2000]),
                    min_p=rng.choice([0.0, 0.1]),
                    sampling_seed=rng.choice([None, 7]),
                    logit_bias=rng.choice([None, {"3": 1.5, "03": -2.0}]),
                    custom_params=rng.choice([None, {"token_ids": [1]}]),
                )
                req.return_sampling_mask = rng.random() < 0.5
                req.sampling_logprobs_mode = rng.choice([None, "support"])
                req.custom_logit_processor = rng.choice([None, *processors])
                req.grammar = rng.choice([None, grammar])
                reqs.append(req)

            info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE, None)

            params = [r.sampling_params for r in reqs]
            top_ks = [p.top_k for p in params]
            self.assertEqual(
                info.temperatures.view(-1).tolist(),
                torch.tensor([p.temperature for p in params]).tolist(),
            )
            self.assertEqual(
                info.top_ps.tolist(), torch.tensor([p.top_p for p in params]).tolist()
            )
            self.assertEqual(info.top_ks.tolist(), top_ks)
            self.assertEqual(
                info.min_ps.tolist(), torch.tensor([p.min_p for p in params]).tolist()
            )
            self.assertEqual(
                info.sampling_seed.tolist(),
                [42 if p.sampling_seed is None else p.sampling_seed for p in params],
            )
            self.assertEqual(
                (
                    info.is_all_greedy,
                    info.is_any_greedy,
                    info.need_top_p_sampling,
                    info.need_top_k_sampling,
                    info.need_min_p_sampling,
                    info.npu_top_k_top_p_eligible,
                ),
                (
                    all(k <= 1 for k in top_ks),
                    any(k <= 1 for k in top_ks),
                    any(p.top_p != 1.0 for p in params),
                    any(k != TOP_K_ALL for k in top_ks),
                    any(p.min_p > 0 for p in params),
                    all(1 <= k <= 1024 for k in top_ks),
                ),
            )
            self.assertEqual(
                info.max_top_k,
                max(top_ks) if all(1 <= k < TOP_K_ALL for k in top_ks) else None,
            )

            biased = [i for i, p in enumerate(params) if p.logit_bias is not None]
            if biased:
                expected_bias = torch.zeros(len(reqs), VOCAB_SIZE)
                expected_bias[biased, 3] = -2.0
                torch.testing.assert_close(info.logit_bias, expected_bias)
            else:
                self.assertIsNone(info.logit_bias)

            masks = [r.return_sampling_mask for r in reqs]
            mask_rows = [i for i, m in enumerate(masks) if m]
            support_rows = [
                i for i in mask_rows if reqs[i].sampling_logprobs_mode == "support"
            ]
            self.assertEqual(info.return_sampling_masks, masks)
            self.assertEqual(
                info.return_sampling_support_logprobs,
                [i in support_rows for i in range(len(reqs))],
            )
            self.assertEqual(
                info.sampling_mask_top_ks,
                [k if m else 1 for k, m in zip(top_ks, masks)],
            )
            for indices, expected in (
                (info.sampling_mask_batch_indices, mask_rows),
                (
                    info.sampling_support_logprobs_capture_indices,
                    [mask_rows.index(i) for i in support_rows],
                ),
            ):
                if expected:
                    self.assertEqual(indices.tolist(), expected)
                else:
                    self.assertIsNone(indices)

            rows_of = {
                hash(p): [
                    i for i, r in enumerate(reqs) if r.custom_logit_processor == p
                ]
                for p in processors
            }
            rows_of = {key: rows for key, rows in rows_of.items() if rows}
            if rows_of:
                self.assertTrue(info.has_custom_logit_processor)
                self.assertEqual(
                    {k: e.rows for k, e in info.custom_logit_processor.items()},
                    rows_of,
                )
                for entry in info.custom_logit_processor.values():
                    self.assertEqual(entry.indices.tolist(), entry.rows)
                self.assertEqual(info.custom_params, [p.custom_params for p in params])
            else:
                self.assertFalse(info.has_custom_logit_processor)
                self.assertIsNone(info.custom_logit_processor)
                self.assertIsNone(info.custom_params)

            grammars = [r.grammar for r in reqs]
            self.assertEqual(info.grammars, grammars if any(grammars) else None)

    def test_build_twice_gives_same_penalties_and_no_side_effects(self):
        store = _store()
        reqs = [_req(1, frequency_penalty=0.5, output_ids=[2, 2, 5]), _req(2)]
        store.alloc(reqs[0])
        first = StepPenalties.build(reqs, _slots(reqs), store, DEVICE)
        second = StepPenalties.build(reqs, _slots(reqs), store, DEVICE)
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
        store.alloc(req)
        previous = StepPenalties.build([req], _slots([req]), store, DEVICE)
        current = StepPenalties.build([req], _slots([req]), store, DEVICE)
        previous.commit(torch.tensor([7]))

        additive, _ = current.logit_penalties()
        expected, _ = _expected_penalties([req], [[2, 7]])
        torch.testing.assert_close(additive, expected)


if __name__ == "__main__":
    unittest.main()
