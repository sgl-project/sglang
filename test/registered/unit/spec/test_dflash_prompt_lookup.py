"""DFLASH prompt lookup (speculative/dflash_prompt_lookup.py): the lookup against
its definition, the token table through a decode loop on the scheduler's own
request, batch and request-pool classes, and looked-up blocks through the
worker's stock accept."""

import random
import types
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import dflash_prompt_lookup as lookup_mod
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.srt.speculative.dflash_prompt_lookup import (
    DFlashPromptLookup,
    HitRing,
    LookupSchedule,
    PromptLookupConfig,
    PromptLookupTable,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CPU = torch.device("cpu")
BLOCK = 8
VOCAB = 32


def _reference_lookup(history, last, max_match, out_len):
    """(match, cont) straight from the definition; -1 is an unknown token."""

    def match_at(end):
        k = 0
        while k < max_match and end - k >= 0 and history[last - k] >= 0:
            if history[end - k] != history[last - k]:
                break
            k += 1
        return k

    if last == 0:
        return 0, None
    matches = [match_at(end) for end in range(last)]
    longest = max(matches)
    end = max(e for e in range(last) if matches[e] == longest)
    return longest, [history[end + 1 + j % (last - end)] for j in range(out_len)]


def _copy_heavy(rng, vocab, n):
    """Tokens that often repeat earlier spans, runs of one token included."""
    tokens = []
    while len(tokens) < n:
        r = rng.random()
        if r < 0.4 and tokens:
            start = rng.randrange(len(tokens))
            tokens += tokens[start : start + rng.randint(1, 30)]
        elif r < 0.5:
            tokens += [rng.randrange(vocab)] * rng.randint(1, 12)
        else:
            tokens.append(rng.randrange(vocab))
    return tokens[:n]


def _pool():
    return ReqToTokenPool(
        size=64, max_context_len=1, device="cpu", enable_memory_saver=False
    )


def _req(rid, prompt, pool):
    req = Req(rid, "", array("q", prompt), SamplingParams())
    pool.alloc([req])
    return req


def _sampling_info(top_ks, **fields):
    bs = len(top_ks)
    return SamplingBatchInfo(
        temperatures=torch.ones(bs, 1),
        top_ps=torch.ones(bs),
        top_ks=torch.tensor(top_ks, dtype=torch.int32),
        min_ps=torch.zeros(bs),
        is_all_greedy=all(k <= 1 for k in top_ks),
        is_any_greedy=any(k <= 1 for k in top_ks),
        need_top_p_sampling=False,
        need_top_k_sampling=any(k > 1 for k in top_ks),
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        device="cpu",
        **fields,
    )


def _decode_step(reqs, histories, top_ks=None, info_fields=None, **batch_fields):
    """The batch and draft input of a decode step in which request i has committed
    histories[i][:-1] and carries the bonus token histories[i][-1]."""
    prefix_lens = torch.tensor([len(h) - 1 for h in histories])
    batch = ScheduleBatch(
        reqs=reqs,
        seq_lens=prefix_lens,
        seq_lens_cpu=prefix_lens.clone(),
        sampling_info=_sampling_info(top_ks or [1] * len(reqs), **(info_fields or {})),
        **batch_fields,
    )
    draft_input = make_draft_input_v2(
        bonus_tokens=torch.tensor([h[-1] for h in histories]),
        new_seq_lens=prefix_lens,
    )
    draft_input.nxt_kv_lens_cpu = (prefix_lens + 2 * BLOCK).to(torch.int32)
    return batch, draft_input


def _lookup(max_bs=4, positions=512, **config):
    return DFlashPromptLookup(
        config=PromptLookupConfig(max_bs=max_bs, **config),
        vocab_size=VOCAB,
        max_rows=64,
        positions=positions,
        device=CPU,
        log_rank=False,
    )


def _propose(lookup, pool, batch, draft_input, drafts, selector_sample=None):
    return lookup.propose(
        batch=batch,
        draft_input=draft_input,
        draft_tokens=drafts,
        selector_sample=selector_sample,
        req_generation=pool.req_generation,
    )


def _fixed_drafts(bs, token=0):
    return torch.full((bs, BLOCK), token, dtype=torch.int64)


class TestPromptLookupTable(CustomTestCase):
    def test_lookup_matches_its_definition(self):
        """The longest suffix ending at last that also ends earlier, the tokens that
        followed its latest occurrence (repeated where they run into last), with
        unknown tokens never matching and tokens past last never read."""
        rng = random.Random(0)
        for _ in range(200):
            max_match = rng.choice([1, 4, 16])
            vocab = rng.choice([2, 5, VOCAB])
            positions = rng.randint(1, 100)
            table = PromptLookupTable(
                rows=4, positions=positions, max_match=max_match, device=CPU
            )
            rows = rng.sample(range(4), rng.randint(1, 4))
            lasts, histories = [], []
            for row in rows:
                last = rng.randrange(positions)
                history = _copy_heavy(rng, vocab, last + 1)
                if rng.random() < 0.3:
                    history[rng.randrange(last + 1)] = -1
                junk = [rng.randrange(vocab) for _ in range(positions - last - 1)]
                table.fill(row=row, tokens=np.array(history + junk, dtype=np.int32))
                lasts.append(last)
                histories.append(history)
            out_len = rng.randint(1, BLOCK - 1)
            span = min(positions, max(lasts) + 1 + rng.randint(0, 8))
            match, cont = table.lookup(
                rows=torch.tensor(rows),
                last=torch.tensor(lasts),
                span=span,
                out_len=out_len,
            )
            for i, (history, last) in enumerate(zip(histories, lasts)):
                want_match, want_cont = _reference_lookup(
                    history, last, max_match, out_len
                )
                self.assertEqual(int(match[i]), want_match)
                if want_match:
                    self.assertEqual(cont[i].tolist(), want_cont)


class TestPromptLookupRows(CustomTestCase):
    def test_rows_hold_each_requests_tokens_through_a_decode_loop(self):
        """Through finishes, slot reuse, retraction, refills from a host that lags a
        step behind, eviction and batches larger than the table, every looked-up row
        holds its request's token (or -1) at each position up to the bonus, and the
        lookup proposes exactly its reference continuation.

        It runs on the scheduler's own Req, ScheduleBatch, SamplingBatchInfo and
        ReqToTokenPool, so a field these classes lack fails here, not on a GPU."""
        rng = random.Random(1)
        generator = torch.Generator().manual_seed(1)
        pool = _pool()
        lookup = _lookup(max_bs=3, min_match=3, min_match_sampled=3)
        # Every step looks up, so every step is checked.
        lookup._schedule = LookupSchedule(quiet=1 << 30)
        reqs, truth, running = {}, {}, []
        num_proposed = num_over_capacity = 0
        for step in range(600):
            if not running or rng.random() < 0.1:
                rid = f"r{step}"
                prompt = _copy_heavy(rng, 6, rng.randint(1, 40))
                reqs[rid] = _req(rid, prompt, pool)
                truth[rid] = prompt + [rng.randrange(6)]  # the prefill's bonus
                running.append(rid)
            if len(running) > 1 and rng.random() < 0.1:
                pool.free(reqs[running.pop(rng.randrange(len(running)))])
            if rng.random() < 0.03:
                # Retracted, then prefilled again in a new slot allocation.
                req = reqs[rng.choice(running)]
                pool.free(req)
                pool.alloc([req])
            for rid in [rid for rid in running if len(truth[rid]) >= 400]:
                running.remove(rid)
                pool.free(reqs[rid])
            if not running:
                continue
            batch_reqs = [reqs[rid] for rid in running]
            histories = [truth[rid] for rid in running]
            batch, draft_input = _decode_step(batch_reqs, histories)
            drafts = _fixed_drafts(len(running), token=VOCAB - 1)
            _propose(lookup, pool, batch, draft_input, drafts)
            table = lookup._table
            rows = lookup._step_rows.tolist()
            over_capacity = len(running) > table.rows
            num_over_capacity += over_capacity
            self.assertEqual(table.spare_row in rows, over_capacity)
            for i, history in enumerate(histories):
                last = len(history) - 1
                proposal = drafts[i, 1:].tolist()
                if rows[i] != table.spare_row:
                    held = table.tokens[rows[i], table._pad : table._pad + last + 1]
                    held = held.tolist()
                    self.assertTrue(all(h in (t, -1) for h, t in zip(held, history)))
                    self.assertEqual(held[-1], history[-1])
                if over_capacity:
                    self.assertEqual(proposal, [VOCAB - 1] * (BLOCK - 1))
                    continue
                match, cont = _reference_lookup(held, last, 16, BLOCK - 1)
                if match >= 3 and min(cont) >= 0:
                    num_proposed += 1
                    self.assertEqual(proposal, cont)
                else:
                    self.assertEqual(proposal, [VOCAB - 1] * (BLOCK - 1))
            commit_lens = [rng.randint(1, BLOCK) for _ in running]
            out_tokens = torch.randint(0, 6, (len(running), BLOCK), generator=generator)
            lookup.record(prefix_lens=batch.seq_lens, out_tokens=out_tokens)
            for i, rid in enumerate(running):
                truth[rid] = truth[rid] + out_tokens[i, : commit_lens[i]].tolist()
                # The host learns a step's tokens one step late.
                known = truth[rid][len(reqs[rid].origin_input_ids) : -commit_lens[i]]
                reqs[rid].output_ids.extend(known[len(reqs[rid].output_ids) :])
        self.assertGreater(num_proposed, 100)
        self.assertGreater(num_over_capacity, 10)

    def test_a_row_belongs_to_one_allocation_of_a_request_slot(self):
        """A request keeps its row, without a refill, from one decode step to the
        next for as long as it holds its slot; two live requests never share a row,
        even with the same rid; and a reused slot fills a fresh row from its new
        request's own tokens."""
        pool = _pool()
        lookup = _lookup()
        prompts = [list(range(1, 11)) * 2, list(range(11, 21)) * 2]
        reqs = [_req("dup", prompt, pool) for prompt in prompts]

        def step(reqs, histories):
            batch, draft_input = _decode_step(reqs, histories)
            drafts = _fixed_drafts(len(reqs), token=VOCAB - 1)
            _propose(lookup, pool, batch, draft_input, drafts)
            rows = lookup._step_rows.tolist()
            lookup.record(prefix_lens=batch.seq_lens, out_tokens=drafts)
            return rows, [d[1:] for d in drafts.tolist()]

        with mock.patch.object(
            PromptLookupTable,
            "fill",
            autospec=True,
            side_effect=PromptLookupTable.fill,
        ) as fill:
            histories = [p + p[:1] for p in prompts]
            rows, proposals = step(reqs, histories)
            self.assertEqual(fill.call_count, 2)
            self.assertNotEqual(rows[0], rows[1])
            self.assertEqual(proposals, [p[1:BLOCK] for p in prompts])

            self.assertEqual(step(reqs, histories)[0], rows)
            self.assertEqual(fill.call_count, 2)

            slot = reqs[0].kv.req_pool_idx
            pool.free(reqs[0])
            prompt = list(range(21, 31)) * 2
            reused = _req("reused", prompt, pool)
            self.assertEqual(reused.kv.req_pool_idx, slot)
            _, proposals = step([reqs[1], reused], [histories[1], prompt + prompt[:1]])
            self.assertEqual(fill.call_count, 3)
            self.assertEqual(proposals[1], prompt[1:BLOCK])


class TestAcceptWithLookup(CustomTestCase):
    @staticmethod
    def _worker():
        return SimpleNamespace(
            _selector_sample=None,
            _tp_sync=SimpleNamespace(sync=lambda site, tensor: tensor),
            _use_triton_accept_bonus=False,
            _block_verification=False,
            _draft_probs_buf=None,
            block_size=BLOCK,
        )

    def test_greedy_commits_the_target_chain_in_fewer_steps(self):
        """Looked-up blocks go through the worker's stock greedy accept: the
        committed tokens are the target's own greedy continuation."""

        def target_next(context):
            # Copies what followed the last earlier occurrence of the last 4 tokens.
            key = tuple(context[-4:])
            for end in range(len(context) - 2, 2, -1):
                if tuple(context[end - 3 : end + 1]) == key:
                    return context[end + 1]
            return (context[-1] * 7 + context[-2] * 3 + 1) % VOCAB

        prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4], [5, 5, 6, 5, 5, 6, 5]]
        num_new = 120
        expected = []
        for prompt in prompts:
            context = list(prompt)
            for _ in range(num_new + 1):
                context.append(target_next(context))
            expected.append(context[len(prompt) + 1 :][:num_new])

        worker = self._worker()
        pool = _pool()
        lookup = _lookup(min_match=4)
        reqs = [_req(f"g{i}", p, pool) for i, p in enumerate(prompts)]
        histories = [p + [target_next(p)] for p in prompts]
        steps = 0
        while min(len(h) - len(p) - 1 for h, p in zip(histories, prompts)) < num_new:
            steps += 1
            batch, draft_input = _decode_step(reqs, histories)
            drafts = _fixed_drafts(len(reqs))
            drafts[:, 0] = draft_input.bonus_tokens
            _propose(lookup, pool, batch, draft_input, drafts)
            target_predict = torch.tensor(
                [
                    [target_next(h[:-1] + d[: j + 1]) for j in range(BLOCK)]
                    for h, d in zip(histories, drafts.tolist())
                ]
            )
            _, commit_lens, _, out_tokens, _, _ = (
                worker_mod.DFlashWorkerV2._accept_block(
                    worker,
                    candidates=drafts,
                    next_token_logits=None,
                    sampling_info=None,
                    draft_input=draft_input,
                    prefix_lens=batch.seq_lens,
                    bs=len(reqs),
                    target_predict=target_predict,
                )
            )
            lookup.record(prefix_lens=batch.seq_lens, out_tokens=out_tokens)
            for i in range(len(reqs)):
                histories[i] = histories[i] + out_tokens[i, : commit_lens[i]].tolist()
        for history, prompt, want in zip(histories, prompts, expected):
            self.assertEqual(history[len(prompt) + 1 :][:num_new], want)
        # One committed token per step without the lookup.
        self.assertLess(steps, num_new // 2)

    def test_sampled_rows_reach_the_accept_as_point_masses(self):
        """The selector accept scatters a looked-up position into a dense q that is
        1.0 at its token and 0 elsewhere, so the speculative accept takes it with
        its target probability; other rows keep their q, and the buffer is zeroed
        after the accept."""
        generator = torch.Generator().manual_seed(2)
        pattern = list(range(3, 23))
        prompts = [pattern * 2, list(range(20))]  # a 16-token match; no match
        pool = _pool()
        reqs = [_req(f"s{i}", p, pool) for i, p in enumerate(prompts)]
        lookup = _lookup()
        batch, draft_input = _decode_step(reqs, prompts, top_ks=[20, 20])
        drafts = _fixed_drafts(2)
        gamma, top_k = BLOCK - 1, 4
        # A selector's top-k candidates are distinct tokens.
        candidate_ids = torch.stack(
            [
                torch.randperm(VOCAB, generator=generator)[:top_k]
                for _ in range(2 * gamma)
            ]
        ).view(2, gamma, top_k)
        q_rows = torch.rand(2, gamma, top_k, generator=generator)
        selector_sample = _propose(
            lookup, pool, batch, draft_input, drafts, (candidate_ids, q_rows)
        )
        looked_up = [pattern[(len(prompts[0]) + j) % 20] for j in range(gamma)]
        self.assertEqual(drafts[0, 1:].tolist(), looked_up)
        self.assertEqual(drafts[1, 1:].tolist(), [0] * gamma)

        worker = self._worker()
        seen = {}

        def accept_sampling(*, draft_probs, **kwargs):
            seen["q"] = draft_probs.clone()
            zeros = torch.zeros(2, dtype=torch.int32)
            return zeros, zeros.long(), zeros

        with mock.patch.object(worker_mod, "accept_sampling", accept_sampling):
            worker_mod.DFlashWorkerV2._selector_sampling_accept(
                worker,
                candidates=drafts,
                next_token_logits=torch.zeros(2 * BLOCK, VOCAB),
                candidate_ids=selector_sample[0],
                q_rows=selector_sample[1],
                sampling_info=batch.sampling_info,
                draft_input=draft_input,
            )
        q = seen["q"]
        expected_one_hot = torch.zeros(gamma, VOCAB)
        expected_one_hot[torch.arange(gamma), torch.tensor(looked_up)] = 1.0
        torch.testing.assert_close(q[0], expected_one_hot, rtol=0, atol=0)
        expected_kept = torch.zeros(gamma, VOCAB).scatter_(
            -1, candidate_ids[1], q_rows[1]
        )
        torch.testing.assert_close(q[1], expected_kept, rtol=0, atol=0)
        self.assertEqual(int(worker._draft_probs_buf.count_nonzero()), 0)

    def test_sampled_requests_need_the_sampled_minimum(self):
        """A 15-token match proposes for a greedy request (minimum 8) but not for
        a sampled one (minimum 16)."""
        history = list(range(1, 16)) + [0] + list(range(1, 16))
        pool = _pool()
        reqs = [_req("greedy", history, pool), _req("sampled", history, pool)]
        lookup = _lookup()
        batch, draft_input = _decode_step(reqs, [history, history], top_ks=[1, 20])
        drafts = _fixed_drafts(2, token=VOCAB - 1)
        _propose(lookup, pool, batch, draft_input, drafts)
        self.assertEqual(drafts[0, 1:].tolist(), [0] + list(range(1, 7)))
        self.assertEqual(drafts[1, 1:].tolist(), [VOCAB - 1] * (BLOCK - 1))


class TestPromptLookupGating(CustomTestCase):
    HISTORY = list(range(10, 30)) * 2  # a 16-token match at every step

    def _propose(self, lookup, *, info_fields=None, **batch_fields):
        pool = _pool()
        reqs = [_req("x", self.HISTORY, pool)]
        batch, draft_input = _decode_step(
            reqs, [self.HISTORY], info_fields=info_fields, **batch_fields
        )
        drafts = _fixed_drafts(1, token=VOCAB - 1)
        selector_sample = (
            torch.zeros(1, BLOCK - 1, 2, dtype=torch.int64),
            torch.full((1, BLOCK - 1, 2), 0.5),
        )
        returned = _propose(lookup, pool, batch, draft_input, drafts, selector_sample)
        unchanged = drafts[0, 1:].tolist() == [VOCAB - 1] * (BLOCK - 1)
        self.assertEqual(unchanged, returned is selector_sample)
        return unchanged

    def test_batches_whose_outputs_depend_on_block_boundaries_keep_their_drafts(self):
        """Verify logit adjustments see a request's history as of its block's start,
        so moving block boundaries would change outputs; grammar and logprob batches
        stay stock too. The table is still kept current."""
        self.assertFalse(self._propose(_lookup()))
        cases = {
            "grammar": {"has_grammar": True},
            "logprobs": {"return_logprob": True},
            "penalties": {"info_fields": {"step_penalties": mock.sentinel.penalties}},
            "custom logit processor": {
                "info_fields": {"has_custom_logit_processor": True}
            },
            "logit bias": {"info_fields": {"logit_bias": torch.zeros(1, VOCAB)}},
            "sampling mask": {
                "info_fields": {"sampling_mask_batch_indices": torch.tensor([0])}
            },
        }
        for name, fields in cases.items():
            with self.subTest(name):
                lookup = _lookup()
                self.assertTrue(self._propose(lookup, **fields))
                self.assertIsNotNone(lookup._step_rows)

    def test_lookups_thin_out_while_no_draft_is_replaced(self):
        """Every step looks up until 8 lookups in a row have replaced nothing, then
        every 8th step does; one hit brings back every step. A match too short for
        a sampled request is a miss. The ring hands each step the hit of the step
        two before it."""
        schedule = LookupSchedule(quiet=8, probe=8)
        due = []
        for step in range(48):
            hit = None if step < 2 or not due[step - 2] else False
            due.append(schedule.next_step(hit))
        self.assertEqual(due[:10], [True] * 9 + [False])
        self.assertEqual(
            [step for step in range(10, 48) if due[step]], [16, 24, 32, 40]
        )
        self.assertTrue(schedule.next_step(True))
        self.assertTrue(all(schedule.next_step(None) for _ in range(7)))

        ring = HitRing(device=CPU)
        pushed = {}
        for step in range(1, 50):
            self.assertEqual(ring.take(), pushed.pop(step - 2, None))
            if step % 3:
                ring.push(torch.tensor(step % 2 == 0))
                pushed[step] = step % 2 == 0

        def num_lookups(history, top_k):
            pool = _pool()
            reqs = [_req("t", history[:-1], pool)]
            lookup = _lookup()
            with mock.patch.object(
                lookup_mod, "_lookup_block", wraps=lookup_mod._lookup_block
            ) as lookup_block:
                for _ in range(64):
                    batch, draft_input = _decode_step(reqs, [history], top_ks=[top_k])
                    drafts = _fixed_drafts(1)
                    _propose(lookup, pool, batch, draft_input, drafts)
                    lookup.record(prefix_lens=batch.seq_lens, out_tokens=drafts)
            return lookup_block.call_count

        repeat = list(range(1, 11))
        ten_token_match = repeat + [11] + list(range(12, 30)) + [30] + repeat
        no_match = list(range(1, 31))
        self.assertEqual(num_lookups(ten_token_match, top_k=1), 64)
        self.assertLess(num_lookups(no_match, top_k=1), 64)
        self.assertEqual(
            num_lookups(ten_token_match, top_k=20), num_lookups(no_match, top_k=1)
        )


class TestPromptLookupStartup(CustomTestCase):
    def _init(
        self,
        *,
        pd_mode="null",
        cuda=True,
        confidential_compute=False,
        attn_dp=False,
        pp_size=1,
        mamba=None,
        simulate_acc_len=-1,
        **worker_fields,
    ):
        worker = SimpleNamespace(
            _is_domino=False,
            lilicorr=None,
            _block_verification=False,
            _verify_planner=None,
            _runtime=None,
            block_size=BLOCK,
            device="cpu",
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=VOCAB, context_len=64)
            ),
        )
        vars(worker).update(worker_fields)
        worker._prompt_lookup_unsupported_reason = types.MethodType(
            worker_mod.DFlashWorkerV2._prompt_lookup_unsupported_reason, worker
        )
        parallel = SimpleNamespace(tp_rank=0, attn_dp_enabled=attn_dp, pp_size=pp_size)
        with (
            mock.patch.object(worker_mod, "get_parallel", return_value=parallel),
            mock.patch.object(
                worker_mod,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=pd_mode),
            ),
            mock.patch.object(
                worker_mod,
                "get_schedule",
                return_value=SimpleNamespace(max_running_requests=48),
            ),
            mock.patch.object(
                worker_mod, "get_req_to_token_extra_context_len", return_value=20
            ),
            mock.patch.object(worker_mod, "is_cuda", return_value=cuda),
            mock.patch.object(
                worker_mod,
                "is_confidential_compute",
                return_value=confidential_compute,
            ),
            mock.patch.object(worker_mod, "mambaish_config", return_value=mamba),
            mock.patch.object(worker_mod, "SIMULATE_ACC_LEN", simulate_acc_len),
        ):
            return worker_mod.DFlashWorkerV2._init_prompt_lookup(worker)

    def test_on_only_when_enabled_and_supported(self):
        """Off unless set; then on, ignored on a PD prefill server, and a startup
        error naming the setting wherever it is not supported."""
        with envs.SGLANG_ENABLE_DFLASH_PROMPT_LOOKUP.override(False):
            self.assertIsNone(self._init())
        with envs.SGLANG_ENABLE_DFLASH_PROMPT_LOOKUP.override(True):
            self.assertIsInstance(self._init(), DFlashPromptLookup)
            self.assertIsNone(self._init(pd_mode="prefill"))
            unsupported = {
                "not CUDA": {"cuda": False},
                "Confidential Computing": {"confidential_compute": True},
                "attention DP": {"attn_dp": True},
                "pipeline parallelism": {"pp_size": 2},
                "block verification": {"_block_verification": True},
                "confidence-scheduled verify": {"_verify_planner": object()},
                "bound runtime": {"_runtime": object()},
                "LiLiCorr draft": {"lilicorr": object()},
                "Domino draft": {"_is_domino": True},
                "hybrid Mamba target": {"mamba": object()},
                "simulated acceptance": {"simulate_acc_len": 3},
                "no draft tokens": {"block_size": 1},
            }
            for name, fields in unsupported.items():
                with self.subTest(name):
                    with self.assertRaisesRegex(
                        ValueError, "SGLANG_ENABLE_DFLASH_PROMPT_LOOKUP=1, but"
                    ):
                        self._init(**fields)


if __name__ == "__main__":
    unittest.main()
