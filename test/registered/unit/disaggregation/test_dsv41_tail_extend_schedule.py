"""Decode-side scheduling of the --dsv41-encoder-only-prefill tail extend."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.utils import (
    dsv41_encoder_only_unsupported,
    dsv41_encoder_prefix_len,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Mixin = SchedulerDisaggregationDecodeMixin


def _req(prompt_len):
    tail_start = dsv41_encoder_prefix_len(prompt_len, 128)
    return SimpleNamespace(
        full_untruncated_fill_ids=list(range(prompt_len)),
        dsv41_tail_start=tail_start,
        kv=SimpleNamespace(kv_allocated_len=tail_start),
        prefix_len=0,
        extend_end=tail_start,
        already_computed=0,
        _cache_breakdown_computed=False,
        logprob_start_len=prompt_len,
    )


def _batch(mode, size):
    batch = MagicMock()
    batch.forward_mode = mode
    batch.batch_size.return_value = size
    batch.is_empty.return_value = size == 0
    return batch


def _scheduler(
    waiting, running, last_batch=None, max_prefill_tokens=16384, free_tokens=1 << 20
):
    return SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=list(waiting),
        last_batch=last_batch,
        req_to_token_pool=MagicMock(size=64),
        max_running_requests=64,
        max_prefill_tokens=max_prefill_tokens,
        token_to_kv_pool_allocator=MagicMock(
            page_size=256, available_size=MagicMock(return_value=free_tokens)
        ),
        tree_cache=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        running_batch=running,
        forward_ct=0,
    )


@patch("sglang.srt.disaggregation.decode.set_time_batch")
@patch("sglang.srt.disaggregation.decode.ScheduleBatch")
class TestDsv41TailExtendSchedule(CustomTestCase):
    def test_split_keeps_one_window_for_decode(self, *_):
        self.assertEqual(dsv41_encoder_prefix_len(1000, 128), 872)
        # Even, so no ratio-2 half-pair is pending at the split.
        self.assertEqual(dsv41_encoder_prefix_len(1001, 128), 872)
        self.assertEqual(dsv41_encoder_prefix_len(129, 128), 0)
        self.assertEqual(dsv41_encoder_prefix_len(128, 128), 0)
        self.assertEqual(dsv41_encoder_prefix_len(5, 128), 0)

    def test_tail_extends_on_top_of_the_transferred_prefix(self, schedule_batch, _):
        short, long = _req(40), _req(1000)
        sched = _scheduler([short, long], _batch(ForwardMode.DECODE, 0))
        batch = Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)

        self.assertIs(batch, schedule_batch.init_new.return_value)
        batch.prepare_for_extend.assert_called_once()
        self.assertEqual(schedule_batch.init_new.call_args.args[0], [short, long])
        self.assertEqual(sched.waiting_queue, [])
        self.assertEqual((long.prefix_len, long.extend_end), (872, 1000))
        self.assertEqual(long.already_computed, 872)
        self.assertIsNone(long.dsv41_tail_start)
        # Nothing was transferred for the short prompt: its row is released so
        # the extend allocates it whole.
        sched.req_to_token_pool.free.assert_called_once_with(short)
        self.assertEqual((short.prefix_len, short.extend_end), (0, 40))

    def test_admission_respects_the_prefill_token_budget(self, schedule_batch, _):
        reqs = [_req(1000) for _ in range(3)]
        sched = _scheduler(reqs, _batch(ForwardMode.DECODE, 0), max_prefill_tokens=300)
        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], reqs[:2])
        self.assertEqual(sched.waiting_queue, reqs[2:])

    def test_admission_waits_for_free_pages(self, schedule_batch, _):
        # [972, 1100) needs one page past the transferred prefix's last page.
        reqs = [_req(1100) for _ in range(3)]
        sched = _scheduler(reqs, _batch(ForwardMode.DECODE, 0), free_tokens=512)
        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], reqs[:2])

        sched = _scheduler(reqs[2:], _batch(ForwardMode.DECODE, 4), free_tokens=0)
        self.assertIsNone(
            Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        )
        self.assertEqual(sched.waiting_queue, reqs[2:])

    def test_rebootstrap_tail_scores_no_replayed_tokens(self, *_):
        # Retracted after 9 tokens: prefill covers 1000 + 8, the tail replays
        # the emitted boundary token and samples the tenth.
        req = _req(1009)
        req.dsv41_tail_start = dsv41_encoder_prefix_len(1008, 128)
        req.pd_rebootstrap_in_progress = True
        req.logprob_start_len = 1000
        sched = _scheduler([req], _batch(ForwardMode.DECODE, 0))
        Mixin._get_new_dsv41_tail_extend_batch(sched, sched.running_batch)
        self.assertEqual((req.prefix_len, req.extend_end), (880, 1009))
        self.assertFalse(req.pd_rebootstrap_in_progress)
        self.assertEqual(req.logprob_start_len, 1009)

    def test_decode_runs_between_tail_extends(self, schedule_batch, _):
        running = _batch(ForwardMode.DECODE, 4)
        sched = _scheduler(
            [_req(1000)], running, last_batch=_batch(ForwardMode.EXTEND, 1)
        )
        self.assertIsNone(Mixin._get_new_dsv41_tail_extend_batch(sched, running))
        schedule_batch.init_new.assert_not_called()

    def test_tails_run_once_per_hold_period_while_decode_runs(self, schedule_batch, _):
        running = _batch(ForwardMode.DECODE, 4)
        # 60 s at the nominal 20 ms step: periods of 3000 forward steps.
        with envs.SGLANG_DSV41_TAIL_HOLD_MS.override(60_000):
            # Mid-period: held.
            sched = _scheduler([_req(1000)], running)
            sched.forward_ct, sched.dsv41_tail_period = 4500, 1
            self.assertIsNone(Mixin._get_new_dsv41_tail_extend_batch(sched, running))
            schedule_batch.init_new.assert_not_called()

            # The first step of a period runs every waiting tail.
            reqs = [_req(1000), _req(2000)]
            sched = _scheduler(reqs, running)
            sched.forward_ct, sched.dsv41_tail_period = 6000, 1
            Mixin._get_new_dsv41_tail_extend_batch(sched, running)
            self.assertEqual(schedule_batch.init_new.call_args.args[0], reqs)
            self.assertEqual(sched.dsv41_tail_period, 2)

            # A full token budget, or idle decode, runs at once.
            for req, kwargs, batch in (
                (_req(1000), {"max_prefill_tokens": 128}, running),
                (_req(1000), {}, _batch(ForwardMode.DECODE, 0)),
            ):
                sched = _scheduler([req], batch, **kwargs)
                sched.forward_ct, sched.dsv41_tail_period = 4500, 1
                Mixin._get_new_dsv41_tail_extend_batch(sched, batch)
                self.assertEqual(schedule_batch.init_new.call_args.args[0], [req])

    def test_tp_ranks_make_the_same_hold_decision(self, schedule_batch, _):
        # The TP ranks of one decode schedule every step alike; a rank that runs a
        # tail extend while another decodes desyncs their collectives (a TP2 decode
        # hung this way under AgentX). Two ranks whose clocks are 3 ms apart, one
        # step 1 ms before a 500 ms boundary, run 60 identical 20 ms steps.
        running = _batch(ForwardMode.DECODE, 4)
        runs = {}
        with envs.SGLANG_DSV41_TAIL_HOLD_MS.override(500):
            for rank, skew in ((0, 0.0), (1, 0.003)):
                period, runs[rank] = None, []
                for step in range(60):
                    sched = _scheduler([_req(1000)], running)
                    sched.forward_ct = step
                    if period is not None:
                        sched.dsv41_tail_period = period
                    clock = 999.999 + 0.020 * step + skew
                    with patch("sglang.srt.disaggregation.decode.time") as fake_time:
                        fake_time.time.return_value = clock
                        fake_time.monotonic.return_value = clock
                        batch = Mixin._get_new_dsv41_tail_extend_batch(sched, running)
                    period = sched.dsv41_tail_period
                    runs[rank].append(batch is not None)
        self.assertEqual(runs[0], runs[1])
        # The hold holds: tails run on the first step of each 25-step period only.
        self.assertEqual([i for i, ran in enumerate(runs[0]) if ran], [0, 25, 50])

    def test_finished_tail_extend_merges_into_decode(self, *_):
        running, tail = _batch(ForwardMode.DECODE, 4), _batch(ForwardMode.EXTEND, 2)
        sched = _scheduler([], running, last_batch=tail)
        self.assertIs(Mixin._merge_dsv41_tail_extend(sched, running), running)
        tail.filter_batch.assert_called_once()
        running.merge_batch.assert_called_once_with(tail)

        empty_running = _batch(ForwardMode.DECODE, 0)
        sched = _scheduler([], empty_running, last_batch=tail)
        self.assertIs(Mixin._merge_dsv41_tail_extend(sched, empty_running), tail)

        sched = _scheduler([], running, last_batch=running)
        self.assertIs(Mixin._merge_dsv41_tail_extend(sched, running), running)
        self.assertEqual(running.merge_batch.call_count, 1)


class TestDsv41EncoderOnlyRequests(CustomTestCase):
    @staticmethod
    def _req(**fields):
        values = dict(
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            return_logprob=False,
            logprob_start_len=-1,
            origin_input_ids=list(range(300)),
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            return_sampling_mask=False,
        )
        values.update(fields)
        return SimpleNamespace(**values)

    def test_peers_reject_what_the_prefix_cannot_serve(self):
        self.assertIsNone(dsv41_encoder_only_unsupported(self._req()))
        # Decode computes every output token's logprob.
        self.assertIsNone(
            dsv41_encoder_only_unsupported(
                self._req(return_logprob=True, logprob_start_len=300)
            )
        )
        for fields in (
            {"return_logprob": True, "logprob_start_len": 0},
            {"input_embeds": [[0.0]]},
            {"return_hidden_states": True},
            {"return_routed_experts": True},
            {"return_sampling_mask": True},
        ):
            with self.subTest(**fields):
                self.assertIsNotNone(
                    dsv41_encoder_only_unsupported(self._req(**fields))
                )


if __name__ == "__main__":
    unittest.main()
