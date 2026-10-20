"""Decode-side scheduling of the --dsv41-encoder-only-prefill tail extend."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.utils import dsv41_encoder_prefix_len
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
    )


def _batch(mode, size):
    batch = MagicMock()
    batch.forward_mode = mode
    batch.batch_size.return_value = size
    batch.is_empty.return_value = size == 0
    return batch


def _scheduler(waiting, running, last_batch=None, max_prefill_tokens=16384):
    return SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=list(waiting),
        last_batch=last_batch,
        req_to_token_pool=MagicMock(size=64),
        max_running_requests=64,
        max_prefill_tokens=max_prefill_tokens,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        running_batch=running,
    )


@patch("sglang.srt.disaggregation.decode.set_time_batch")
@patch("sglang.srt.disaggregation.decode.ScheduleBatch")
class TestDsv41TailExtendSchedule(CustomTestCase):
    def test_split_keeps_one_window_for_decode(self, *_):
        self.assertEqual(dsv41_encoder_prefix_len(1000, 128), 872)
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

    def test_decode_runs_between_tail_extends(self, schedule_batch, _):
        running = _batch(ForwardMode.DECODE, 4)
        sched = _scheduler(
            [_req(1000)], running, last_batch=_batch(ForwardMode.EXTEND, 1)
        )
        self.assertIsNone(Mixin._get_new_dsv41_tail_extend_batch(sched, running))
        schedule_batch.init_new.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
