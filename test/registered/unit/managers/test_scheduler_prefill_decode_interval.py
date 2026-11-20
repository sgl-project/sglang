"""Tests for gating the prefill/decode interval on the running batch."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    RecentPrefillBatchSizeTracker,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_scheduler(
    *, interval: int, min_bs: int, require_mlp_sync: bool = False
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.prefill_decode_interval_min_bs = min_bs
    scheduler.require_mlp_sync = require_mlp_sync
    return scheduler


def _arm(scheduler: Scheduler) -> None:
    scheduler._arm_prefill_decode_interval(
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=True,
        )
    )


def _make_running_batch(batch_size: int, *, prefill_only: bool = False):
    return SimpleNamespace(
        batch_size=lambda: batch_size,
        is_empty=lambda: batch_size == 0,
        is_prefill_only=prefill_only,
    )


class TestPrefillDecodeIntervalMinBs(unittest.TestCase):
    def test_small_running_batch_admits_prefill_and_clears_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(7)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)
        # The cleared interval does not resume once the batch grows.
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(8)))

    def test_large_running_batch_defers_prefill(self):
        scheduler = _make_scheduler(interval=2, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(9)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(9)))

    def test_running_batch_shrinking_below_threshold_ends_interval(self):
        scheduler = _make_scheduler(interval=4, min_bs=8)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(8)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(5)))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)


class TestPrefillDecodeIntervalNothingDecodes(unittest.TestCase):
    # These use the default min_bs=0, where the interval applies at any batch size.

    def test_empty_running_batch_does_not_defer(self):
        # The next chunk of a long prompt on an idle server runs right away.
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))

    def test_prefill_only_running_batch_does_not_defer(self):
        scheduler = _make_scheduler(interval=4, min_bs=0)
        _arm(scheduler)

        running = _make_running_batch(3, prefill_only=True)
        self.assertFalse(scheduler._should_defer_prefill(running))

    def test_single_decoding_request_keeps_interval(self):
        scheduler = _make_scheduler(interval=2, min_bs=0)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(1)))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(1)))

    def test_dp_defers_without_local_decode(self):
        # Under DP attention, peer ranks may still be decoding.
        scheduler = _make_scheduler(interval=2, min_bs=0, require_mlp_sync=True)
        _arm(scheduler)

        self.assertTrue(scheduler._should_defer_prefill(_make_running_batch(0)))
        running = _make_running_batch(3, prefill_only=True)
        self.assertTrue(scheduler._should_defer_prefill(running))
        self.assertFalse(scheduler._should_defer_prefill(_make_running_batch(0)))


class _CpuGroup:
    """A CPU group of DP ranks, one thread per rank. A collective completes
    only when every rank enters the same one, otherwise all of them fail, as a
    mismatched gloo collective does."""

    def __init__(self, world_size: int):
        self._barrier = threading.Barrier(world_size, timeout=10)
        self._entered = [None] * world_size
        self._thread = threading.local()
        self.log = [[] for _ in range(world_size)]

    def run(self, steps):
        errors = [None] * len(steps)

        def target(rank):
            self._thread.rank = rank
            try:
                steps[rank]()
            except Exception as e:
                errors[rank] = e

        threads = [
            threading.Thread(target=target, args=(r,)) for r in range(len(steps))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors

    def collective(self, name, local):
        rank = self._thread.rank
        self.log[rank].append(name)
        self._entered[rank] = (name, local)
        self._barrier.wait()
        entered = list(self._entered)
        self._barrier.wait()
        if len({n for n, _ in entered}) > 1:
            raise RuntimeError(f"collective mismatch: {[n for n, _ in entered]}")
        return [row for _, row in entered]

    def delayer_all_gather(self, output, local, group=None):
        rows = self.collective("prefill_delayer", local)
        output.view(len(rows), -1).copy_(torch.stack(rows))


class _DecodingBatch:
    def __init__(self, num_reqs: int):
        self.reqs = [SimpleNamespace(finished=False) for _ in range(num_reqs)]
        self.forward_mode = SimpleNamespace(is_extend=lambda: False)
        self.is_extend_in_batch = False
        self.is_prefill_only = False
        self.batch_is_full = False

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        pass


def _make_dp_rank(group, *, min_bs, delayer, waiting=0) -> Scheduler:
    """A DP-attention rank whose prefill planning only talks to the delayer:
    with `waiting` requests it asks to prefill and admits when allowed."""
    s = _make_scheduler(interval=2, min_bs=min_bs, require_mlp_sync=True)
    s.scheduler_stage_metrics = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.process_pending_chunked_abort = Mock()
    s._process_hicache_events = Mock()
    s.enable_fpm = False
    s.dllm_config = None
    s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
    s.enable_hisparse = False
    s.invariant_checker = Mock()
    s.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    s.prefill_delayer = delayer
    s.pool_stats_observer = SimpleNamespace(
        get_pool_stats=lambda: SimpleNamespace(get_max_pool_usage=lambda: 0.5)
    )
    s.prefill_bs_tracker = RecentPrefillBatchSizeTracker()
    s.max_prefill_bs = 0
    s.max_running_requests = 8
    s.admitted = []

    def plan_prefill(prefill_delayer_single_pass, running_batch):
        if not waiting:
            return None, running_batch
        if prefill_delayer_single_pass is None or (
            prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=running_batch.batch_size(),
                max_prefill_bs=s.max_prefill_bs,
                max_running_requests=s.max_running_requests,
                waiting_queue_len=waiting,
            )
        ):
            s.admitted.append(waiting)
        return None, running_batch

    def mlp_sync(batch, need_sync=None):
        group.collective("mlp_sync", None)
        return batch

    s._get_new_batch_prefill_raw = Mock(side_effect=plan_prefill)
    s.update_running_batch = lambda batch: batch
    s.dp_attn_adapter = Mock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = mlp_sync
    s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: b
    s.ngram_embedding_manager = Mock()
    s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b
    return s


class TestPrefillDecodeIntervalMinBsCollectives(CustomTestCase):
    """In the round after an extend, a DP rank at or above the min batch size
    defers prefill and a rank below it plans one. With the prefill delayer on,
    planning runs the delayer's all-gather, so the deferring rank must join it
    or the two ranks enter different collectives on the same group."""

    def setUp(self):
        super().setUp()
        self.enterContext(published_topology(tp_size=2, attn_dp_size=2))
        self.enterContext(
            get_parallel().override(
                tp_group=SimpleNamespace(cpu_group=None, device_group=None)
            )
        )
        self.group = _CpuGroup(world_size=2)
        self.enterContext(
            patch(
                "sglang.srt.managers.prefill_delayer.all_gather_single",
                side_effect=self.group.delayer_all_gather,
            )
        )

    def _delayer(self):
        return PrefillDelayer(max_delay_passes=30, token_usage_low_watermark=None)

    def _run_round_after_extend(self, ranks, running_sizes):
        for s in ranks:
            _arm(s)
        steps = [
            (lambda s=s, n=n: s.get_next_batch_to_run(_DecodingBatch(n), None))
            for s, n in zip(ranks, running_sizes)
        ]
        errors = self.group.run(steps)
        self.assertEqual(errors, [None, None])

    def test_ranks_on_either_side_of_min_bs_enter_the_same_collectives(self):
        ranks = [
            _make_dp_rank(self.group, min_bs=2, delayer=self._delayer())
            for _ in range(2)
        ]
        self._run_round_after_extend(ranks, running_sizes=[3, 1])

        # Rank 0 defers, rank 1 is below min-bs and plans a prefill.
        ranks[0]._get_new_batch_prefill_raw.assert_not_called()
        ranks[1]._get_new_batch_prefill_raw.assert_called_once()
        self.assertEqual(self.group.log[0], ["prefill_delayer", "mlp_sync"])
        self.assertEqual(self.group.log[1], ["prefill_delayer", "mlp_sync"])

    def test_deferring_rank_votes_nothing_prefillable(self):
        delayers = [self._delayer(), self._delayer()]
        ranks = [
            _make_dp_rank(self.group, min_bs=2, delayer=delayers[0], waiting=1),
            _make_dp_rank(self.group, min_bs=2, delayer=delayers[1], waiting=1),
        ]
        self._run_round_after_extend(ranks, running_sizes=[3, 1])

        # Only rank 1 can prefill, so the delayer holds it ("mixed") and both
        # ranks count the same delayed pass.
        self.assertEqual(ranks[1].admitted, [])
        self.assertEqual([d._curr_state.delayed_count for d in delayers], [1, 1])

    def test_min_bs_zero_ranks_defer_together_without_delayer_gather(self):
        ranks = [
            _make_dp_rank(self.group, min_bs=0, delayer=self._delayer())
            for _ in range(2)
        ]
        self._run_round_after_extend(ranks, running_sizes=[3, 1])

        for s in ranks:
            s._get_new_batch_prefill_raw.assert_not_called()
        self.assertEqual(self.group.log, [["mlp_sync"], ["mlp_sync"]])

    def test_no_delayer_adds_no_collective(self):
        ranks = [_make_dp_rank(self.group, min_bs=2, delayer=None) for _ in range(2)]
        self._run_round_after_extend(ranks, running_sizes=[3, 1])

        ranks[0]._get_new_batch_prefill_raw.assert_not_called()
        ranks[1]._get_new_batch_prefill_raw.assert_called_once()
        self.assertEqual(self.group.log, [["mlp_sync"], ["mlp_sync"]])


if __name__ == "__main__":
    unittest.main()
