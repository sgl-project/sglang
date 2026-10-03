"""Unit tests for what a draining retiree counts as work of its own.

A retiree can only leave from a tick where it reports idle. Under dp-attention it is
pulled into a request-free forward on every iteration any survivor holds tokens, and
that batch lands in ``result_queue``, which the strict predicate refuses on. So the
retiree's idle test discounts request-free batches and every other caller does not.

That is what makes holding only the pinned requests safe, so the shrink gate's scope is
covered here too.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.test_utils import CustomTestCase


class FakeBatch:
    def __init__(self, empty: bool):
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


def _sched(*batches: FakeBatch) -> SimpleNamespace:
    """Enough of a Scheduler for the unbound predicate: just the queue."""
    return SimpleNamespace(result_queue=[(b, object()) for b in batches])


class TestPendingResults(CustomTestCase):
    def test_strict_count_includes_idle_batches(self):
        sched = _sched(FakeBatch(empty=True), FakeBatch(empty=False))
        self.assertEqual(Scheduler._pending_results(sched, False), 2)

    def test_idle_batches_discounted(self):
        sched = _sched(FakeBatch(empty=True), FakeBatch(empty=False))
        self.assertEqual(Scheduler._pending_results(sched, True), 1)

    def test_all_idle_reads_as_drained(self):
        """The retiree's actual shape: forwards in flight, none of them its own."""
        sched = _sched(*[FakeBatch(empty=True)] * 4)
        self.assertEqual(Scheduler._pending_results(sched, True), 0)
        self.assertEqual(Scheduler._pending_results(sched, False), 4)

    def test_real_work_still_blocks(self):
        sched = _sched(FakeBatch(empty=False))
        self.assertEqual(Scheduler._pending_results(sched, True), 1)

    def test_empty_queue(self):
        sched = _sched()
        for ignore in (False, True):
            with self.subTest(ignore_idle_batches=ignore):
                self.assertEqual(Scheduler._pending_results(sched, ignore), 0)


class TestShrinkGateScope(CustomTestCase):
    """Only a pin onto a departing slot waits. Everything else goes through."""

    def _gate(self, *, closed: bool, retiring=(1, 2)) -> SimpleNamespace:
        event = asyncio.Event()
        if not closed:
            event.set()
        return SimpleNamespace(
            _elastic_shrink_pause_event=event,
            _elastic_shrink_retiring=set(retiring),
        )

    def _admitted(self, tm, obj, *, timeout=0.05) -> bool:
        """True if the gate let the request through without waiting."""

        async def run():
            try:
                await asyncio.wait_for(
                    TokenizerManager._await_elastic_shrink_gate(tm, obj), timeout
                )
                return True
            except asyncio.TimeoutError:
                return False

        return asyncio.new_event_loop().run_until_complete(run())

    def test_open_gate_admits_everything(self):
        tm = self._gate(closed=False)
        self.assertTrue(self._admitted(tm, SimpleNamespace(routed_dp_rank=1)))

    def test_closed_gate_admits_unpinned(self):
        """The DPC routes these, and it already skips the draining slots."""
        tm = self._gate(closed=True)
        self.assertTrue(self._admitted(tm, SimpleNamespace(routed_dp_rank=None)))

    def test_closed_gate_admits_a_surviving_pin(self):
        tm = self._gate(closed=True, retiring=(2, 3))
        self.assertTrue(self._admitted(tm, SimpleNamespace(routed_dp_rank=0)))

    def test_closed_gate_holds_a_retiring_pin(self):
        tm = self._gate(closed=True, retiring=(2, 3))
        self.assertFalse(self._admitted(tm, SimpleNamespace(routed_dp_rank=3)))

    def test_a_request_with_no_pin_attribute_is_admitted(self):
        """Non-generate inputs carry no routed_dp_rank at all."""
        tm = self._gate(closed=True)
        self.assertTrue(self._admitted(tm, SimpleNamespace()))

    def test_close_records_the_retiring_slots_and_open_clears_them(self):
        tm = self._gate(closed=False, retiring=())
        TokenizerManager._close_elastic_shrink_gate(tm, range(2, 4))
        self.assertEqual(tm._elastic_shrink_retiring, {2, 3})
        self.assertFalse(tm._elastic_shrink_pause_event.is_set())
        TokenizerManager._open_elastic_shrink_gate(tm)
        self.assertEqual(tm._elastic_shrink_retiring, set())
        self.assertTrue(tm._elastic_shrink_pause_event.is_set())


if __name__ == "__main__":
    unittest.main()
