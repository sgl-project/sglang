import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Larger than any free-slot count below, so only the threshold decides.
_MANY_WAITING = 1000


class TestResolveMinFreeSlots(unittest.TestCase):
    def test_unset_non_dflash_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 512, is_dflash_family=False))

    def test_unset_dflash_auto_enables(self):
        self.assertEqual(resolve_min_free_slots(None, 512, is_dflash_family=True), 4)
        self.assertEqual(resolve_min_free_slots(None, 8, is_dflash_family=True), 2)

    def test_unset_dflash_small_cluster_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 7, is_dflash_family=True))
        self.assertIsNone(resolve_min_free_slots(None, 0, is_dflash_family=True))

    def test_le_one_disables(self):
        # <= 1 can never batch, so it is a no-op.
        self.assertIsNone(resolve_min_free_slots(1, 512))
        self.assertIsNone(resolve_min_free_slots(0, 512))

    def test_explicit_value_survives_small_cluster(self):
        # The < 8 guard belongs to the DFlash auto-default, not explicit values.
        self.assertEqual(resolve_min_free_slots(4, 7), 4)
        self.assertEqual(resolve_min_free_slots(4, 7, is_dflash_family=True), 4)

    def test_non_dflash_uses_explicit_value(self):
        self.assertEqual(resolve_min_free_slots(2, 8), 2)
        self.assertEqual(resolve_min_free_slots(3, 512), 3)
        self.assertEqual(resolve_min_free_slots(8, 512), 8)
        self.assertEqual(resolve_min_free_slots(16, 512), 16)

    def test_explicit_value_is_capped_to_max_running_requests(self):
        self.assertEqual(resolve_min_free_slots(16, 8), 8)

    def test_user_value_overrides_dflash_default(self):
        self.assertEqual(resolve_min_free_slots(3, 512, is_dflash_family=True), 3)
        self.assertEqual(resolve_min_free_slots(16, 512, is_dflash_family=True), 16)

    def test_explicit_one_disables_dflash_default(self):
        self.assertIsNone(resolve_min_free_slots(1, 512, is_dflash_family=True))


def _should_delay(delayer, running, allocatable, waiting=_MANY_WAITING):
    return delayer.should_delay(
        running_bs=running, num_allocatable_reqs=allocatable, num_waiting_reqs=waiting
    )


class TestMinFreeSlotsDelayer(unittest.TestCase):
    def test_delays_below_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertTrue(_should_delay(delayer, running=100, allocatable=2))

    def test_no_delay_at_or_above_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertFalse(_should_delay(delayer, running=100, allocatable=4))
        self.assertFalse(_should_delay(delayer, running=100, allocatable=8))

    def test_no_delay_when_idle(self):
        # Nothing running: no decode batch to protect, prefill at once.
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertFalse(_should_delay(delayer, running=0, allocatable=0))

    def test_admits_last_request_into_nearly_full_batch(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=16)
        self.assertFalse(_should_delay(delayer, running=63, allocatable=1, waiting=1))

    def test_admits_when_threshold_equals_max_running(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=2)
        self.assertFalse(_should_delay(delayer, running=1, allocatable=1, waiting=1))

    def test_delays_burst_larger_than_free_slots(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=16)
        self.assertTrue(_should_delay(delayer, running=60, allocatable=4, waiting=40))

    def test_delays_until_every_waiting_request_fits(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=16)
        self.assertTrue(_should_delay(delayer, running=60, allocatable=4, waiting=5))
        self.assertFalse(_should_delay(delayer, running=59, allocatable=5, waiting=5))


class _PastDelayGate(Exception):
    pass


class TestSchedulerDelayGate(unittest.TestCase):
    """The scheduler hands the delayer the current waiting-queue length."""

    def _held(self, *, running, allocatable, waiting):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = [object()] * waiting
        s.chunked_reqs = []
        s.min_free_slots_delayer = MinFreeSlotsDelayer(min_free_slots=16)
        # The gate reads the free slots once; the next read is past the gate.
        s.get_num_allocatable_reqs = Mock(side_effect=[allocatable, _PastDelayGate])
        running_batch = SimpleNamespace(
            reqs=[object()] * running, batch_is_full=False, is_empty=lambda: False
        )
        try:
            new_batch, _ = s._get_new_batch_prefill_raw(None, running_batch)
        except _PastDelayGate:
            return False
        self.assertIsNone(new_batch)
        return True

    def test_admits_waiting_requests_that_fit(self):
        self.assertFalse(self._held(running=63, allocatable=1, waiting=1))

    def test_holds_waiting_requests_that_do_not_fit(self):
        self.assertTrue(self._held(running=60, allocatable=4, waiting=40))


if __name__ == "__main__":
    unittest.main()
