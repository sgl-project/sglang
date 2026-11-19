import unittest

from sglang.srt.mem_cache.prefetch_admission_budget import PrefetchAdmissionBudget
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefetchAdmissionBudget(CustomTestCase):
    def test_ready_request_bypasses_waiter_using_reserved_capacity(self):
        b = PrefetchAdmissionBudget()
        b.enqueue("owner", 6, "owner")
        self.assertEqual(b.grant("owner", 10, 4), "owner")
        b.enqueue("blocked", 4, "blocked")
        self.assertIsNone(b.grant("blocked", 10, 4))
        b.enqueue("ready", 4, "ready")
        b.ready = {"ready"}
        self.assertEqual(b.candidates(), ["ready", "blocked"])
        self.assertEqual(b.grant("ready", 10, 4), "ready")
        b.release("owner")
        b.release("ready")
        self.assertEqual(b.grant("blocked", 10, 4), "blocked")

    def test_mamba_reservation_preserves_anchor_and_incoming_slot(self):
        b = PrefetchAdmissionBudget()
        b.enqueue("owner", 1, "owner")
        self.assertEqual(b.grant("owner", 100, 10, 4), "owner")
        b.enqueue("waiter", 1, "waiter")
        self.assertIsNone(b.grant("waiter", 100, 10, 4))
        b.ready = {"waiter"}
        self.assertEqual(b.grant("waiter", 100, 10, 4), "waiter")

    def test_cancellation_and_generation_reuse(self):
        b = PrefetchAdmissionBudget()
        old, new = ("r", 0), ("r", 1)
        b.enqueue(old, 4, "old")
        b.enqueue(new, 4, "new")
        b.release(old)
        b.release(old)
        self.assertEqual(b.grant(new, 8, 4), "new")
        self.assertEqual(b.granted, {new: 4})
        b.release(new)
        self.assertFalse(b.granted)


if __name__ == "__main__":
    unittest.main()
