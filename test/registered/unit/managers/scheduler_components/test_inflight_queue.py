import unittest

from sglang.srt.managers.scheduler_components.inflight_queue import InflightQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestInflightQueue(CustomTestCase):
    def test_depth_zero_commits_in_the_launching_step(self):
        q = InflightQueue(depth=0)
        self.assertTrue(q.can_launch())
        q.push("b0", "r0")
        self.assertTrue(q.has_due())
        record = q.pop_due()
        self.assertEqual((record.batch, record.result), ("b0", "r0"))
        q.end_step("b0")
        self.assertEqual(len(q), 0)
        self.assertEqual(q.last_launched, "b0")

    def test_depth_one_commits_one_completed_step_later(self):
        q = InflightQueue(depth=1)
        q.push("b0", "r0")
        self.assertFalse(q.has_due())
        q.end_step("b0")
        # The next step launches b1 before committing b0; b1 is not due yet.
        self.assertTrue(q.can_launch())
        q.push("b1", "r1")
        self.assertTrue(q.has_due())
        self.assertEqual(q.pop_due().batch, "b0")
        self.assertFalse(q.has_due())
        q.end_step("b1")
        # A step that launches nothing still commits the previous record.
        self.assertEqual(q.pop_due().batch, "b1")
        q.end_step(None)
        self.assertFalse(q.has_due())
        self.assertIsNone(q.last_launched)

    def test_launch_cap_is_depth_plus_one(self):
        q = InflightQueue(depth=1, blocking=False)
        q.push("b0", "r0")
        self.assertTrue(q.can_launch())
        q.push("b1", "r1")
        self.assertFalse(q.can_launch())
        self.assertEqual(q.head().batch, "b0")
        self.assertEqual(q.newest().batch, "b1")

    def test_drain_takes_every_record_oldest_first(self):
        q = InflightQueue(depth=1)
        q.push("b0", "r0")
        q.end_step("b0")
        q.push("b1", "r1")
        self.assertEqual([record.batch for record in q.drain()], ["b0", "b1"])
        self.assertEqual(len(q), 0)
        self.assertIsNone(q.newest())
        self.assertIsNone(q.last_launched)


if __name__ == "__main__":
    unittest.main()
