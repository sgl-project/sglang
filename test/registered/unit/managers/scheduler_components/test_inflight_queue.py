import random
import unittest
from collections import deque

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

    def test_step_rule_matches_last_batch_gate(self):
        # Differential against the gate the loops used before the queue: commit
        # the oldest launched batch iff last_batch (what the previous completed
        # step launched) is set. Depth 0 commits in the launching step instead;
        # the overlap loop may commit early before a launch; a retract pause
        # commits what is pending; the PD prefill polling loop may wait a pass
        # for the result without ending the step.
        rng = random.Random(0)
        for _ in range(3000):
            mode = rng.choice(("normal", "overlap", "polling"))
            q = InflightQueue(
                depth=0 if mode == "normal" else 1, blocking=mode != "polling"
            )
            pending = deque()  # launched and not yet committed, oldest first
            last_batch = batch = None
            waiting = False
            for i in range(rng.randint(1, 40)):
                if mode != "normal" and not waiting and rng.random() < 0.1:
                    self.assertEqual(len(q), int(last_batch is not None))
                    drained = [record.batch for record in q.drain()]
                    self.assertEqual(drained, list(pending))
                    pending.clear()
                    last_batch = None
                    continue

                launched = False
                # The polling loop keeps its previous batch when it cannot plan.
                if mode != "polling" or q.can_launch():
                    launched = rng.random() < 0.6
                    batch = f"b{i}" if launched else None
                disable = (
                    mode == "overlap" and launched and pending and rng.random() < 0.3
                )
                if disable:
                    self.assertEqual(q.pop_due().batch, pending.popleft())
                if launched:
                    q.push(batch, None)
                    pending.append(batch)

                if mode == "normal":
                    gate = launched
                else:
                    gate = last_batch is not None and not disable
                self.assertEqual(q.has_due(), gate)
                if gate:
                    if mode == "polling" and rng.random() < 0.5:
                        waiting = True
                        continue
                    self.assertEqual(q.pop_due().batch, pending.popleft())
                last_batch = batch
                q.end_step(batch)
                waiting = False
                self.assertLessEqual(len(q), q.depth)


if __name__ == "__main__":
    unittest.main()
