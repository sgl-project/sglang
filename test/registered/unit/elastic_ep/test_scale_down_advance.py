"""Unit tests for the per-tick scale-down coordinator — no server, no ModelRunner."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from sglang.srt.elastic_ep.scale_down_driver import ScaleDownDriver, advance_scale_down
from sglang.srt.elastic_ep.scale_down_state import ScaleDownState
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import RecordingDriver  # noqa: E402

PENDING_SIZE = 3
EFFECTIVE_SIZE = 4
RETIREE_RANK = 3
SURVIVOR_RANK = 0


def advance(sm, driver, failures=None, *, timed_out=False):
    failures = [] if failures is None else failures
    return advance_scale_down(
        sm=sm,
        driver=driver,
        pending_size=PENDING_SIZE,
        effective_size=EFFECTIVE_SIZE,
        pending_since=0.0 if timed_out else time.monotonic(),
        my_global_rank=SURVIVOR_RANK,
        scale_timeout=1.0 if timed_out else 60.0,
        fail_scale=lambda error, effective_size: failures.append(error),
    )


class TestAdvanceScaleDown(CustomTestCase):
    def test_hands_back_the_same_machine_until_it_is_terminal(self):
        driver = RecordingDriver()
        sm = advance(None, driver)
        # Retiring the tail slots is derived from the width pair, not passed in.
        self.assertEqual(sm.ranks_to_retire, [RETIREE_RANK])
        self.assertFalse(sm.is_retiree)
        for _ in range(20):
            nxt = advance(sm, driver)
            if nxt is None:
                break
            self.assertIs(nxt, sm)
        # None means the caller drops it, and the survivor got its reconfig.
        self.assertIsNone(nxt)
        self.assertEqual(sm.state, ScaleDownState.COMPLETE)
        self.assertIn("on_reconfig", driver.calls)

    def test_timeout_fails_the_scale_and_releases_a_held_barrier(self):
        driver, failures = RecordingDriver(barrier_ready=False), []
        sm = advance(None, driver)
        for _ in range(3):
            sm = advance(sm, driver)
        # Parked in DRAIN with the barrier it posted still outstanding.
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        self.assertEqual(driver.consumed, [])
        self.assertIsNone(advance(sm, driver, failures, timed_out=True))
        self.assertEqual(len(failures), 1)
        self.assertIn("retire barrier", failures[0])
        # Abandoned, not leaked: the epoch this rank posted is reset.
        self.assertEqual(len(driver.consumed), 1)

    def test_a_failure_before_any_machine_exists_still_fails_the_scale(self):
        """A timeout, or a driver exception: FAILED is terminal, so dropped at once."""
        failing = RecordingDriver()
        failing.on_prepare = Mock(side_effect=RuntimeError("prepare blew up"))
        for driver, timed_out, error in (
            (RecordingDriver(), True, "retire barrier"),
            (failing, False, "prepare blew up"),
        ):
            with self.subTest(error=error):
                failures = []
                self.assertIsNone(advance(None, driver, failures, timed_out=timed_out))
                self.assertEqual(len(failures), 1)
                self.assertIn(error, failures[0])


class TestScaleDownDriverNeedsNoRunner(CustomTestCase):
    def test_the_two_runner_tails_are_plain_callables(self):
        calls = []
        driver = ScaleDownDriver(
            lambda **kw: calls.append(("finalize", kw)),
            lambda: calls.append(("retire", {})),
        )
        sm = advance(None, RecordingDriver())
        driver.on_reconfig(sm)
        driver.on_exit(sm)
        self.assertEqual([name for name, _ in calls], ["finalize", "retire"])
        self.assertEqual(
            calls[0][1],
            {
                "ranks_to_retire": [RETIREE_RANK],
                "target_size": PENDING_SIZE,
                "effective_size": EFFECTIVE_SIZE,
            },
        )

    def test_an_unwired_idle_predicate_reads_as_idle(self):
        # A runner driving the FSM from forward() has no scheduler to ask, and a
        # retiree that reported busy forever would hold the whole cohort.
        unwired = ScaleDownDriver(lambda **kw: None, lambda: None)
        self.assertTrue(unwired.local_idle(sm=None))
        busy = ScaleDownDriver(lambda **kw: None, lambda: None, lambda: False)
        self.assertFalse(busy.local_idle(sm=None))


if __name__ == "__main__":
    unittest.main()
