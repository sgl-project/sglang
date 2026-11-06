"""Unit tests for the elastic scale-down FSM, driven through a recording driver stub."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from sglang.srt.elastic_ep.scale_down_state import (
    ScaleDownState,
    ScaleDownStateMachine,
)
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import RecordingDriver  # noqa: E402


def make_sm(is_retiree):
    return ScaleDownStateMachine(
        is_retiree=is_retiree,
        target_size=3,
        effective_size=4,
        ranks_to_retire=[3],
        my_global_rank=3 if is_retiree else 0,
    )


def run(sm, driver, max_ticks=20):
    """Tick until terminal, at most ``max_ticks`` times."""
    for _ in range(max_ticks):
        if sm.is_terminal():
            return
        sm.tick(driver)


class TestScaleDownStateMachine(CustomTestCase):
    def test_survivor_flips_after_the_nixl_barrier_and_reconfigures(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        run(sm, driver)
        self.assertEqual(sm.state, ScaleDownState.COMPLETE)
        calls = list(driver.calls)
        # The flip narrows the device-side expert bound, so every peer must have
        # crossed the NIXL barrier before it happens.
        self.assertLess(
            calls.index("post_nixl_retire_barrier"), calls.index("on_flip_mask")
        )
        self.assertLess(calls.index("on_flip_mask"), calls.index("on_reconfig"))
        self.assertEqual(calls.count("announce_departure"), 1)
        # A survivor never runs retiree-only side effects.
        self.assertNotIn("on_local_cleanup", calls)
        self.assertNotIn("on_exit", calls)
        sm.tick(driver)  # a terminal state ignores further ticks
        self.assertEqual(driver.calls, calls)

    def test_retiree_reaches_exit_after_cleanup(self):
        sm, driver = make_sm(is_retiree=True), RecordingDriver()
        run(sm, driver)
        self.assertEqual(sm.state, ScaleDownState.EXIT)
        self.assertLess(
            driver.calls.index("on_local_cleanup"), driver.calls.index("on_exit")
        )
        self.assertNotIn("on_reconfig", driver.calls)

    def test_busy_retiree_does_not_post_the_drain_barrier(self):
        sm, driver = make_sm(is_retiree=True), RecordingDriver(idle=False)
        run(sm, driver, 5)
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        self.assertNotIn("post_drain_barrier", driver.calls)
        self.assertFalse(sm.is_terminal())

    def test_busy_survivor_still_posts_the_drain_barrier(self):
        # Only a retiree's own queue strands work; a survivor keeps serving.
        sm, driver = make_sm(is_retiree=False), RecordingDriver(idle=False)
        run(sm, driver)
        self.assertEqual(sm.state, ScaleDownState.COMPLETE)

    def test_retiree_rechecks_idle_before_announcing_departure(self):
        """Work admitted while the barrier was in flight must hold the departure."""
        sm, driver = make_sm(is_retiree=True), RecordingDriver()
        run(sm, driver, 2)  # PREPARE -> DRAIN, then posts the barrier while idle
        self.assertIn("post_drain_barrier", driver.calls)
        driver.idle = False  # a request lands while the rank waits on the barrier
        run(sm, driver, 2)  # the barrier completes
        self.assertNotIn("announce_departure", driver.calls)
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        driver.idle = True
        run(sm, driver)
        self.assertIn("announce_departure", driver.calls)
        self.assertEqual(sm.state, ScaleDownState.EXIT)

    def test_drain_barrier_is_posted_once_per_cycle(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver(barrier_ready=False)
        run(sm, driver, 6)
        # Re-posting would double-count arrivals and let a subset pass.
        self.assertEqual(driver.calls.count("post_drain_barrier"), 1)

    def test_barrier_that_never_arms_fails_instead_of_flipping(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver(barrier_ready=False)
        sm.BARRIER_POST_MAX_ATTEMPTS = 2
        sm.tick(driver)  # PREPARE -> DRAIN
        for _ in range(5):  # force a re-post each tick to exhaust the attempts
            sm._drain_barrier_handle = None
            sm.tick(driver)
        self.assertTrue(sm.is_failed())
        self.assertIn("failed to arm", sm.last_error)
        self.assertNotIn("on_flip_mask", driver.calls)

    def test_abandon_releases_a_held_barrier(self):
        """A leaked barrier leader wedges every later scale. Idempotent."""
        sm, driver = make_sm(is_retiree=False), RecordingDriver(barrier_ready=False)
        make_sm(is_retiree=False).abandon(driver)  # nothing held: no release
        self.assertEqual(driver.consumed, [])
        run(sm, driver, 2)
        held = sm._drain_barrier_handle
        self.assertIsNotNone(held)
        sm.abandon(driver)
        sm.abandon(driver)
        self.assertEqual(driver.consumed, [held])
        self.assertIsNone(sm._drain_barrier_handle)

    def test_abandon_survives_a_failing_consume(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver(barrier_ready=False)
        driver.consume_barrier = Mock(side_effect=RuntimeError("store is gone"))
        run(sm, driver, 2)
        sm.abandon(driver)  # must not raise during teardown
        self.assertIsNone(sm._drain_barrier_handle)

    def test_driver_exception_is_captured_as_failure(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        driver.on_prepare = Mock(side_effect=ValueError("prepare exploded"))
        sm.tick(driver)
        self.assertTrue(sm.is_failed())
        self.assertIn("prepare exploded", sm.last_error)
        self.assertTrue(sm.is_terminal())

    def test_departure_not_cleared_holds_the_cohort_in_drain(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver(departure_ready=False)
        run(sm, driver, 6)
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        self.assertNotIn("on_depart_drain", driver.calls)


if __name__ == "__main__":
    unittest.main()
