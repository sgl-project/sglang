"""Unit tests for the elastic scale-down FSM — no server, no model loading.

The machine is driven entirely through the driver protocol, so a recording stub is
enough to assert the ordering guarantees the real driver depends on.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.elastic_ep.scale_down_state import (
    ScaleDownState,
    ScaleDownStateMachine,
)
from sglang.test.test_utils import CustomTestCase


class RecordingDriver:
    """Driver stub that records calls and lets each gate be opened on demand."""

    def __init__(self, *, idle=True, barrier_ready=True, departure_ready=True):
        self.calls = []
        self.idle = idle
        self.barrier_ready = barrier_ready
        self.departure_ready = departure_ready
        self.consumed = []
        self._handle_seq = 0

    def _record(self, name):
        self.calls.append(name)

    def _new_handle(self, tag):
        self._handle_seq += 1
        return f"{tag}-{self._handle_seq}"

    def on_prepare(self, sm):
        self._record("on_prepare")

    def local_idle(self, sm):
        self._record("local_idle")
        return self.idle

    def post_drain_barrier(self, sm):
        self._record("post_drain_barrier")
        return self._new_handle("drain")

    def announce_departure(self, sm):
        self._record("announce_departure")

    def departure_cleared(self, sm):
        return self.departure_ready

    def on_depart_drain(self, sm):
        self._record("on_depart_drain")

    def check_barrier(self, handle, *, block_s=None, keep_serving=False):
        return self.barrier_ready

    def consume_barrier(self, handle):
        self.consumed.append(handle)

    def on_retiree_quiesce(self, sm):
        self._record("on_retiree_quiesce")

    def on_nixl_retire_pre(self, sm):
        self._record("on_nixl_retire_pre")

    def post_nixl_retire_barrier(self, sm):
        self._record("post_nixl_retire_barrier")
        return self._new_handle("nixl")

    def on_flip_mask(self, sm):
        self._record("on_flip_mask")

    def on_reconfig(self, sm):
        self._record("on_reconfig")

    def on_local_cleanup(self, sm):
        self._record("on_local_cleanup")

    def on_exit(self, sm):
        self._record("on_exit")


def make_sm(is_retiree):
    return ScaleDownStateMachine(
        is_retiree=is_retiree,
        target_size=3,
        effective_size=4,
        ranks_to_retire=[3],
        my_global_rank=3 if is_retiree else 0,
    )


def run_to_terminal(sm, driver, max_ticks=20):
    for _ in range(max_ticks):
        if sm.is_terminal():
            return
        sm.tick(driver)


class TestScaleDownStateMachine(CustomTestCase):
    def test_survivor_reaches_complete_and_reconfigures(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        run_to_terminal(sm, driver)
        self.assertEqual(sm.state, ScaleDownState.COMPLETE)
        self.assertIn("on_reconfig", driver.calls)
        # A survivor never runs retiree-only side effects.
        self.assertNotIn("on_local_cleanup", driver.calls)
        self.assertNotIn("on_exit", driver.calls)

    def test_retiree_reaches_exit_after_cleanup(self):
        sm, driver = make_sm(is_retiree=True), RecordingDriver()
        run_to_terminal(sm, driver)
        self.assertEqual(sm.state, ScaleDownState.EXIT)
        self.assertLess(
            driver.calls.index("on_local_cleanup"), driver.calls.index("on_exit")
        )
        self.assertNotIn("on_reconfig", driver.calls)

    def test_mask_flips_only_after_the_nixl_barrier_is_consumed(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        run_to_terminal(sm, driver)
        # The flip narrows the device-side expert bound, so every peer must have
        # crossed the NIXL barrier before it happens.
        self.assertLess(
            driver.calls.index("post_nixl_retire_barrier"),
            driver.calls.index("on_flip_mask"),
        )
        self.assertLess(
            driver.calls.index("on_flip_mask"), driver.calls.index("on_reconfig")
        )

    def test_busy_retiree_does_not_post_the_drain_barrier(self):
        sm, driver = make_sm(is_retiree=True), RecordingDriver(idle=False)
        for _ in range(5):
            sm.tick(driver)
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        self.assertNotIn("post_drain_barrier", driver.calls)
        self.assertFalse(sm.is_terminal())

    def test_busy_survivor_still_posts_the_drain_barrier(self):
        # Only a retiree's own queue strands work; a survivor keeps serving.
        sm, driver = make_sm(is_retiree=False), RecordingDriver(idle=False)
        run_to_terminal(sm, driver)
        self.assertEqual(sm.state, ScaleDownState.COMPLETE)

    def test_retiree_rechecks_idle_before_announcing_departure(self):
        """Work admitted while the barrier was in flight must hold the departure."""
        sm, driver = make_sm(is_retiree=True), RecordingDriver()
        sm.tick(driver)  # PREPARE -> DRAIN
        sm.tick(driver)  # posts the drain barrier (idle at this point)
        self.assertIn("post_drain_barrier", driver.calls)

        # A request lands while the rank waits on the barrier.
        driver.idle = False
        sm.tick(driver)  # barrier completes
        sm.tick(driver)
        self.assertNotIn("announce_departure", driver.calls)
        self.assertEqual(sm.state, ScaleDownState.DRAIN)

        # Once it drains again, departure proceeds.
        driver.idle = True
        run_to_terminal(sm, driver)
        self.assertIn("announce_departure", driver.calls)
        self.assertEqual(sm.state, ScaleDownState.EXIT)

    def test_departure_is_announced_once(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        run_to_terminal(sm, driver)
        self.assertEqual(driver.calls.count("announce_departure"), 1)

    def test_drain_barrier_is_posted_once_per_cycle(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver(barrier_ready=False)
        for _ in range(6):
            sm.tick(driver)
        # Re-posting would double-count arrivals and let a subset pass.
        self.assertEqual(driver.calls.count("post_drain_barrier"), 1)

    def test_barrier_that_never_arms_fails_instead_of_flipping(self):
        sm = make_sm(is_retiree=False)
        sm.BARRIER_POST_MAX_ATTEMPTS = 2
        driver = RecordingDriver(barrier_ready=False)
        sm.tick(driver)  # PREPARE -> DRAIN
        # Force a re-post on each tick to exhaust the bounded attempts.
        for _ in range(5):
            sm._drain_barrier_handle = None
            sm.tick(driver)
        self.assertTrue(sm.is_failed())
        self.assertIn("failed to arm", sm.last_error)
        self.assertNotIn("on_flip_mask", driver.calls)

    def test_abandon_releases_a_held_barrier(self):
        """A leaked barrier leader wedges every later scale."""
        sm, driver = make_sm(is_retiree=False), RecordingDriver(barrier_ready=False)
        sm.tick(driver)
        sm.tick(driver)
        held = sm._drain_barrier_handle
        self.assertIsNotNone(held)

        sm.abandon(driver)
        self.assertEqual(driver.consumed, [held])
        self.assertIsNone(sm._drain_barrier_handle)

    def test_abandon_is_idempotent_and_safe_with_nothing_held(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        sm.abandon(driver)
        sm.abandon(driver)
        self.assertEqual(driver.consumed, [])

    def test_abandon_survives_a_failing_consume(self):
        class Boom(RecordingDriver):
            def consume_barrier(self, handle):
                raise RuntimeError("store is gone")

        sm, driver = make_sm(is_retiree=False), Boom(barrier_ready=False)
        sm.tick(driver)
        sm.tick(driver)
        sm.abandon(driver)  # must not raise during teardown
        self.assertIsNone(sm._drain_barrier_handle)

    def test_terminal_state_ignores_further_ticks(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver()
        run_to_terminal(sm, driver)
        before = list(driver.calls)
        sm.tick(driver)
        self.assertEqual(driver.calls, before)

    def test_driver_exception_is_captured_as_failure(self):
        class Boom(RecordingDriver):
            def on_prepare(self, sm):
                raise ValueError("prepare exploded")

        sm, driver = make_sm(is_retiree=False), Boom()
        sm.tick(driver)
        self.assertTrue(sm.is_failed())
        self.assertIn("prepare exploded", sm.last_error)
        self.assertTrue(sm.is_terminal())

    def test_departure_not_cleared_holds_the_cohort_in_drain(self):
        sm, driver = make_sm(is_retiree=False), RecordingDriver(departure_ready=False)
        for _ in range(6):
            sm.tick(driver)
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        self.assertNotIn("on_depart_drain", driver.calls)


if __name__ == "__main__":
    unittest.main()
