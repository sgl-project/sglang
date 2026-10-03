"""Unit tests for the per-tick scale-down coordinator — no server, no ModelRunner.

``advance_scale_down`` owns the tick: it builds the machine on the first pending
tick, hands back whatever the caller should keep, and returns None once the cycle is
over. ``ScaleDownDriver`` takes its two runner-owned tails as callables, so both can
be exercised here with plain functions standing in for the runner.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import time
import unittest

from sglang.srt.elastic_ep.scale_down_driver import ScaleDownDriver, advance_scale_down
from sglang.srt.elastic_ep.scale_down_state import ScaleDownState
from sglang.test.test_utils import CustomTestCase

PENDING_SIZE = 3
EFFECTIVE_SIZE = 4
RETIREE_RANK = 3
SURVIVOR_RANK = 0


class StubDriver:
    """Opens every gate, so one advance_scale_down call runs a full stage."""

    def __init__(self):
        self.consumed = []
        self._seq = 0
        self.reconfigured = False
        self.exited = False

    def _handle(self, tag):
        self._seq += 1
        return f"{tag}-{self._seq}"

    def on_prepare(self, sm):
        pass

    def local_idle(self, sm):
        return True

    def post_drain_barrier(self, sm):
        return self._handle("drain")

    def announce_departure(self, sm):
        pass

    def departure_cleared(self, sm):
        return True

    def on_depart_drain(self, sm):
        pass

    def check_barrier(self, handle, *, block_s=None, keep_serving=False):
        return True

    def consume_barrier(self, handle):
        self.consumed.append(handle)

    def on_retiree_quiesce(self, sm):
        pass

    def on_nixl_retire_pre(self, sm):
        pass

    def post_nixl_retire_barrier(self, sm):
        return self._handle("nixl")

    def on_flip_mask(self, sm):
        pass

    def on_reconfig(self, sm):
        self.reconfigured = True

    def on_local_cleanup(self, sm):
        pass

    def on_exit(self, sm):
        self.exited = True


class StuckDriver(StubDriver):
    """Never clears the drain barrier, so the machine stays mid-cycle holding it."""

    def check_barrier(self, handle, *, block_s=None, keep_serving=False):
        return False


class FailingDriver(StubDriver):
    def on_prepare(self, sm):
        raise RuntimeError("prepare blew up")


def advance(sm, driver, *, pending_since=None, scale_timeout=60.0, failures=None):
    def fail_scale(error, effective_size):
        if failures is not None:
            failures.append(error)

    return advance_scale_down(
        sm=sm,
        driver=driver,
        pending_size=PENDING_SIZE,
        effective_size=EFFECTIVE_SIZE,
        pending_since=time.monotonic() if pending_since is None else pending_since,
        my_global_rank=SURVIVOR_RANK,
        scale_timeout=scale_timeout,
        fail_scale=fail_scale,
    )


class TestAdvanceScaleDown(CustomTestCase):
    def test_builds_the_machine_on_the_first_tick(self):
        sm = advance(None, StubDriver())
        self.assertIsNotNone(sm)
        # Retiring the tail slots is derived from the width pair, not passed in.
        self.assertEqual(sm.ranks_to_retire, [RETIREE_RANK])
        self.assertFalse(sm.is_retiree)

    def test_hands_back_the_same_machine_until_it_is_terminal(self):
        driver = StubDriver()
        sm = advance(None, driver)
        for _ in range(20):
            nxt = advance(sm, driver)
            if nxt is None:
                break
            self.assertIs(nxt, sm)
        # None means the caller drops it, and the survivor got its reconfig.
        self.assertIsNone(nxt)
        self.assertEqual(sm.state, ScaleDownState.COMPLETE)
        self.assertTrue(driver.reconfigured)

    def test_timeout_fails_the_scale_and_releases_a_held_barrier(self):
        driver, failures = StuckDriver(), []
        sm = advance(None, driver)
        for _ in range(3):
            sm = advance(sm, driver)
        # Parked in DRAIN with the barrier it posted still outstanding.
        self.assertEqual(sm.state, ScaleDownState.DRAIN)
        self.assertEqual(driver.consumed, [])

        dropped = advance(
            sm, driver, pending_since=0.0, scale_timeout=1.0, failures=failures
        )
        self.assertIsNone(dropped)
        self.assertEqual(len(failures), 1)
        self.assertIn("retire barrier", failures[0])
        # Abandoned, not leaked: the epoch this rank posted is reset.
        self.assertEqual(len(driver.consumed), 1)

    def test_timeout_before_any_machine_exists_is_not_an_error(self):
        failures = []
        self.assertIsNone(
            advance(
                None,
                StubDriver(),
                pending_since=0.0,
                scale_timeout=1.0,
                failures=failures,
            )
        )
        self.assertEqual(len(failures), 1)

    def test_a_driver_exception_surfaces_as_a_failed_scale(self):
        failures = []
        dropped = advance(None, FailingDriver(), failures=failures)
        # FAILED is terminal, so the machine is dropped in the same tick.
        self.assertIsNone(dropped)
        self.assertEqual(len(failures), 1)
        self.assertIn("prepare blew up", failures[0])


class TestScaleDownDriverNeedsNoRunner(CustomTestCase):
    def test_the_two_runner_tails_are_plain_callables(self):
        calls = []
        driver = ScaleDownDriver(
            lambda **kw: calls.append(("finalize", kw)),
            lambda: calls.append(("retire", {})),
        )
        sm = advance(None, StubDriver())
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
