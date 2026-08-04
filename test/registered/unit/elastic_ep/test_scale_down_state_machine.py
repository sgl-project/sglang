"""Unit tests: ScaleDownStateMachine state transitions.

Validates the finite-state-machine that drives Mooncake-native
scale-DOWN across scheduler ticks. Uses an in-memory driver stub so we
can exercise the transitions without any distributed backend.
"""

from __future__ import annotations

import unittest

from sglang.srt.elastic_ep.scale_down_state import (
    ScaleDownRetireeState,
    ScaleDownStateMachine,
    ScaleDownStateMachineDriver,
    ScaleDownSurvivorState,
)


class _RecordingDriver(ScaleDownStateMachineDriver):
    """Records callback invocations for FSM assertion tests.

    Barrier post/check/consume return synchronously-ready values by
    default so the happy path collapses into the "check_ok on first
    poll" branch of the FSM, which folds NIXL_RETIRE + FLIP_MASK into
    the same tick that consumes the drain barrier. Individual tests
    can flip ``is_drained_result``, ``drain_check_result``, or
    ``nixl_check_result`` to exercise the wait-for-cohort branches.
    """

    _DRAIN_HANDLE = object()
    _NIXL_HANDLE = object()

    def __init__(
        self,
        is_drained_result: bool = True,
        drain_check_result: bool = True,
        nixl_check_result: bool = True,
    ):
        self.events = []
        self._is_drained_result = is_drained_result
        self._drain_check_result = drain_check_result
        self._nixl_check_result = nixl_check_result
        self.raise_on = None  # optional: event name that should raise

    def _maybe_raise(self, name: str) -> None:
        if self.raise_on == name:
            raise RuntimeError(f"forced failure at {name}")

    def on_prepare(self, sm):
        self.events.append("prepare")
        self._maybe_raise("prepare")

    def is_drained(self, sm):
        return self._is_drained_result

    def post_drain_barrier(self, sm):
        self.events.append("post_drain_barrier")
        self._maybe_raise("post_drain_barrier")
        return self._DRAIN_HANDLE

    def check_drain_barrier(self, handle):
        return self._drain_check_result

    def consume_drain_barrier(self, handle):
        self.events.append("consume_drain_barrier")
        self._maybe_raise("consume_drain_barrier")

    def on_nixl_retire_pre(self, sm):
        self.events.append("nixl_retire_pre")
        self._maybe_raise("nixl_retire_pre")

    def post_nixl_retire_barrier(self, sm):
        self.events.append("post_nixl_retire_barrier")
        self._maybe_raise("post_nixl_retire_barrier")
        return self._NIXL_HANDLE

    def check_nixl_retire_barrier(self, handle):
        return self._nixl_check_result

    def consume_nixl_retire_barrier(self, handle):
        self.events.append("consume_nixl_retire_barrier")
        self._maybe_raise("consume_nixl_retire_barrier")

    def on_flip_mask(self, sm):
        self.events.append("flip_mask")
        self._maybe_raise("flip_mask")

    def on_reconfig(self, sm):
        self.events.append("reconfig")
        self._maybe_raise("reconfig")

    def on_local_cleanup(self, sm):
        self.events.append("local_cleanup")
        self._maybe_raise("local_cleanup")

    def on_exit(self, sm):
        # Real driver never returns from this (``sys.exit`` inside
        # ``ModelRunner._retire_and_exit``); the test stub just records
        # it and lets tick() complete normally.
        self.events.append("exit")
        self._maybe_raise("exit")


class TestSurvivorFsm(unittest.TestCase):
    def _sm(self):
        return ScaleDownStateMachine(
            is_retiree=False,
            target_size=3,
            effective_size=4,
            ranks_to_retire=[3],
            my_global_rank=0,
        )

    def test_full_lifecycle_reaches_complete(self):
        """Happy path: 4 ticks from PREPARE to COMPLETE.

        Tick 1: PREPARE -> DRAIN (calls on_prepare).
        Tick 2: DRAIN, is_drained -> post drain barrier, return.
        Tick 3: DRAIN, drain_check -> consume drain, then fold
                NIXL_RETIRE (post/check/consume + on_nixl_retire_pre)
                + FLIP_MASK (on_flip_mask) into the same tick, exit
                on RECONFIG.
        Tick 4: RECONFIG -> on_reconfig -> COMPLETE.
        """
        sm = self._sm()
        driver = _RecordingDriver()

        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.PREPARE)

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
        self.assertEqual(driver.events, ["prepare"])

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
        self.assertEqual(driver.events, ["prepare", "post_drain_barrier"])

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.RECONFIG)
        self.assertEqual(
            driver.events,
            [
                "prepare",
                "post_drain_barrier",
                "consume_drain_barrier",
                "post_nixl_retire_barrier",
                "consume_nixl_retire_barrier",
                "nixl_retire_pre",
                "flip_mask",
            ],
        )

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.COMPLETE)
        self.assertIn("reconfig", driver.events)
        self.assertTrue(sm.is_terminal())
        self.assertFalse(sm.is_failed())

        events_at_terminal = list(driver.events)
        sm.tick(driver)
        self.assertEqual(
            driver.events,
            events_at_terminal,
            "terminal state must be a no-op",
        )

    def test_drain_gates_barrier_post(self):
        """DRAIN must not post the barrier until ``is_drained`` is True."""
        sm = self._sm()
        driver = _RecordingDriver(is_drained_result=False)

        sm.tick(driver)  # PREPARE -> DRAIN
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)

        for _ in range(5):
            sm.tick(driver)
            self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
            self.assertEqual(driver.events, ["prepare"])
            self.assertNotIn("post_drain_barrier", driver.events)

        driver._is_drained_result = True
        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
        self.assertEqual(driver.events, ["prepare", "post_drain_barrier"])

    def test_drain_barrier_waits_for_check(self):
        """DRAIN posts the barrier but stays put while ``check_drain`` is False.

        This validates the async retire-barrier busy-poll: the FSM does
        not consume + fold into NIXL_RETIRE until every cohort rank
        has posted.
        """
        sm = self._sm()
        driver = _RecordingDriver(drain_check_result=False)

        sm.tick(driver)  # PREPARE -> DRAIN
        sm.tick(driver)  # DRAIN posts barrier
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
        self.assertEqual(driver.events, ["prepare", "post_drain_barrier"])

        for _ in range(3):
            sm.tick(driver)
            self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
            self.assertNotIn("consume_drain_barrier", driver.events)

        driver._drain_check_result = True
        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.RECONFIG)
        self.assertIn("consume_drain_barrier", driver.events)
        self.assertIn("flip_mask", driver.events)

    def test_nixl_retire_stalls_stay_in_nixl_retire(self):
        """When the folded NIXL_RETIRE check fails, the FSM parks in
        NIXL_RETIRE and re-polls without re-posting."""
        sm = self._sm()
        driver = _RecordingDriver(nixl_check_result=False)

        sm.tick(driver)  # PREPARE -> DRAIN
        sm.tick(driver)  # DRAIN posts barrier
        sm.tick(driver)  # DRAIN consumes -> NIXL_RETIRE (barrier posted, check False)
        self.assertEqual(
            sm.survivor_state, ScaleDownSurvivorState.NIXL_RETIRE
        )
        self.assertEqual(
            driver.events.count("post_nixl_retire_barrier"),
            1,
            "barrier must not be re-posted while check is stalled",
        )

        for _ in range(3):
            sm.tick(driver)
            self.assertEqual(
                sm.survivor_state, ScaleDownSurvivorState.NIXL_RETIRE
            )
            self.assertEqual(
                driver.events.count("post_nixl_retire_barrier"), 1
            )

        driver._nixl_check_result = True
        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.RECONFIG)

    def test_failure_in_reconfig_marks_failed(self):
        sm = self._sm()
        driver = _RecordingDriver()
        driver.raise_on = "reconfig"

        sm.tick(driver)  # PREPARE -> DRAIN
        sm.tick(driver)  # DRAIN posts barrier
        sm.tick(driver)  # DRAIN -> NIXL_RETIRE -> FLIP_MASK -> RECONFIG (folded)
        sm.tick(driver)  # RECONFIG -> raises
        self.assertTrue(sm.is_terminal())
        self.assertTrue(sm.is_failed())
        self.assertIn("forced failure at reconfig", sm.last_error or "")


class TestRetireeFsm(unittest.TestCase):
    def _sm(self):
        return ScaleDownStateMachine(
            is_retiree=True,
            target_size=3,
            effective_size=4,
            ranks_to_retire=[3],
            my_global_rank=3,
        )

    def test_full_lifecycle_reaches_exit(self):
        """Retiree happy path: DRAIN -> NIXL_RETIRE -> FLIP_MASK all
        folded, then LOCAL_CLEANUP -> EXIT.

        Retiree does NOT call ``on_nixl_retire_pre`` (survivor-side
        NIXL peer disconnect). Retiree's ``on_exit`` hook fires on the
        same tick that runs ``on_local_cleanup``.
        """
        sm = self._sm()
        driver = _RecordingDriver()

        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.PREPARE)

        sm.tick(driver)
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.DRAIN)

        sm.tick(driver)  # DRAIN posts barrier
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.DRAIN)

        sm.tick(driver)
        self.assertEqual(
            sm.retiree_state, ScaleDownRetireeState.LOCAL_CLEANUP
        )
        self.assertNotIn(
            "nixl_retire_pre",
            driver.events,
            "retirees must not run the survivor-side NIXL disconnect",
        )
        self.assertIn("consume_nixl_retire_barrier", driver.events)
        self.assertIn("flip_mask", driver.events)

        sm.tick(driver)
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.EXIT)
        self.assertIn("local_cleanup", driver.events)
        self.assertIn("exit", driver.events)
        self.assertTrue(sm.is_terminal())

    def test_failure_in_flip_mask_marks_failed(self):
        sm = self._sm()
        driver = _RecordingDriver()
        driver.raise_on = "flip_mask"

        sm.tick(driver)  # PREPARE -> DRAIN
        sm.tick(driver)  # DRAIN posts barrier
        sm.tick(driver)  # DRAIN -> NIXL_RETIRE fold, then flip_mask raises
        self.assertTrue(sm.is_terminal())
        self.assertTrue(sm.is_failed())


class TestFsmObservability(unittest.TestCase):
    def test_phase_name_reports_role_and_stage(self):
        surv = ScaleDownStateMachine(
            is_retiree=False,
            target_size=3,
            effective_size=4,
            ranks_to_retire=[3],
            my_global_rank=0,
        )
        self.assertEqual(surv.phase_name, "survivor:prepare")

        ret = ScaleDownStateMachine(
            is_retiree=True,
            target_size=3,
            effective_size=4,
            ranks_to_retire=[3],
            my_global_rank=3,
        )
        self.assertEqual(ret.phase_name, "retiree:prepare")


if __name__ == "__main__":
    unittest.main(verbosity=2)
