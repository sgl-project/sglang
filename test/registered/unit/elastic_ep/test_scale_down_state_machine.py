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
    """Records callback invocations for FSM assertion tests."""

    def __init__(self, is_drained_result: bool = True):
        self.events = []
        self._is_drained_result = is_drained_result
        self.raise_on = None  # optional: event name that should raise

    def _maybe_raise(self, name: str) -> None:
        if self.raise_on == name:
            raise RuntimeError(f"forced failure at {name}")

    def on_prepare(self, sm):
        self.events.append("prepare")
        self._maybe_raise("prepare")

    def is_drained(self, sm):
        return self._is_drained_result

    def on_drain_complete(self, sm):
        self.events.append("drain_complete")
        self._maybe_raise("drain_complete")

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
        # Real driver never returns from this (sys.exit(0)); the test stub
        # just records it and lets tick() complete normally.
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

    def test_full_lifecycle_advances_one_state_per_tick(self):
        sm = self._sm()
        driver = _RecordingDriver()

        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.PREPARE)

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
        self.assertEqual(driver.events, ["prepare"])

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.FLIP_MASK)
        self.assertEqual(driver.events, ["prepare", "drain_complete"])

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.RECONFIG)
        self.assertEqual(
            driver.events, ["prepare", "drain_complete", "flip_mask"]
        )

        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.COMPLETE)
        self.assertEqual(
            driver.events,
            ["prepare", "drain_complete", "flip_mask", "reconfig"],
        )
        self.assertTrue(sm.is_terminal())
        self.assertFalse(sm.is_failed())

        # Terminal state is a no-op.
        sm.tick(driver)
        self.assertEqual(
            driver.events,
            ["prepare", "drain_complete", "flip_mask", "reconfig"],
        )

    def test_drain_gates_flip_mask(self):
        """DRAIN must stay put until is_drained returns True."""
        sm = self._sm()
        driver = _RecordingDriver(is_drained_result=False)

        sm.tick(driver)  # PREPARE -> DRAIN
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)

        for _ in range(5):
            sm.tick(driver)
            self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.DRAIN)
            self.assertEqual(driver.events, ["prepare"])  # no advance

        driver._is_drained_result = True
        sm.tick(driver)
        self.assertEqual(sm.survivor_state, ScaleDownSurvivorState.FLIP_MASK)
        self.assertEqual(driver.events, ["prepare", "drain_complete"])

    def test_failure_in_reconfig_marks_failed(self):
        sm = self._sm()
        driver = _RecordingDriver()
        driver.raise_on = "reconfig"

        sm.tick(driver)  # PREPARE
        sm.tick(driver)  # DRAIN
        sm.tick(driver)  # FLIP_MASK
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

    def test_full_lifecycle_ends_at_exit(self):
        sm = self._sm()
        driver = _RecordingDriver()

        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.PREPARE)

        sm.tick(driver)
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.DRAIN)

        sm.tick(driver)
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.FLIP_MASK)

        sm.tick(driver)
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.LOCAL_CLEANUP)
        self.assertIn("flip_mask", driver.events)

        # LOCAL_CLEANUP tick fires both local_cleanup and (never-returning
        # in real code) exit hooks. Our stub records both.
        sm.tick(driver)
        self.assertEqual(sm.retiree_state, ScaleDownRetireeState.EXIT)
        self.assertIn("local_cleanup", driver.events)
        self.assertIn("exit", driver.events)
        self.assertTrue(sm.is_terminal())

    def test_failure_in_flip_mask_marks_failed(self):
        sm = self._sm()
        driver = _RecordingDriver()
        driver.raise_on = "flip_mask"

        sm.tick(driver)  # PREPARE
        sm.tick(driver)  # DRAIN
        sm.tick(driver)  # FLIP_MASK -> raises
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
