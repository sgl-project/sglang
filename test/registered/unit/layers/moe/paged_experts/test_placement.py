"""Unit tests for srt/layers/moe/paged_experts/placement.py (decode-placement selection; no GPU)."""

import unittest

from sglang.srt.layers.moe.paged_experts.placement import (
    CapturedPlacement,
    CapturedWindowedBCGPlacement,
    CapturedWindowedPlacement,
    EagerPlacement,
    Placement,
    make_placement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlacement(CustomTestCase):
    def test_factory_selects_placement(self):
        captured = make_placement(use_ondevice=True)
        eager = make_placement(use_ondevice=False)
        self.assertIsInstance(captured, CapturedPlacement)
        self.assertIsInstance(eager, EagerPlacement)
        self.assertIsInstance(captured, Placement)
        self.assertIsInstance(eager, Placement)

    def test_factory_selects_windowed_placement(self):
        # A >pin-ceiling (windowed) store picks the replay-twice variant on-device, or the
        # break-and-page-in (BCG) variant when decode runs under the breakable backend.
        replay_twice = make_placement(True, windowed=True, breakable_decode=False)
        bcg = make_placement(True, windowed=True, breakable_decode=True)
        self.assertIsInstance(replay_twice, CapturedWindowedPlacement)
        self.assertIsInstance(bcg, CapturedWindowedBCGPlacement)
        # windowed is only meaningful on the captured path; eager ignores it.
        self.assertIsInstance(
            make_placement(False, windowed=True, breakable_decode=True), EagerPlacement
        )

    def test_only_captured_needs_ondevice_store(self):
        # setup_pager allocates the on-device residency state iff the placement needs it.
        self.assertTrue(make_placement(True).needs_ondevice_store)
        self.assertTrue(make_placement(True, windowed=True).needs_ondevice_store)
        self.assertTrue(
            make_placement(
                True, windowed=True, breakable_decode=True
            ).needs_ondevice_store
        )
        self.assertFalse(make_placement(False).needs_ondevice_store)

    def test_placement_is_abstract(self):
        with self.assertRaises(TypeError):
            Placement()  # apply is the contract subclasses must implement


if __name__ == "__main__":
    unittest.main()
