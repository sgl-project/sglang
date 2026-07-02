"""Unit tests for srt/layers/moe/paged_experts/placement.py (decode-placement selection; no GPU)."""

import unittest

from sglang.srt.layers.moe.paged_experts.placement import (
    CapturedPlacement,
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

    def test_only_captured_needs_ondevice_store(self):
        # setup_pager allocates the on-device residency state iff the placement needs it.
        self.assertTrue(make_placement(True).needs_ondevice_store)
        self.assertFalse(make_placement(False).needs_ondevice_store)

    def test_placement_is_abstract(self):
        with self.assertRaises(TypeError):
            Placement()  # apply is the contract subclasses must implement


if __name__ == "__main__":
    unittest.main()
