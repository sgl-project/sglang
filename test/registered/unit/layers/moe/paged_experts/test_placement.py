"""Unit tests for srt/layers/moe/paged_experts/placement.py (decode-placement selection; no GPU)."""

import unittest

from sglang.srt.layers.moe.paged_experts.placement import (
    EagerPlacement,
    Placement,
    make_placement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlacement(CustomTestCase):
    def test_factory_selects_eager(self):
        # Paged Experts is eager-only for now (requires --disable-cuda-graph); the captured on-device
        # placement is a follow-up.
        placement = make_placement()
        self.assertIsInstance(placement, EagerPlacement)
        self.assertIsInstance(placement, Placement)

    def test_placement_is_abstract(self):
        with self.assertRaises(TypeError):
            Placement()  # apply is the contract subclasses must implement


if __name__ == "__main__":
    unittest.main()
