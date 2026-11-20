"""Unit tests for append_trivial_expert_slots — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location import (
    append_trivial_expert_slots,
    build_trivial_physical_to_logical_map,
)
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 4
NUM_LOGICAL = 8


class TestAppendTrivialExpertSlots(CustomTestCase):
    def test_negative_count_returns_the_map_unchanged(self):
        """Only appends: narrowing is not offered and the caller must not rely on it."""
        base = torch.arange(0, 16).repeat(NUM_LAYERS, 1) % NUM_LOGICAL
        out = append_trivial_expert_slots(base, -8, NUM_LOGICAL)
        self.assertIs(out, base)
        self.assertEqual(out.shape, (NUM_LAYERS, 16))

    def test_build_reaches_every_target_width(self):
        """init_trivial's layout, called not re-implemented: narrower, equal, wider."""
        for target in (8, 16, 31, 32, 33, 48):
            with self.subTest(target=target):
                out = build_trivial_physical_to_logical_map(
                    base_num_physical_experts=32,
                    num_physical_experts=target,
                    num_layers=NUM_LAYERS,
                    num_logical_experts=NUM_LOGICAL,
                )
                self.assertEqual(out.shape, (NUM_LAYERS, target))
                self.assertTrue(bool(((out >= 0) & (out < NUM_LOGICAL)).all()))


if __name__ == "__main__":
    unittest.main()
