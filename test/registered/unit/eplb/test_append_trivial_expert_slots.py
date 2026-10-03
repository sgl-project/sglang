"""Unit tests for append_trivial_expert_slots — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location import append_trivial_expert_slots
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 4
NUM_LOGICAL = 8


def _base_map(width: int) -> torch.Tensor:
    return torch.arange(0, width).repeat(NUM_LAYERS, 1) % NUM_LOGICAL


class TestAppendTrivialExpertSlots(CustomTestCase):
    """The widening helper, and the one shape it silently declines to produce.

    It only appends, so a caller that lays the map out at one width and asks this to
    reach a smaller one gets the original back with no error. That is how a cohort
    narrower than the launch width ended up holding a map that disagreed with its own
    ep_size, surfacing far away in a tensor view inside the expert map store copy.
    """

    def test_negative_count_returns_the_map_unchanged(self):
        """Narrowing is not offered. Pinned because the caller must not rely on it."""
        base = _base_map(16)
        out = append_trivial_expert_slots(base, -8, NUM_LOGICAL)
        self.assertIs(out, base)
        self.assertEqual(out.shape, (NUM_LAYERS, 16))

    def test_build_then_append_reaches_every_target_width(self):
        """The caller's contract: pick the narrower width first, then append.

        Covers narrowing, exact and widening against a fixed base, which is the
        invariant init_trivial now holds and did not before.
        """
        base_width = 32
        for target in (8, 16, 31, 32, 33, 48):
            with self.subTest(target=target):
                built = _base_map(min(base_width, target))
                out = append_trivial_expert_slots(
                    built, target - built.shape[-1], NUM_LOGICAL
                )
                self.assertEqual(out.shape, (NUM_LAYERS, target))
                self.assertTrue(bool((out >= 0).all()))
                self.assertTrue(bool((out < NUM_LOGICAL).all()))


if __name__ == "__main__":
    unittest.main()
