#!/usr/bin/env python3

import unittest

from nightly_skip_policy import allowed_skips


class TestNightlySkipPolicy(unittest.TestCase):
    def test_exact_policy_is_bounded_and_unique(self):
        skips = allowed_skips()
        self.assertEqual(len(skips), 145)
        self.assertEqual(len(set(skips)), len(skips))
        self.assertEqual(
            skips[0],
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-80-False-1]",
        )
        self.assertEqual(
            skips[-1],
            "registered.kernels.ops.diffusion.test_timestep_embedding.test_timestep_embedding_perf",
        )

    def test_rope_policy_only_covers_invalid_head_dimensions(self):
        skips = allowed_skips()
        self.assertEqual(
            sum(
                node.startswith("registered.kernels.ops.attention.test_rope.test_partial_rope[")
                for node in skips
            ),
            144,
        )
        self.assertIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[64-128-True-2049]",
            skips,
        )
        self.assertNotIn(
            "registered.kernels.ops.attention.test_rope.test_partial_rope[128-128-False-1]",
            skips,
        )


if __name__ == "__main__":
    unittest.main()
