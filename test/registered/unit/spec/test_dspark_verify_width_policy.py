import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable
from sglang.srt.speculative.dspark_components.dspark_verify_width import (
    VerifyWidthPolicy,
    parse_verify_widths,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FULL = 6
# DeepSeek-V4.1-Flash DSpark target verify on MI355X x4 (TP4/EP4, 4K prompts).
TABLE = SpsCostTable(
    sample_batch_tokens=[
        6,
        12,
        24,
        48,
        96,
        144,
        192,
        240,
        288,
        336,
        384,
        480,
        576,
        672,
        768,
        960,
        1152,
        1344,
        1536,
    ],
    sample_steps_per_sec=[
        94.4,
        82.3,
        74.3,
        63.7,
        52.4,
        43.5,
        39.7,
        37.1,
        33.3,
        31.2,
        29.8,
        25.5,
        22.8,
        20.8,
        19.5,
        16.3,
        14.8,
        13.2,
        12.2,
    ],
    max_batch_tokens=1536,
)


def _delta(policy, width, correct_counts):
    delta = torch.zeros((len(policy.widths), FULL), dtype=torch.int64)
    row = policy.widths.index(width)
    delta[row, : len(correct_counts)] = torch.tensor(correct_counts)
    return delta


class TestVerifyWidthPolicy(CustomTestCase):
    def test_hazards_from_censored_widths(self):
        """A narrow width observes only its own drafts; deeper positions keep the
        estimate measured at full width."""
        policy = VerifyWidthPolicy(widths=[3], full_width=FULL, sps_table=TABLE)
        # 100 full-width requests: 20 with 0 correct drafts, then 20 each for 1..4.
        policy.update(_delta(policy, FULL, [20, 20, 20, 20, 20]))
        full = policy.hazards()
        self.assertAlmostEqual(full[0], 80 / 100)
        self.assertAlmostEqual(full[3], 20 / 40)
        # Width 3 verifies drafts 0 and 1: all 50 requests pass draft 0, none draft 1.
        policy.update(_delta(policy, 3, [0, 50, 0]))
        after = policy.hazards()
        self.assertGreater(after[0], full[0])
        self.assertLess(after[1], full[1])
        self.assertEqual(after[2:], full[2:])

    def test_full_width_until_every_position_is_measured(self):
        policy = VerifyWidthPolicy(widths=[3, 4], full_width=FULL, sps_table=TABLE)
        self.assertEqual(policy.choose(256), FULL)
        policy.update(_delta(policy, 3, [10, 10, 10]))
        self.assertEqual(policy.choose(256), FULL)

    def test_width_shrinks_with_batch_size(self):
        """Per-position acceptance of the measured V4.1 sweep; the policy must pick
        the full width at small batches and a narrow one at large batches."""
        # Correct-draft histogram whose mean accept length is ~3.25 at width 6.
        counts = [220, 200, 170, 130, 110, 170]
        results = {}
        for bs in (1, 8, 256):
            policy = VerifyWidthPolicy(
                widths=[3, 4], full_width=FULL, sps_table=TABLE, margin=0.0
            )
            policy.update(_delta(policy, FULL, counts))
            results[bs] = policy.choose(bs)
        self.assertEqual(results[1], FULL)
        self.assertEqual(results[8], FULL)
        self.assertLess(results[256], FULL)

    def test_margin_holds_the_current_width(self):
        policy = VerifyWidthPolicy(
            widths=[3, 4], full_width=FULL, sps_table=TABLE, margin=10.0
        )
        policy.update(_delta(policy, FULL, [220, 200, 170, 130, 110, 170]))
        self.assertEqual(policy.choose(256), FULL)

    def test_forced_width(self):
        policy = VerifyWidthPolicy(
            widths=[3, 4], full_width=FULL, sps_table=TABLE, forced_width=3
        )
        self.assertEqual(policy.choose(1), 3)
        with self.assertRaises(ValueError):
            VerifyWidthPolicy(
                widths=[3], full_width=FULL, sps_table=TABLE, forced_width=5
            )

    def test_parse_rejects_widths_outside_the_draft_block(self):
        self.assertEqual(parse_verify_widths(("4", "3", "4"), full_width=FULL), [3, 4])
        for bad in (("1",), ("6",), ("7",)):
            with self.assertRaises(ValueError):
                parse_verify_widths(bad, full_width=FULL)


if __name__ == "__main__":
    unittest.main()
