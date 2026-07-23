import math
import unittest

from sglang.srt.speculative.spec_accept_metrics import SpecAcceptMetrics
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpecAcceptMetrics(unittest.TestCase):
    def test_per_position_rates_and_mean_accept_length(self):
        metrics = SpecAcceptMetrics()
        metrics.observe([0, 1, 3], num_draft_tokens=4)

        snapshot = metrics.snapshot_and_reset_window()

        self.assertEqual(snapshot.window.num_requests, 3)
        self.assertEqual(snapshot.window.accept_rates, (2 / 3, 1 / 2, 1.0))
        self.assertAlmostEqual(snapshot.window.mean_accept_length, 7 / 3)
        self.assertEqual(snapshot.window, snapshot.lifetime)

    def test_smaller_adaptive_budget_is_censored_at_deeper_positions(self):
        metrics = SpecAcceptMetrics()
        # All three positions were eligible and accepted in the first batch.
        metrics.observe([3], num_draft_tokens=4)
        metrics.snapshot_and_reset_window()

        # The next batch only proposes position 0.  It must not lower the
        # lifetime rate at positions 1 and 2.
        metrics.observe([1], num_draft_tokens=2)
        snapshot = metrics.snapshot_and_reset_window()

        self.assertEqual(snapshot.lifetime.accept_rates, (1.0, 1.0, 1.0))
        self.assertEqual(snapshot.window.accept_rates[0], 1.0)
        self.assertTrue(math.isnan(snapshot.window.accept_rates[1]))
        self.assertTrue(math.isnan(snapshot.window.accept_rates[2]))
        self.assertEqual(snapshot.lifetime.mean_accept_length, 3.0)

    def test_rejection_updates_only_positions_that_were_eligible(self):
        metrics = SpecAcceptMetrics()
        metrics.observe([0, 1, 2], num_draft_tokens=4)

        rates = metrics.snapshot_and_reset_window().window.accept_rates

        # position 0: 2 / 3; position 1: 1 / 2; position 2: 0 / 1.
        self.assertAlmostEqual(rates[0], 2 / 3)
        self.assertAlmostEqual(rates[1], 1 / 2)
        self.assertEqual(rates[2], 0.0)

    def test_window_resets_but_lifetime_persists(self):
        metrics = SpecAcceptMetrics()
        metrics.observe([1, 1], num_draft_tokens=2)
        first = metrics.snapshot_and_reset_window()
        self.assertEqual(first.window.num_requests, 2)

        metrics.observe([0], num_draft_tokens=2)
        second = metrics.snapshot_and_reset_window()

        self.assertEqual(second.window.num_requests, 1)
        self.assertEqual(second.window.accept_rates, (0.0,))
        self.assertEqual(second.lifetime.num_requests, 3)
        self.assertAlmostEqual(second.lifetime.accept_rates[0], 2 / 3)

    def test_reset_clears_window_and_lifetime(self):
        metrics = SpecAcceptMetrics()
        metrics.observe([1], num_draft_tokens=2)
        metrics.reset()

        snapshot = metrics.snapshot_and_reset_window()
        self.assertEqual(snapshot.window.num_requests, 0)
        self.assertEqual(snapshot.lifetime.num_requests, 0)
        self.assertEqual(snapshot.window.accept_rates, ())
        self.assertEqual(snapshot.lifetime.accept_rates, ())

    def test_missing_budget_is_ignored(self):
        metrics = SpecAcceptMetrics()
        metrics.observe([1], num_draft_tokens=None)

        snapshot = metrics.snapshot_and_reset_window()
        self.assertEqual(snapshot.window.num_requests, 0)

    def test_bonus_only_round_contributes_to_accept_length(self):
        metrics = SpecAcceptMetrics()
        metrics.observe([0, 0], num_draft_tokens=1)

        snapshot = metrics.snapshot_and_reset_window()
        self.assertEqual(snapshot.window.accept_rates, ())
        self.assertEqual(snapshot.window.num_requests, 2)
        self.assertEqual(snapshot.window.mean_accept_length, 1.0)


if __name__ == "__main__":
    unittest.main()
