import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.acceptance_estimator import (
    MAX_LOG_ODDS,
    DraftTopLogOdds,
)
from sglang.srt.speculative.dspark_components import dspark_config
from sglang.srt.speculative.online_acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _two_token_logits(features: torch.Tensor) -> torch.Tensor:
    """[bs, gamma, 2] logits whose T = 1 top log-odds equal ``features`` (>= 0)."""
    return torch.stack([features, torch.zeros_like(features)], dim=-1)


class TestDraftTopLogOdds(CustomTestCase):
    def test_matches_top_probability_log_odds(self):
        torch.manual_seed(0)
        bs, gamma, vocab = 3, 4, 50
        logits = torch.randn(bs * gamma, vocab) * 3
        temperatures = torch.tensor([0.5, 1.0, 1.7])
        out = DraftTopLogOdds.execute(
            logits=logits, temperatures=temperatures, rows_per_request=gamma
        )
        q = torch.softmax(
            logits.double() / temperatures.double().repeat_interleave(gamma)[:, None],
            dim=-1,
        )
        q_max = q.max(dim=-1).values
        expected = torch.log(q_max / (1 - q_max))
        torch.testing.assert_close(out.double(), expected, rtol=1e-4, atol=1e-4)

    def test_point_mass_row_is_clamped(self):
        logits = torch.full((1, 8), -1e4)
        logits[0, 3] = 0.0
        out = DraftTopLogOdds.execute(
            logits=logits, temperatures=torch.ones(1), rows_per_request=1
        )
        self.assertEqual(float(out), MAX_LOG_ODDS)

    def test_masked_logits(self):
        logits = torch.tensor([[2.0, float("-inf"), 0.0, float("-inf")]])
        out = DraftTopLogOdds.execute(
            logits=logits, temperatures=torch.ones(1), rows_per_request=1
        )
        torch.testing.assert_close(out, torch.tensor([2.0]))


class TestOnlineAcceptanceEstimator(CustomTestCase):
    def _estimator(self, gamma=3):
        return OnlineAcceptanceEstimator(
            gamma=gamma, device=torch.device("cpu"), log_rank=False
        )

    def test_predict_uses_current_coefficients(self):
        est = self._estimator()
        features = torch.tensor([[0.0, 1.0, 4.0], [2.0, 2.0, 0.5]])
        conf = est.predict(
            step_logits=_two_token_logits(features),
            temperatures=torch.ones(2),
            greedy_mask=torch.zeros(2, dtype=torch.bool),
        )
        torch.testing.assert_close(
            conf, torch.sigmoid(0.5 * features - 0.2), rtol=1e-5, atol=1e-5
        )

    def test_greedy_rows_score_the_unit_temperature_distribution(self):
        est = self._estimator(gamma=1)
        features = torch.tensor([[1.5], [1.5]])
        conf = est.predict(
            step_logits=_two_token_logits(features),
            temperatures=torch.tensor([1e-5, 1e-5]),
            greedy_mask=torch.tensor([True, False]),
        )
        self.assertAlmostEqual(
            float(conf[0]), float(torch.sigmoid(torch.tensor(0.55))), places=5
        )
        # The sampled row at T = 1e-5 is a point mass: its log-odds saturate.
        self.assertAlmostEqual(float(conf[1]), 1.0, places=5)

    def test_labels_stop_at_first_rejection_and_graded_rows(self):
        est = self._estimator(gamma=4)
        features = torch.ones(3, 4)
        est.predict(
            step_logits=_two_token_logits(features),
            temperatures=torch.ones(3),
            greedy_mask=torch.zeros(3, dtype=torch.bool),
        )
        est.observe(
            # Row 0 rejects draft 1; row 1 accepts all four; row 2 accepts three
            # but only its first two drafts met a real target row.
            num_correct_drafts=torch.tensor([1, 4, 3]),
            num_graded_drafts=torch.tensor([4, 4, 2]),
        )
        graded = est._stats[:, 5].tolist()
        accepted = est._stats[:, 6].tolist()
        self.assertEqual(graded, [3.0, 3.0, 1.0, 1.0])
        self.assertEqual(accepted, [3.0, 2.0, 1.0, 1.0])

    def test_converges_to_generating_model(self):
        torch.manual_seed(1)
        gamma, bs, steps = 4, 256, 1500
        true_slope = 0.55
        true_intercepts = torch.tensor([0.6, 0.2, -0.3, -0.8])
        est = self._estimator(gamma=gamma)
        for _ in range(steps):
            features = torch.rand(bs, gamma) * 6.0
            est.predict(
                step_logits=_two_token_logits(features),
                temperatures=torch.ones(bs),
                greedy_mask=torch.zeros(bs, dtype=torch.bool),
            )
            accept_prob = torch.sigmoid(true_slope * features + true_intercepts)
            accepted = torch.rand(bs, gamma) < accept_prob
            # Drafts before the first rejection.
            num_correct = torch.cumprod(accepted.to(torch.int64), dim=1).sum(dim=1)
            est.observe(
                num_correct_drafts=num_correct,
                num_graded_drafts=torch.full((bs,), gamma),
            )
        self.assertEqual(est.num_refits, steps // est.REFIT_INTERVAL)
        self.assertAlmostEqual(float(est.slope), true_slope, delta=0.03)
        torch.testing.assert_close(est.intercepts, true_intercepts, atol=0.08, rtol=0)

    def test_calibrated_in_the_large_when_acceptance_saturates(self):
        """Acceptance of a confident draft tops out at the target's own entropy,
        which logit(q_max) cannot express. Each position's mean prediction must
        still match its observed acceptance: the budget sums these."""
        torch.manual_seed(2)
        gamma, bs, steps, ceiling = 3, 128, 1500, 0.8
        true_intercepts = torch.tensor([0.5, 0.0, -0.5])
        est = self._estimator(gamma=gamma)

        def batch():
            confident = torch.rand(bs, 1) < 0.5
            centers = torch.where(confident, torch.tensor(12.0), torch.tensor(1.0))
            features = (centers + torch.randn(bs, gamma) * 3.0).clamp(min=0.0)
            accept_prob = ceiling * torch.sigmoid(0.5 * features + true_intercepts)
            return features, accept_prob

        for _ in range(steps):
            features, accept_prob = batch()
            est.predict(
                step_logits=_two_token_logits(features),
                temperatures=torch.ones(bs),
                greedy_mask=torch.zeros(bs, dtype=torch.bool),
            )
            accepted = torch.rand(bs, gamma) < accept_prob
            est.observe(
                num_correct_drafts=torch.cumprod(accepted.long(), dim=1).sum(dim=1),
                num_graded_drafts=torch.full((bs,), gamma),
            )
        features, accept_prob = batch()
        predicted = est.predict(
            step_logits=_two_token_logits(features),
            temperatures=torch.ones(bs),
            greedy_mask=torch.zeros(bs, dtype=torch.bool),
        )
        est.skip_observe()
        # Expected accepted drafts per request, the quantity the budget ranks.
        expected = torch.cumprod(accept_prob, dim=1).sum(dim=1).mean()
        self.assertAlmostEqual(
            float(torch.cumprod(predicted, dim=1).sum(dim=1).mean()),
            float(expected),
            delta=0.08,
        )

    def test_unlabeled_positions_do_not_move(self):
        est = self._estimator(gamma=3)
        for _ in range(est.REFIT_INTERVAL):
            est.predict(
                step_logits=_two_token_logits(torch.ones(8, 3)),
                temperatures=torch.ones(8),
                greedy_mask=torch.zeros(8, dtype=torch.bool),
            )
            est.observe(
                num_correct_drafts=torch.zeros(8, dtype=torch.int64),
                num_graded_drafts=torch.full((8,), 3),
            )
        self.assertEqual(est.num_refits, 1)
        self.assertLess(float(est.intercepts[0]), -0.2)
        torch.testing.assert_close(est.intercepts[1:], torch.full((2,), -0.2))
        self.assertGreaterEqual(float(est.slope), 0.0)

    def test_skip_observe_drops_the_prediction(self):
        est = self._estimator()
        est.predict(
            step_logits=_two_token_logits(torch.ones(1, 3)),
            temperatures=torch.ones(1),
            greedy_mask=torch.zeros(1, dtype=torch.bool),
        )
        est.skip_observe()
        with self.assertRaises(AssertionError):
            est.observe(
                num_correct_drafts=torch.zeros(1, dtype=torch.int64),
                num_graded_drafts=torch.full((1,), 3),
            )


class TestConfidenceSource(unittest.TestCase):
    def _builds(self, *, source, mode, enable_field):
        config = SimpleNamespace()
        if enable_field is not None:
            config.enable_confidence_head = enable_field
        with (
            mock.patch.object(
                dspark_config,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_dspark_confidence_source=source
                ),
            ),
            mock.patch(
                "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                return_value=mode,
            ),
        ):
            return dspark_config.builds_confidence_head(config)

    def test_static_mode_builds_no_head(self):
        for source in ("auto", "head", "online"):
            self.assertFalse(
                self._builds(
                    source=source, mode=RaggedVerifyMode.STATIC, enable_field=True
                )
            )

    def test_source_selection(self):
        compact = RaggedVerifyMode.COMPACT
        self.assertTrue(self._builds(source="auto", mode=compact, enable_field=True))
        self.assertTrue(self._builds(source="auto", mode=compact, enable_field=None))
        self.assertFalse(self._builds(source="auto", mode=compact, enable_field=False))
        self.assertTrue(self._builds(source="head", mode=compact, enable_field=False))
        self.assertFalse(self._builds(source="online", mode=compact, enable_field=True))


if __name__ == "__main__":
    unittest.main()
