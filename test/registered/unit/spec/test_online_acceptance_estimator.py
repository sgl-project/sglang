import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.acceptance_estimator import MAX_LOG_ODDS
from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_config
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.online_acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _two_token_logits(features: torch.Tensor) -> torch.Tensor:
    """[bs, positions, 2] logits whose T = 1 top log-odds equal ``features`` (>= 0)."""
    return torch.stack([features, torch.zeros_like(features)], dim=-1)


def _estimator(num_positions=3):
    return OnlineAcceptanceEstimator(
        num_positions=num_positions, device=torch.device("cpu"), log_rank=False
    )


def _run_steps(est, *, steps, bs, sample):
    for _ in range(steps):
        features, accept_prob = sample(bs)
        x, _ = est.predict(
            logits=_two_token_logits(features), temperatures=torch.ones(bs)
        )
        correct = torch.rand(features.shape) < accept_prob
        est.observe(
            features=x,
            num_correct_drafts=torch.cumprod(correct.long(), dim=1).sum(dim=1),
            num_graded_drafts=torch.full((bs,), features.shape[1]),
        )


class TestPredict(CustomTestCase):
    def test_feature_is_top_probability_log_odds(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 4, 50) * 3
        temperatures = torch.tensor([0.5, 1.0, 1.7])
        features, predictions = _estimator(4).predict(
            logits=logits, temperatures=temperatures
        )
        q = torch.softmax(logits.double() / temperatures.double().view(-1, 1, 1), -1)
        q_max = q.max(dim=-1).values
        expected = torch.log(q_max / (1 - q_max))
        torch.testing.assert_close(features.double(), expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(predictions, torch.sigmoid(0.5 * features - 0.2))

    def test_greedy_rows_score_the_unit_temperature_distribution(self):
        features, _ = _estimator(1).predict(
            logits=_two_token_logits(torch.tensor([[1.5], [1.5]])),
            temperatures=torch.tensor([0.0, 1e-5]),
        )
        self.assertAlmostEqual(float(features[0]), 1.5, places=5)
        # The sampled row at T = 1e-5 is a point mass: its log-odds saturate.
        self.assertEqual(float(features[1]), MAX_LOG_ODDS)

    def test_masked_logits(self):
        logits = torch.tensor([[[2.0, float("-inf"), 0.0, float("-inf")]]])
        features, _ = _estimator(1).predict(logits=logits, temperatures=torch.ones(1))
        torch.testing.assert_close(features, torch.tensor([[2.0]]))

    def test_writes_into_given_buffers(self):
        est = _estimator(2)
        features_out = torch.zeros(4, 2)
        predictions_out = torch.zeros(4, 2)
        est.predict(
            logits=_two_token_logits(torch.ones(2, 2)),
            temperatures=torch.ones(2),
            features_out=features_out[:2],
            predictions_out=predictions_out[:2],
        )
        torch.testing.assert_close(features_out[:2], torch.ones(2, 2))
        self.assertEqual(float(features_out[2:].abs().sum()), 0.0)
        self.assertGreater(float(predictions_out[:2].min()), 0.0)


class TestObserve(CustomTestCase):
    def test_labels_stop_at_first_rejection_and_graded_rows(self):
        est = _estimator(4)
        est.observe(
            features=torch.ones(3, 4),
            # Row 0 rejects draft 1; row 1 accepts all four; row 2 accepts
            # three but only its first two drafts met a real target row.
            num_correct_drafts=torch.tensor([1, 4, 3]),
            num_graded_drafts=torch.tensor([4, 4, 2]),
        )
        graded = est._round_counts[..., 0].sum(dim=1).tolist()
        correct = est._round_counts[..., 1].sum(dim=1).tolist()
        self.assertEqual(graded, [3.0, 3.0, 1.0, 1.0])
        self.assertEqual(correct, [3.0, 2.0, 1.0, 1.0])

    def test_converges_to_generating_model(self):
        torch.manual_seed(1)
        true_slope = 0.55
        true_intercepts = torch.tensor([0.6, 0.2, -0.3, -0.8])

        def sample(bs):
            features = torch.rand(bs, 4) * 6.0
            return features, torch.sigmoid(true_slope * features + true_intercepts)

        est = _estimator(4)
        _run_steps(est, steps=1500, bs=256, sample=sample)
        self.assertEqual(est.refit_ct, 1500 // est.REFIT_INTERVAL)
        self.assertAlmostEqual(float(est.slope), true_slope, delta=0.03)
        torch.testing.assert_close(est.intercepts, true_intercepts, atol=0.08, rtol=0)

    def test_calibrated_in_the_large_when_acceptance_saturates(self):
        """Acceptance of a confident draft tops out at the target's own entropy,
        which logit(q_max) cannot express. The expected accepted drafts per
        request, which the verify budget ranks, must still match."""
        torch.manual_seed(2)
        ceiling = 0.8
        true_intercepts = torch.tensor([0.5, 0.0, -0.5])

        def sample(bs):
            confident = torch.rand(bs, 1) < 0.5
            centers = torch.where(confident, torch.tensor(12.0), torch.tensor(1.0))
            features = (centers + torch.randn(bs, 3) * 3.0).clamp(min=0.0)
            return features, ceiling * torch.sigmoid(0.5 * features + true_intercepts)

        est = _estimator(3)
        _run_steps(est, steps=1500, bs=128, sample=sample)
        features, accept_prob = sample(128)
        _, predicted = est.predict(
            logits=_two_token_logits(features), temperatures=torch.ones(128)
        )
        self.assertAlmostEqual(
            float(torch.cumprod(predicted, dim=1).sum(dim=1).mean()),
            float(torch.cumprod(accept_prob, dim=1).sum(dim=1).mean()),
            delta=0.08,
        )

    def test_unlabeled_positions_do_not_move(self):
        est = _estimator(3)
        for _ in range(est.REFIT_INTERVAL):
            est.observe(
                features=torch.ones(8, 3),
                num_correct_drafts=torch.zeros(8, dtype=torch.int64),
                num_graded_drafts=torch.full((8,), 3),
            )
        self.assertEqual(est.refit_ct, 1)
        self.assertLess(float(est.intercepts[0]), -0.2)
        torch.testing.assert_close(est.intercepts[1:], torch.full((2,), -0.2))
        self.assertGreaterEqual(float(est.slope), 0.0)
        self.assertEqual(float(est._round_counts.abs().sum()), 0.0)


class TestDSparkObserve(CustomTestCase):
    def _observe(self, *, layout):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.gamma = 4
        worker._acceptance_estimator = mock.Mock()
        worker._observe_acceptance(
            features=torch.zeros(2, 4),
            accept=SimpleNamespace(
                correct_len=torch.tensor([1, 2]), cap_trim_lens=torch.tensor([0, 1])
            ),
            layout=layout,
        )
        return worker._acceptance_estimator.observe.call_args.kwargs

    def test_full_verify_grades_every_draft(self):
        kwargs = self._observe(layout=None)
        self.assertEqual(kwargs["num_graded_drafts"].tolist(), [4, 4])
        # The cap trims the accept, not the label.
        self.assertEqual(kwargs["num_correct_drafts"].tolist(), [1, 3])

    def test_compact_grades_only_verified_rows(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([5, 3, 1], dtype=torch.int32), graph_num_tokens=9
        )
        kwargs = self._observe(layout=layout)
        self.assertEqual(kwargs["num_graded_drafts"].tolist(), [4, 2])


class TestConfidenceSource(CustomTestCase):
    def _builds(self, *, mode, enable_field, use_online=False):
        config = SimpleNamespace()
        if enable_field is not None:
            config.enable_confidence_head = enable_field
        with (
            envs.SGLANG_DSPARK_USE_ONLINE_ESTIMATOR.override(use_online),
            mock.patch.object(
                dspark_config, "read_ragged_verify_mode", return_value=mode
            ),
        ):
            return dspark_config.builds_confidence_head(config)

    def test_static_mode_builds_no_head(self):
        for use_online in (False, True):
            self.assertFalse(
                self._builds(
                    mode=RaggedVerifyMode.STATIC,
                    enable_field=True,
                    use_online=use_online,
                )
            )

    def test_source_selection(self):
        compact = RaggedVerifyMode.COMPACT
        self.assertTrue(self._builds(mode=compact, enable_field=True))
        self.assertTrue(self._builds(mode=compact, enable_field=None))
        self.assertFalse(self._builds(mode=compact, enable_field=False))
        self.assertFalse(self._builds(mode=compact, enable_field=True, use_online=True))


if __name__ == "__main__":
    unittest.main()
