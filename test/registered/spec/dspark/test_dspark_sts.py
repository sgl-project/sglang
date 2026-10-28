import tempfile
import unittest
from pathlib import Path

import torch

from sglang.benchmark.dspark_sts_fit import (
    default_temperature_grid,
    expected_calibration_error,
    fit_sts_temperatures,
)
from sglang.srt.models.dspark import DSparkConfidenceHead, VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_sts import (
    DSparkStsCalibration,
    StsDataRecorder,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestApplySts(CustomTestCase):
    def test_per_position_temperature_scales_each_column(self):
        head = DSparkConfidenceHead(hidden_size=8, markov_rank=4, with_markov=False)
        head.sts_temperatures = torch.tensor([0.5, 1.0, 2.0])
        confidence_raw = torch.full((2, 3), 2.0)
        out = head.apply_sts(confidence_raw)
        # Hand-computed sigmoid(2.0 / T) per column: identical raw logits with
        # distinct per-column T catch wrong-axis broadcasts, and dividing (not
        # multiplying) by T is what separates 0.982 from 0.731 in column 0.
        expected_row = [0.98201379, 0.88079708, 0.73105858]
        for row in out.tolist():
            for got, want in zip(row, expected_row):
                self.assertAlmostEqual(got, want, places=6)


class TestConfidencePredict(CustomTestCase):
    def test_predict_applies_sts_to_the_raw_logit(self):
        torch.manual_seed(0)
        head = DSparkConfidenceHead(hidden_size=8, markov_rank=4, with_markov=False)
        head.sts_temperatures = torch.tensor([0.5, 1.0, 2.0])
        confidence, confidence_raw = head.predict(
            torch.randn(2 * 3, 8),
            anchor_tokens=torch.tensor([1, 2]),
            draft_tokens=torch.tensor([[3, 4, 5], [6, 7, 8]]),
            markov_head=None,
        )
        self.assertEqual(confidence_raw.shape, (2, 3))
        self.assertTrue(torch.equal(confidence, head.apply_sts(confidence_raw)))

    def test_markov_features_are_the_previous_tokens(self):
        # Step i reads the embedding of the token before it: the anchor, then drafts.
        torch.manual_seed(0)
        head = DSparkConfidenceHead(hidden_size=8, markov_rank=4, with_markov=True)
        markov_head = VanillaMarkov(vocab_size=16, markov_rank=4)
        features = torch.randn(2, 3, 8)
        anchor_tokens = torch.tensor([1, 2])
        draft_tokens = torch.tensor([[3, 4, 5], [6, 7, 8]])
        _, confidence_raw = head.predict(
            features.view(6, 8),
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            markov_head=markov_head,
        )
        prev_tokens = torch.tensor([[1, 3, 4], [2, 6, 7]])
        expected = head(features, markov_head.get_prev_embeddings(prev_tokens))
        self.assertTrue(torch.equal(confidence_raw, expected))


class TestDSparkStsCalibration(CustomTestCase):
    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            DSparkStsCalibration(temperatures=[1.0, 0.0, 2.0])
        with self.assertRaises(ValueError):
            DSparkStsCalibration(temperatures=[1.0, -0.5])


class TestExpectedCalibrationError(CustomTestCase):
    def test_perfectly_calibrated_probs_have_low_ece(self):
        torch.manual_seed(0)
        probs = torch.full((20000,), 0.3, dtype=torch.float64)
        targets = (torch.rand(20000) < 0.3).to(torch.float64)
        ece = expected_calibration_error(probs=probs, targets=targets, num_bins=15)
        self.assertLess(ece, 0.02)

    def test_overconfident_probs_have_high_ece(self):
        probs = torch.full((20000,), 0.95, dtype=torch.float64)
        targets = torch.full((20000,), 0.3, dtype=torch.float64)
        ece = expected_calibration_error(probs=probs, targets=targets, num_bins=15)
        self.assertGreater(ece, 0.5)


class TestFitStsTemperatures(CustomTestCase):
    def test_recovers_scale_and_reduces_ece(self):
        torch.manual_seed(0)
        num_samples, gamma, scale = 60000, 4, 2.5
        base_logit = torch.tensor([2.0, 1.2, 0.8, 0.4])
        true_logit = base_logit[None, :] + torch.randn(num_samples, gamma) * 0.5
        true_prob = torch.sigmoid(true_logit)
        accept = (torch.rand(num_samples, gamma) < true_prob).to(torch.float64)
        prefix_mask = torch.cumprod(accept, dim=1)
        overconfident_logits = true_logit * scale

        result = fit_sts_temperatures(
            logits=overconfident_logits,
            prefix_mask=prefix_mask,
            grid=default_temperature_grid(),
            num_bins=15,
        )

        self.assertEqual(len(result["temperatures"]), gamma)
        for temperature in result["temperatures"]:
            self.assertGreater(temperature, scale / 1.5)
            self.assertLess(temperature, scale * 1.5)
        mean_before = sum(result["ece_before"]) / gamma
        mean_after = sum(result["ece_after"]) / gamma
        self.assertLess(mean_after, 0.25 * mean_before)


class TestStsDataRecorder(CustomTestCase):
    def test_builds_prefix_mask_and_writes_shard(self):
        gamma = 4
        confidence_raw = torch.randn(4, gamma)
        num_correct_drafts = torch.tensor([0, 2, 4, 1], dtype=torch.int32)
        expected_prefix_mask = torch.tensor(
            [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
            dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "shard")
            recorder = StsDataRecorder(path_stem=stem, gamma=gamma, flush_every=10)
            recorder.record(
                confidence_raw=confidence_raw,
                num_correct_drafts=num_correct_drafts,
            )
            recorder.flush()
            shard = torch.load(f"{stem}.0.pt")
        self.assertTrue(torch.equal(shard["prefix_mask"], expected_prefix_mask))
        self.assertTrue(torch.equal(shard["logits"], confidence_raw.to(torch.float32)))


if __name__ == "__main__":
    unittest.main()
