import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.benchmark.dspark_sts_fit import (
    default_temperature_grid,
    expected_calibration_error,
    fit,
    fit_sts_temperatures,
)
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
)
from sglang.srt.speculative.dspark_components.dspark_sts import (
    DSparkStsCalibration,
    StsDataRecorder,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestApplySts(CustomTestCase):
    def test_default_buffer_is_identity_sigmoid(self):
        head = DSparkConfidenceHead(hidden_size=8, markov_rank=4, with_markov=False)
        confidence_raw = torch.randn(3, 5) * 9.0
        out = head.apply_sts(confidence_raw)
        self.assertTrue(torch.equal(out, torch.sigmoid(confidence_raw.float())))

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

    def test_apply_sts_stashes_raw_logit(self):
        head = DSparkConfidenceHead(hidden_size=8, markov_rank=4, with_markov=False)
        confidence_raw = torch.randn(2, 5)
        head.apply_sts(confidence_raw)
        self.assertIs(head._last_confidence_raw, confidence_raw)


class TestDSparkStsCalibration(CustomTestCase):
    def test_json_round_trip_preserves_fields(self):
        calibration = DSparkStsCalibration(
            temperatures=[1.5, 2.0, 0.5],
            dataset="shards.*.pt",
            num_samples=1234,
            ece_before=[0.3, 0.2, 0.1],
            ece_after=[0.02, 0.01, 0.03],
        )
        restored = DSparkStsCalibration.from_json(calibration.to_json())
        self.assertEqual(restored.temperatures, calibration.temperatures)
        self.assertEqual(restored.dataset, calibration.dataset)
        self.assertEqual(restored.num_samples, calibration.num_samples)
        self.assertEqual(restored.ece_before, calibration.ece_before)
        self.assertEqual(restored.ece_after, calibration.ece_after)

    def test_rejects_empty_temperatures(self):
        with self.assertRaises(ValueError):
            DSparkStsCalibration(temperatures=[])

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

    def test_compact_verify_censoring_is_masked(self):
        # Compact verify checks only the first verify_len - 1 drafts; positions
        # past that cap must be masked, not counted as rejections.
        torch.manual_seed(0)
        num_samples, gamma, scale = 60000, 4, 2.5
        base_logit = torch.tensor([2.0, 1.2, 0.8, 0.4])
        true_logit = base_logit[None, :] + torch.randn(num_samples, gamma) * 0.5
        accept = torch.rand(num_samples, gamma) < torch.sigmoid(true_logit)
        caps = torch.randint(1, gamma + 1, (num_samples, 1))
        num_correct = torch.minimum(
            accept.cumprod(dim=1).sum(dim=1, keepdim=True), caps
        )
        positions = torch.arange(gamma)[None, :]

        shard = {
            "logits": (true_logit * scale).to(torch.float32),
            "prefix_mask": (positions < num_correct).to(torch.float32),
        }

        def fit_shard(shard):
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(shard, Path(tmp) / "sts.0.pt")
                out = Path(tmp) / "sts.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    fit(data_glob=str(Path(tmp) / "sts.*.pt"), out=out)
                return DSparkStsCalibration.from_json(out.read_text()).temperatures

        observed_mask = (positions < caps).to(torch.float32)
        for temperature in fit_shard({**shard, "observed_mask": observed_mask}):
            self.assertGreater(temperature, scale / 1.5)
            self.assertLess(temperature, scale * 1.5)
        # Without the mask, capped positions count as rejections and the fitted
        # temperatures inflate.
        self.assertGreater(fit_shard(shard)[-1], scale * 1.5)


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
        self.assertTrue(torch.equal(shard["observed_mask"], torch.ones(4, gamma)))

    def test_masks_positions_past_the_verify_window(self):
        gamma = 4
        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "shard")
            recorder = StsDataRecorder(path_stem=stem, gamma=gamma, flush_every=10)
            recorder.record(
                confidence_raw=torch.randn(4, gamma),
                num_correct_drafts=torch.tensor([0, 2, 2, 1]),
                verify_lens=torch.tensor([3, 3, 5, 2], dtype=torch.int32),
            )
            recorder.flush()
            shard = torch.load(f"{stem}.0.pt")
        expected_prefix_mask = torch.tensor(
            [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]],
            dtype=torch.float32,
        )
        expected_observed_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(shard["prefix_mask"], expected_prefix_mask))
        self.assertTrue(torch.equal(shard["observed_mask"], expected_observed_mask))


class TestStsCollectLabels(CustomTestCase):
    def test_labels_come_from_the_step_acceptance(self):
        # Greedy argmax accepts every draft, but the step accepted fewer
        # (rejection sampling); compact verify checked only verify_len - 1.
        gamma, bs = 3, 2
        width = gamma + 1
        drafts = torch.tensor([[11, 12, 13], [21, 22, 23]])
        verify_ids_2d = torch.cat([torch.tensor([[1], [2]]), drafts], dim=1)
        target_predict = torch.cat([drafts, torch.zeros(bs, 1, dtype=torch.int64)], 1)
        target_logits = torch.nn.functional.one_hot(target_predict, 32).float()
        confidence_raw = torch.randn(bs, gamma)

        observers = object.__new__(DsparkStepObservers)
        observers._gamma = gamma
        observers._verify_num_draft_tokens = width
        observers._planner = SimpleNamespace(
            carries_confidence=True,
            last_confidence_raw=confidence_raw,
            is_compact_mode=True,
        )
        observers._confidence_probe = SimpleNamespace(maybe_observe=lambda **_: None)
        observers._block_accept_recorder = None
        observers._info_dumper = SimpleNamespace(enabled=False)
        observers._sts_recorder = None
        with tempfile.TemporaryDirectory() as tmp:
            observers._sts_collect_path = str(Path(tmp) / "shard")
            observers.observe_verify_step(
                forward_ct=0,
                reqs=[],
                bs=bs,
                proposal_folded=False,
                verify_ids_2d=verify_ids_2d,
                target_logits=target_logits.view(bs * width, -1),
                layout=SimpleNamespace(verify_lens=torch.tensor([4, 2])),
                confidence=None,
                prefix_lens=None,
                draft_tokens=drafts,
                draft_block=None,
                sampling_info=None,
                correct_len=torch.tensor([1, 0]),
                cap_trim_lens=None,
                bonus=None,
                commit_lens=None,
                verify_token_budget=None,
                req_pool_indices=None,
                verify_tier_num_tokens=bs * width,
                dp_tier_num_tokens=None,
            )
            observers._sts_recorder.flush()
            shard = torch.load(str(Path(tmp) / "shard.0.pt"))
        self.assertTrue(torch.equal(shard["logits"], confidence_raw))
        self.assertTrue(
            torch.equal(shard["prefix_mask"], torch.tensor([[1.0, 0, 0], [0, 0, 0]]))
        )
        self.assertTrue(
            torch.equal(shard["observed_mask"], torch.tensor([[1.0, 1, 1], [1, 0, 0]]))
        )


if __name__ == "__main__":
    unittest.main()
