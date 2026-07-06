import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_confidence_metrics import (
    ConfidenceMetricsProbe,
    PerPositionConfidenceMetrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _cpu_metrics(gamma: int) -> PerPositionConfidenceMetrics:
    return PerPositionConfidenceMetrics(gamma=gamma, device=torch.device("cpu"))


class TestPerPositionConfidenceMetrics(CustomTestCase):
    def test_perfectly_calibrated_has_low_ece(self):
        torch.manual_seed(0)
        n = 40000
        survival = torch.full((n, 1), 0.3, dtype=torch.float64)
        prefix_mask = (torch.rand(n, 1) < 0.3).to(torch.float64)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        row = metrics.compute()[0]
        self.assertLess(row["ece"], 0.03)
        self.assertAlmostEqual(row["pred_mean"], 0.3, places=4)

    def test_overconfident_has_high_ece_and_pred_above_target(self):
        torch.manual_seed(0)
        n = 40000
        survival = torch.full((n, 1), 0.9, dtype=torch.float64)
        prefix_mask = (torch.rand(n, 1) < 0.3).to(torch.float64)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        row = metrics.compute()[0]
        self.assertGreater(row["ece"], 0.4)
        self.assertGreater(row["pred_mean"], row["target_mean"])

    def test_separable_scores_give_auc_near_one(self):
        torch.manual_seed(0)
        n = 20000
        pos = torch.rand(n, 1) * 0.3 + 0.7
        neg = torch.rand(n, 1) * 0.3
        survival = torch.cat([pos, neg], dim=0)
        prefix_mask = torch.cat([torch.ones(n, 1), torch.zeros(n, 1)], dim=0)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        self.assertGreater(metrics.compute()[0]["auc"], 0.99)

    def test_random_scores_give_auc_near_half(self):
        torch.manual_seed(0)
        n = 40000
        survival = torch.rand(n, 1)
        prefix_mask = (torch.rand(n, 1) < 0.5).to(torch.float64)
        metrics = _cpu_metrics(gamma=1)
        metrics.update(survival=survival, prefix_mask=prefix_mask)
        auc = metrics.compute()[0]["auc"]
        self.assertGreater(auc, 0.45)
        self.assertLess(auc, 0.55)

    def test_batched_update_matches_per_sample_update(self):
        torch.manual_seed(0)
        bs, gamma = 32, 5
        survival = torch.rand(bs, gamma)
        prefix_mask = (torch.rand(bs, gamma) < 0.5).to(torch.float64)

        batched = _cpu_metrics(gamma=gamma)
        batched.update(survival=survival, prefix_mask=prefix_mask)

        per_sample = _cpu_metrics(gamma=gamma)
        for row_idx in range(bs):
            per_sample.update(
                survival=survival[row_idx : row_idx + 1],
                prefix_mask=prefix_mask[row_idx : row_idx + 1],
            )

        for name in (
            "coarse_count",
            "coarse_pred",
            "coarse_target",
            "fine_pos",
            "fine_neg",
            "brier_num",
        ):
            self.assertTrue(
                torch.allclose(
                    getattr(batched, name), getattr(per_sample, name), atol=1e-9
                ),
                msg=name,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_on_device_accumulation_matches_cpu(self):
        torch.manual_seed(0)
        bs, gamma = 24, 4
        survival = torch.rand(bs, gamma)
        prefix_mask = (torch.rand(bs, gamma) < 0.5).to(torch.float64)

        cpu = _cpu_metrics(gamma=gamma)
        cpu.update(survival=survival, prefix_mask=prefix_mask)

        gpu = PerPositionConfidenceMetrics(gamma=gamma, device=torch.device("cuda"))
        gpu.update(survival=survival.cuda(), prefix_mask=prefix_mask.cuda())

        for cpu_row, gpu_row in zip(cpu.compute(), gpu.compute()):
            for key in ("ece", "auc", "brier", "pred_mean", "target_mean"):
                self.assertAlmostEqual(cpu_row[key], gpu_row[key], places=9, msg=key)


def _probe_inputs(bs: int, gamma: int):
    torch.manual_seed(0)
    verify_num_draft_tokens = gamma + 1
    vocab = 16
    verify_ids_2d = torch.randint(0, vocab, (bs, verify_num_draft_tokens))
    target_logits = torch.randn(bs * verify_num_draft_tokens, vocab)
    confidence_raw = torch.randn(bs, gamma)
    return verify_ids_2d, target_logits, confidence_raw


class TestConfidenceMetricsProbe(CustomTestCase):
    def _observe(self, probe, *, carries_confidence=True, is_compact_mode=False):
        verify_ids_2d, target_logits, confidence_raw = _probe_inputs(bs=3, gamma=4)
        probe.maybe_observe(
            carries_confidence=carries_confidence,
            is_compact_mode=is_compact_mode,
            confidence_raw=confidence_raw,
            verify_ids_2d=verify_ids_2d,
            target_logits=target_logits,
            bs=3,
        )

    def test_disabled_env_is_noop(self):
        probe = ConfidenceMetricsProbe(gamma=4, verify_num_draft_tokens=5, tp_rank=0)
        self._observe(probe)
        self.assertIsNone(probe._metrics)
        self.assertEqual(probe._step_ct, 0)

    def test_non_rank0_is_noop(self):
        probe = ConfidenceMetricsProbe(gamma=4, verify_num_draft_tokens=5, tp_rank=1)
        with envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.override(True):
            self._observe(probe)
        self.assertIsNone(probe._metrics)

    def test_missing_confidence_head_is_noop(self):
        probe = ConfidenceMetricsProbe(gamma=4, verify_num_draft_tokens=5, tp_rank=0)
        with envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.override(True):
            self._observe(probe, carries_confidence=False)
        self.assertIsNone(probe._metrics)

    def test_compact_mode_warns_once_and_skips(self):
        probe = ConfidenceMetricsProbe(gamma=4, verify_num_draft_tokens=5, tp_rank=0)
        with envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.override(True):
            self._observe(probe, is_compact_mode=True)
            self.assertTrue(probe._compact_warned)
            self._observe(probe, is_compact_mode=True)
        self.assertIsNone(probe._metrics)
        self.assertEqual(probe._step_ct, 0)

    def test_enabled_path_accumulates_and_prints(self):
        probe = ConfidenceMetricsProbe(
            gamma=4, verify_num_draft_tokens=5, tp_rank=0, print_every=2
        )
        with envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.override(True):
            self._observe(probe)
            self.assertIsInstance(probe._metrics, PerPositionConfidenceMetrics)
            self.assertEqual(probe._step_ct, 1)
            self._observe(probe)
        self.assertEqual(probe._step_ct, 2)


if __name__ == "__main__":
    unittest.main()
