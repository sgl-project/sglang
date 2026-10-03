# SPDX-License-Identifier: Apache-2.0
"""The lossless-tier admission template admits re-association, not precision loss.

Each fusion used to carry its own hand-picked atol/rtol, which cannot tell a
reassociated reduction from a kernel that quietly dropped a few mantissa bits:
both land "close". The template scores every implementation against the same
fp64 evaluation and judges the candidate relative to the path the model
already runs, so the question it answers is the tier's own question.
"""

import sys
import unittest

import pytest
import torch

from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
    assert_special_values_match,
    discrete_flip_rate,
    score_against,
)


def _row_sums(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Sequential row sum in ``dtype`` -- the association under test."""
    return x.to(dtype).sum(dim=-1)


class TestOperatorLevelGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Outlier channels are the point: DiT activations have them, and they
        # decide whether a reassociated reduction stays within the reference.
        self.x = torch.randn(64, 512, dtype=torch.float32)
        self.x[:, ::97] *= 300.0
        self.reference_fp64 = self.x.to(torch.float64).sum(dim=-1)

    def test_admits_a_reassociated_reduction(self):
        baseline = _row_sums(self.x, torch.float32)
        # Pairwise summation: same operands and accumulation precision, a
        # different order -- the shape of a lossless-tier fusion.
        pairwise = self.x.to(torch.float32).reshape(64, -1, 2).sum(dim=-1).sum(dim=-1)

        base, cand = assert_error_no_worse_than_reference(
            reference_fp64=self.reference_fp64,
            baseline=baseline,
            candidate=pairwise,
        )
        self.assertGreater(base.rms, 0.0)
        self.assertLess(cand.rms, base.rms * 1.5)

    def test_rejects_a_reduction_that_accumulates_in_bf16(self):
        baseline = _row_sums(self.x, torch.float32)
        degraded = _row_sums(self.x, torch.bfloat16).to(torch.float32)

        with self.assertRaisesRegex(AssertionError, "not lossless-tier"):
            assert_error_no_worse_than_reference(
                reference_fp64=self.reference_fp64,
                baseline=baseline,
                candidate=degraded,
            )

    def test_rejects_a_rounding_that_always_leans_one_way(self):
        # A one-sided error small enough to stay inside the magnitude budget:
        # the shape of truncation instead of round-to-nearest-even, or of an
        # approximate transcendental whose error keeps its sign. It cannot be
        # caught by a tolerance, because per-element it is tiny -- but it
        # accumulates over layers and steps where a symmetric error cancels.
        baseline = _row_sums(self.x, torch.float32)
        leaning = baseline + 5e-5

        base = score_against(self.reference_fp64, baseline)
        cand = score_against(self.reference_fp64, leaning)
        self.assertLess(cand.rms, base.rms * 1.5, "stays inside the magnitude budget")

        with self.assertRaisesRegex(AssertionError, "biased rounding"):
            assert_error_no_worse_than_reference(
                reference_fp64=self.reference_fp64,
                baseline=baseline,
                candidate=leaning,
            )

    def test_score_reports_signed_mean_separately_from_magnitude(self):
        score = score_against(
            torch.zeros(4, dtype=torch.float64), torch.full((4,), 0.5)
        )
        self.assertAlmostEqual(score.mean, 0.5)
        self.assertAlmostEqual(score.rms, 0.5)
        self.assertAlmostEqual(score.max_abs, 0.5)


class TestSpecialValues(unittest.TestCase):
    def test_a_fast_path_that_swallows_nan_is_rejected(self):
        cases = {"nan row": (torch.tensor([1.0, float("nan"), 3.0]),)}

        assert_special_values_match(
            baseline=torch.relu, candidate=torch.relu, cases=cases
        )

        with self.assertRaisesRegex(AssertionError, "moves NaNs"):
            assert_special_values_match(
                baseline=torch.relu,
                candidate=lambda x: torch.nan_to_num(torch.relu(x)),
                cases=cases,
            )

    def test_a_fast_path_that_ignores_strides_is_rejected(self):
        # Reading the wrong elements is a magnitude error, not a structural
        # one, so it is the operator-level gate that has to catch it.
        torch.manual_seed(0)
        strided = torch.randn(8, 16)[:, ::2]
        reference = (strided.double() * 2).contiguous()
        baseline = strided * 2
        ignores_stride = strided.flatten()[: strided.numel()].view_as(strided) * 2

        with self.assertRaisesRegex(AssertionError, "not lossless-tier"):
            assert_error_no_worse_than_reference(
                reference_fp64=reference,
                baseline=baseline,
                candidate=ignores_stride,
                label="stride-ignoring kernel",
            )

    def test_a_rounding_level_difference_is_not_a_special_value_failure(self):
        # The gate asks whether the special values behave the same, not
        # whether the numbers match: a lossless-tier path differs there by
        # construction, and judging that is the operator gate's job.
        torch.manual_seed(0)
        x = torch.randn(32, 8)
        cases = {"plain": (x,), "with a NaN": (torch.cat([x[:1] * float("nan"), x]),)}

        assert_special_values_match(
            baseline=lambda t: t * 3.0,
            candidate=lambda t: (t.to(torch.bfloat16) * 3.0).to(t.dtype),
            cases=cases,
        )

    def test_raising_where_the_reference_returns_is_rejected(self):
        cases = {"empty": (torch.zeros(0),)}

        def explodes(x):
            raise RuntimeError("no empty support")

        with self.assertRaisesRegex(AssertionError, "RuntimeError"):
            assert_special_values_match(
                baseline=lambda x: x, candidate=explodes, cases=cases
            )


class TestDiscreteDownstream(unittest.TestCase):
    def test_flip_rate_counts_decisions_not_distance(self):
        baseline = torch.tensor([0, 1, 2, 3])
        self.assertEqual(discrete_flip_rate(baseline, baseline.clone()), 0.0)
        self.assertEqual(discrete_flip_rate(baseline, torch.tensor([0, 1, 2, 9])), 0.25)

    def test_empty_input_has_no_flips(self):
        empty = torch.zeros(0, dtype=torch.long)
        self.assertEqual(discrete_flip_rate(empty, empty), 0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
