from __future__ import annotations

import unittest
from unittest.mock import patch

from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceLog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


_GOOD_LINE: str = SwaDivergenceLog(
    forward_ct=120,
    verify_full=10000,
    verify_swa=4200,
    swa_full_idx_divergence=512,
    swa_out_of_window_tokens=8192,
).format()
_LATER_LINE: str = SwaDivergenceLog(
    forward_ct=240,
    verify_full=20000,
    verify_swa=8400,
    swa_full_idx_divergence=1024,
    swa_out_of_window_tokens=16384,
).format()


class _DummyHarness(CanaryE2EBase):
    model_mode = "swa"
    kv_canary_mode = "log"

    @classmethod
    def setUpClass(cls) -> None:
        return

    @classmethod
    def tearDownClass(cls) -> None:
        return


class TestAssertSwaDivergenceObserved(CustomTestCase):
    def _make_harness(
        self, log_text_or_sequence
    ) -> tuple[_DummyHarness, "patch._patch[None]"]:
        harness = _DummyHarness()
        harness._stderr_buf = None
        harness._stdout_buf = None
        if isinstance(log_text_or_sequence, list):
            patcher = patch.object(
                _DummyHarness, "_captured_log_text", side_effect=log_text_or_sequence
            )
        else:
            patcher = patch.object(
                _DummyHarness,
                "_captured_log_text",
                return_value=log_text_or_sequence,
            )
        return harness, patcher

    def test_assert_swa_divergence_observed_passes_when_above_threshold(self) -> None:
        harness, patcher = self._make_harness(_LATER_LINE + "\n" + _GOOD_LINE + "\n")
        with patcher:
            harness.assert_swa_divergence_observed(
                min_swa_full_idx_divergence=100,
                require_verify_lag=True,
                flush_wait_seconds=0.0,
                max_retries=1,
            )

    def test_assert_swa_divergence_observed_uses_peak_out_of_window(self) -> None:
        diverged = SwaDivergenceLog(
            forward_ct=120,
            verify_full=10000,
            verify_swa=4200,
            swa_full_idx_divergence=512,
            swa_out_of_window_tokens=8192,
        ).format()
        trailing_zero = SwaDivergenceLog(
            forward_ct=240,
            verify_full=20000,
            verify_swa=8400,
            swa_full_idx_divergence=0,
            swa_out_of_window_tokens=0,
        ).format()
        harness, patcher = self._make_harness(diverged + "\n" + trailing_zero + "\n")
        with patcher:
            harness.assert_swa_divergence_observed(
                min_swa_out_of_window_tokens=1,
                min_swa_full_idx_divergence=1,
                require_verify_lag=True,
                flush_wait_seconds=0.0,
                max_retries=1,
            )

    def test_assert_swa_divergence_observed_checks_verify_lag_on_latest_line(
        self,
    ) -> None:
        lagging = SwaDivergenceLog(
            forward_ct=120,
            verify_full=10000,
            verify_swa=4200,
            swa_full_idx_divergence=512,
            swa_out_of_window_tokens=8192,
        ).format()
        no_lag = SwaDivergenceLog(
            forward_ct=240,
            verify_full=20000,
            verify_swa=20000,
            swa_full_idx_divergence=1024,
            swa_out_of_window_tokens=16384,
        ).format()
        harness, patcher = self._make_harness(lagging + "\n" + no_lag + "\n")
        with patcher:
            with self.assertRaisesRegex(AssertionError, "verify_swa=20000"):
                harness.assert_swa_divergence_observed(
                    min_swa_full_idx_divergence=1,
                    require_verify_lag=True,
                    flush_wait_seconds=0.0,
                    max_retries=1,
                )

    def test_assert_swa_divergence_observed_raises_when_below_threshold(self) -> None:
        zero_mapping_line = SwaDivergenceLog(
            forward_ct=100,
            verify_full=5000,
            verify_swa=2000,
            swa_full_idx_divergence=0,
            swa_out_of_window_tokens=8192,
        ).format()
        harness, patcher = self._make_harness(zero_mapping_line + "\n")
        with patcher:
            with self.assertRaisesRegex(AssertionError, "swa_full_idx_divergence=0"):
                harness.assert_swa_divergence_observed(
                    min_swa_full_idx_divergence=1,
                    require_verify_lag=False,
                    flush_wait_seconds=0.0,
                    max_retries=1,
                )

    def test_assert_swa_divergence_observed_raises_when_no_verify_lag(self) -> None:
        equal_verify_line = SwaDivergenceLog(
            forward_ct=100,
            verify_full=5000,
            verify_swa=5000,
            swa_full_idx_divergence=200,
            swa_out_of_window_tokens=8192,
        ).format()
        harness, patcher = self._make_harness(equal_verify_line + "\n")
        with patcher:
            with self.assertRaisesRegex(AssertionError, "verify_swa=5000"):
                harness.assert_swa_divergence_observed(
                    min_swa_full_idx_divergence=1,
                    require_verify_lag=True,
                    flush_wait_seconds=0.0,
                    max_retries=1,
                )

    def test_assert_swa_divergence_observed_retries_until_stats_emitted(self) -> None:
        sequence = ["", "", "", _GOOD_LINE + "\n", _GOOD_LINE + "\n"]
        harness, patcher = self._make_harness(sequence)
        with patcher:
            harness.assert_swa_divergence_observed(
                min_swa_full_idx_divergence=1,
                require_verify_lag=True,
                flush_wait_seconds=0.0,
                max_retries=5,
            )

    def test_assert_swa_divergence_observed_raises_when_no_stats_emitted(self) -> None:
        harness, patcher = self._make_harness("nothing here\n")
        with patcher:
            with self.assertRaisesRegex(AssertionError, "No kv_canary swa_divergence"):
                harness.assert_swa_divergence_observed(
                    min_swa_full_idx_divergence=1,
                    require_verify_lag=True,
                    flush_wait_seconds=0.0,
                    max_retries=2,
                )

    def test_assert_swa_divergence_observed_catches_zero_swa_full_idx_divergence(
        self,
    ) -> None:
        zero_divergence_line = SwaDivergenceLog(
            forward_ct=200,
            verify_full=10000,
            verify_swa=2000,
            swa_full_idx_divergence=0,
            swa_out_of_window_tokens=8192,
        ).format()
        harness, patcher = self._make_harness(zero_divergence_line + "\n")
        with patcher:
            with self.assertRaisesRegex(AssertionError, "swa_full_idx_divergence=0"):
                harness.assert_swa_divergence_observed(
                    min_swa_full_idx_divergence=1,
                    require_verify_lag=True,
                    flush_wait_seconds=0.0,
                    max_retries=1,
                )


if __name__ == "__main__":
    unittest.main()
