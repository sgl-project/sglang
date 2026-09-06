"""CPU unit tests for async invariant probes."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import async_probe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


@contextmanager
def probe_env(*, async_assert: bool, sanitize: bool = False):
    with (
        patch.object(envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", return_value=async_assert),
        patch.object(envs.SGLANG_SANITIZE_NAN_LOGITS, "get", return_value=sanitize),
    ):
        yield


class TestAsyncProbe(CustomTestCase):
    def test_disabled_assertions_are_noops(self):
        probes = [
            lambda: async_probe.maybe_assert_async(
                torch.tensor(False), "disabled assertion"
            ),
            lambda: async_probe.maybe_detect_nan(
                torch.tensor([float("nan")]), "disabled NaN check"
            ),
            lambda: async_probe.maybe_detect_inf(
                torch.tensor([float("inf")]), "disabled Inf check"
            ),
            lambda: async_probe.maybe_detect_in_closed_range(
                torch.tensor([-1.0]), 0.0, 1.0, "disabled range check"
            ),
            lambda: async_probe.maybe_detect_oob(
                torch.tensor([-1, 4]), 0, 4, "disabled OOB check"
            ),
            lambda: async_probe.maybe_detect_page_aligned(
                torch.tensor([3]), 2, "disabled alignment check"
            ),
        ]

        with probe_env(async_assert=False):
            for probe in probes:
                with self.subTest(probe=probe):
                    probe()

    def test_nan_and_inf_detection(self):
        with probe_env(async_assert=True):
            async_probe.maybe_detect_nan(torch.tensor([0.0, 1.0]), "activations")
            async_probe.maybe_detect_inf(torch.tensor([0.0, 1.0]), "activations")

            with self.assertRaisesRegex(RuntimeError, "NaN detected! activations"):
                async_probe.maybe_detect_nan(
                    torch.tensor([0.0, float("nan")]), "activations"
                )
            with self.assertRaisesRegex(RuntimeError, "Inf detected! activations"):
                async_probe.maybe_detect_inf(
                    torch.tensor([0.0, float("inf")]), "activations"
                )

    def test_closed_range_includes_both_boundaries(self):
        with probe_env(async_assert=True):
            async_probe.maybe_detect_in_closed_range(
                torch.tensor([-2.0, 0.0, 3.0]), -2.0, 3.0, "scores"
            )

            with self.assertRaisesRegex(RuntimeError, "outside"):
                async_probe.maybe_detect_in_closed_range(
                    torch.tensor([-2.01]), -2.0, 3.0, "scores"
                )
            with self.assertRaisesRegex(RuntimeError, "outside"):
                async_probe.maybe_detect_in_closed_range(
                    torch.tensor([3.01]), -2.0, 3.0, "scores"
                )

    def test_oob_uses_inclusive_low_and_exclusive_high(self):
        with probe_env(async_assert=True):
            async_probe.maybe_detect_oob(
                torch.tensor([0, 4]), low=0, high=5, msg="token ids"
            )

            with self.assertRaisesRegex(RuntimeError, "index < 0"):
                async_probe.maybe_detect_oob(
                    torch.tensor([-1, 4]), low=0, high=5, msg="token ids"
                )
            with self.assertRaisesRegex(RuntimeError, "index >= 5"):
                async_probe.maybe_detect_oob(
                    torch.tensor([0, 5]), low=0, high=5, msg="token ids"
                )

    def test_page_alignment_and_skip_cases(self):
        with probe_env(async_assert=True):
            async_probe.maybe_detect_page_aligned(
                torch.tensor([0, 4, 8]), page_size=4, msg="slots"
            )
            async_probe.maybe_detect_page_aligned(None, page_size=4, msg="slots")
            async_probe.maybe_detect_page_aligned(
                torch.tensor([], dtype=torch.int64), page_size=4, msg="slots"
            )
            async_probe.maybe_detect_page_aligned(
                torch.tensor([3]), page_size=1, msg="slots"
            )

            with self.assertRaisesRegex(RuntimeError, "page-misaligned"):
                async_probe.maybe_detect_page_aligned(
                    torch.tensor([0, 5]), page_size=4, msg="slots"
                )

    def test_none_and_empty_tensors_are_ignored(self):
        with probe_env(async_assert=True):
            async_probe.maybe_detect_nan(None, "missing")
            async_probe.maybe_detect_inf(None, "missing")
            async_probe.maybe_detect_in_closed_range(None, 0.0, 1.0, "missing")
            async_probe.maybe_detect_in_closed_range(
                torch.tensor([]), 0.0, 1.0, "empty"
            )
            async_probe.maybe_detect_oob(None, 0, 1, "missing")
            async_probe.maybe_detect_oob(
                torch.tensor([], dtype=torch.int64), 0, 1, "empty"
            )

    def test_sanitization_disabled_preserves_logits(self):
        logits = torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0])

        with (
            probe_env(async_assert=False, sanitize=False),
            patch.object(async_probe._nan_warner, "check") as warn,
        ):
            async_probe.sanitize_nan_logits(logits, "sampler")

        self.assertTrue(torch.isnan(logits[0]))
        self.assertTrue(torch.isposinf(logits[1]))
        self.assertTrue(torch.isneginf(logits[2]))
        self.assertEqual(logits[3].item(), 2.0)
        warn.assert_not_called()

    def test_sanitization_replaces_non_finite_logits_in_place(self):
        logits = torch.tensor([float("nan"), float("inf"), float("-inf"), 2.0])
        data_ptr = logits.data_ptr()

        with (
            probe_env(async_assert=False, sanitize=True),
            patch.object(async_probe._nan_warner, "check") as warn,
        ):
            async_probe.sanitize_nan_logits(logits, "sampler")

        self.assertEqual(logits.data_ptr(), data_ptr)
        torch.testing.assert_close(
            logits, torch.tensor([-1e30, 1e30, -1e30, 2.0]), rtol=0, atol=0
        )
        warn.assert_called_once_with(logits, "sampler")

    def test_hard_assert_precedes_sanitization(self):
        logits = torch.tensor([float("nan")])

        with probe_env(async_assert=True, sanitize=True):
            with self.assertRaisesRegex(RuntimeError, "NaN detected! sampler"):
                async_probe.sanitize_nan_logits(logits, "sampler")

        self.assertTrue(torch.isnan(logits[0]))


if __name__ == "__main__":
    unittest.main()
