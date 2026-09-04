"""Unit tests for srt/utils/invariants.py -- CPU, no server.

Covers the (bucket x level) crash matrix, the unconditional data layer, the
throttled reporter, and self-registration / injection-coverage.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.utils import invariants as ic
from sglang.srt.utils.invariants import (
    Bucket,
    Finite,
    InRange,
    Invariant,
    expect,
    registered_invariants,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# "test." namespace so the coverage meta-test isolates these from real invariants.
_SOFTEN = Invariant("test.soften", Bucket.SOFTEN, Finite(), recover=torch.nan_to_num)
_GUARD = Invariant("test.guard", Bucket.GUARD, Finite(), recover=torch.nan_to_num)
_FATAL_C = Invariant("test.fatal_containable", Bucket.FATAL_CONTAINABLE, InRange(0, 10))
_FATAL_U = Invariant(
    "test.fatal_uncontainable", Bucket.FATAL_UNCONTAINABLE, InRange(0, 10)
)

# Injection manifest: every "test." invariant must have a test that triggers it.
_INJECTION_COVERAGE = {
    "test.soften": "test_soften_never_crashes",
    "test.guard": "test_guard_crashes_in_strict",
    "test.fatal_containable": "test_fatal_containable_crashes_in_strict",
    "test.fatal_uncontainable": "test_fatal_uncontainable_crashes_in_warn",
}


class TestInvariants(CustomTestCase):
    def _strict(self):
        return envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.STRICT))

    def _warn(self):
        return envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.WARN))

    def _off(self):
        return envs.SGLANG_INVARIANT_CHECK.override(int(InvariantCheckLevel.OFF))

    def test_crash_matrix(self):
        L = InvariantCheckLevel
        expected = {
            (Bucket.SOFTEN, L.OFF): False,
            (Bucket.SOFTEN, L.WARN): False,
            (Bucket.SOFTEN, L.STRICT): False,
            (Bucket.GUARD, L.OFF): False,
            (Bucket.GUARD, L.WARN): False,
            (Bucket.GUARD, L.STRICT): True,
            (Bucket.FATAL_CONTAINABLE, L.OFF): False,
            (Bucket.FATAL_CONTAINABLE, L.WARN): False,
            (Bucket.FATAL_CONTAINABLE, L.STRICT): True,
            (Bucket.FATAL_UNCONTAINABLE, L.OFF): False,
            (Bucket.FATAL_UNCONTAINABLE, L.WARN): True,
            (Bucket.FATAL_UNCONTAINABLE, L.STRICT): True,
        }
        for (bucket, level), want in expected.items():
            self.assertEqual(ic._crashes(bucket, level), want, f"{bucket} @ {level}")

    def test_recover_applied_even_when_off(self):
        bad = torch.tensor([[float("nan"), 1.0]])
        with self._off():
            with mock.patch.object(torch, "_assert_async") as m:
                out = expect(_GUARD, bad.clone())
        self.assertTrue(torch.isfinite(out).all())
        m.assert_not_called()

    def test_soften_never_crashes(self):
        bad = torch.tensor([[float("nan"), 1.0]])
        with self._strict():
            with mock.patch.object(torch, "_assert_async") as m:
                out = expect(_SOFTEN, bad.clone())
        m.assert_not_called()
        self.assertTrue(torch.isfinite(out).all())

    def test_guard_crashes_in_strict(self):
        bad = torch.tensor([[float("nan"), 1.0]])
        with self._strict():
            with mock.patch.object(torch, "_assert_async") as m:
                expect(_GUARD, bad.clone())
        m.assert_called_once()
        (cond, _msg), _ = m.call_args
        self.assertFalse(bool(cond))

    def test_fatal_containable_crashes_in_strict(self):
        bad = torch.tensor([5, 20], dtype=torch.int64)  # 20 is out of [0, 10)
        with self._strict():
            with mock.patch.object(torch, "_assert_async") as m:
                expect(_FATAL_C, bad)
        m.assert_called_once()

    def test_fatal_uncontainable_crashes_in_warn(self):
        bad = torch.tensor([5, 20], dtype=torch.int64)  # 20 is out of [0, 10)
        with self._warn():
            with mock.patch.object(torch, "_assert_async") as m:
                expect(_FATAL_U, bad)
        m.assert_called_once()

    def test_warn_counts_and_logs(self):
        bad = torch.tensor([[float("nan"), 1.0]])
        with self._warn():
            with mock.patch.object(torch, "_assert_async") as m:
                with self.assertLogs(ic.logger, level="WARNING") as cap:
                    expect(_GUARD, bad.clone())
        m.assert_not_called()
        self.assertTrue(any("test.guard" in line for line in cap.output))

    def test_registry_and_validation(self):
        reg = registered_invariants()
        for name in _INJECTION_COVERAGE:
            self.assertIn(name, reg)

        with self.assertRaises(ValueError):  # duplicate name
            Invariant("test.guard", Bucket.GUARD, Finite(), recover=torch.nan_to_num)

        with self.assertRaises(ValueError):  # uncontainable FATAL forbids recover
            Invariant(
                "test.bad",
                Bucket.FATAL_UNCONTAINABLE,
                Finite(),
                recover=torch.nan_to_num,
            )

    def test_injection_coverage_meta(self):
        """Every registered test-namespace invariant must have an injection test.

        A later change broadens this filter to all namespaces so no real
        GUARD/SOFTEN ships without a triggering test.
        """
        uncovered = [
            name
            for name in registered_invariants()
            if name.startswith("test.") and name not in _INJECTION_COVERAGE
        ]
        self.assertEqual(
            uncovered, [], f"invariants without an injection test: {uncovered}"
        )


if __name__ == "__main__":
    unittest.main()
