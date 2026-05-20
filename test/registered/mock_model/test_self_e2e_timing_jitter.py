"""Timing-jitter self-e2e."""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="requires timing jitter fuzzer which is not currently implemented"
)


def test_timing_jitter() -> None:
    pass
