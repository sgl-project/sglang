"""Timing-jitter self-e2e (type-c): v1 out-of-scope marker."""

from __future__ import annotations

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


def test_skipped_v1_oos() -> None:
    pytest.skip("phase-2 timing jitter infrastructure")
