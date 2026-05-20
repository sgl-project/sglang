"""Timing-jitter self-e2e: placeholder until the jitter-measurement harness is implemented."""

from __future__ import annotations

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="MockEngine harness not yet implemented.")


def test_timing_jitter_not_yet_implemented() -> None:
    pytest.skip("timing jitter measurement infrastructure not yet implemented")
