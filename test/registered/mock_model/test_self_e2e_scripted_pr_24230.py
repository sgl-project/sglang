"""Scripted-PR regression self-e2e for #24230."""

from __future__ import annotations

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(reason="MockEngine harness not yet implemented.")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_pr_24230_regression_placeholder() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_24230_fix=False,
    )
    req = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until(req, n=4)
    violations = engine.canary_violations()

    assert len(violations) > 0
    engine.shutdown()


def test_pr_24230_with_fix_placeholder() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_24230_fix=True,
    )
    req = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until(req, n=4)
    engine.assert_no_canary_violations()

    engine.shutdown()
