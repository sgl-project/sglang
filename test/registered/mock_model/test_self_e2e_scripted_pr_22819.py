"""Scripted-PR regression self-e2e for #22819."""

from __future__ import annotations

import pytest

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="requires force_preempt API which is not currently implemented"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_pr_22819_regression_placeholder() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_22819_fix=False,
    )
    req = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until(req, n=4)
    violations = engine.canary_violations()

    assert len(violations) > 0
    engine.shutdown()


def test_pr_22819_with_fix_placeholder() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_22819_fix=True,
    )
    req = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until(req, n=4)
    engine.assert_no_canary_violations()

    engine.shutdown()
