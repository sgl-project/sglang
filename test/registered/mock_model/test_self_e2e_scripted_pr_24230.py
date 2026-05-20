"""Scripted-PR regression self-e2e for #24230."""

from __future__ import annotations

import pytest

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="scripted regression scenario for PR #24230 is not supported"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_pr_24230_regression_placeholder() -> None:
    engine = SteppableEngine.launch(
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
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_24230_fix=True,
    )
    req = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until(req, n=4)
    engine.assert_no_canary_violations()

    engine.shutdown()
